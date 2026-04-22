#!/usr/bin/env python3
"""
Deteccion multi-persona de "mano con objeto" usando:
- YOLO Pose para personas + keypoints
- Clasificador VLM/CLIP inyectado (ver test_new_handobject_*.py)

Flujo:
1) Detecta personas y keypoints.
2) Genera crop por muñeca (con apoyo de codo si existe).
3) Clasifica cada crop: mano con objeto vs mano vacia.
4) Aplica histeresis temporal por track/persona.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import re
import statistics
import socket
import sys
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import deque

_APPROACHES_DIR = Path(__file__).resolve().parent
if str(_APPROACHES_DIR) not in sys.path:
    sys.path.insert(0, str(_APPROACHES_DIR))

from yolo_weights_download import YOLO_PT_DOWNLOAD_URL, download_yolo_pt_if_missing


def _terminal_frame_progress_line(label: str, video_path: Path, frame_i: int, total_frames: int) -> None:
    """Una linea con \\r: misma logica que la barra usada con --save (vista en headless / orquestador)."""
    if total_frames > 0:
        pct = min(100.0, (100.0 * frame_i) / max(1, total_frames))
        bar_w = 28
        fill = int(round((pct / 100.0) * bar_w))
        bar = ("#" * fill) + ("-" * (bar_w - fill))
        print(
            f"\r{label} {video_path.name} [{bar}] {frame_i}/{total_frames} ({pct:5.1f}%)",
            end="",
            flush=True,
        )
    else:
        print(
            f"\r{label} {video_path.name} frames={frame_i}",
            end="",
            flush=True,
        )

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    import transformers
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "No se pudo importar transformers. Instala dependencias: pip install transformers pillow"
    ) from e


KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
PERSON_CLASS_ID = 0
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
KP_NAMES = {
    0: "nariz",
    1: "ojo_izq",
    2: "ojo_der",
    3: "oreja_izq",
    4: "oreja_der",
    5: "hombro_izq",
    6: "hombro_der",
    7: "codo_izq",
    8: "codo_der",
    9: "muneca_izq",
    10: "muneca_der",
    11: "cadera_izq",
    12: "cadera_der",
    13: "rodilla_izq",
    14: "rodilla_der",
    15: "tobillo_izq",
    16: "tobillo_der",
}
EXCLUDED_KP_FOR_DROP = {KP_LEFT_WRIST, KP_RIGHT_WRIST}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_resolve_path(s: str) -> str:
    p = str(s).strip()
    if not p:
        return ""
    try:
        return str(Path(p).expanduser().resolve())
    except OSError:
        return p


def namespace_to_experiment_args(ns: argparse.Namespace) -> dict[str, Any]:
    """Argumentos CLI con rutas resueltas a absolutas cuando aplica."""
    d = vars(ns).copy()
    path_keys = (
        "video",
        "videos",
        "save",
        "output",
        "vlm_model",
        "pose_weights",
        "personal_weights",
        "roi_region",
    )
    for k in path_keys:
        if k in d and isinstance(d[k], str):
            d[k] = _safe_resolve_path(d[k]) if d[k].strip() else ""
    return d


class NvidiaGpuPoller:
    """Muestrea utilizacion y memoria NVIDIA durante el bucle de video (opcional)."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = max(0.1, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            row = _nvidia_smi_sample()
            if row is not None:
                self.samples.append(row)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _nvidia_smi_sample() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None
    try:
        return {
            "gpu_index": int(parts[0]),
            "gpu_name": parts[1],
            "utilization_gpu_percent": float(parts[2]),
            "memory_used_mib": float(parts[3]),
            "memory_total_mib": float(parts[4]),
            "temperature_c": float(parts[5]) if parts[5] != "[N/A]" else None,
            "power_draw_w": float(parts[6]) if parts[6] != "[N/A]" else None,
            "t_wall": time.perf_counter(),
        }
    except (ValueError, IndexError):
        return None


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


@dataclass
class HandState:
    prob: float = 0.0
    raw_prob: float = 0.0
    raw_high_count: int = 0
    weak_count: int = 0
    raw_low_count: int = 0
    on_count: int = 0
    off_count: int = 0
    holding: bool = False
    last_seen_frame: int = 0
    last_crop_box: tuple[int, int, int, int] | None = None


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    last_seen_frame: int
    hands: dict[str, HandState] = field(
        default_factory=lambda: {"left": HandState(), "right": HandState()}
    )
    prompt_yes: bool = False
    last_object_xy: tuple[int, int] | None = None
    object_traj: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=12))
    active_object_box: tuple[int, int, int, int] | None = None
    pending_drop_steps: int = 0
    drop_info_text: str = ""
    drop_info_frames_left: int = 0


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    a_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    b_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1e-6, a_area + b_area - inter)


def build_parser(
    *,
    description: str = "Deteccion mano-objeto con YOLO Pose + clasificador VLM por crop.",
    default_vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    vlm_model_help: str = "HF id o carpeta local del modelo VLM.",
    default_vlm_prompt: str | None = None,
) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--video", default="", help="Ruta a un video.")
    ap.add_argument("--videos", default="", help="Carpeta con videos a procesar.")
    ap.add_argument("--pose-weights", default="yolo11n-pose.pt", help="Pesos YOLO pose.")
    ap.add_argument(
        "--vlm-model",
        default=default_vlm_model,
        help=vlm_model_help,
    )
    _vlm_prompt_default = default_vlm_prompt or (
        "Look at this image. Is the person clearly holding an object in either hand? "
        "Answer with one word only: YES or NO."
    )
    ap.add_argument(
        "--vlm-prompt",
        default=_vlm_prompt_default,
        help="Prompt de clasificacion SÍ/NO.",
    )
    ap.add_argument(
        "--per-hand-fast",
        action="store_true",
        help="Modo rapido por mano: clasifica una mano y solo evalua la otra en zona gris.",
    )
    ap.add_argument(
        "--fast-gray-zone",
        type=float,
        default=0.20,
        help="Margen de duda para lanzar 2a inferencia en --per-hand-fast.",
    )
    ap.add_argument("--device", default="cpu", help="cpu o cuda:0")
    ap.add_argument(
        "--no-tensorrt",
        action="store_true",
        help="No usar TensorRT (.engine junto al .pt) aunque exista; evita fallos si CUDA/TRT no van.",
    )
    ap.add_argument("--stride", type=int, default=2, help="Evalua cada N frames.")
    ap.add_argument("--display", default="1280x720", choices=["1280x720", "1920x1080"])
    ap.add_argument(
        "--save",
        default="",
        help="Ruta de salida para modo --video. Si se indica, guarda video y no abre ventana.",
    )
    ap.add_argument(
        "--display-sync-fps",
        action="store_true",
        help=(
            "Solo ventana (sin --save): tras cada frame espera hasta alinear ~1/FPS del clip "
            "(metadata); si no, la vista va al ritmo del procesamiento (acelerada si la GPU va sobrada)."
        ),
    )
    ap.add_argument("--wrist-conf-th", type=float, default=0.35)
    ap.add_argument("--elbow-conf-th", type=float, default=0.25)
    ap.add_argument("--crop-size", type=int, default=200, help="Tamano base de crop de mano.")
    ap.add_argument("--crop-min", type=int, default=150)
    ap.add_argument("--crop-max", type=int, default=260)
    ap.add_argument(
        "--crop-mode",
        default="hand",
        choices=["hand", "upper-torso-hands"],
        help=(
            "Modo de crop para clasificacion: hand (muneca+codo) o "
            "upper-torso-hands (torso superior + mano de cada lado)."
        ),
    )
    ap.add_argument("--hold-frames", type=int, default=3)
    ap.add_argument("--drop-frames", type=int, default=3)
    ap.add_argument(
        "--robbery-th",
        type=float,
        default=0.80,
        help="Umbral de robo: se considera YES solo si probabilidad > este valor.",
    )
    ap.add_argument(
        "--robbery-on-frames",
        type=int,
        default=3,
        help="Frames consecutivos con prob > --robbery-th para activar robo.",
    )
    ap.add_argument(
        "--robbery-off-frames",
        type=int,
        default=3,
        help="Frames consecutivos con prob < --robbery-th para desactivar robo.",
    )
    ap.add_argument(
        "--raw-on-th",
        type=float,
        default=-1.0,
        help="Si >0, activa compuerta de entrada por probabilidad raw (criterio estricto).",
    )
    ap.add_argument(
        "--raw-on-frames",
        type=int,
        default=0,
        help="Frames consecutivos raw > --raw-on-th para activar 'holding' (0 desactiva).",
    )
    ap.add_argument("--yes-th", type=float, default=0.55, help="Umbral probabilidad mano con objeto.")
    ap.add_argument(
        "--force-drop-th",
        type=float,
        default=0.42,
        help="Si probabilidad cae por debajo de este valor de forma sostenida, forzar soltado.",
    )
    ap.add_argument(
        "--force-drop-frames",
        type=int,
        default=3,
        help="Frames consecutivos por debajo de --force-drop-th para forzar soltado.",
    )
    ap.add_argument(
        "--raw-drop-th",
        type=float,
        default=0.40,
        help="Umbral de probabilidad raw para cortar estado pegajoso.",
    )
    ap.add_argument(
        "--raw-drop-frames",
        type=int,
        default=2,
        help="Ciclos consecutivos con raw<th para forzar soltado.",
    )
    ap.add_argument(
        "--no-hands-drop-steps",
        type=int,
        default=2,
        help="Ciclos de inferencia sin manos visibles para forzar SIN OBJETO COGIDO.",
    )
    ap.add_argument("--iou-track-th", type=float, default=0.30)
    ap.add_argument("--max-track-lost", type=int, default=20)
    ap.add_argument(
        "--drop-zone-mode",
        default="keypoint-first",
        choices=["keypoint-first", "segment-mix"],
        help="Modo de zona de soltado: keypoint-first (recomendado) o segment-mix.",
    )
    ap.add_argument(
        "--drop-window-steps",
        type=int,
        default=2,
        help="Ciclos extra tras el primer NO para estimar mejor punto de guardado.",
    )
    ap.add_argument(
        "--traj-smooth-len",
        type=int,
        default=6,
        help="Numero de puntos recientes para suavizar trayectoria del objeto.",
    )
    ap.add_argument(
        "--roi-region",
        default="",
        help="ROI: ruta a JSON o camera_id (ej. cam_6). Activa deteccion tras salir de ROI.",
    )
    ap.add_argument("--personal-weights", default="yolo11n.pt", help="Pesos YOLO para detectar bolso/mochila.")
    ap.add_argument("--personal-conf", type=float, default=0.35, help="Confianza minima para bolso/mochila.")
    ap.add_argument("--personal-stride", type=int, default=2, help="Evaluar bolso/mochila cada N ciclos de inferencia.")
    ap.add_argument(
        "--personal-near-px",
        type=float,
        default=80.0,
        help="Si el punto de guardado cae a <= este umbral de bolso/mochila, prioriza zona 'mochila'.",
    )
    ap.add_argument(
        "--output",
        default="",
        help=(
            "Carpeta donde guardar experiment_<video>_<fecha>.json, o ruta a un .json concreto "
            "(solo con un solo --video; se crean directorios padre si hace falta)."
        ),
    )
    return ap


def parse_args(ap: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    if ap is None:
        ap = build_parser()
    return ap.parse_args()


def build_hand_crop(
    frame: np.ndarray,
    person_box: tuple[int, int, int, int],
    wrist_xy: tuple[int, int],
    elbow_xy: tuple[int, int] | None,
    crop_size: int,
    crop_min: int,
    crop_max: int,
    crop_mode: str = "hand",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    px1, py1, px2, py2 = person_box
    if crop_mode == "upper-torso-hands":
        person_w = max(1, px2 - px1)
        person_h = max(1, py2 - py1)
        wx, _ = wrist_xy
        side_sign = -1 if wx <= (px1 + px2) // 2 else 1
        half_w = int(np.clip(person_w * 0.38, crop_min, crop_max * 2))
        torso_h = int(np.clip(person_h * 0.68, crop_min, crop_max * 2))
        cx = int(wx + side_sign * 0.08 * person_w)
        cy = int(py1 + 0.35 * person_h)
        x1 = cx - half_w
        x2 = cx + half_w
        y1 = cy - torso_h // 2
        y2 = cy + torso_h // 2
        x1 = max(px1, x1)
        y1 = max(py1, y1)
        x2 = min(px2, x2)
        y2 = min(py2, y2)
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    wx, wy = wrist_xy
    if elbow_xy is not None:
        ex, ey = elbow_xy
        side = int(np.clip(1.2 * math.hypot(wx - ex, wy - ey), crop_min, crop_max))
    else:
        side = int(np.clip(crop_size, crop_min, crop_max))
    x1 = int(wx - side)
    y1 = int(wy - side)
    x2 = int(wx + side)
    y2 = int(wy + side)
    x1 = max(px1, x1)
    y1 = max(py1, y1)
    x2 = min(px2, x2)
    y2 = min(py2, y2)
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def load_roi_polygon(roi_region: str) -> np.ndarray:
    raw = Path(roi_region)
    stem = raw.name if raw.suffix else raw.name
    json_name = stem if str(stem).endswith(".json") else f"{stem}.json"
    candidates: list[Path] = []
    if raw.exists():
        candidates.append(raw)
    else:
        candidates.append(Path("roi_regions") / json_name)
        _here = Path(__file__).resolve().parent
        candidates.append(_here / "roi_regions" / json_name)
        candidates.append(_here.parent / "roi_regions" / json_name)
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise FileNotFoundError(f"No existe ROI: {roi_region} (buscado en: {candidates})")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    regions = data.get("regions") or []
    if not regions:
        raise RuntimeError(f"ROI sin regiones: {p}")
    poly = np.array(regions[0].get("polygon") or [], dtype=np.float32)
    if len(poly) < 3:
        raise RuntimeError(f"ROI invalida (menos de 3 puntos): {p}")
    return poly


def scale_roi_polygon(poly_1080: np.ndarray, w: int, h: int) -> np.ndarray:
    p = poly_1080.copy()
    p[:, 0] = p[:, 0] * float(w) / 1920.0
    p[:, 1] = p[:, 1] * float(h) / 1080.0
    return p.astype(np.int32)


def nearest_body_point_label(
    point_xy: tuple[int, int],
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    conf_th: float = 0.20,
) -> tuple[str, float]:
    px, py = point_xy
    best_name = "desconocido"
    best_d = float("inf")
    n = min(len(kps_xy), len(kps_conf))
    for i in range(n):
        if i in EXCLUDED_KP_FOR_DROP:
            continue
        if float(kps_conf[i]) < conf_th:
            continue
        kx, ky = float(kps_xy[i][0]), float(kps_xy[i][1])
        d = math.hypot(px - kx, py - ky)
        if d < best_d:
            best_d = d
            best_name = KP_NAMES.get(i, f"kp_{i}")
    if best_d == float("inf"):
        return "desconocido", 0.0
    return best_name, best_d


def _point_to_segment_distance(
    p: tuple[int, int],
    a: tuple[float, float],
    b: tuple[float, float],
    wy: float = 1.8,
) -> float:
    """Distancia punto-segmento con ponderación vertical (y)."""
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    # Escalado anisotrópico para dar más peso a vertical.
    py *= wy
    ay *= wy
    by *= wy
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    den = (abx * abx) + (aby * aby)
    if den <= 1e-6:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, ((apx * abx) + (apy * aby)) / den))
    qx = ax + (t * abx)
    qy = ay + (t * aby)
    return math.hypot(px - qx, py - qy)


def nearest_body_zone_label(
    point_xy: tuple[int, int],
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    conf_th: float = 0.20,
    wy: float = 1.8,
) -> tuple[str, float]:
    """
    Zona corporal más cercana usando:
    - puntos (keypoints sin muñecas)
    - segmentos corporales (hombro-codo, codo-muñeca, cadera-rodilla, rodilla-tobillo, etc.)
    Distancia ponderada en Y para mejorar coherencia vertical.
    """
    px, py = point_xy
    best_name = "desconocido"
    best_d = float("inf")
    n = min(len(kps_xy), len(kps_conf))

    def valid(i: int) -> bool:
        return 0 <= i < n and float(kps_conf[i]) >= conf_th

    # 1) Distancia a puntos (sin muñecas).
    for i in range(n):
        if i in EXCLUDED_KP_FOR_DROP:
            continue
        if not valid(i):
            continue
        kx, ky = float(kps_xy[i][0]), float(kps_xy[i][1])
        d = math.hypot(px - kx, (py - ky) * wy)
        if d < best_d:
            best_d = d
            best_name = KP_NAMES.get(i, f"kp_{i}")

    # 2) Distancia a segmentos.
    segments = [
        (5, 7, "brazo_izq"),
        (7, 9, "antebrazo_izq"),
        (6, 8, "brazo_der"),
        (8, 10, "antebrazo_der"),
        (11, 13, "muslo_izq"),
        (13, 15, "pierna_izq"),
        (12, 14, "muslo_der"),
        (14, 16, "pierna_der"),
        (11, 12, "cintura"),
        (5, 11, "torso_izq"),
        (6, 12, "torso_der"),
    ]
    for i1, i2, lbl in segments:
        if not (valid(i1) and valid(i2)):
            continue
        a = (float(kps_xy[i1][0]), float(kps_xy[i1][1]))
        b = (float(kps_xy[i2][0]), float(kps_xy[i2][1]))
        d = _point_to_segment_distance(point_xy, a, b, wy=wy)
        if d < best_d:
            best_d = d
            best_name = lbl

    if best_d == float("inf"):
        return "desconocido", 0.0
    return best_name, best_d


def nearest_body_keypoint_label(
    point_xy: tuple[int, int],
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    conf_th: float = 0.20,
    wy: float = 1.8,
) -> tuple[str, float]:
    px, py = point_xy
    best_name = "desconocido"
    best_d = float("inf")
    n = min(len(kps_xy), len(kps_conf))
    for i in range(n):
        if i in EXCLUDED_KP_FOR_DROP:
            continue
        if float(kps_conf[i]) < conf_th:
            continue
        kx, ky = float(kps_xy[i][0]), float(kps_xy[i][1])
        d = math.hypot(px - kx, (py - ky) * wy)
        if d < best_d:
            best_d = d
            best_name = KP_NAMES.get(i, f"kp_{i}")
    if best_d == float("inf"):
        return "desconocido", 0.0
    return best_name, best_d


def estimate_drop_xy_from_traj(traj: deque[tuple[int, int]], smooth_len: int) -> tuple[int, int] | None:
    if not traj:
        return None
    pts = list(traj)[-max(1, smooth_len) :]
    if not pts:
        return None
    # Suavizado temporal ponderado (más peso a puntos recientes).
    w_sum = 0.0
    x_acc = 0.0
    y_acc = 0.0
    for i, (x, y) in enumerate(pts, start=1):
        w = float(i)
        w_sum += w
        x_acc += w * float(x)
        y_acc += w * float(y)
    x_avg = x_acc / max(1e-6, w_sum)
    y_avg = y_acc / max(1e-6, w_sum)
    # Mezclar con el punto más bajo para capturar "bajada" al soltar.
    x_low, y_low = max(pts, key=lambda p: p[1])
    x_out = int(round((x_avg + float(x_low)) * 0.5))
    y_out = int(round((y_avg + float(y_low)) * 0.5))
    return (x_out, y_out)


def _point_to_bbox_distance(p: tuple[int, int], b: tuple[int, int, int, int]) -> float:
    px, py = p
    x1, y1, x2, y2 = b
    dx = max(float(x1 - px), 0.0, float(px - x2))
    dy = max(float(y1 - py), 0.0, float(py - y2))
    return math.hypot(dx, dy)


def _personal_label_to_zone(label: str) -> str:
    l = str(label).lower()
    if "handbag" in l or "purse" in l:
        return "bolso de mano"
    if "backpack" in l:
        return "mochila"
    if "suitcase" in l:
        return "maleta"
    if "bag" in l:
        return "bolso de colgar"
    if "cart" in l or "trolley" in l:
        return "carrito de supermercado"
    return "objeto personal"


def detect_personal_objects(
    frame: np.ndarray,
    yolo_model: YOLO,
    conf_th: float,
    *,
    predict_device: int | str | None = None,
) -> list[tuple[int, int, int, int, str, float]]:
    out: list[tuple[int, int, int, int, str, float]] = []
    infer_kw: dict[str, Any] = {"conf": conf_th, "verbose": False}
    if predict_device is not None:
        infer_kw["device"] = predict_device
    res = yolo_model(frame, **infer_kw)
    r0 = res[0] if res else None
    if r0 is None or r0.boxes is None or len(r0.boxes) == 0:
        return out
    names = r0.names if hasattr(r0, "names") else {}
    keep_tokens = ("backpack", "handbag", "purse", "bag", "suitcase")
    for box in r0.boxes:
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        if isinstance(names, dict):
            raw_label = names.get(cls_id, f"cls_{cls_id}")
        elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            raw_label = names[cls_id]
        else:
            raw_label = f"cls_{cls_id}"
        label = str(raw_label).lower().strip()
        if not any(tok in label for tok in keep_tokens):
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        out.append((x1, y1, x2, y2, label, conf))
    return out


def compute_drop_zone(
    drop_xy: tuple[int, int] | None,
    kps_xy: np.ndarray,
    kps_conf: np.ndarray,
    mode: str,
    personal_boxes: list[tuple[int, int, int, int, str, float]] | None = None,
    personal_near_px: float = 80.0,
) -> tuple[str, float]:
    if drop_xy is None:
        return "desconocido", 0.0
    if personal_boxes:
        best_d = float("inf")
        best_lbl = ""
        for b in personal_boxes:
            d = _point_to_bbox_distance(drop_xy, (b[0], b[1], b[2], b[3]))
            if d < best_d:
                best_d = d
                best_lbl = str(b[4])
        if best_d <= float(personal_near_px):
            return _personal_label_to_zone(best_lbl), float(best_d)
    if mode == "keypoint-first":
        kp_name, kp_dist = nearest_body_keypoint_label(drop_xy, kps_xy, kps_conf)
    else:
        kp_name, kp_dist = nearest_body_zone_label(drop_xy, kps_xy, kps_conf)
    if kp_name == "desconocido":
        kp_name, kp_dist = nearest_body_zone_label(drop_xy, kps_xy, kps_conf)
    return kp_name, kp_dist


def extract_people_and_hands(
    frame: np.ndarray,
    pose_model: YOLO,
    wrist_conf_th: float,
    elbow_conf_th: float,
    *,
    predict_device: int | str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    infer_kw: dict[str, Any] = {"verbose": False}
    if predict_device is not None:
        infer_kw["device"] = predict_device
    res = pose_model(frame, **infer_kw)
    r0 = res[0] if res else None
    if r0 is None or r0.boxes is None or len(r0.boxes) == 0 or r0.keypoints is None:
        return out

    boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
    cls = r0.boxes.cls.cpu().numpy() if r0.boxes.cls is not None else np.zeros((len(boxes_xyxy),), dtype=np.float32)
    kps_xy_all = r0.keypoints.xy.cpu().numpy()
    if r0.keypoints.conf is not None:
        kps_conf_all = r0.keypoints.conf.cpu().numpy()
    else:
        kps_conf_all = np.ones((kps_xy_all.shape[0], kps_xy_all.shape[1]), dtype=np.float32)

    h, w = frame.shape[:2]
    for i, box in enumerate(boxes_xyxy):
        if int(cls[i]) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        person_box = clamp_box(x1, y1, x2, y2, w, h)
        kxy = kps_xy_all[i]
        kcf = kps_conf_all[i]

        hands: dict[str, dict[str, Any]] = {}
        for side, wi, ei in (("left", KP_LEFT_WRIST, KP_LEFT_ELBOW), ("right", KP_RIGHT_WRIST, KP_RIGHT_ELBOW)):
            if wi >= len(kxy):
                continue
            wc = float(kcf[wi]) if wi < len(kcf) else 0.0
            if wc < wrist_conf_th:
                continue
            wx, wy = kxy[wi]
            wrist = (int(np.clip(wx, 0, w - 1)), int(np.clip(wy, 0, h - 1)))
            elbow = None
            if ei < len(kxy):
                ec = float(kcf[ei]) if ei < len(kcf) else 0.0
                if ec >= elbow_conf_th:
                    ex, ey = kxy[ei]
                    elbow = (int(np.clip(ex, 0, w - 1)), int(np.clip(ey, 0, h - 1)))
            hands[side] = {"wrist": wrist, "elbow": elbow}

        out.append({"person_box": person_box, "hands": hands, "kps_xy": kxy, "kps_conf": kcf})
    return out


def assign_tracks(
    detections: list[dict[str, Any]],
    tracks: dict[int, TrackState],
    frame_i: int,
    next_track_id: int,
    iou_th: float,
    max_lost: int,
) -> int:
    det_used = [False] * len(detections)
    active_track_ids = [tid for tid, tr in tracks.items() if (frame_i - tr.last_seen_frame) <= max_lost]

    for tid in active_track_ids:
        tr = tracks[tid]
        best_i = -1
        best_iou = 0.0
        for i, det in enumerate(detections):
            if det_used[i]:
                continue
            iou = bbox_iou(tr.bbox, det["person_box"])
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0 and best_iou >= iou_th:
            det = detections[best_i]
            det["track_id"] = tid
            tr.bbox = det["person_box"]
            tr.last_seen_frame = frame_i
            det_used[best_i] = True

    for i, det in enumerate(detections):
        if det_used[i]:
            continue
        tid = next_track_id
        next_track_id += 1
        tracks[tid] = TrackState(track_id=tid, bbox=det["person_box"], last_seen_frame=frame_i)
        det["track_id"] = tid

    stale_ids = [tid for tid, tr in tracks.items() if (frame_i - tr.last_seen_frame) > max_lost]
    for tid in stale_ids:
        del tracks[tid]
    return next_track_id


def update_temporal_state(
    hs: HandState,
    yes_prob: float,
    frame_i: int,
    yes_th: float,
    hold_frames: int,
    drop_frames: int,
    force_drop_th: float,
    force_drop_frames: int,
    raw_drop_th: float,
    raw_drop_frames: int,
    raw_on_th: float = -1.0,
    raw_on_frames: int = 0,
) -> None:
    hs.last_seen_frame = frame_i
    hs.raw_prob = yes_prob
    hs.prob = 0.65 * hs.prob + 0.35 * yes_prob
    use_raw_on_gate = (raw_on_th > 0.0) and (raw_on_frames > 0)
    if use_raw_on_gate:
        # Modo estricto: entrada/salida gobernada por la probabilidad raw de este frame.
        if hs.raw_prob > raw_on_th:
            hs.raw_high_count += 1
            hs.off_count = 0
            hs.on_count = hs.raw_high_count
        elif hs.raw_prob < raw_on_th:
            hs.raw_high_count = 0
            hs.off_count += 1
            hs.on_count = 0
        else:
            # En igualdad exacta no activar ni desactivar.
            hs.raw_high_count = 0
            hs.on_count = 0
            hs.off_count = 0
        is_yes_now = hs.raw_prob > raw_on_th
    else:
        hs.raw_high_count = 0
        is_yes_now = hs.prob >= yes_th

    if not use_raw_on_gate:
        if is_yes_now:
            hs.on_count += 1
            hs.off_count = 0
        else:
            hs.off_count += 1
            hs.on_count = 0
    # Anti-stick: si la evidencia es claramente baja varios ciclos, soltar.
    if hs.prob < force_drop_th:
        hs.weak_count += 1
    else:
        hs.weak_count = 0
    if hs.raw_prob < raw_drop_th:
        hs.raw_low_count += 1
    else:
        hs.raw_low_count = 0
    on_frames_needed = int(raw_on_frames) if use_raw_on_gate else int(hold_frames)
    if not hs.holding and hs.on_count >= max(1, on_frames_needed):
        hs.holding = True
    if hs.holding and hs.off_count >= drop_frames:
        hs.holding = False
    if hs.holding and hs.weak_count >= force_drop_frames:
        hs.holding = False
        hs.on_count = 0
        hs.off_count = max(hs.off_count, drop_frames)
    if hs.holding and hs.raw_low_count >= raw_drop_frames:
        hs.holding = False
        hs.on_count = 0
        hs.off_count = max(hs.off_count, drop_frames)


def resolve_yolo_weights_for_runtime(weights_arg: str, *, allow_tensorrt: bool = True) -> str:
    """
    Si la ruta apunta a un .pt y existe un .engine en el mismo directorio, usar el .engine (TensorRT).
    Si allow_tensorrt es False (p. ej. --device cpu o --no-tensorrt), se mantiene el .pt.
    Busca el .pt en cwd, junto a handobject_shared.py (approaches/) y ruta absoluta.
    Para yolo11n*.pt conocidos, si no existen en esas rutas, los descarga en approaches/.
    """
    ws = str(weights_arg).strip()
    if not ws:
        return ws
    _here = Path(__file__).resolve().parent
    candidates = [
        Path(ws).expanduser(),
        _here / ws,
        Path.cwd() / ws,
    ]
    p = next((c for c in candidates if c.is_file()), None)
    if p is None:
        base_name = Path(ws).name
        if base_name in YOLO_PT_DOWNLOAD_URL:
            dest = _here / base_name
            dl_info = download_yolo_pt_if_missing(dest)
            if dl_info.get("downloaded"):
                print(f"[yolo] Peso descargado automáticamente: {dest}", flush=True)
            if dest.is_file():
                p = dest
    if p is None:
        return ws
    if p.suffix.lower() == ".pt":
        eng = p.with_suffix(".engine")
        if eng.is_file() and allow_tensorrt:
            return str(eng.resolve())
    return str(p.resolve())


def _weights_is_tensorrt(path: str) -> bool:
    return Path(path).suffix.lower() == ".engine"


def yolo_predict_device_for_args(args: argparse.Namespace) -> int | str:
    """
    Dispositivo para YOLO predict. Los .engine (TensorRT) no admiten model.to();
    hay que pasar device= en cada predict.
    """
    d = str(getattr(args, "device", None) or "cpu").strip()
    low = d.lower()
    if low in ("", "cpu"):
        return "cpu"
    if low.startswith("cuda:"):
        try:
            return int(d.split(":", 1)[1])
        except ValueError:
            return 0
    if low == "cuda":
        return 0
    return "cpu"


def run_pipeline(
    args: argparse.Namespace,
    classifier: Any,
    *,
    window_title: str = "hand-object",
    batch_output_suffix: str = "_qwen",
    experiment_backend: str = "",
) -> None:
    if bool(str(args.video).strip()) == bool(str(args.videos).strip()):
        raise RuntimeError("Debes indicar exactamente uno: --video o --videos.")
    predict_dev = yolo_predict_device_for_args(args)
    robbery_th = float(getattr(args, "robbery_th", -1.0))
    robbery_on_frames = max(1, int(getattr(args, "robbery_on_frames", 3)))
    robbery_off_frames = max(1, int(getattr(args, "robbery_off_frames", 3)))
    allow_trt = (
        not bool(getattr(args, "no_tensorrt", False))
        and predict_dev != "cpu"
        and torch.cuda.is_available()
    )
    if not allow_trt:
        if bool(getattr(args, "no_tensorrt", False)):
            print("[yolo] --no-tensorrt: usando .pt (no TensorRT .engine).", flush=True)
        elif predict_dev == "cpu":
            print(
                "[yolo] Dispositivo cpu: usando .pt (no TensorRT .engine aunque exista).",
                flush=True,
            )
    args.pose_weights = resolve_yolo_weights_for_runtime(str(args.pose_weights), allow_tensorrt=allow_trt)
    args.personal_weights = resolve_yolo_weights_for_runtime(str(args.personal_weights), allow_tensorrt=allow_trt)
    pose_trt = _weights_is_tensorrt(str(args.pose_weights))
    personal_trt = _weights_is_tensorrt(str(args.personal_weights))

    # task explícito evita el aviso "Unable to automatically guess model task" de ultralytics
    pose_model = YOLO(str(args.pose_weights), task="pose")
    if args.device and not pose_trt:
        pose_model.to(args.device)
    personal_model = YOLO(str(args.personal_weights), task="detect")
    if args.device and not personal_trt:
        personal_model.to(args.device)

    if args.display == "1280x720":
        view_w, view_h = 1280, 720
    else:
        view_w, view_h = 1920, 1080

    def run_single_video(video_path: Path, save_path: str) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir video: {video_path}")
        fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        writer = None
        if save_path:
            src_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            fourcc = src_fourcc if src_fourcc != 0 else cv2.VideoWriter_fourcc(*"mp4v")
            out_w = src_w if src_w > 0 else view_w
            out_h = src_h if src_h > 0 else view_h
            writer = cv2.VideoWriter(save_path, fourcc, fps_src if fps_src > 0 else 12.0, (out_w, out_h))
            if not writer.isOpened():
                # Fallback robusto si el codec fuente no escribe en salida.
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, fps_src if fps_src > 0 else 12.0, (out_w, out_h))
            if not writer.isOpened():
                raise RuntimeError(f"No se pudo abrir salida de video: {save_path}")
            print(f"[video] guardando salida en: {save_path}")

        tracks: dict[int, TrackState] = {}
        next_track_id = 1
        frame_i = 0
        t0 = time.perf_counter()
        hands_visible_now = False
        no_hands_steps = 0
        roi_poly_1080: np.ndarray | None = None
        if str(args.roi_region).strip():
            roi_poly_1080 = load_roi_polygon(str(args.roi_region).strip())
            print(f"[video] ROI activa: {args.roi_region}")
        roi_hold_steps = max(1, int(round(((fps_src if fps_src > 0 else 12.0) * 0.5) / max(1, args.stride))))
        side_state: dict[str, dict[str, Any]] = {
            "left": {"inside": False, "in_count": 0, "out_count": 0, "active_after_exit": False},
            "right": {"inside": False, "in_count": 0, "out_count": 0, "active_after_exit": False},
        }
        require_roi_rearm = False
        prompt_yes_state = False
        nearest_zone_text = "N/A"
        nearest_zone_frames_left = 0
        personal_boxes: list[tuple[int, int, int, int, str, float]] = []
        personal_tick = 0

        experiment_start_iso = _utc_now_iso()
        experiment_wall_t0 = time.perf_counter()
        vlm_calls: list[dict[str, Any]] = []
        gpu_poller: NvidiaGpuPoller | None = None
        if classifier.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            gpu_poller = NvidiaGpuPoller(0.5)
            gpu_poller.start()

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_i += 1
            t_frame_start = time.perf_counter()
            vis = frame.copy()
            h, w = frame.shape[:2]
            roi_poly_i: np.ndarray | None = None
            if roi_poly_1080 is not None:
                roi_poly_i = scale_roi_polygon(roi_poly_1080, w, h)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [roi_poly_i], (40, 170, 40))
                vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)
                cv2.polylines(vis, [roi_poly_i], True, (40, 220, 40), 2)
                cv2.putText(
                    vis,
                    "ROI",
                    (int(roi_poly_i[:, 0].min()), max(18, int(roi_poly_i[:, 1].min()) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (40, 220, 40),
                    2,
                )

            prev_any_holding = any(
                tr.hands["left"].holding or tr.hands["right"].holding
                for tr in tracks.values()
                if (frame_i - tr.last_seen_frame) <= int(args.max_track_lost)
            )
            if frame_i % max(1, args.stride) == 0:
                detections = extract_people_and_hands(
                    frame=frame,
                    pose_model=pose_model,
                    wrist_conf_th=float(args.wrist_conf_th),
                    elbow_conf_th=float(args.elbow_conf_th),
                    predict_device=predict_dev,
                )
                trigger_roi_rearm_now = False
                hands_visible_now = any(len(det["hands"]) > 0 for det in detections)
                if hands_visible_now:
                    no_hands_steps = 0
                else:
                    no_hands_steps += 1
                # Maquina ROI por lado: activar inferencia al salir de ROI.
                active_sides: set[str] = {"left", "right"}
                if roi_poly_i is not None:
                    inside_now = {"left": False, "right": False}
                    for det in detections:
                        for side, hand in det["hands"].items():
                            wx, wy = hand["wrist"]
                            if cv2.pointPolygonTest(roi_poly_i, (float(wx), float(wy)), False) >= 0:
                                inside_now[side] = True
                    for side in ("left", "right"):
                        st = side_state[side]
                        if inside_now[side]:
                            st["in_count"] += 1
                            st["out_count"] = 0
                            if (not st["inside"]) and st["in_count"] >= roi_hold_steps:
                                st["inside"] = True
                                st["active_after_exit"] = False
                        else:
                            st["out_count"] += 1
                            st["in_count"] = 0
                            if st["inside"] and st["out_count"] >= roi_hold_steps:
                                st["inside"] = False
                                st["active_after_exit"] = True
                                if require_roi_rearm:
                                    require_roi_rearm = False
                    active_sides = {s for s, st in side_state.items() if st["active_after_exit"]}
                can_infer = (not require_roi_rearm) and (len(active_sides) > 0)
                if can_infer:
                    personal_tick += 1
                    if personal_tick % max(1, int(args.personal_stride)) == 0:
                        personal_boxes = detect_personal_objects(
                            frame=frame,
                            yolo_model=personal_model,
                            conf_th=float(args.personal_conf),
                            predict_device=predict_dev,
                        )
                else:
                    personal_boxes = []

                next_track_id = assign_tracks(
                    detections=detections,
                    tracks=tracks,
                    frame_i=frame_i,
                    next_track_id=next_track_id,
                    iou_th=float(args.iou_track_th),
                    max_lost=int(args.max_track_lost),
                )

                for det in detections:
                    tid = int(det["track_id"])
                    tr = tracks[tid]
                    prev_track_hold = tr.hands["left"].holding or tr.hands["right"].holding
                    prev_prompt_yes = tr.prompt_yes
                    if not can_infer:
                    # Forzar estado vacio mientras no este armada la ROI.
                        for hs in tr.hands.values():
                            hs.prob = 0.0
                            hs.on_count = 0
                            hs.off_count = 0
                            hs.holding = False
                        tr.active_object_box = None
                        tr.prompt_yes = False
                        continue
                    seen_sides: set[str] = set()
                    det_active_sides = [side for side in det["hands"].keys() if side in active_sides]
                    for side in det_active_sides:
                        seen_sides.add(side)

                    px1, py1, px2, py2 = det["person_box"]
                    person_crop = frame[py1:py2, px1:px2]
                    if len(det_active_sides) > 0 and person_crop.size > 0:
                        side_probs: dict[str, float] = {}
                        if args.per_hand_fast:
                            # Clasificacion por mano: 1 inferencia principal + 2a opcional en zona gris.
                            # Elegir mano principal por historial de mayor probabilidad.
                            primary_side = max(
                                det_active_sides,
                                key=lambda s: tr.hands[s].prob,
                            )
                            ph = det["hands"][primary_side]
                            primary_crop, primary_box = build_hand_crop(
                                frame=frame,
                                person_box=det["person_box"],
                                wrist_xy=ph["wrist"],
                                elbow_xy=ph["elbow"],
                                crop_size=int(args.crop_size),
                                crop_min=int(args.crop_min),
                                crop_max=int(args.crop_max),
                                crop_mode=str(args.crop_mode),
                            )
                            p_prob = (
                                classifier.predict_yes_prob(primary_crop, frame_i, vlm_calls)
                                if primary_crop.size > 0
                                else 0.0
                            )
                            side_probs[primary_side] = p_prob
                            tr.hands[primary_side].last_crop_box = primary_box if primary_crop.size > 0 else None
                            answer_txt = getattr(classifier, "last_answer_text", "")
                            print(
                                f"[f={frame_i} id={tid} side={primary_side}] RESPUESTA: {answer_txt} | yes_prob={p_prob:.3f}"
                            )
                            # Si hay segunda mano activa y la principal esta en zona gris, evaluarla tambien.
                            need_second = abs(p_prob - 0.5) <= float(args.fast_gray_zone)
                            other_sides = [s for s in det_active_sides if s != primary_side]
                            if need_second and other_sides:
                                secondary_side = other_sides[0]
                                sh = det["hands"][secondary_side]
                                sec_crop, sec_box = build_hand_crop(
                                    frame=frame,
                                    person_box=det["person_box"],
                                    wrist_xy=sh["wrist"],
                                    elbow_xy=sh["elbow"],
                                    crop_size=int(args.crop_size),
                                    crop_min=int(args.crop_min),
                                    crop_max=int(args.crop_max),
                                    crop_mode=str(args.crop_mode),
                                )
                                s_prob = (
                                    classifier.predict_yes_prob(sec_crop, frame_i, vlm_calls)
                                    if sec_crop.size > 0
                                    else 0.0
                                )
                                side_probs[secondary_side] = s_prob
                                tr.hands[secondary_side].last_crop_box = sec_box if sec_crop.size > 0 else None
                                answer_txt2 = getattr(classifier, "last_answer_text", "")
                                print(
                                    f"[f={frame_i} id={tid} side={secondary_side}] RESPUESTA: {answer_txt2} | yes_prob={s_prob:.3f}"
                                )
                            # Lados no evaluados este ciclo: usar su propio historial suavizado degradado.
                            for s in det_active_sides:
                                if s not in side_probs:
                                    side_probs[s] = 0.0
                                    tr.hands[s].last_crop_box = None
                        else:
                            yes_prob = classifier.predict_yes_prob(person_crop, frame_i, vlm_calls)
                            for side in det_active_sides:
                                side_probs[side] = yes_prob
                                tr.hands[side].last_crop_box = det["person_box"]
                            answer_txt = getattr(classifier, "last_answer_text", "")
                            dbg_txt = getattr(classifier, "last_debug", "")
                            p_used = getattr(classifier, "last_prompt_used", args.vlm_prompt)
                            print(f"[f={frame_i} id={tid} side=person] PROMPT: {p_used}")
                            print(
                                f"[f={frame_i} id={tid} side=person] RESPUESTA: {answer_txt} | yes_prob={yes_prob:.3f}"
                            )
                            if dbg_txt:
                                print(f"[f={frame_i} id={tid} side=person] DEBUG: {dbg_txt}")

                        prompt_yes_th = robbery_th if robbery_th > 0.0 else 0.5
                        tr.prompt_yes = any(p > prompt_yes_th for p in side_probs.values())
                        if tr.prompt_yes:
                            prompt_yes_state = True
                            side_obj = max(det_active_sides, key=lambda s: side_probs.get(s, 0.0))
                            tr.active_object_box = tr.hands[side_obj].last_crop_box
                        else:
                            tr.active_object_box = None
                        for side in det_active_sides:
                            p_side = float(side_probs.get(side, 0.0))
                            update_temporal_state(
                                hs=tr.hands[side],
                                yes_prob=p_side,
                                frame_i=frame_i,
                                yes_th=robbery_th if robbery_th > 0.0 else float(args.yes_th),
                                hold_frames=robbery_on_frames if robbery_th > 0.0 else int(args.hold_frames),
                                drop_frames=robbery_off_frames if robbery_th > 0.0 else int(args.drop_frames),
                                force_drop_th=robbery_th if robbery_th > 0.0 else float(args.force_drop_th),
                                force_drop_frames=robbery_off_frames
                                if robbery_th > 0.0
                                else int(args.force_drop_frames),
                                raw_drop_th=robbery_th if robbery_th > 0.0 else float(args.raw_drop_th),
                                raw_drop_frames=robbery_off_frames
                                if robbery_th > 0.0
                                else int(args.raw_drop_frames),
                                raw_on_th=robbery_th if robbery_th > 0.0 else float(args.raw_on_th),
                                raw_on_frames=robbery_on_frames
                                if robbery_th > 0.0
                                else int(args.raw_on_frames),
                            )
                        if tr.prompt_yes:
                            # VALOR1: guardar posicion de muñeca de la mano con mayor probabilidad YES.
                            side_best = max(det_active_sides, key=lambda s: side_probs.get(s, 0.0))
                            # Preferir centro del crop (proxy del objeto) frente a muñeca.
                            lb = tr.hands[side_best].last_crop_box
                            if lb is not None:
                                x1b, y1b, x2b, y2b = lb
                                tr.last_object_xy = (int((x1b + x2b) / 2), int((y1b + y2b) / 2))
                            else:
                                tr.last_object_xy = det["hands"][side_best]["wrist"]
                            tr.object_traj.append(tr.last_object_xy)
                        else:
                            # En NO, mantener ventana corta de seguimiento de muñecas visibles.
                            for s in det_active_sides:
                                tr.object_traj.append(det["hands"][s]["wrist"])
                # Si una mano no se ve en este ciclo, degradar su estado para evitar "stick".
                    for side in ("left", "right"):
                        if side in seen_sides:
                            continue
                        tr.hands[side].last_crop_box = None
                        update_temporal_state(
                            hs=tr.hands[side],
                            yes_prob=0.0,
                            frame_i=frame_i,
                            yes_th=robbery_th if robbery_th > 0.0 else float(args.yes_th),
                            hold_frames=robbery_on_frames if robbery_th > 0.0 else int(args.hold_frames),
                            drop_frames=robbery_off_frames if robbery_th > 0.0 else int(args.drop_frames),
                            force_drop_th=robbery_th if robbery_th > 0.0 else float(args.force_drop_th),
                            force_drop_frames=1,
                            raw_drop_th=robbery_th if robbery_th > 0.0 else float(args.raw_drop_th),
                            raw_drop_frames=1,
                            raw_on_th=robbery_th if robbery_th > 0.0 else float(args.raw_on_th),
                            raw_on_frames=robbery_on_frames if robbery_th > 0.0 else int(args.raw_on_frames),
                        )
                    curr_track_hold = tr.hands["left"].holding or tr.hands["right"].holding
                    # Abrir ventana de refinado tras transición YES->NO o hold->no-hold.
                    if (prev_prompt_yes and (not tr.prompt_yes)) or (prev_track_hold and (not curr_track_hold)):
                        # Mostrar resultado inmediato para no perder el texto en casos límite.
                        drop_xy_now = estimate_drop_xy_from_traj(tr.object_traj, int(args.traj_smooth_len))
                        if drop_xy_now is None:
                            drop_xy_now = tr.last_object_xy
                        kp_name, kp_dist = compute_drop_zone(
                            drop_xy_now,
                            det["kps_xy"],
                            det["kps_conf"],
                            str(args.drop_zone_mode),
                            personal_boxes=personal_boxes,
                            personal_near_px=float(args.personal_near_px),
                        )
                        tr.drop_info_text = f"Guardado cerca de {kp_name} (d={kp_dist:.1f}px)"
                        tr.drop_info_frames_left = max(1, int(args.drop_frames) * max(1, int(args.stride)))
                        nearest_zone_text = kp_name
                        nearest_zone_frames_left = tr.drop_info_frames_left
                        print(f"[f={frame_i} id={tid}] {tr.drop_info_text}")
                        trigger_roi_rearm_now = True
                        tr.pending_drop_steps = max(tr.pending_drop_steps, int(args.drop_window_steps))
                    # Cerrar ventana y calcular zona con trayectoria suavizada.
                    if tr.pending_drop_steps > 0:
                        tr.pending_drop_steps -= 1
                        if tr.pending_drop_steps == 0:
                            drop_xy = estimate_drop_xy_from_traj(tr.object_traj, int(args.traj_smooth_len))
                            if drop_xy is not None:
                                kp_name, kp_dist = compute_drop_zone(
                                    drop_xy,
                                    det["kps_xy"],
                                    det["kps_conf"],
                                    str(args.drop_zone_mode),
                                    personal_boxes=personal_boxes,
                                    personal_near_px=float(args.personal_near_px),
                                )
                                tr.drop_info_text = f"Guardado cerca de {kp_name} (d={kp_dist:.1f}px)"
                                tr.drop_info_frames_left = max(1, int(args.drop_frames) * max(1, int(args.stride)))
                                nearest_zone_text = kp_name
                                nearest_zone_frames_left = tr.drop_info_frames_left
                                print(f"[f={frame_i} id={tid}] {tr.drop_info_text}")
                                trigger_roi_rearm_now = True
                            tr.object_traj.clear()

            # Si no hay manos visibles varios ciclos, vaciar estado global de agarre.
                if no_hands_steps >= max(1, int(args.no_hands_drop_steps)):
                    for tr in tracks.values():
                        for hs in tr.hands.values():
                            hs.prob = 0.0
                            hs.raw_prob = 0.0
                            hs.on_count = 0
                            hs.off_count = max(hs.off_count, int(args.drop_frames))
                            hs.weak_count = max(hs.weak_count, int(args.force_drop_frames))
                            hs.raw_low_count = max(hs.raw_low_count, int(args.raw_drop_frames))
                            hs.holding = False
                        tr.prompt_yes = False
                if not any(
                    (det.get("hands") and len(det["hands"]) > 0 and any(det["hands"].keys()))
                    for det in detections
                ):
                    prompt_yes_state = False
                elif not any(
                    tracks[int(det["track_id"])].prompt_yes
                    for det in detections
                    if int(det["track_id"]) in tracks
                ):
                    prompt_yes_state = False
                if roi_poly_1080 is not None and trigger_roi_rearm_now:
                    # Tras informar la zona de guardado, detener monitorización
                    # hasta que vuelva a entrar/salir de ROI.
                    require_roi_rearm = True
                    for side in ("left", "right"):
                        side_state[side]["active_after_exit"] = False
                    for tr in tracks.values():
                        tr.prompt_yes = False

            for tid, tr in tracks.items():
                if (frame_i - tr.last_seen_frame) > int(args.max_track_lost):
                    continue
                x1, y1, x2, y2 = tr.bbox
                any_hold = tr.hands["left"].holding or tr.hands["right"].holding
                # Visual coherente con el criterio temporal final: verde solo si "holding" confirmado.
                color = (0, 255, 0) if any_hold else (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                txt = (
                    f"ID {tid} | L:{'YES' if tr.hands['left'].holding else 'NO'}({tr.hands['left'].prob:.2f}/{tr.hands['left'].raw_prob:.2f}) "
                    f"R:{'YES' if tr.hands['right'].holding else 'NO'}({tr.hands['right'].prob:.2f}/{tr.hands['right'].raw_prob:.2f}) "
                    f"P:{'YES' if tr.prompt_yes else 'NO'}"
                )
                cv2.putText(vis, txt, (x1, max(16, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if tr.active_object_box is not None:
                    ox1, oy1, ox2, oy2 = tr.active_object_box
                    cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), (255, 255, 0), 2)
                    cv2.putText(
                        vis,
                        "objeto/mano",
                        (ox1, max(14, oy1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 0),
                        1,
                    )
                if tr.drop_info_frames_left > 0 and tr.drop_info_text:
                    cv2.putText(
                        vis,
                        tr.drop_info_text,
                        (x1, min(h - 10, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )
                    tr.drop_info_frames_left -= 1
            for bx1, by1, bx2, by2, lbl, conf in personal_boxes:
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 0, 255), 2)
                lbl_es = _personal_label_to_zone(lbl)
                cv2.putText(
                    vis,
                    f"{lbl_es} {conf:.2f}",
                    (bx1, max(16, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )

            any_holding_global = any(
                tr.hands["left"].holding or tr.hands["right"].holding
                for tr in tracks.values()
                if (frame_i - tr.last_seen_frame) <= int(args.max_track_lost)
            )
            if roi_poly_1080 is not None and prev_any_holding and (not any_holding_global):
                # Tras soltar, exigir rearme ROI (entrar/salir de nuevo).
                require_roi_rearm = True
                for side in ("left", "right"):
                    side_state[side]["active_after_exit"] = False
            if not hands_visible_now:
                status_text = "MANOS NO VISIBLES"
                status_color = (0, 200, 255)
            else:
                status_text = "Objeto cogido" if any_holding_global else "Sin objeto cogido"
                status_color = (0, 255, 0) if any_holding_global else (0, 0, 255)
            cv2.putText(vis, status_text, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.95, status_color, 3)
            if nearest_zone_frames_left > 0:
                cv2.putText(
                    vis,
                    f"Zona mas cercana: {nearest_zone_text}",
                    (12, 114),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 0),
                    2,
                )
                nearest_zone_frames_left -= 1
            if roi_poly_1080 is not None:
                left_st = side_state["left"]
                right_st = side_state["right"]
                roi_state_txt = (
                    f"ROI L(in={left_st['inside']} act={left_st['active_after_exit']}) "
                    f"R(in={right_st['inside']} act={right_st['active_after_exit']}) "
                    f"rearm={require_roi_rearm}"
                )
                cv2.putText(vis, roi_state_txt, (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 2)

            elapsed = max(1e-6, time.perf_counter() - t0)
            fps = frame_i / elapsed
            cv2.putText(vis, f"FPS {fps:.1f}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(
                vis,
                f"modelo: {args.vlm_model.split('/')[-1]}",
                (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
            )

            if writer is not None:
                # Guardar en resolución original.
                if src_w > 0 and src_h > 0 and (vis.shape[1] != src_w or vis.shape[0] != src_h):
                    out_frame = cv2.resize(vis, (src_w, src_h), interpolation=cv2.INTER_AREA)
                else:
                    out_frame = vis
                writer.write(out_frame)
                _terminal_frame_progress_line("[save]", video_path, frame_i, total_frames)
            else:
                if os.environ.get("DISPLAY", "").strip():
                    vis = cv2.resize(vis, (view_w, view_h), interpolation=cv2.INTER_AREA)
                    cv2.imshow(window_title, vis)
                    if (
                        getattr(args, "display_sync_fps", False)
                        and fps_src > 0
                        and not save_path
                    ):
                        spf = 1.0 / fps_src
                        dt = time.perf_counter() - t_frame_start
                        if dt < spf:
                            time.sleep(spf - dt)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break
                else:
                    # Sin --save (orquestador) y sin X11: la barra [save] no se imprimia; replicar en [run]
                    _terminal_frame_progress_line("[run]", video_path, frame_i, total_frames)

        if gpu_poller is not None:
            gpu_poller.stop()
        peak_alloc_b: int | None = None
        peak_reserved_b: int | None = None
        if classifier.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc_b = int(torch.cuda.max_memory_allocated())
            peak_reserved_b = int(torch.cuda.max_memory_reserved())

        experiment_end_iso = _utc_now_iso()
        wall_clock_sec = float(time.perf_counter() - experiment_wall_t0)

        cap.release()
        if writer is not None:
            writer.release()
        if writer is not None or (
            not str(save_path).strip() and not os.environ.get("DISPLAY", "").strip()
        ):
            # Nueva linea tras [save] o [run] (misma consola, \\r)
            print()
        cv2.destroyAllWindows()

        out_dir_s = str(args.output).strip()
        if out_dir_s:
            out_spec = Path(out_dir_s).expanduser().resolve()
            if out_spec.suffix.lower() == ".json":
                if str(args.videos).strip():
                    raise RuntimeError(
                        "Con --videos usa --output como carpeta, no como fichero .json (varios videos)."
                    )
                out_spec.parent.mkdir(parents=True, exist_ok=True)
                json_path = out_spec
            else:
                out_spec.mkdir(parents=True, exist_ok=True)
                json_path = None  # se asigna abajo con nombre autogenerado
            latencies = [float(c["latency_sec"]) for c in vlm_calls]
            gpu_utils: list[float] = []
            gpu_mem_used: list[float] = []
            if gpu_poller is not None:
                for s in gpu_poller.samples:
                    if "utilization_gpu_percent" in s:
                        gpu_utils.append(float(s["utilization_gpu_percent"]))
                    if "memory_used_mib" in s:
                        gpu_mem_used.append(float(s["memory_used_mib"]))
            gpu_total_mib = None
            if gpu_poller is not None and gpu_poller.samples:
                gpu_total_mib = float(gpu_poller.samples[0].get("memory_total_mib", 0) or 0) or None

            peak_torch_mib = (peak_alloc_b / (1024.0**2)) if peak_alloc_b is not None else None
            est_parallel = None
            if gpu_total_mib and peak_torch_mib and peak_torch_mib > 0:
                est_parallel = max(1, int(gpu_total_mib // peak_torch_mib))

            video_dur_sec = (total_frames / fps_src) if fps_src > 0 else None
            proc_fps = (frame_i / wall_clock_sec) if wall_clock_sec > 0 else None
            rt_factor = None
            if fps_src > 0 and proc_fps and proc_fps > 0:
                rt_factor = float(fps_src / proc_fps)

            report: dict[str, Any] = {
                "ok_correct": False,
                "experiment_backend": experiment_backend
                or getattr(classifier, "experiment_backend", None)
                or "",
                "experiment_start_utc": experiment_start_iso,
                "experiment_end_utc": experiment_end_iso,
                "wall_clock_processing_sec": round(wall_clock_sec, 6),
                "hostname": socket.gethostname(),
                "versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "transformers": transformers.__version__,
                    "numpy": np.__version__,
                    "opencv": cv2.__version__,
                },
                "video_input_path": str(video_path.resolve()),
                "video_output_path": _safe_resolve_path(save_path) if save_path else "",
                "video_fps": fps_src,
                "video_duration_frames_metadata": total_frames,
                "video_duration_sec_metadata": video_dur_sec,
                "frames_processed": frame_i,
                "pose_inference_cycles": frame_i // max(1, int(args.stride)),
                "processing_fps_effective": round(proc_fps, 4) if proc_fps is not None else None,
                "realtime_factor_vs_video_fps": round(rt_factor, 4) if rt_factor is not None else None,
                "vlm_device": str(classifier.device),
                "vlm_gpu_name": (
                    torch.cuda.get_device_name(
                        classifier.device.index if classifier.device.index is not None else 0
                    )
                    if classifier.device.type == "cuda" and torch.cuda.is_available()
                    else None
                ),
                "torch_cuda_peak_memory_allocated_bytes": peak_alloc_b,
                "torch_cuda_peak_memory_reserved_bytes": peak_reserved_b,
                "torch_cuda_peak_memory_allocated_mib": round(peak_torch_mib, 4) if peak_torch_mib is not None else None,
                "nvidia_smi_samples": gpu_poller.samples if gpu_poller is not None else [],
                "nvidia_gpu_utilization_mean_percent": round(_mean(gpu_utils), 4) if gpu_utils else None,
                "nvidia_gpu_memory_used_mean_mib": round(_mean(gpu_mem_used), 4) if gpu_mem_used else None,
                "nvidia_gpu_memory_total_mib_sample": gpu_total_mib,
                "est_max_parallel_streams_by_memory_naive": est_parallel,
                "est_max_parallel_streams_note": (
                    "Heuristica: VRAM total nvidia-smi / pico PyTorch del proceso. "
                    "No incluye fragmentacion ni otros procesos; validar en entorno real."
                ),
                "vlm_calls": vlm_calls,
                "vlm_inference_count": len(vlm_calls),
                "vlm_latency_mean_sec": round(_mean(latencies), 6) if latencies else None,
                "vlm_latency_stdev_sec": round(statistics.stdev(latencies), 6) if len(latencies) > 1 else None,
                "vlm_latency_min_sec": round(min(latencies), 6) if latencies else None,
                "vlm_latency_max_sec": round(max(latencies), 6) if latencies else None,
                "args": namespace_to_experiment_args(args),
            }
            if json_path is None:
                json_name = (
                    f"experiment_{video_path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
                )
                json_path = out_spec / json_name
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(report, jf, indent=2, ensure_ascii=False)
            print(f"[experiment] JSON guardado: {json_path}")

    if str(args.videos).strip():
        videos_dir = Path(str(args.videos).strip()).expanduser().resolve()
        if not videos_dir.exists() or not videos_dir.is_dir():
            raise RuntimeError(f"--videos no es una carpeta valida: {videos_dir}")
        video_files = sorted(
            [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        )
        if not video_files:
            raise RuntimeError(f"No hay videos en: {videos_dir}")
        for vp in video_files:
            out_path = str(vp.with_name(f"{vp.stem}{batch_output_suffix}{vp.suffix}"))
            print(f"[batch] procesando: {vp.name}")
            run_single_video(vp, out_path)
    else:
        video_path = Path(str(args.video).strip()).expanduser().resolve()
        if not video_path.exists():
            raise RuntimeError(f"--video no existe: {video_path}")
        run_single_video(video_path, str(args.save).strip())


if __name__ == "__main__":
    main()

