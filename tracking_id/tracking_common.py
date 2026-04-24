#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class IdentityProfile:
    canonical_id: int
    embeddings: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=120))
    last_seen_frame: int = -1
    last_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    last_raw_id: int = -1
    life_frames: int = 0
    misses: int = 0

    def mean_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        return np.mean(np.stack(self.embeddings, axis=0), axis=0)

    def max_similarity(self, emb: np.ndarray) -> float:
        if not self.embeddings:
            return 0.0
        best = 0.0
        for ref in self.embeddings:
            an = float(np.linalg.norm(emb))
            bn = float(np.linalg.norm(ref))
            if an < 1e-8 or bn < 1e-8:
                continue
            sim = float(np.dot(emb, ref) / (an * bn))
            if sim > best:
                best = sim
        return best


class IdentityStabilizer:
    """
    Capa de estabilizacion para intentar mantener IDs consistentes ante:
    - cambios espurios de id del tracker base.
    - desapariciones prolongadas (hasta el limite configurable).
    """

    def __init__(
        self,
        max_absence_frames: int,
        similarity_threshold: float = 0.83,
        iou_weight: float = 0.15,
    ) -> None:
        self.max_absence_frames = int(max_absence_frames)
        self.similarity_threshold = float(similarity_threshold)
        self.iou_weight = float(iou_weight)
        self.profiles: dict[int, IdentityProfile] = {}
        self.raw_to_canonical: dict[int, int] = {}
        self.next_canonical_id = 1
        self.scene_cut_grace_left = 0
        self.pending_switches: dict[int, tuple[int, int, int]] = {}

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an < 1e-8 or bn < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    @staticmethod
    def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
        return inter / max(1e-6, area_a + area_b - inter)

    @staticmethod
    def _make_embedding(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((96,), dtype=np.float32)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((96,), dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
        emb = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-8:
            emb = emb / norm
        return emb

    def _new_identity(self, embedding: np.ndarray, bbox: tuple[int, int, int, int], frame_idx: int) -> int:
        cid = self.next_canonical_id
        self.next_canonical_id += 1
        prof = IdentityProfile(canonical_id=cid, last_seen_frame=frame_idx, last_bbox=bbox, life_frames=1)
        prof.embeddings.append(embedding)
        self.profiles[cid] = prof
        return cid

    def _best_profile_match(
        self,
        embedding: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_idx: int,
    ) -> tuple[int | None, float]:
        best_cid = None
        best_score = -1.0
        for cid, prof in self.profiles.items():
            absence = frame_idx - prof.last_seen_frame
            if absence > self.max_absence_frames:
                continue
            mean_emb = prof.mean_embedding()
            if mean_emb is None:
                continue
            cos = self._cosine_similarity(embedding, mean_emb)
            iou = self._bbox_iou(bbox, prof.last_bbox)
            temporal_penalty = min(0.25, absence / max(1.0, self.max_absence_frames) * 0.25)
            score = (1.0 - self.iou_weight) * cos + self.iou_weight * iou - temporal_penalty
            if score > best_score:
                best_score = score
                best_cid = cid
        return best_cid, best_score

    def _score_candidate(
        self,
        raw_id: int,
        embedding: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_idx: int,
        cid: int,
    ) -> float:
        prof = self.profiles[cid]
        absence = frame_idx - prof.last_seen_frame
        if absence > self.max_absence_frames:
            return -1.0
        app_sim = prof.max_similarity(embedding)
        if app_sim <= 0.0:
            return -1.0
        iou = self._bbox_iou(bbox, prof.last_bbox)
        temporal_penalty = min(0.25, absence / max(1.0, self.max_absence_frames) * 0.25)
        local_iou_weight = self.iou_weight if self.scene_cut_grace_left <= 0 else 0.03
        score = (1.0 - local_iou_weight) * app_sim + local_iou_weight * iou - temporal_penalty
        prev_cid = self.raw_to_canonical.get(raw_id)
        if prev_cid == cid:
            score += 0.03  # continuidad suave, evitando fijar swaps
        return score

    def _commit(self, raw_id: int, cid: int, bbox: tuple[int, int, int, int], emb: np.ndarray, frame_idx: int) -> None:
        self.raw_to_canonical[raw_id] = cid
        prof = self.profiles[cid]
        mean_emb = prof.mean_embedding()
        sim_prev = self._cosine_similarity(emb, mean_emb) if mean_emb is not None else 1.0
        prof.last_seen_frame = frame_idx
        prof.last_bbox = bbox
        prof.last_raw_id = raw_id
        prof.life_frames += 1
        prof.misses = 0
        # Evita contaminar el perfil con una apariencia claramente distinta
        # (caso tipico de ID switch momentaneo por cruce/oclusion).
        if sim_prev >= (self.similarity_threshold - 0.18):
            prof.embeddings.append(emb)

    def notify_scene_cut(self, grace_frames: int = 25) -> None:
        self.scene_cut_grace_left = max(self.scene_cut_grace_left, int(grace_frames))

    def best_similarity_for_detection(
        self,
        raw_id: int,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_idx: int,
    ) -> float:
        emb = self._make_embedding(frame_bgr, bbox)
        best = 0.0
        prev_cid = self.raw_to_canonical.get(raw_id)
        if prev_cid is not None and prev_cid in self.profiles:
            prof = self.profiles[prev_cid]
            if (frame_idx - prof.last_seen_frame) <= self.max_absence_frames:
                best = max(best, prof.max_similarity(emb))
        for cid, prof in self.profiles.items():
            if cid == prev_cid:
                continue
            if (frame_idx - prof.last_seen_frame) > self.max_absence_frames:
                continue
            best = max(best, prof.max_similarity(emb))
        return best

    def assign_batch(
        self,
        detections: list[tuple[int, tuple[int, int, int, int]]],
        frame_bgr: np.ndarray,
        frame_idx: int,
    ) -> list[tuple[int, int, tuple[int, int, int, int]]]:
        if not detections:
            return []
        if self.scene_cut_grace_left > 0:
            self.scene_cut_grace_left -= 1

        emb_by_raw: dict[int, np.ndarray] = {}
        bbox_by_raw: dict[int, tuple[int, int, int, int]] = {}
        for raw_id, bbox in detections:
            emb_by_raw[raw_id] = self._make_embedding(frame_bgr, bbox)
            bbox_by_raw[raw_id] = bbox

        assigned_raws: set[int] = set()
        used_cids: set[int] = set()
        assign_map: dict[int, int] = {}
        score_cache: dict[tuple[int, int], float] = {}

        # Paso 1: mantener mapeos previos estables (anti ID-switch durante cruces).
        for raw_id, _ in detections:
            prev_cid = self.raw_to_canonical.get(raw_id)
            if prev_cid is None or prev_cid not in self.profiles or prev_cid in used_cids:
                continue
            keep_score = self._score_candidate(
                raw_id=raw_id,
                embedding=emb_by_raw[raw_id],
                bbox=bbox_by_raw[raw_id],
                frame_idx=frame_idx,
                cid=prev_cid,
            )
            score_cache[(raw_id, prev_cid)] = keep_score
            # Importante: NO fijamos por "reciente" sin comprobar score, porque eso
            # perpetua intercambios cuando el tracker raw hace swap de IDs.
            keep_th = self.similarity_threshold - (0.08 if self.scene_cut_grace_left > 0 else 0.02)
            if keep_score >= keep_th:
                assigned_raws.add(raw_id)
                used_cids.add(prev_cid)
                assign_map[raw_id] = prev_cid

        # Paso 2: matching global greedy 1-a-1 para evitar que dos raws compartan ID.
        candidates: list[tuple[float, int, int]] = []
        for raw_id, _ in detections:
            if raw_id in assigned_raws:
                continue
            emb = emb_by_raw[raw_id]
            bbox = bbox_by_raw[raw_id]
            for cid in self.profiles:
                if cid in used_cids:
                    continue
                score = self._score_candidate(raw_id, emb, bbox, frame_idx, cid)
                score_cache[(raw_id, cid)] = score
                base_th = self.similarity_threshold - (0.10 if self.scene_cut_grace_left > 0 else 0.0)
                if score >= base_th:
                    candidates.append((score, raw_id, cid))
        candidates.sort(reverse=True, key=lambda x: x[0])

        for score, raw_id, cid in candidates:
            if raw_id in assigned_raws or cid in used_cids:
                continue
            assigned_raws.add(raw_id)
            used_cids.add(cid)
            assign_map[raw_id] = cid

        # Paso 2.5: correccion de swap momentaneo A<->B tras cruce.
        # Si dos asignaciones tienen mejor score cruzado que propio, se invierten.
        raw_ids = list(assign_map.keys())
        swap_margin = 0.035
        for i in range(len(raw_ids)):
            ra = raw_ids[i]
            ca = assign_map[ra]
            if ca not in self.profiles:
                continue
            for j in range(i + 1, len(raw_ids)):
                rb = raw_ids[j]
                cb = assign_map[rb]
                if cb not in self.profiles or ca == cb:
                    continue
                saa = score_cache.get((ra, ca), self._score_candidate(ra, emb_by_raw[ra], bbox_by_raw[ra], frame_idx, ca))
                sbb = score_cache.get((rb, cb), self._score_candidate(rb, emb_by_raw[rb], bbox_by_raw[rb], frame_idx, cb))
                sab = score_cache.get((ra, cb), self._score_candidate(ra, emb_by_raw[ra], bbox_by_raw[ra], frame_idx, cb))
                sba = score_cache.get((rb, ca), self._score_candidate(rb, emb_by_raw[rb], bbox_by_raw[rb], frame_idx, ca))
                # Swap si ambos mejoran de forma consistente.
                if (sab >= saa + swap_margin) and (sba >= sbb + swap_margin):
                    assign_map[ra], assign_map[rb] = cb, ca

        # Paso 2.6: histeresis temporal para evitar switches espurios.
        # Si hay cambio de ID, pedimos confirmacion de varios frames salvo
        # que la mejora de score sea claramente superior.
        required_confirm = 2 if self.scene_cut_grace_left > 0 else 3
        strong_margin = 0.12
        for raw_id in list(assign_map.keys()):
            new_cid = assign_map[raw_id]
            prev_cid = self.raw_to_canonical.get(raw_id)
            if prev_cid is None or prev_cid == new_cid or prev_cid not in self.profiles:
                self.pending_switches.pop(raw_id, None)
                continue
            # Si el ID previo ya esta ocupado por otro raw, no podemos retenerlo.
            owner_prev = next((r for r, c in assign_map.items() if c == prev_cid), None)
            if owner_prev is not None and owner_prev != raw_id:
                self.pending_switches.pop(raw_id, None)
                continue

            s_new = score_cache.get(
                (raw_id, new_cid),
                self._score_candidate(raw_id, emb_by_raw[raw_id], bbox_by_raw[raw_id], frame_idx, new_cid),
            )
            s_prev = score_cache.get(
                (raw_id, prev_cid),
                self._score_candidate(raw_id, emb_by_raw[raw_id], bbox_by_raw[raw_id], frame_idx, prev_cid),
            )

            if s_new >= s_prev + strong_margin:
                self.pending_switches.pop(raw_id, None)
                continue

            pending = self.pending_switches.get(raw_id)
            if pending is not None and pending[0] == prev_cid and pending[1] == new_cid:
                cnt = pending[2] + 1
            else:
                cnt = 1
            self.pending_switches[raw_id] = (prev_cid, new_cid, cnt)
            if cnt < required_confirm:
                assign_map[raw_id] = prev_cid
            else:
                self.pending_switches.pop(raw_id, None)

        # Paso 2.7: reparacion diferida de identidad (1-2s) para recuperar
        # IDs historicos tras un switch que no se corrige en el mismo frame.
        reclaim_min_life = 20
        reclaim_recent_window = 45
        reclaim_margin = 0.01
        occupied_cids = set(assign_map.values())
        for raw_id in list(assign_map.keys()):
            current_cid = assign_map[raw_id]
            emb = emb_by_raw[raw_id]
            bbox = bbox_by_raw[raw_id]
            score_current = score_cache.get(
                (raw_id, current_cid),
                self._score_candidate(raw_id, emb, bbox, frame_idx, current_cid),
            )
            best_old_cid = None
            best_old_score = -1.0
            for cid, prof in self.profiles.items():
                if cid == current_cid:
                    continue
                if cid in occupied_cids:
                    continue
                if prof.life_frames < reclaim_min_life:
                    continue
                if (frame_idx - prof.last_seen_frame) > reclaim_recent_window:
                    continue
                s_old = score_cache.get(
                    (raw_id, cid),
                    self._score_candidate(raw_id, emb, bbox, frame_idx, cid),
                )
                if s_old > best_old_score:
                    best_old_score = s_old
                    best_old_cid = cid
            if best_old_cid is None:
                continue
            if best_old_score >= max(self.similarity_threshold - 0.04, score_current + reclaim_margin):
                occupied_cids.discard(current_cid)
                assign_map[raw_id] = best_old_cid
                occupied_cids.add(best_old_cid)

        # Paso 3: nuevas identidades para lo no asignado.
        for raw_id, _ in detections:
            if raw_id in assigned_raws:
                continue
            bbox = bbox_by_raw[raw_id]
            emb = emb_by_raw[raw_id]
            new_cid = self._new_identity(emb, bbox, frame_idx)
            assigned_raws.add(raw_id)
            used_cids.add(new_cid)
            assign_map[raw_id] = new_cid

        # Commit final ya corregido.
        output: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for raw_id, _ in detections:
            cid = assign_map[raw_id]
            bbox = bbox_by_raw[raw_id]
            emb = emb_by_raw[raw_id]
            self._commit(raw_id, cid, bbox, emb, frame_idx)
            output.append((raw_id, cid, bbox))

        return output

    def step_frame(self, frame_idx: int) -> None:
        dead_profiles = []
        for cid, prof in self.profiles.items():
            if prof.last_seen_frame < frame_idx:
                prof.misses += 1
            if frame_idx - prof.last_seen_frame > (self.max_absence_frames * 2):
                dead_profiles.append(cid)
        for cid in dead_profiles:
            del self.profiles[cid]

        dead_raw = []
        for rid, cid in self.raw_to_canonical.items():
            prof = self.profiles.get(cid)
            if prof is None or (frame_idx - prof.last_seen_frame > self.max_absence_frames * 2):
                dead_raw.append(rid)
        for rid in dead_raw:
            del self.raw_to_canonical[rid]


def draw_box_with_big_id(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    canonical_id: int,
    raw_id: int,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {canonical_id} (raw {raw_id})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 3
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    ly1 = max(0, y1 - th - 14)
    ly2 = min(frame.shape[0], y1 - 2)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 12, ly2), color, -1)
    cv2.putText(frame, label, (x1 + 6, ly2 - 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def color_for_id(identity: int) -> tuple[int, int, int]:
    hue = (identity * 37) % 180
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tracking de personas con YOLO11 + estabilizacion de IDs.")
    parser.add_argument("--video", required=True, help="Ruta al video de entrada.")
    parser.add_argument("--algorithm", choices=["bytetrack", "botsort", "deepsort"], required=True)
    parser.add_argument("--yolo-model", default="yolo11n.pt", help="Modelo YOLO (por defecto yolo11n.pt).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device para YOLO: auto (cuda si disponible, si no cpu), o 0/cpu manual.",
    )
    parser.add_argument("--show", action="store_true", help="Mostrar ventana en tiempo real.")
    parser.add_argument("--save-video", action="store_true", help="Guardar video de salida en 1280x720.")
    parser.add_argument("--output", default="", help="Ruta de salida si se guarda video.")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument(
        "--output-fps",
        type=float,
        default=0.0,
        help="FPS del video de salida. 0 = usar FPS original del video.",
    )
    parser.add_argument(
        "--max-absence-sec",
        type=float,
        default=900.0,
        help="Segundos maximos de ausencia para intentar conservar/reasignar el mismo ID (default recomendado: 900s).",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.76)
    parser.add_argument("--botsort-with-reid", action="store_true")
    parser.add_argument("--deepsort-max-age", type=int, default=9000)
    parser.add_argument("--deepsort-n-init", type=int, default=3)
    parser.add_argument("--deepsort-max-cosine-distance", type=float, default=0.18)
    parser.add_argument(
        "--require-full-body",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="No asignar ID a personas no completas, salvo alta similitud con una identidad existente.",
    )
    parser.add_argument(
        "--partial-min-height-ratio",
        type=float,
        default=0.22,
        help="Altura minima relativa de bbox para considerar persona completa.",
    )
    parser.add_argument(
        "--partial-border-margin",
        type=int,
        default=8,
        help="Margen de seguridad con bordes para considerar bbox completa.",
    )
    parser.add_argument(
        "--partial-allow-sim-th",
        type=float,
        default=0.90,
        help="Si bbox no completa, solo permitir ID cuando similitud >= este valor.",
    )
    return parser


def _is_full_body_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    min_height_ratio: float,
    border_margin: int,
) -> bool:
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    touches_left = x1 <= border_margin
    touches_right = x2 >= (w - border_margin)
    touches_top = y1 <= border_margin
    touches_bottom = y2 >= (h - border_margin)
    if touches_left or touches_right or touches_top or touches_bottom:
        return False
    if (bh / max(1.0, float(h))) < float(min_height_ratio):
        return False
    aspect = bw / max(1.0, float(bh))
    # Aspect ratio humano aproximado (ancho/alto).
    if aspect < 0.15 or aspect > 0.85:
        return False
    return True


def _deepsort_tracks_from_frame(
    tracker: object,
    frame: np.ndarray,
    model: YOLO,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, device=device, classes=[0], verbose=False)
    result = results[0]
    xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes and result.boxes.xyxy is not None else np.empty((0, 4))
    scores = result.boxes.conf.cpu().numpy() if result.boxes and result.boxes.conf is not None else np.empty((0,))

    detections = []
    for box, score in zip(xyxy, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        detections.append(([x1, y1, w, h], float(score), "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    out: list[tuple[int, tuple[int, int, int, int]]] = []
    for tr in tracks:
        if not tr.is_confirmed():
            continue
        # Evita "cajas fantasma": solo usamos tracks actualizados en este frame.
        if getattr(tr, "time_since_update", 1) != 0:
            continue
        ltrb = tr.to_ltrb()
        x1, y1, x2, y2 = [int(v) for v in ltrb]
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((int(tr.track_id), (x1, y1, x2, y2)))
    return out


def _ultralytics_tracks_from_frame(
    frame: np.ndarray,
    model: YOLO,
    tracker_yaml: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    results = model.track(
        source=frame,
        persist=True,
        tracker=tracker_yaml,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        classes=[0],
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or result.boxes.id is None:
        return []
    ids = result.boxes.id.int().cpu().tolist()
    boxes = result.boxes.xyxy.int().cpu().tolist()
    out = []
    for raw_id, box in zip(ids, boxes):
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((int(raw_id), (int(x1), int(y1), int(x2), int(y2))))
    return out


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.video).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el video: {input_path}")

    resolved_device = args.device
    if str(args.device).lower() == "auto":
        resolved_device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(args.yolo_model)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if (not fps or math.isnan(fps) or fps <= 0) else float(fps)
    max_absence_frames = int(max(1, args.max_absence_sec * fps))
    stabilizer = IdentityStabilizer(
        max_absence_frames=max_absence_frames,
        similarity_threshold=args.similarity_threshold,
    )

    tracker_name = args.algorithm
    tracker_impl = None
    tracker_yaml_path = None
    base_dir = Path(__file__).resolve().parent

    if tracker_name == "deepsort":
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            raise ImportError(
                "Para usar deepsort instala: pip install deep-sort-realtime"
            ) from exc
        use_gpu_for_embedder = str(resolved_device).lower() != "cpu" and torch.cuda.is_available()
        tracker_impl = DeepSort(
            max_age=args.deepsort_max_age,
            n_init=args.deepsort_n_init,
            max_cosine_distance=args.deepsort_max_cosine_distance,
            embedder="mobilenet",
            half=use_gpu_for_embedder,
            bgr=True,
            embedder_gpu=use_gpu_for_embedder,
        )
    elif tracker_name == "botsort":
        tracker_yaml_path = str(base_dir / ("botsort_reid.yaml" if args.botsort_with_reid else "botsort_long.yaml"))
    elif tracker_name == "bytetrack":
        tracker_yaml_path = str(base_dir / "bytetrack_long.yaml")
    else:
        raise ValueError(f"Algoritmo no soportado: {tracker_name}")

    if args.save_video:
        out_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_name(
            f"{input_path.stem}_{tracker_name}_stable.mp4"
        )
        output_fps = float(args.output_fps) if float(args.output_fps) > 0 else fps
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (args.window_width, args.window_height),
        )
    else:
        out_path = None
        writer = None

    if args.show:
        cv2.namedWindow("tracking_id_eval", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tracking_id_eval", args.window_width, args.window_height)

    frame_idx = 0
    per_id_seen = defaultdict(int)
    start = time.time()
    prev_scene_sig: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Deteccion simple de cambio brusco de escena (video concatenado entre camaras).
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        scene_sig = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]).flatten().astype(np.float32)
        nrm = float(np.linalg.norm(scene_sig))
        if nrm > 1e-8:
            scene_sig /= nrm
        if prev_scene_sig is not None:
            scene_sim = float(np.dot(scene_sig, prev_scene_sig))
            if scene_sim < 0.55:
                stabilizer.notify_scene_cut(grace_frames=35)
        prev_scene_sig = scene_sig

        if tracker_name == "deepsort":
            tracks = _deepsort_tracks_from_frame(
                tracker=tracker_impl,
                frame=frame,
                model=model,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=resolved_device,
            )
        else:
            tracks = _ultralytics_tracks_from_frame(
                frame=frame,
                model=model,
                tracker_yaml=tracker_yaml_path,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=resolved_device,
            )

        filtered_tracks: list[tuple[int, tuple[int, int, int, int]]] = []
        for raw_id, bbox in tracks:
            if not args.require_full_body:
                filtered_tracks.append((raw_id, bbox))
                continue
            full_ok = _is_full_body_bbox(
                bbox=bbox,
                frame_shape=frame.shape,
                min_height_ratio=float(args.partial_min_height_ratio),
                border_margin=int(args.partial_border_margin),
            )
            if full_ok:
                filtered_tracks.append((raw_id, bbox))
                continue
            sim = stabilizer.best_similarity_for_detection(
                raw_id=raw_id,
                frame_bgr=frame,
                bbox=bbox,
                frame_idx=frame_idx,
            )
            if sim >= float(args.partial_allow_sim_th):
                filtered_tracks.append((raw_id, bbox))

        stabilizer.step_frame(frame_idx)
        frame_assignments: dict[int, tuple[int, int, int, int]] = {}
        assignments = stabilizer.assign_batch(detections=filtered_tracks, frame_bgr=frame, frame_idx=frame_idx)
        for raw_id, canonical_id, bbox in assignments:
            # Si en el mismo frame entran dos raws con el mismo canónico, nos quedamos
            # con la caja de mayor area para evitar parpadeos visuales.
            prev = frame_assignments.get(canonical_id)
            if prev is not None:
                px1, py1, px2, py2 = prev
                prev_area = max(1, (px2 - px1) * (py2 - py1))
                x1, y1, x2, y2 = bbox
                area = max(1, (x2 - x1) * (y2 - y1))
                if area <= prev_area:
                    continue
            per_id_seen[canonical_id] += 1
            frame_assignments[canonical_id] = bbox
            draw_box_with_big_id(frame, bbox, canonical_id, raw_id, color_for_id(canonical_id))

        info = (
            f"Alg: {tracker_name} | frame {frame_idx} | dets: {len(tracks)} filtro: {len(filtered_tracks)} | "
            f"IDs canonicos: {len(stabilizer.profiles)}"
        )
        cv2.putText(frame, info, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        out_frame = cv2.resize(frame, (args.window_width, args.window_height), interpolation=cv2.INTER_LINEAR)
        if args.show:
            cv2.imshow("tracking_id_eval", out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        if writer is not None:
            writer.write(out_frame)

    elapsed = time.time() - start
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Procesado completo en {elapsed:.1f}s ({frame_idx} frames).")
    print(f"IDs canonicos totales: {len(per_id_seen)}")
    if out_path is not None:
        print(f"Video guardado en: {out_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
