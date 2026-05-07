#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + SigLIP (embeddings positivo/negativo)."""

from handobject_classifiers import ClipLikeClassifier
from handobject_shared import (
    build_hand_crop,
    build_parser,
    extract_people_and_hands,
    parse_args,
    resolve_yolo_weights_for_runtime,
    run_pipeline,
    yolo_predict_device_for_args,
)
from ultralytics import YOLO

_SESSION_VLM_NULLS: dict[str, Any] = {
    "vlm_vector_prompt_probs": None,
    "vlm_vector_prompt_logits": None,
    "fused_image_embedding": None,
    "gated_yes_prob": None,
    "latency_sec": None,
    "hands_in_shopping_basket_prob": None,
    "hands_in_shopping_cart_prob": None,
    "hands_in_shopping_basket_or_cart_prob": None,
}

# Alineado con el orden de SiglipSO400MClassifier.texts (sesión + probe).
_LABELS_ES_PROBE: list[str] = [
    "objeto empaquetado / compra en las manos (tienda)",
    "manos vacías, sin coger nada",
    "manos en bolsillos u ocultas",
    "gestos o ropa, sin producto en la mano",
    "manos dentro de la cesta",
    "manos dentro del carro",
]

_PROMPT_PROFILES: dict[str, list[str]] = {
    "default": [
        "A shopper holding a packaged grocery item or product box with their hands in a store aisle.",
        "Empty hands with palms open, not grasping any package, bottle or box.",
        "Hands in pockets or fully hidden, not visible holding anything.",
        "Hands gesturing or touching clothing with no product, bottle or box in hand.",
        "Hands inside a shopping basket among food or products.",
        "Hands inside a shopping cart or store trolley.",
    ],
    # Más estricto en negativos ambiguos para recortar falsos positivos.
    "hard_negative": [
        "Hands clearly holding a product, package, bottle, or boxed item.",
        "Empty hands, no object held at all.",
        "Hands not visible or hidden in pockets; no clear object in hands.",
        "Ambiguous hand motion or touching clothes, with no clear evidence of holding a product.",
        "Hands reaching into or inside a shopping basket.",
        "Hands reaching into or inside a shopping cart or trolley.",
    ],
    # Variante con lenguaje tipo unknown en clases negativas.
    "unknown_like": [
        "A shopper clearly carrying an item in hand (package, box, bottle, or product).",
        "No object in hands; hands are empty.",
        "Hands hidden or out of view, action uncertain.",
        "Hands visible but interaction is unclear or non-object-related.",
        "Hands inside a shopping basket among food or products.",
        "Hands inside a shopping cart or store trolley.",
    ],
    # Cesta/carro solo si se ve claramente el contenedor (reduce FPs en estanterías).
    "strict_container_visible": [
        "Hands clearly holding a product, package, box, or bottle.",
        "Hands empty with no item held.",
        "Hands hidden in pockets or not visible.",
        "Hands touching clothes or gesturing, no item in hand.",
        "Hands inside a clearly visible portable shopping basket with basket rim or handle visible.",
        "Hands inside a clearly visible shopping cart with cart frame or handle visible.",
    ],
}


class SiglipSO400MClassifier(ClipLikeClassifier):
    """SigLIP SO400M con batch multiescala y score diferencial."""

    def __init__(
        self,
        model_name: str,
        device: str,
        prompt: str,
        *,
        prompt_profile: str = "default",
        net_th: float = 0.35,
        net_margin_th: float = 0.30,
        decision_mode: str = "weighted",
        multicrop_mode: str = "batch3",
        torso_weight: float = 0.45,
        left_weight: float = 0.275,
        right_weight: float = 0.275,
    ) -> None:
        super().__init__(model_name, device, prompt, backend_name="siglip")
        # Softmax de 6 anclas (orden fijo). Inglés para SigLIP; ver siglip_prompt_labels_es en JSON.
        # [0] Enfoque tienda / producto empaquetado: "visible object" era demasiado genérico y perdía
        #     frente a otras clases en recortes de cámara cenital.
        profile_key = str(prompt_profile).strip().lower()
        if profile_key not in _PROMPT_PROFILES:
            raise RuntimeError(
                f"Perfil de prompts no válido: {prompt_profile!r}. "
                f"Opciones: {', '.join(sorted(_PROMPT_PROFILES.keys()))}"
            )
        self.prompt_profile = profile_key
        self.texts = list(_PROMPT_PROFILES[profile_key])
        # Calibración rápida anti-FP: en perfil estricto pedimos más evidencia para cesta/carro.
        self.container_logit_penalty = 3.0 if profile_key == "strict_container_visible" else 0.0
        self._prompt_idx_basket = 4
        self._prompt_idx_cart = 5
        self.net_th = float(max(0.0, net_th))
        self.net_margin_th = float(max(0.0, net_margin_th))
        self.decision_mode = str(decision_mode)
        self.multicrop_mode = str(multicrop_mode)
        wsum = max(1e-6, float(torso_weight + left_weight + right_weight))
        self.crop_weights = [
            float(torso_weight / wsum),
            float(left_weight / wsum),
            float(right_weight / wsum),
        ]
        with torch.no_grad():
            txt = self.processor(text=self.texts, return_tensors="pt", padding=True).to(self.device)
            txt_feat = self._encode_text(txt)
            self.text_features = self._l2_normalize(txt_feat)

    @staticmethod
    def _center_zoom(bgr: np.ndarray, scale: float = 0.85) -> np.ndarray:
        h, w = bgr.shape[:2]
        zh, zw = max(16, int(h * scale)), max(16, int(w * scale))
        y1 = max(0, (h - zh) // 2)
        x1 = max(0, (w - zw) // 2)
        y2 = min(h, y1 + zh)
        x2 = min(w, x1 + zw)
        return bgr[y1:y2, x1:x2]

    @staticmethod
    def _hand_half_crop(bgr: np.ndarray, side: str) -> np.ndarray:
        h, w = bgr.shape[:2]
        if side == "left":
            x1, x2 = 0, max(1, int(0.60 * w))
        else:
            x1, x2 = min(w - 1, int(0.40 * w)), w
        y1, y2 = int(0.18 * h), int(0.96 * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        return bgr[y1:y2, x1:x2]

    def _build_crops(self, bgr: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        if self.multicrop_mode == "off":
            return [bgr], [1.0]
        if self.multicrop_mode == "light":
            return [bgr, self._center_zoom(bgr, scale=0.85)], [0.65, 0.35]
        if self.multicrop_mode == "full":
            return [bgr, self._center_zoom(bgr, scale=0.85), cv2.convertScaleAbs(bgr, alpha=1.15, beta=6.0)], [0.50, 0.30, 0.20]
        # batch3 (recomendado): torso + mano izquierda + mano derecha
        return [
            bgr,
            self._hand_half_crop(bgr, "left"),
            self._hand_half_crop(bgr, "right"),
        ], self.crop_weights

    def _score_crops_batch(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inp = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
        img_feat = self._l2_normalize(self._encode_image(inp))
        logits = (100.0 * img_feat @ self.text_features.T)
        probs = torch.softmax(logits, dim=1)
        p_obj = probs[:, 0]
        p_empty = probs[:, 1]
        p_other_max = torch.max(probs[:, 2:], dim=1).values
        return (
            p_obj.detach().cpu().numpy(),
            p_empty.detach().cpu().numpy(),
            p_other_max.detach().cpu().numpy(),
        )

    def _forward_multicrop(self, bgr: np.ndarray) -> dict[str, Any]:
        """Una pasada SigLIP: crops, logits/probs por prompt, embedding fusionado, gated_yes."""
        crops, weights = self._build_crops(bgr)
        pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inp = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            t0 = time.perf_counter()
            img_feat = self._l2_normalize(self._encode_image(inp))
            logits_t = 100.0 * img_feat @ self.text_features.T
            if self.container_logit_penalty > 0.0:
                logits_t = logits_t.clone()
                logits_t[:, self._prompt_idx_basket] -= float(self.container_logit_penalty)
                logits_t[:, self._prompt_idx_cart] -= float(self.container_logit_penalty)
            probs_t = torch.softmax(logits_t, dim=1)
            latency = time.perf_counter() - t0
        w = np.array(weights, dtype=np.float32)
        w = w / max(1e-6, float(np.sum(w)))
        p_obj = probs_t[:, 0].detach().cpu().numpy()
        p_empty = probs_t[:, 1].detach().cpu().numpy()
        p_other = probs_t[:, 2:].detach().cpu().numpy()
        p_other_max = np.max(p_other, axis=1)
        logits_np = logits_t.detach().cpu().numpy()
        probs_np = probs_t.detach().cpu().numpy()
        img_np = img_feat.detach().cpu().numpy()
        net = p_obj - p_empty
        best_crop_idx = int(np.argmax(net)) if net.size > 0 else 0
        if self.decision_mode == "best_crop":
            fused_logits = np.asarray(logits_np[best_crop_idx], dtype=np.float64)
            fused_probs = np.asarray(probs_np[best_crop_idx], dtype=np.float64)
            fused_probs = fused_probs / max(1e-9, float(np.sum(fused_probs)))
            fused_emb = np.asarray(img_np[best_crop_idx], dtype=np.float64)
            fused_emb = fused_emb / max(1e-9, float(np.linalg.norm(fused_emb)))
            net_global = float(net[best_crop_idx])
            p_obj_global = float(p_obj[best_crop_idx])
            p_other_global = float(p_other_max[best_crop_idx])
            votes_yes = 0
            votes_needed = 0
            margin_ok = (net_global > self.net_margin_th) and (p_obj_global > (p_other_global + 0.05))
            gated_yes = p_obj_global if (margin_ok and net_global > self.net_th) else 0.0
        else:
            fused_logits = np.sum(w[:, None] * logits_np, axis=0)
            fused_probs = np.sum(w[:, None] * probs_np, axis=0)
            fused_probs = fused_probs / max(1e-9, float(np.sum(fused_probs)))
            fused_emb = np.sum(w[:, None] * img_np, axis=0)
            fused_emb = fused_emb / max(1e-9, float(np.linalg.norm(fused_emb)))
            net_global = float(np.sum(w * net))
            p_obj_global = float(np.sum(w * p_obj))
            p_other_global = float(np.sum(w * p_other_max))
            margin_ok = (net_global > self.net_margin_th) and (p_obj_global > (p_other_global + 0.05))
        if self.decision_mode == "majority":
            positives = ((net > self.net_th) & (net > self.net_margin_th) & (p_obj > (p_other_max + 0.05))).astype(
                np.int32
            )
            votes_yes = int(np.sum(positives))
            votes_needed = (len(crops) // 2) + 1
            gated_yes = p_obj_global if votes_yes >= votes_needed else 0.0
        elif self.decision_mode != "best_crop":
            votes_yes = 0
            votes_needed = 0
            gated_yes = p_obj_global if (margin_ok and net_global > self.net_th) else 0.0
        pb = float(fused_probs[self._prompt_idx_basket])
        pc = float(fused_probs[self._prompt_idx_cart])
        return {
            "latency_sec": latency,
            "fused_prompt_logits": fused_logits,
            "fused_prompt_probs": fused_probs,
            "fused_image_embedding": fused_emb,
            "gated_yes_prob": float(gated_yes),
            "net_global": net_global,
            "p_obj_global": p_obj_global,
            "p_other_global": p_other_global,
            "votes_yes": votes_yes,
            "votes_needed": votes_needed,
            "best_crop_idx": best_crop_idx,
            "hands_in_shopping_basket_prob": pb,
            "hands_in_shopping_cart_prob": pc,
            "hands_in_shopping_basket_or_cart_prob": float(pb + pc),
        }

    def encode_frame_vectors(self, bgr: np.ndarray) -> dict[str, Any]:
        """Vectores por frame (logits/probs por prompt + embedding fusionado + gated_yes)."""
        return self._forward_multicrop(bgr)

    def predict_yes_prob(
        self,
        bgr: np.ndarray,
        frame_index: int | None = None,
        vlm_calls: list[dict[str, Any]] | None = None,
    ) -> float:
        self.last_prompt_used = (
            "SigLIP v2: score diferencial (objeto-vacio) con batch multiescala "
            "(torso + mano izq + mano der)."
        )
        out = self._forward_multicrop(bgr)
        latency = float(out["latency_sec"])
        gated_yes = float(out["gated_yes_prob"])
        self.last_answer_text = "YES" if gated_yes > 0.0 else "NO"
        self.last_debug = (
            f"p_obj={out['p_obj_global']:.3f} net={out['net_global']:.3f} net_th={self.net_th:.3f} "
            f"net_margin={self.net_margin_th:.3f} p_other={out['p_other_global']:.3f} "
            f"multi={self.multicrop_mode} decision={self.decision_mode} "
            f"votes={out['votes_yes']}/{out['votes_needed']}"
        )
        if frame_index is not None and vlm_calls is not None:
            vlm_calls.append(
                {
                    "frame_prompt": frame_index,
                    "frame_response": frame_index,
                    "latency_sec": round(latency, 6),
                    "stage": "siglip_v2_batch3_differential",
                    "note": self.last_debug,
                }
            )
        return gated_yes


def _discover_chunk_dirs(session_root: Path) -> list[Path]:
    if not session_root.is_dir():
        raise RuntimeError(f"No es un directorio: {session_root}")
    chunks: list[Path] = []
    for p in sorted(session_root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if (p / "frames").is_dir() and (p / "frames_meta.json").is_file():
            chunks.append(p)
    if not chunks:
        raise RuntimeError(
            f"No se encontraron subcarpetas con frames/ y frames_meta.json bajo {session_root}"
        )
    return chunks


def _frame_has_wrist_visibility(meta_row: dict[str, Any]) -> bool:
    if bool(meta_row.get("left_wrist_visible")) or bool(meta_row.get("right_wrist_visible")):
        return True
    try:
        return int(meta_row.get("visible_wrists_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def _load_frames_meta(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"frames_meta.json debe ser una lista: {path}")
    rows = [r for r in raw if isinstance(r, dict)]
    rows.sort(key=lambda r: int(r.get("sample_idx", 0)))
    return rows


def _session_progress_line(
    session_label: str,
    cur: int,
    total: int,
    chunk_name: str,
    image_key: str,
    phase: str,
    *,
    elapsed_s: float,
    eta_s: float | None,
) -> None:
    """Una línea con \\r (barra + chunk + fase + ETA)."""
    if total > 0:
        pct = min(100.0, (100.0 * cur) / max(1, total))
    else:
        pct = 0.0
    bar_w = 28
    fill = int(round((pct / 100.0) * bar_w))
    bar = ("#" * fill) + ("-" * (bar_w - fill))
    eta_part = ""
    if eta_s is not None and eta_s > 0.5:
        eta_part = f" ETA ~{int(eta_s)}s"
    print(
        f"\r[session] {session_label} [{bar}] {cur}/{total} ({pct:5.1f}%) "
        f"{chunk_name}/{image_key} | {phase} | {elapsed_s:6.1f}s{eta_part}   ",
        end="",
        flush=True,
    )


def run_session_siglip_v2(
    session_dir: Path,
    clf: SiglipSO400MClassifier,
    *,
    output_json: Path,
    emit_image_embedding: bool,
) -> None:
    chunk_dirs = _discover_chunk_dirs(session_dir)
    session_label = session_dir.name or str(session_dir)
    texts = list(clf.texts)

    plan: list[tuple[Path, dict[str, Any]]] = []
    for chunk_dir in chunk_dirs:
        meta_path = chunk_dir / "frames_meta.json"
        for row in _load_frames_meta(meta_path):
            if not str(row.get("image_key") or "").strip():
                continue
            plan.append((chunk_dir, row))

    total = len(plan)
    print(
        f"[session] Sesión: {session_dir} | chunks={len(chunk_dirs)} | entradas en meta: {total}",
        flush=True,
    )
    print(
        "[session] Fases por frame: omitir (sin muñeca) | omitir (sin archivo) | SigLIP (inferencia).",
        flush=True,
    )
    if total == 0:
        print("[session] No hay filas con image_key; nada que hacer.", flush=True)

    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, (chunk_dir, row) in enumerate(plan, start=1):
        frames_dir = chunk_dir / "frames"
        image_key = str(row.get("image_key") or "").strip()
        img_path = frames_dir / image_key
        base_rec: dict[str, Any] = {
            "chunk": chunk_dir.name,
            "sample_idx": row.get("sample_idx"),
            "image_key": image_key,
            "image_path": str(img_path.resolve()),
            "left_wrist_visible": bool(row.get("left_wrist_visible")),
            "right_wrist_visible": bool(row.get("right_wrist_visible")),
            "visible_wrists_count": row.get("visible_wrists_count"),
        }
        if not _frame_has_wrist_visibility(row):
            records.append(
                {
                    **base_rec,
                    "vlm_applied": False,
                    "skip_reason": "no_wrist_visibility",
                    **_SESSION_VLM_NULLS,
                }
            )
            elapsed = time.perf_counter() - t0
            eta = (elapsed / i) * (total - i) if i > 0 else None
            _session_progress_line(
                session_label, i, total, chunk_dir.name, image_key, "omitir (sin muñeca)", elapsed_s=elapsed, eta_s=eta
            )
            continue
        if not img_path.is_file():
            records.append(
                {
                    **base_rec,
                    "vlm_applied": False,
                    "skip_reason": "missing_image_file",
                    **_SESSION_VLM_NULLS,
                }
            )
            elapsed = time.perf_counter() - t0
            eta = (elapsed / i) * (total - i) if i > 0 else None
            _session_progress_line(
                session_label, i, total, chunk_dir.name, image_key, "omitir (sin archivo)", elapsed_s=elapsed, eta_s=eta
            )
            continue
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            records.append(
                {
                    **base_rec,
                    "vlm_applied": False,
                    "skip_reason": "imread_failed",
                    **_SESSION_VLM_NULLS,
                }
            )
            elapsed = time.perf_counter() - t0
            eta = (elapsed / i) * (total - i) if i > 0 else None
            _session_progress_line(
                session_label, i, total, chunk_dir.name, image_key, "omitir (lectura img)", elapsed_s=elapsed, eta_s=eta
            )
            continue
        elapsed_pre = time.perf_counter() - t0
        eta_pre = (elapsed_pre / (i - 1)) * (total - (i - 1)) if i > 1 and total > 0 else None
        _session_progress_line(
            session_label,
            i,
            total,
            chunk_dir.name,
            image_key,
            "SigLIP…",
            elapsed_s=elapsed_pre,
            eta_s=eta_pre,
        )
        out = clf.encode_frame_vectors(bgr)
        rec: dict[str, Any] = {
            **base_rec,
            "vlm_applied": True,
            "skip_reason": None,
            "vlm_vector_prompt_probs": np.asarray(out["fused_prompt_probs"], dtype=np.float64).tolist(),
            "vlm_vector_prompt_logits": np.asarray(out["fused_prompt_logits"], dtype=np.float64).tolist(),
            "gated_yes_prob": out["gated_yes_prob"],
            "latency_sec": round(float(out["latency_sec"]), 6),
            "hands_in_shopping_basket_prob": float(out["hands_in_shopping_basket_prob"]),
            "hands_in_shopping_cart_prob": float(out["hands_in_shopping_cart_prob"]),
            "hands_in_shopping_basket_or_cart_prob": float(out["hands_in_shopping_basket_or_cart_prob"]),
        }
        if emit_image_embedding:
            rec["fused_image_embedding"] = np.asarray(out["fused_image_embedding"], dtype=np.float64).tolist()
        else:
            rec["fused_image_embedding"] = None
        records.append(rec)
        elapsed = time.perf_counter() - t0
        eta = (elapsed / i) * (total - i) if i > 0 and total > 0 else None
        siglip_ms = float(out["latency_sec"]) * 1000.0
        _session_progress_line(
            session_label,
            i,
            total,
            chunk_dir.name,
            image_key,
            f"SigLIP ok ({siglip_ms:.0f}ms)",
            elapsed_s=elapsed,
            eta_s=eta,
        )

    if total > 0:
        print(flush=True)

    print("[session] Escribiendo JSON de salida…", flush=True)
    payload: dict[str, Any] = {
        "session_dir": str(session_dir.resolve()),
        "vlm_model": clf.model_name,
        "siglip_prompt_texts": texts,
        "siglip_prompt_labels_es": list(_LABELS_ES_PROBE),
        "vector_description": (
            "vlm_vector_prompt_probs: softmax sobre los 6 textos en siglip_prompt_texts (mismo orden que siglip_prompt_labels_es). "
            "Indice 0 = lleva objeto en manos (no implica robo). 1 = manos vacías. 2-3 = confusores. "
            "4 = cesta, 5 = carrito. "
            "hands_in_shopping_basket_prob y hands_in_shopping_cart_prob repiten probs[4] y probs[5]; "
            "hands_in_shopping_basket_or_cart_prob es su suma (masa conjunta sobre esas dos hipótesis). "
            "Fusion de crops v2. fused_image_embedding opcional."
        ),
        "frames": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    t_write = time.perf_counter()
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_s = time.perf_counter() - t_write
    n_infer = sum(1 for r in records if r.get("vlm_applied"))
    wall = time.perf_counter() - t0
    print(
        f"[session] Listo en {wall:.1f}s (escritura JSON {write_s:.2f}s). "
        f"chunks={len(chunk_dirs)} frames_total={len(records)} vlm_infer={n_infer} -> {output_json}",
        flush=True,
    )


def run_single_image_probe(
    clf: SiglipSO400MClassifier,
    image_path: Path,
    *,
    args: Any,
    yolo_pose_model: Any | None,
    image_roi: tuple[int, int, int, int] | None,
    dump_crops_dir: Path | None,
    output_json: Path | None,
) -> dict[str, Any]:
    """Una imagen → mismos vectores que en sesión; imprime tabla y opcionalmente guarda JSON."""
    path = image_path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"No existe la imagen: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise RuntimeError(f"No se pudo leer la imagen: {path}")
    yolo_applied = False
    yolo_crop_box_xyxy: tuple[int, int, int, int] | None = None
    if yolo_pose_model is not None:
        detections = extract_people_and_hands(
            bgr,
            yolo_pose_model,
            float(getattr(args, "wrist_conf_th", 0.15)),
            float(getattr(args, "elbow_conf_th", 0.10)),
            predict_device=yolo_predict_device_for_args(args),
        )
        if detections:
            # Preferimos la persona con mayor área (escenario típico: una sola persona).
            det = max(
                detections,
                key=lambda d: max(
                    1,
                    (int(d["person_box"][2]) - int(d["person_box"][0]))
                    * (int(d["person_box"][3]) - int(d["person_box"][1])),
                ),
            )
            hand_boxes: list[tuple[int, int, int, int]] = []
            for side in ("left", "right"):
                hd = (det.get("hands") or {}).get(side)
                if not hd:
                    continue
                crop_img, crop_box = build_hand_crop(
                    bgr,
                    tuple(det["person_box"]),
                    tuple(hd["wrist"]),
                    tuple(hd["elbow"]) if hd.get("elbow") is not None else None,
                    int(getattr(args, "crop_size", 220)),
                    int(getattr(args, "crop_min", 180)),
                    int(getattr(args, "crop_max", 420)),
                    crop_mode="upper-torso-hands",
                )
                if crop_img is not None and crop_img.size > 0:
                    hand_boxes.append(crop_box)
            if hand_boxes:
                x1 = min(b[0] for b in hand_boxes)
                y1 = min(b[1] for b in hand_boxes)
                x2 = max(b[2] for b in hand_boxes)
                y2 = max(b[3] for b in hand_boxes)
                yolo_crop_box_xyxy = (int(x1), int(y1), int(x2), int(y2))
            else:
                yolo_crop_box_xyxy = tuple(det["person_box"])
            x1, y1, x2, y2 = yolo_crop_box_xyxy
            bgr = bgr[y1:y2, x1:x2]
            yolo_applied = True
    if image_roi is not None:
        x1, y1, x2, y2 = image_roi
        h, w = bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        bgr = bgr[y1:y2, x1:x2]
    if dump_crops_dir is not None:
        dump_crops_dir.mkdir(parents=True, exist_ok=True)
        crops, _weights = clf._build_crops(bgr)
        cv2.imwrite(str(dump_crops_dir / "probe_input.jpg"), bgr)
        for i, c in enumerate(crops):
            cv2.imwrite(str(dump_crops_dir / f"probe_crop_{i:02d}.jpg"), c)
    t0 = time.perf_counter()
    out = clf.encode_frame_vectors(bgr)
    wall = time.perf_counter() - t0
    probs = np.asarray(out["fused_prompt_probs"], dtype=np.float64)
    logits = np.asarray(out["fused_prompt_logits"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    if len(clf.texts) != len(_LABELS_ES_PROBE):
        raise RuntimeError("Desajuste entre prompts y etiquetas ES en probe.")
    for i, (txt, lab_es) in enumerate(zip(clf.texts, _LABELS_ES_PROBE)):
        rows.append(
            {
                "idx": i,
                "prob": float(probs[i]),
                "logit": float(logits[i]),
                "prompt_en": txt,
                "label_es": lab_es,
            }
        )
    report: dict[str, Any] = {
        "image_path": str(path),
        "image_roi_xyxy": list(image_roi) if image_roi is not None else None,
        "image_yolo_crop_applied": bool(yolo_applied),
        "image_yolo_crop_xyxy": list(yolo_crop_box_xyxy) if yolo_crop_box_xyxy is not None else None,
        "probe_input_shape_hwc": [int(bgr.shape[0]), int(bgr.shape[1]), int(bgr.shape[2])],
        "vlm_model": clf.model_name,
        "device": str(clf.device),
        "multicrop_mode": clf.multicrop_mode,
        "wall_time_sec": round(wall, 4),
        "siglip_latency_sec": float(out["latency_sec"]),
        "gated_yes_prob": float(out["gated_yes_prob"]),
        "hands_in_shopping_basket_prob": float(out["hands_in_shopping_basket_prob"]),
        "hands_in_shopping_cart_prob": float(out["hands_in_shopping_cart_prob"]),
        "hands_in_shopping_basket_or_cart_prob": float(out["hands_in_shopping_basket_or_cart_prob"]),
        "prompts": rows,
    }
    print(f"\n[probe] Imagen: {path}", flush=True)
    print(f"[probe] Modelo: {clf.model_name} | dispositivo: {clf.device} | multicrop: {clf.multicrop_mode}", flush=True)
    # ^ bug: clf.model_model doesn't exist, should be clf.model_name
    print(
        f"[probe] gated_yes_prob={out['gated_yes_prob']:.4f} | "
        f"cesta={out['hands_in_shopping_basket_prob']:.4f} | carro={out['hands_in_shopping_cart_prob']:.4f} | "
        f"{wall:.3f}s wall\n",
        flush=True,
    )
    w = max(40, max(len(r["label_es"]) for r in rows))
    for r in rows:
        bar = "#" * int(round(r["prob"] * 20))
        print(
            f"  [{r['idx']}] {r['prob']*100:5.1f}%  {bar:<20}  {r['label_es']}",
            flush=True,
        )
    print(f"\n  (Prueba --multicrop-mode off si el recorte ya es solo torso/manos.)", flush=True)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[probe] JSON: {output_json}", flush=True)
    if dump_crops_dir is not None:
        print(f"[probe] Crops guardados en: {dump_crops_dir}", flush=True)
    return report


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + SigLIP SO400M (v2).",
        default_vlm_model="google/siglip-so400m-patch14-384",
        vlm_model_help="Id HF modelo SigLIP.",
    )
    p.add_argument(
        "--net-th",
        type=float,
        default=0.35,
        help="Umbral base del score diferencial ponderado (objeto-vacio).",
    )
    p.add_argument(
        "--net-margin-th",
        type=float,
        default=0.30,
        help="Diferencia minima objeto-vacio para validar positivo robusto.",
    )
    p.add_argument(
        "--multicrop-mode",
        default="batch3",
        choices=["off", "light", "full", "batch3"],
        help="Modo multi-crop: batch3 (torso+mano izq+mano der) recomendado.",
    )
    p.add_argument(
        "--decision-mode",
        default="weighted",
        choices=["weighted", "majority", "best_crop"],
        help="weighted: score ponderado; majority: voto por mayoría; best_crop: usa solo el crop con mayor net=objeto-vacío.",
    )
    p.add_argument(
        "--prompt-profile",
        default="default",
        choices=sorted(_PROMPT_PROFILES.keys()),
        help="Perfil rápido de prompts para A/B de falsos positivos.",
    )
    p.add_argument(
        "--session-dir",
        default="",
        help=(
            "Carpeta de sesión con subcarpetas chunk_*/frames/ y chunk_*/frames_meta.json. "
            "Si se indica, no se usa --video/--videos; se escribe un JSON por frame."
        ),
    )
    p.add_argument(
        "--session-output-json",
        default="",
        help="Ruta del JSON de salida con vectores por frame (modo --session-dir). Por defecto: <session-dir>/siglip_v2_session_vectors.json",
    )
    p.add_argument(
        "--emit-image-embedding",
        action="store_true",
        help="Incluir fused_image_embedding (alto dimensional) en el JSON de sesión.",
    )
    p.add_argument(
        "--image",
        default="",
        help="Ruta a una imagen (jpg/png): aplica SigLIP v2 y muestra probs por prompt (exclusivo con --video/--videos/--session-dir).",
    )
    p.add_argument(
        "--image-output-json",
        default="",
        help="Opcional con --image: guarda el informe en este .json.",
    )
    p.add_argument(
        "--image-use-yolo",
        action="store_true",
        help="Solo con --image: aplica YOLO Pose y recorta automáticamente región torso+manos antes del multicrop.",
    )
    p.add_argument(
        "--image-roi",
        default="",
        help="Solo con --image: ROI manual x1,y1,x2,y2 (pixel) para recortar antes del multicrop.",
    )
    p.add_argument(
        "--dump-crops-dir",
        default="",
        help="Solo con --image: carpeta donde guardar probe_input.jpg y probe_crop_XX.jpg para depurar.",
    )
    args = parse_args(p)
    # Variante v2: ROI local por lado (torso superior + manos), evitando frame completo.
    args.per_hand_fast = True
    args.crop_mode = "upper-torso-hands"
    # Defaults mas robustos para distancia larga (4m aprox); se pueden sobrescribir por CLI.
    args.crop_min = max(int(args.crop_min), 180)
    args.crop_max = max(int(args.crop_max), 420)
    args.crop_size = max(int(args.crop_size), 260)
    args.fast_gray_zone = max(float(args.fast_gray_zone), 0.24)
    # Defaults recomendados para estabilidad temporal en este backend.
    if str(args.temporal_mode) == "consecutive":
        args.temporal_mode = "accumulator"
    image_s = str(getattr(args, "image", "") or "").strip()
    session_s = str(getattr(args, "session_dir", "") or "").strip()
    has_video_mode = bool(str(args.video).strip()) != bool(str(args.videos).strip())
    if image_s and (session_s or has_video_mode):
        raise RuntimeError("--image es exclusivo con --session-dir, --video y --videos.")
    if session_s and has_video_mode:
        raise RuntimeError("Con --session-dir no uses --video ni --videos.")
    if image_s or session_s:
        print(
            "[siglip_v2] Cargando modelo (primera vez: descarga HF; CPU lento)…",
            flush=True,
        )
    clf = SiglipSO400MClassifier(
        args.vlm_model,
        args.device,
        args.vlm_prompt,
        prompt_profile=str(args.prompt_profile),
        net_th=float(args.net_th),
        net_margin_th=float(args.net_margin_th),
        decision_mode=str(args.decision_mode),
        multicrop_mode=str(args.multicrop_mode),
    )
    if image_s:
        print(f"[probe] Modelo listo: {args.vlm_model} | dispositivo: {clf.device}", flush=True)
        img_json = str(getattr(args, "image_output_json", "") or "").strip()
        roi_s = str(getattr(args, "image_roi", "") or "").strip()
        dump_s = str(getattr(args, "dump_crops_dir", "") or "").strip()
        use_yolo_for_image = bool(getattr(args, "image_use_yolo", False))
        roi_xyxy: tuple[int, int, int, int] | None = None
        pose_model = None
        if use_yolo_for_image and roi_s:
            raise RuntimeError("Usa solo uno: --image-use-yolo o --image-roi (no ambos).")
        if use_yolo_for_image:
            args.pose_weights = resolve_yolo_weights_for_runtime(str(args.pose_weights), allow_tensorrt=False)
            print(f"[probe] Cargando YOLO pose: {args.pose_weights}", flush=True)
            pose_model = YOLO(str(args.pose_weights), task="pose")
            if args.device and not str(args.pose_weights).lower().endswith(".engine"):
                pose_model.to(args.device)
        if roi_s:
            toks = [t.strip() for t in roi_s.split(",")]
            if len(toks) != 4:
                raise RuntimeError("--image-roi debe ser x1,y1,x2,y2")
            try:
                x1, y1, x2, y2 = (int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3]))
            except ValueError as e:
                raise RuntimeError("--image-roi debe contener enteros: x1,y1,x2,y2") from e
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError("--image-roi inválido: se requiere x2>x1 e y2>y1")
            roi_xyxy = (x1, y1, x2, y2)
        run_single_image_probe(
            clf,
            Path(image_s),
            args=args,
            yolo_pose_model=pose_model,
            image_roi=roi_xyxy,
            dump_crops_dir=Path(dump_s).expanduser().resolve() if dump_s else None,
            output_json=Path(img_json).expanduser().resolve() if img_json else None,
        )
        return
    if session_s:
        print(f"[session] Modelo listo: {args.vlm_model} | dispositivo: {clf.device}", flush=True)
        session_path = Path(session_s).expanduser().resolve()
        out_s = str(getattr(args, "session_output_json", "") or "").strip()
        if out_s:
            out_path = Path(out_s).expanduser().resolve()
        else:
            out_path = session_path / "siglip_v2_session_vectors.json"
        run_session_siglip_v2(
            session_path,
            clf,
            output_json=out_path,
            emit_image_embedding=bool(getattr(args, "emit_image_embedding", False)),
        )
        return
    run_pipeline(
        args,
        clf,
        window_title="hand-object siglip so400m v2",
        batch_output_suffix="_siglip_so400m_v2",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
