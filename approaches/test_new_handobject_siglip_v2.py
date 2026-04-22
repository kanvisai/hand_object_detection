#!/usr/bin/env python3
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
from handobject_shared import build_parser, parse_args, run_pipeline


class SiglipSO400MClassifier(ClipLikeClassifier):
    """SigLIP SO400M con batch multiescala y score diferencial."""

    def __init__(
        self,
        model_name: str,
        device: str,
        prompt: str,
        *,
        net_th: float = 0.35,
        net_margin_th: float = 0.30,
        multicrop_mode: str = "batch3",
        torso_weight: float = 0.45,
        left_weight: float = 0.275,
        right_weight: float = 0.275,
    ) -> None:
        super().__init__(model_name, device, prompt, backend_name="siglip")
        self.texts = [
            "A person holding a stolen object in their hands.",
            "A person with empty hands walking.",
            "A person with hands in their pockets.",
            "A person clapping or rubbing their hands together.",
        ]
        self.net_th = float(max(0.0, net_th))
        self.net_margin_th = float(max(0.0, net_margin_th))
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
        crops, weights = self._build_crops(bgr)
        with torch.no_grad():
            t0 = time.perf_counter()
            p_obj, p_empty, p_other = self._score_crops_batch(crops)
            latency = time.perf_counter() - t0
        w = np.array(weights, dtype=np.float32)
        w = w / max(1e-6, float(np.sum(w)))
        net = p_obj - p_empty
        net_global = float(np.sum(w * net))
        p_obj_global = float(np.sum(w * p_obj))
        p_other_global = float(np.sum(w * p_other))
        margin_ok = (net_global > self.net_margin_th) and (p_obj_global > (p_other_global + 0.05))
        gated_yes = p_obj_global if (margin_ok and net_global > self.net_th) else 0.0
        self.last_answer_text = "YES" if gated_yes > 0.0 else "NO"
        self.last_debug = (
            f"p_obj={p_obj_global:.3f} net={net_global:.3f} net_th={self.net_th:.3f} "
            f"net_margin={self.net_margin_th:.3f} p_other={p_other_global:.3f} "
            f"multi={self.multicrop_mode}"
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
    clf = SiglipSO400MClassifier(
        args.vlm_model,
        args.device,
        args.vlm_prompt,
        net_th=float(args.net_th),
        net_margin_th=float(args.net_margin_th),
        multicrop_mode=str(args.multicrop_mode),
    )
    run_pipeline(
        args,
        clf,
        window_title="hand-object siglip so400m v2",
        batch_output_suffix="_siglip_so400m_v2",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
