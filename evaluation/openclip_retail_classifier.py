"""
OpenCLIP / MobileCLIP (vía open_clip) con los mismos prompts retail y multicrop que SigLIP.

Requiere: pip install open_clip_torch
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

_ep = Path(__file__).resolve().parent
if str(_ep) not in sys.path:
    sys.path.insert(0, str(_ep))

import cv2
import numpy as np
import torch
from PIL import Image

from retail_semantic_prompts import (
    CANONICAL_LABELS,
    IDX_PERSONAL_BAG,
    IDX_SHOPPING_BASKET,
    IDX_SHOPPING_CART,
    RETAIL_PROMPT_TEXTS_EN,
)


class OpenClipRetailClassifier:
    """CLIP/OpenCLIP API: encode_image / encode_text, fusión multicrop idéntica a Siglip."""

    def __init__(
        self,
        openclip_model_name: str,
        pretrained: str,
        device: str,
        *,
        prompt_texts_en: list[str] | None = None,
        net_th: float = 0.35,
        net_margin_th: float = 0.30,
        decision_mode: str = "weighted",
        multicrop_mode: str = "batch3",
        torso_weight: float = 0.45,
        left_weight: float = 0.275,
        right_weight: float = 0.275,
    ) -> None:
        try:
            import open_clip
        except ImportError as e:
            raise RuntimeError("Backend openclip/mobileclip requiere: pip install open_clip_torch") from e

        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            openclip_model_name,
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(openclip_model_name)

        self.model_name = f"{openclip_model_name}::{pretrained}"
        texts = list(prompt_texts_en) if prompt_texts_en is not None else list(RETAIL_PROMPT_TEXTS_EN)
        if len(texts) != len(CANONICAL_LABELS):
            raise ValueError(
                f"OpenClipRetailClassifier: hace falta {len(CANONICAL_LABELS)} prompts; "
                f"se recibieron {len(texts)}."
            )
        self.texts = texts
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
            tok = self.tokenizer(self.texts)
            tf = self.model.encode_text(tok.to(self.device))
            self.text_features = tf / tf.norm(dim=-1, keepdim=True)

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
            return (
                [bgr, self._center_zoom(bgr, scale=0.85), cv2.convertScaleAbs(bgr, alpha=1.15, beta=6.0)],
                [0.50, 0.30, 0.20],
            )
        return [
            bgr,
            self._hand_half_crop(bgr, "left"),
            self._hand_half_crop(bgr, "right"),
        ], self.crop_weights

    def _encode_image_rgb_batch(self, pil_images: list[Image.Image]) -> torch.Tensor:
        xs = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
        feat = self.model.encode_image(xs)
        return feat / feat.norm(dim=-1, keepdim=True)

    def _forward_multicrop(self, bgr: np.ndarray) -> dict[str, Any]:
        crops, weights = self._build_crops(bgr)
        pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        with torch.no_grad():
            t0 = time.perf_counter()
            img_feat = self._encode_image_rgb_batch(pil_images)
            logits_t = 100.0 * img_feat @ self.text_features.T
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
        fused_logits = np.sum(w[:, None] * logits_np, axis=0)
        fused_probs = np.sum(w[:, None] * probs_np, axis=0)
        fused_probs = fused_probs / max(1e-9, float(np.sum(fused_probs)))
        fused_emb = np.sum(w[:, None] * img_np, axis=0)
        fused_emb = fused_emb / max(1e-9, float(np.linalg.norm(fused_emb)))
        net = p_obj - p_empty
        net_global = float(np.sum(w * net))
        p_obj_global = float(np.sum(w * p_obj))
        p_other_global = float(np.sum(w * p_other_max))
        margin_ok = (net_global > self.net_margin_th) and (p_obj_global > (p_other_global + 0.05))
        if self.decision_mode == "majority":
            positives = ((net > self.net_th) & (net > self.net_margin_th) & (p_obj > (p_other_max + 0.05))).astype(np.int32)
            votes_yes = int(np.sum(positives))
            votes_needed = (len(crops) // 2) + 1
            gated_yes = p_obj_global if votes_yes >= votes_needed else 0.0
        else:
            votes_yes = 0
            votes_needed = 0
            gated_yes = p_obj_global if (margin_ok and net_global > self.net_th) else 0.0

        pb = float(fused_probs[IDX_SHOPPING_BASKET])
        pc = float(fused_probs[IDX_SHOPPING_CART])
        pbag = float(fused_probs[IDX_PERSONAL_BAG])

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
            "hands_in_shopping_basket_prob": pb,
            "hands_in_shopping_cart_prob": pc,
            "hands_in_personal_bag_prob": pbag,
            "hands_in_shopping_basket_or_cart_prob": float(pb + pc),
            "hands_basket_cart_or_personal_bag_prob": float(pb + pc + pbag),
        }

    def encode_frame_vectors(self, bgr: np.ndarray) -> dict[str, Any]:
        return self._forward_multicrop(bgr)
