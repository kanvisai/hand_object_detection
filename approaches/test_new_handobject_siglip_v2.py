#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + SigLIP (embeddings positivo/negativo)."""

from handobject_classifiers import ClipLikeClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


class SiglipSO400MClassifier(ClipLikeClassifier):
    """SigLIP SO400M con anclas binarias A/B para mano-objeto."""

    def __init__(self, model_name: str, device: str, prompt: str) -> None:
        super().__init__(model_name, device, prompt, backend_name="siglip")
        self.positive_texts = ["A person holding an object in their hands."]
        self.negative_texts = ["A person with empty hands."]
        self.texts = self.positive_texts + self.negative_texts
        self.n_pos = len(self.positive_texts)
        with torch.no_grad():
            txt = self.processor(text=self.texts, return_tensors="pt", padding=True).to(self.device)
            txt_feat = self._encode_text(txt)
            self.text_features = self._l2_normalize(txt_feat)


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + SigLIP SO400M (v2).",
        default_vlm_model="google/siglip-so400m-patch14-384",
        vlm_model_help="Id HF modelo SigLIP.",
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
    # Regla solicitada: YES solo si p_yes > 0.80 en 2 frames consecutivos.
    # En cualquier otro caso, NO/soltado con caida rapida.
    args.yes_th = 0.80
    args.hold_frames = 2
    args.drop_frames = 1
    args.raw_on_th = 0.80
    args.raw_on_frames = 2
    args.force_drop_th = 0.80
    args.force_drop_frames = 1
    args.raw_drop_th = 0.80
    args.raw_drop_frames = 1
    clf = SiglipSO400MClassifier(args.vlm_model, args.device, args.vlm_prompt)
    run_pipeline(
        args,
        clf,
        window_title="hand-object siglip so400m v2",
        batch_output_suffix="_siglip_so400m_v2",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
