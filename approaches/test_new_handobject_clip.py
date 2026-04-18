#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + CLIP (embeddings positivo/negativo; no genera texto libre)."""

from handobject_classifiers import ClipLikeClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + CLIP.",
        default_vlm_model="openai/clip-vit-base-patch32",
        vlm_model_help="Id HF modelo CLIP (ViT+texto).",
    )
    args = parse_args(p)
    clf = ClipLikeClassifier(
        args.vlm_model, args.device, args.vlm_prompt, backend_name="clip"
    )
    run_pipeline(
        args,
        clf,
        window_title="hand-object clip",
        batch_output_suffix="_clip",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
