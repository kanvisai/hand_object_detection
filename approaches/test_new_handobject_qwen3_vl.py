#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + Qwen3-VL (misma logica que Qwen2 si el template es compatible)."""

from handobject_classifiers import Qwen3VLHandClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + Qwen3-VL.",
        default_vlm_model="Qwen/Qwen3-VL-2B-Instruct",
        vlm_model_help="Id HF o carpeta local Qwen3-VL (debe contener qwen3-vl en el nombre).",
    )
    args = parse_args(p)
    clf = Qwen3VLHandClassifier(args.vlm_model, args.device, args.vlm_prompt)
    run_pipeline(
        args,
        clf,
        window_title="hand-object qwen3-vl",
        batch_output_suffix="_qwen3vl",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
