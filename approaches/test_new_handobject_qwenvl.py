#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto con YOLO Pose + Qwen2-VL. Ver handobject_shared / handobject_classifiers."""

from handobject_classifiers import Qwen2VLHandClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + Qwen2-VL-2B-Instruct.",
        default_vlm_model="Qwen/Qwen2-VL-2B-Instruct",
        vlm_model_help="Carpeta local o id HF del modelo Qwen2-VL.",
    )
    args = parse_args(p)
    clf = Qwen2VLHandClassifier(args.vlm_model, args.device, args.vlm_prompt)
    run_pipeline(
        args,
        clf,
        window_title="hand-object qwen2-vl",
        batch_output_suffix="_qwen2vl",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
