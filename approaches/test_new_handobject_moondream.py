#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + Moondream2 (vikhyatk/moondream2)."""

from handobject_classifiers import MoondreamHandClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + Moondream2.",
        default_vlm_model="vikhyatk/moondream2",
        vlm_model_help="Id HF o carpeta local de Moondream2.",
    )
    args = parse_args(p)
    clf = MoondreamHandClassifier(args.vlm_model, args.device, args.vlm_prompt)
    run_pipeline(
        args,
        clf,
        window_title="hand-object moondream2",
        batch_output_suffix="_moondream",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
