#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + Florence-2 base (HF florence-community)."""

from handobject_classifiers import Florence2HandClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + Florence-2 base.",
        default_vlm_model="florence-community/Florence-2-base",
        vlm_model_help="Id HF o carpeta local (microsoft/Florence-2-base se redirige a florence-community).",
        # Formato "Y/N — ...": la generación suele ser "Y — ..." o "N — ..."; parseable y menos eco.
        default_vlm_prompt="Y/N — object clearly grasped in this crop?",
    )
    args = parse_args(p)
    clf = Florence2HandClassifier(args.vlm_model, args.device, args.vlm_prompt)
    run_pipeline(
        args,
        clf,
        window_title="hand-object florence2-base",
        batch_output_suffix="_florence2_base",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
