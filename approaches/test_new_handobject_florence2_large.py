#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + Florence-2 large (HF florence-community; misma logica que test_new_handobject_florence2_base)."""

from handobject_classifiers import Florence2HandClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + Florence-2 large.",
        default_vlm_model="florence-community/Florence-2-large",
        vlm_model_help="Id HF o carpeta local (microsoft/Florence-2-large se redirige a florence-community).",
        # Mismo prompt que base: formato "Y/N — ..."; echo + reintento en Florence2HandClassifier.
        default_vlm_prompt="Y/N — object clearly grasped in this crop?",
    )
    args = parse_args(p)
    clf = Florence2HandClassifier(args.vlm_model, args.device, args.vlm_prompt)
    run_pipeline(
        args,
        clf,
        window_title="hand-object florence2-large",
        batch_output_suffix="_florence2_large",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
