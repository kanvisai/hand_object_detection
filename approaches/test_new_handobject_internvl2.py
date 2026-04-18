#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + InternVL2-2B (chat generativo)."""

from handobject_classifiers import GenericChatVLMClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + InternVL2-2B.",
        default_vlm_model="OpenGVLab/InternVL2-2B",
        vlm_model_help="Id HF o carpeta local InternVL2.",
    )
    args = parse_args(p)
    clf = GenericChatVLMClassifier(
        args.vlm_model, args.device, args.vlm_prompt, backend_name="internvl2"
    )
    run_pipeline(
        args,
        clf,
        window_title="hand-object internvl2",
        batch_output_suffix="_internvl2",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
