#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + PaliGemma (Google)."""

from handobject_classifiers import GenericChatVLMClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + PaliGemma.",
        default_vlm_model="google/paligemma2-3b-pt-224",
        vlm_model_help="Id HF PaliGemma (ajusta variante 2B/3B segun necesites).",
    )
    args = parse_args(p)
    clf = GenericChatVLMClassifier(
        args.vlm_model, args.device, args.vlm_prompt, backend_name="paligemma"
    )
    run_pipeline(
        args,
        clf,
        window_title="hand-object paligemma",
        batch_output_suffix="_paligemma",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
