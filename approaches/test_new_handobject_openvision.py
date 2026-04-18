#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + OpenVision (OpenCLIP Hub; embeddings positivo/negativo)."""

from handobject_classifiers import OpenClipLikeClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + OpenVision (open_clip).",
        default_vlm_model="hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224",
        vlm_model_help=(
            "Nombre Hub OpenCLIP, p. ej. hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224 "
            "(requiere open-clip-torch; ver coleccion UCSC-VLAA/OpenVision)."
        ),
    )
    args = parse_args(p)
    clf = OpenClipLikeClassifier(
        args.vlm_model, args.device, args.vlm_prompt, backend_name="openvision"
    )
    run_pipeline(
        args,
        clf,
        window_title="hand-object openvision",
        batch_output_suffix="_openvision",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
