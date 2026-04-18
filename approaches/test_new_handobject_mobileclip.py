#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Mano-objeto + MobileCLIP (OpenCLIP Hub; embeddings positivo/negativo)."""

from handobject_classifiers import OpenClipLikeClassifier
from handobject_shared import build_parser, parse_args, run_pipeline


def main() -> None:
    p = build_parser(
        description="Deteccion mano-objeto con YOLO Pose + MobileCLIP (open_clip).",
        default_vlm_model="hf-hub:apple/MobileCLIP-S1-OpenCLIP",
        vlm_model_help=(
            "Nombre Hub OpenCLIP, p. ej. hf-hub:apple/MobileCLIP-S1-OpenCLIP "
            "(requiere open-clip-torch)."
        ),
    )
    args = parse_args(p)
    clf = OpenClipLikeClassifier(
        args.vlm_model, args.device, args.vlm_prompt, backend_name="mobileclip"
    )
    run_pipeline(
        args,
        clf,
        window_title="hand-object mobileclip",
        batch_output_suffix="_mobileclip",
        experiment_backend=clf.experiment_backend,
    )


if __name__ == "__main__":
    main()
