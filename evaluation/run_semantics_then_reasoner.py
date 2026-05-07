#!/usr/bin/env python3
"""
Ejecuta de un tirón:
1) session_semantics.py sobre --chunk-parent-dir
2) interaction_reasoner.py con los JSON generados

Pensado para una ejecución rápida end-to-end sin montar comandos manuales.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from retail_prompt_variants import list_experiment_variant_ids, list_variant_ids  # noqa: E402
from session_semantics import _semantics_output_filename, discover_chunk_dirs_under_parent  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lanza session_semantics y luego interaction_reasoner en una sola orden."
    )
    p.add_argument("--chunk-parent-dir", required=True, metavar="DIR", help="Carpeta con chunk_001, chunk_002, ...")
    p.add_argument("--skip-semantics", action="store_true", help="No ejecutar session_semantics; solo reasoner.")
    p.add_argument("--vlm-backend", default="siglip", choices=["siglip", "openclip", "mobileclip"])
    p.add_argument("--device", default="cpu", help="cpu o cuda:0")
    p.add_argument("--hf-token", default="", help="Opcional: token HF para SigLIP.")
    p.add_argument("--vlm-model", default="google/siglip-so400m-patch14-384", help="Modelo SigLIP.")
    p.add_argument("--openclip-model", default="ViT-B-32")
    p.add_argument("--openclip-pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--mobileclip-model", default="MobileCLIP2-S0")
    p.add_argument("--mobileclip-pretrained", default="dfndr2b")
    p.add_argument("--multicrop-mode", default="batch3", choices=["off", "light", "full", "batch3"])
    p.add_argument("--decision-mode", default="weighted", choices=["weighted", "majority"])
    p.add_argument("--net-th", type=float, default=0.35)
    p.add_argument("--net-margin-th", type=float, default=0.30)
    p.add_argument("--tau-max-prob", type=float, default=0.35)
    p.add_argument("--min-margin", type=float, default=0.08)
    p.add_argument("--max-entropy", type=float, default=-1.0)
    p.add_argument("--prompt-variant-all", action="store_true", help="Usa todas las variantes únicas.")
    p.add_argument(
        "--prompt-variant",
        nargs="*",
        default=None,
        metavar="ID",
        help=f"Variantes concretas. IDs: {', '.join(list_variant_ids())}",
    )
    p.add_argument("--reasoner-output-dir", default="", help="Si se indica, guarda un JSON de reasoner por variante.")
    p.add_argument("--quiet-semantics", action="store_true")
    p.add_argument("--quiet-summary", action="store_true")
    p.add_argument("--summary-per-chunk-max", type=int, default=2)
    p.add_argument("--summary-global-max", type=int, default=5)
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--min-evaluable-frames", type=int, default=4)
    p.add_argument("--min-run-object", type=int, default=2)
    p.add_argument("--min-run-deposit", type=int, default=2)
    p.add_argument("--treat-unknown-as-gap", action="store_true")
    return p


def _resolve_variants(args: argparse.Namespace) -> list[str]:
    if bool(args.prompt_variant_all) and args.prompt_variant:
        raise SystemExit("No combines --prompt-variant-all con --prompt-variant.")
    if bool(args.prompt_variant_all):
        return list_experiment_variant_ids()
    vals = list(args.prompt_variant or [])
    return vals if vals else ["default"]


def _run_session_semantics(args: argparse.Namespace, variants: list[str]) -> None:
    cmd = [
        sys.executable,
        str(_EVAL / "session_semantics.py"),
        "--chunk-parent-dir",
        str(Path(args.chunk_parent_dir).expanduser().resolve()),
        "--vlm-backend",
        str(args.vlm_backend),
        "--device",
        str(args.device),
        "--vlm-model",
        str(args.vlm_model),
        "--openclip-model",
        str(args.openclip_model),
        "--openclip-pretrained",
        str(args.openclip_pretrained),
        "--mobileclip-model",
        str(args.mobileclip_model),
        "--mobileclip-pretrained",
        str(args.mobileclip_pretrained),
        "--multicrop-mode",
        str(args.multicrop_mode),
        "--decision-mode",
        str(args.decision_mode),
        "--net-th",
        str(args.net_th),
        "--net-margin-th",
        str(args.net_margin_th),
        "--tau-max-prob",
        str(args.tau_max_prob),
        "--min-margin",
        str(args.min_margin),
        "--max-entropy",
        str(args.max_entropy),
        "--prompt-variant",
        *variants,
    ]
    if args.hf_token:
        cmd += ["--hf-token", str(args.hf_token)]
    if bool(args.quiet_semantics):
        cmd.append("--quiet")
    print(f"[runner] Ejecutando session_semantics para {len(variants)} variante(s)...", flush=True)
    subprocess.check_call(cmd, cwd=str(_EVAL))


def _collect_variant_paths(parent: Path, backend: str, variants: list[str]) -> dict[str, list[Path]]:
    chunk_dirs = discover_chunk_dirs_under_parent(parent)
    if not chunk_dirs:
        raise SystemExit(f"No hay chunks válidos bajo: {parent}")
    out: dict[str, list[Path]] = {}
    for v in variants:
        paths: list[Path] = []
        for cd in chunk_dirs:
            fn = _semantics_output_filename(cd.name, backend, v, all_pv_ids=variants)
            p = cd / fn
            if not p.is_file():
                raise SystemExit(f"Falta JSON para variante {v!r}: {p}")
            paths.append(p)
        out[v] = paths
    return out


def _run_reasoner_for_variant(args: argparse.Namespace, variant: str, paths: list[Path]) -> None:
    cmd = [
        sys.executable,
        str(_EVAL / "interaction_reasoner.py"),
        "--input-json",
        *[str(p) for p in paths],
        "--smooth-window",
        str(args.smooth_window),
        "--min-evaluable-frames",
        str(args.min_evaluable_frames),
        "--min-run-object",
        str(args.min_run_object),
        "--min-run-deposit",
        str(args.min_run_deposit),
        "--summary-per-chunk-max",
        str(args.summary_per_chunk_max),
        "--summary-global-max",
        str(args.summary_global_max),
    ]
    if bool(args.treat_unknown_as_gap):
        cmd.append("--treat-unknown-as-gap")
    if bool(args.quiet_summary):
        cmd.append("--quiet-summary")
    if args.reasoner_output_dir:
        out_dir = Path(args.reasoner_output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"reasoner_{args.vlm_backend}_{variant}.json"
        cmd += ["--output-json", str(out_json)]

    print(f"\n[runner] interaction_reasoner para variante: {variant}", flush=True)
    subprocess.check_call(cmd, cwd=str(_EVAL))


def main() -> None:
    args = build_arg_parser().parse_args()
    parent = Path(args.chunk_parent_dir).expanduser().resolve()
    if not parent.is_dir():
        raise SystemExit(f"No es un directorio: {parent}")
    variants = _resolve_variants(args)

    if not bool(args.skip_semantics):
        _run_session_semantics(args, variants)

    grouped = _collect_variant_paths(parent, str(args.vlm_backend), variants)
    for v in variants:
        _run_reasoner_for_variant(args, v, grouped[v])


if __name__ == "__main__":
    main()
