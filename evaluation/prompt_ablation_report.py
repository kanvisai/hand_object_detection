#!/usr/bin/env python3
"""
Pipeline “un clic”: corre `session_semantics` con SigLIP SO400M y **todas** las variantes de prompts
únicas, luego fusiona por variante todos los chunks bajo `--chunk-parent-dir` y muestra el
**resumen de acciones** (interaction_reasoner) por variante para comparar prompts.

Uso típico (desde esta carpeta `evaluation/`):

  python3 prompt_ablation_report.py --chunk-parent-dir /ruta/sesion/

Opciones:
  --skip-semantics     Solo informe (requiere JSON ya generados por variante en cada chunk).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from interaction_reasoner import merge_chunk_semantics_payloads, reason_interaction  # noqa: E402
from retail_prompt_variants import list_experiment_variant_ids  # noqa: E402
from session_semantics import _semantics_output_filename, discover_chunk_dirs_under_parent  # noqa: E402

DEFAULT_VLM = "google/siglip-so400m-patch14-384"


def _reason_defaults() -> dict[str, Any]:
    sw = 3
    if sw % 2 == 0:
        sw += 1
    return {
        "smooth_window": sw,
        "min_evaluable_frames": 4,
        "min_run_object": 2,
        "min_run_deposit": 2,
        "treat_unknown_as_gap": False,
        "summary_per_chunk_max": 2,
        "summary_global_max": 5,
    }


def run_session_semantics(parent: Path, vlm_model: str, quiet: bool, hf_token: str) -> None:
    cmd = [
        sys.executable,
        str(_EVAL / "session_semantics.py"),
        "--chunk-parent-dir",
        str(parent.resolve()),
        "--vlm-backend",
        "siglip",
        "--vlm-model",
        vlm_model,
        "--prompt-variant-all",
    ]
    if quiet:
        cmd.append("--quiet")
    if hf_token.strip():
        cmd.extend(["--hf-token", hf_token.strip()])
    print("[ablation] Ejecutando session_semantics (todas las variantes únicas)…", flush=True)
    subprocess.check_call(cmd, cwd=str(_EVAL))


def print_variant_reports(parent: Path, backend: str = "siglip") -> None:
    variant_ids = list_experiment_variant_ids()
    chunk_dirs = discover_chunk_dirs_under_parent(parent)
    if not chunk_dirs:
        raise SystemExit(f"No hay chunks válidos bajo {parent.resolve()}")

    rd = _reason_defaults()
    print("\n" + "=" * 72, flush=True)
    print(
        f"INFORME — resumen de acciones por variante de prompts ({backend}, todos los chunks fusionados)",
        flush=True,
    )
    print(f"Carpeta padre: {parent.resolve()}", flush=True)
    print(f"Chunks ({len(chunk_dirs)}): {', '.join(c.name for c in chunk_dirs)}", flush=True)
    print(f"Variantes únicas ({len(variant_ids)}): {', '.join(variant_ids)}", flush=True)
    print("=" * 72 + "\n", flush=True)

    for pv in variant_ids:
        paths: list[Path] = []
        missing: list[str] = []
        for cd in chunk_dirs:
            fn = _semantics_output_filename(cd.name, backend, pv, all_pv_ids=variant_ids)
            p = cd / fn
            if p.is_file():
                paths.append(p)
            else:
                missing.append(str(p))

        print(f"### Variante: `{pv}`", flush=True)
        if missing:
            print(f"  (AVISO: faltan {len(missing)} JSON esperados)", flush=True)
            for m in missing[:6]:
                print(f"    - {m}", flush=True)
            if len(missing) > 6:
                print(f"    … (+{len(missing) - 6})", flush=True)

        if len(paths) < len(chunk_dirs):
            print("  → Sin fusionar (faltan ficheros). Saltando reasoner para esta variante.\n", flush=True)
            continue

        payload = merge_chunk_semantics_payloads(paths)
        result = reason_interaction(payload, **rd)
        verdict = result.get("verdict")
        conf = result.get("confidence")
        summ = result.get("session_summary_es") or {}
        print(f"  verdict={verdict}  confidence={conf}", flush=True)
        note = str(summ.get("note") or "").strip()
        if note:
            print(f"  nota: {note}", flush=True)
        ol = summ.get("one_liner_es")
        if ol:
            print(f"  Secuencia compacta: {ol}", flush=True)
        lines = summ.get("lines_es")
        if isinstance(lines, list) and lines:
            print("  Resumen de acciones:", flush=True)
            for line in lines:
                print(f"    {line}", flush=True)
        print(flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Corre session_semantics con todas las variantes de prompts y muestra resumen de acciones por variante.",
    )
    p.add_argument(
        "--chunk-parent-dir",
        required=True,
        metavar="DIR",
        help="Carpeta que contiene chunk_001, chunk_002, …",
    )
    p.add_argument(
        "--skip-semantics",
        action="store_true",
        help="No ejecutar session_semantics; solo generar informe desde JSON ya existentes.",
    )
    p.add_argument("--vlm-model", default=DEFAULT_VLM, help=f"Por defecto: {DEFAULT_VLM}")
    p.add_argument("--quiet-semantics", action="store_true", help="Pasa --quiet a session_semantics.")
    p.add_argument(
        "--hf-token",
        default="",
        help="Opcional: token HF si session_semantics necesita descargar el modelo.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    parent = Path(args.chunk_parent_dir).expanduser().resolve()
    if not parent.is_dir():
        raise SystemExit(f"No es un directorio: {parent}")

    if not args.skip_semantics:
        run_session_semantics(parent, str(args.vlm_model), bool(args.quiet_semantics), str(args.hf_token))

    print_variant_reports(parent, backend="siglip")


if __name__ == "__main__":
    main()
