#!/usr/bin/env python3
"""
Agrega JSON de semántica rc1 por todos los chunks bajo una carpeta y aplica `robbery_rules_score`.

Busca en cada subcarpeta de chunk el JSON generado por `run_semantics_rc1.py` (una ejecución por chunk con `--chunk-dir`).
(por defecto `<chunk_stem>_siglip_<PROMPT_VARIANT_ID>.json` desde `vlm_rc1_prompts.py`), concatena `frames` en orden
y calcula probabilidad de robo con la misma lógica que `robbery_rules_score.py`.

Por defecto guarda `<chunk-main-dir>/vlm_rc1_robbery_evaluation.json` y escribe el **mismo**
objeto JSON en **stdout** (una línea). Mensajes `[vlm_rc1]` en stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_EVAL = Path(__file__).resolve().parent.parent
_RCDIR = Path(__file__).resolve().parent
for _p in (_EVAL, _RCDIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from robbery_rules_score import compute_robbery_rules_score  # noqa: E402
from session_semantics import discover_chunk_dirs_under_parent  # noqa: E402

from vlm_rc1_config import PROMPT_VARIANT_ID, semantics_filename_for_chunk, write_json_stdout  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_frames_from_semantics_files(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    frames: list[dict[str, Any]] = []
    errors: list[str] = []
    for p in paths:
        try:
            data = _load_json(p)
        except Exception as e:
            errors.append(f"{p}: lectura JSON fallida ({e})")
            continue
        frs = data.get("frames")
        if not isinstance(frs, list):
            errors.append(f"{p}: falta lista 'frames'")
            continue
        frames.extend([x for x in frs if isinstance(x, dict)])
    return frames, errors


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="vlm_rc1: reglas de robo agregadas sobre todos los chunks.")
    p.add_argument(
        "--chunk-main-dir",
        "--chunk_main_dir",
        required=True,
        metavar="DIR",
        help="Directorio que contiene chunk_001, chunk_002, … (cada uno con frames/ + frames_meta.json).",
    )
    p.add_argument(
        "--vlm-backend",
        default="siglip",
        choices=["siglip", "openclip", "mobileclip"],
        help="Debe coincidir con el backend usado en run_semantics_rc1.",
    )
    p.add_argument(
        "--output-json",
        default="",
        help="Ruta del JSON de evaluación (por defecto <chunk-main-dir>/vlm_rc1_robbery_evaluation.json).",
    )
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--min-object-run", type=int, default=2)
    p.add_argument("--lookahead-runs", type=int, default=4)
    p.add_argument("--w-severe", type=float, default=0.55)
    p.add_argument("--w-suspicious", type=float, default=0.30)
    p.add_argument("--w-normal", type=float, default=0.30)
    p.add_argument("--base", type=float, default=0.20)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fallar si falta el JSON de semántica en algún chunk.",
    )
    p.add_argument(
        "--pretty-json",
        action="store_true",
        help="JSON formateado en stdout (por defecto: una línea compacta).",
    )
    p.add_argument(
        "--no-write-file",
        action="store_true",
        help="No guardar evaluación en disco; solo stdout.",
    )
    p.add_argument("--quiet", action="store_true", help="Sin mensajes en stderr.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    def _log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    root = Path(args.chunk_main_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"No es un directorio: {root}", file=sys.stderr)
        raise SystemExit(1)

    pv_id = PROMPT_VARIANT_ID
    backend = str(args.vlm_backend)

    chunk_dirs = discover_chunk_dirs_under_parent(root)
    if not chunk_dirs:
        print(
            f"No hay subcarpetas válidas (frames/ + frames_meta.json) bajo:\n  {root}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    semantics_paths: list[Path] = []
    missing: list[str] = []
    for ch in chunk_dirs:
        expected_name = semantics_filename_for_chunk(ch.name, backend, pv_id)
        cand = ch / expected_name
        if cand.is_file():
            semantics_paths.append(cand)
            continue
        # Resolución laxa: un único *.json que coincida con sufijo _backend_variant.json
        alt = sorted(ch.glob(f"*_{backend}_*.json"))
        if len(alt) == 1:
            semantics_paths.append(alt[0])
            continue
        missing.append(f"{ch} (esperado {expected_name})")

    if missing and args.strict:
        print("Faltan JSON de semántica en:\n  " + "\n  ".join(missing), file=sys.stderr)
        raise SystemExit(1)

    if not semantics_paths:
        print(
            "No se encontró ningún JSON de semántica rc1. Ejecuta antes run_semantics_rc1.py por cada chunk "
            f"y revisa `PROMPT_VARIANT_ID` en vlm_rc1_prompts.py / --vlm-backend (esperado por chunk: "
            f"{semantics_filename_for_chunk('chunk_XXX', backend, pv_id)}).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    frames, load_errors = _collect_frames_from_semantics_files(semantics_paths)
    result_body = compute_robbery_rules_score(
        frames,
        smooth_window=int(args.smooth_window),
        min_object_run=int(args.min_object_run),
        lookahead_runs=int(args.lookahead_runs),
        w_severe=float(args.w_severe),
        w_suspicious=float(args.w_suspicious),
        w_normal=float(args.w_normal),
        base=float(args.base),
    )

    out_path_s = str(args.output_json or "").strip()
    if args.no_write_file:
        out_path = Path(os.devnull)
    elif out_path_s:
        out_path = Path(out_path_s).expanduser().resolve()
    else:
        out_path = root / "vlm_rc1_robbery_evaluation.json"

    payload: dict[str, Any] = {
        "schema_version": "vlm_rc1_aggregate_rules_1.0",
        "chunk_main_dir": str(root),
        "prompt_variant_id": pv_id,
        "vlm_backend": backend,
        "semantics_inputs": [str(p) for p in semantics_paths],
        "chunks_discovered": len(chunk_dirs),
        "chunks_with_semantics_json": len(semantics_paths),
        "chunks_missing_semantics_json": missing,
        "load_errors": load_errors,
        "aggregation_note": "Frames de todos los chunks fusionados y ordenados por (chunk, sample_idx).",
        **result_body,
    }

    if not args.no_write_file:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    write_json_stdout(payload, pretty=bool(args.pretty_json))

    _log(f"[vlm_rc1] Evaluación guardada: {out_path.resolve()}" if not args.no_write_file else "[vlm_rc1] Sin escritura en disco (--no-write-file); JSON solo en stdout.")
    _log(
        f"[vlm_rc1] robbery_probability={payload['robbery_probability']} verdict={payload['verdict']} "
        f"(chunks con JSON: {len(semantics_paths)}/{len(chunk_dirs)})"
    )


if __name__ == "__main__":
    main()
