#!/usr/bin/env python3
"""
Agrega JSON de semántica rc1 por todos los chunks bajo una carpeta y aplica `robbery_rules_score`.

Busca en cada subcarpeta el fichero `<chunk_stem>_vlm.json` generado por `run_semantics_rc1.py`,
concatena `frames` en orden y calcula probabilidad de robo.

Por defecto guarda `<chunk-main-dir>/vlm_evaluation.json` y escribe el mismo objeto JSON en **stdout**.
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

from vlm_rc1_config import VLM_EVALUATION_FILENAME, vlm_semantics_basename, write_json_stdout  # noqa: E402
from vlm_rc1_metrics import (  # noqa: E402
    mean_probability_pct,
    object_in_hand_probs_evaluable_frames,
)
from vlm_rc1_prompts import PROMPT_VARIANT_ID  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_semantics_json(chunk_dir: Path) -> Path | None:
    """`<stem>_vlm.json` o un único `*_vlm.json` en la carpeta del chunk."""
    expected = chunk_dir / vlm_semantics_basename(chunk_dir.name)
    if expected.is_file():
        return expected
    alts = sorted(chunk_dir.glob("*_vlm.json"))
    if len(alts) == 1:
        return alts[0]
    return None


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


def _find_last_chunk_with_evaluable_vlm_frames(
    chunk_dirs: list[Path],
    load_errors: list[str],
) -> tuple[Path | None, str | None, list[dict[str, Any]], list[str]]:
    """
    Desde el último chunk (orden temporal = discover), busca el primero que tenga **al menos un
    frame evaluable** donde se aplicó el VLM (p. ej. muñeca visible en meta). Si el último chunk
    solo tiene omisiones (`no_wrist_visibility`, etc.), retrocede hasta encontrar uno válido.

    Devuelve (path_json, nombre_chunk, frames_raw, chunks_saltados_sin_vlm_evaluable).
    Los «saltados» son chunks más recientes que tenían *_vlm.json pero 0 frames con probs VLM.
    """
    skipped_newer: list[str] = []
    for ch in reversed(chunk_dirs):
        p = _resolve_semantics_json(ch)
        if p is None:
            continue
        try:
            data = _load_json(p)
        except Exception as e:
            load_errors.append(f"{p}: último-chunk-válido — {e}")
            continue
        frs = data.get("frames")
        if not isinstance(frs, list):
            continue
        frames = [x for x in frs if isinstance(x, dict)]
        probs = object_in_hand_probs_evaluable_frames(frames)
        if probs:
            return p, ch.name, frames, skipped_newer
        skipped_newer.append(ch.name)
    return None, None, [], skipped_newer


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
        "--output-json",
        default="",
        help=f"Ruta del JSON de evaluación (por defecto <chunk-main-dir>/{VLM_EVALUATION_FILENAME}).",
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
        cand = _resolve_semantics_json(ch)
        if cand is not None:
            semantics_paths.append(cand)
            continue
        missing.append(f"{ch} (esperado {vlm_semantics_basename(ch.name)})")

    if missing and args.strict:
        print("Faltan JSON de semántica en:\n  " + "\n  ".join(missing), file=sys.stderr)
        raise SystemExit(1)

    if not semantics_paths:
        print(
            "No se encontró ningún JSON *_vlm.json. Ejecuta run_semantics_rc1.py por cada chunk "
            f"(fichero esperado: {vlm_semantics_basename('chunk_XXX')}).",
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

    last_path, last_name, last_frames, skipped_newer_empty_vlm = _find_last_chunk_with_evaluable_vlm_frames(
        chunk_dirs,
        load_errors,
    )
    last_probs = object_in_hand_probs_evaluable_frames(last_frames)
    last_pct = mean_probability_pct(last_probs)

    out_path_s = str(args.output_json or "").strip()
    if args.no_write_file:
        out_path = Path(os.devnull)
    elif out_path_s:
        out_path = Path(out_path_s).expanduser().resolve()
    else:
        out_path = root / VLM_EVALUATION_FILENAME

    payload: dict[str, Any] = {
        "schema_version": "vlm_rc1_aggregate_rules_1.2",
        "chunk_main_dir": str(root),
        "prompt_variant_id_internal": PROMPT_VARIANT_ID,
        "semantics_inputs": [str(p) for p in semantics_paths],
        "chunks_discovered": len(chunk_dirs),
        "chunks_with_semantics_json": len(semantics_paths),
        "chunks_missing_semantics_json": missing,
        "load_errors": load_errors,
        "last_chunk_object_visible_hands_pct": last_pct,
        "last_chunk_object_visible_hands": {
            "chunk_used_for_metric": last_name,
            "semantics_json": str(last_path) if last_path else None,
            "evaluable_frames": len(last_probs),
            "mean_softmax_object_in_hand_probability_pct": last_pct,
            "chunks_skipped_newer_no_evaluable_vlm": skipped_newer_empty_vlm,
            "selection_policy": (
                "Último chunk en orden temporal que tenga ≥1 frame con VLM aplicado "
                "(p. ej. muñeca visible en frames_meta). Si el último chunk no tiene persona/"
                "muñecas, todos los frames van sin VLM y se retrocede al chunk válido anterior."
            ),
            "vlm_gating_note": (
                "Sin muñeca visible (visible_wrists_count==0 y ambas false), session_semantics "
                "no ejecuta el VLM (skip_reason no_wrist_visibility)."
            ),
        },
        "aggregation_note": "Frames de todos los chunks fusionados y ordenados por (chunk, sample_idx).",
        **result_body,
    }

    if not args.no_write_file:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    write_json_stdout(payload, pretty=bool(args.pretty_json))

    _log(f"[vlm_rc1] Evaluación guardada: {out_path.resolve()}" if not args.no_write_file else "[vlm_rc1] Sin escritura en disco (--no-write-file); JSON solo en stdout.")
    _log(
        f"[vlm_rc1] robbery_probability={payload['robbery_probability']} "
        f"last_chunk_object_visible_hands_pct={payload['last_chunk_object_visible_hands_pct']} "
        f"(chunks con JSON: {len(semantics_paths)}/{len(chunk_dirs)})"
    )


if __name__ == "__main__":
    main()
