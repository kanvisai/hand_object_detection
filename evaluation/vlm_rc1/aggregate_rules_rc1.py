#!/usr/bin/env python3
"""
Agrega JSON de semántica rc1 por todos los chunks y aplica `robbery_rules_score`.

Modo producción: ante datos ausentes o errores grave, escribe `vlm_evaluation_error.json`
(con robbery_probability = -1) y sigue respondiendo JSON por stdout; proceso termina con código 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

_EVAL = Path(__file__).resolve().parent.parent
_RCDIR = Path(__file__).resolve().parent
for _p in (_EVAL, _RCDIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from robbery_rules_score import compute_robbery_rules_score  # noqa: E402
from session_semantics import discover_chunk_dirs_under_parent  # noqa: E402

from vlm_rc1_config import (  # noqa: E402
    SENTINEL_SCORE_UNAVAILABLE,
    VLM_EVALUATION_ERROR_FILENAME,
    VLM_EVALUATION_FILENAME,
    vlm_error_semantics_basename,
    vlm_semantics_basename,
    write_json_stdout,
)
from vlm_rc1_errors import evaluation_unavailable_payload  # noqa: E402
from vlm_rc1_metrics import (  # noqa: E402
    mean_probability_pct,
    object_in_hand_probs_evaluable_frames,
)
from vlm_rc1_prompts import PROMPT_VARIANT_ID  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_semantics_json(chunk_dir: Path) -> Path | None:
    expected = chunk_dir / vlm_semantics_basename(chunk_dir.name)
    if expected.is_file():
        return expected
    alts = sorted(chunk_dir.glob("*_vlm.json"))
    if len(alts) == 1:
        return alts[0]
    return None


def _resolve_error_marker(chunk_dir: Path) -> Path | None:
    p = chunk_dir / vlm_error_semantics_basename(chunk_dir.name)
    return p if p.is_file() else None


def _collect_frames_from_semantics_files(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    frames: list[dict[str, Any]] = []
    errors: list[str] = []
    for p in paths:
        try:
            data = _load_json(p)
        except Exception as e:
            errors.append(f"{p}: lectura JSON fallida ({e})")
            continue
        if str(data.get("vlm_rc1_pipeline_status") or "") == "error":
            errors.append(f"{p}: JSON marcado como error de semántica (omitido para reglas)")
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
        if str(data.get("vlm_rc1_pipeline_status") or "") == "error":
            skipped_newer.append(ch.name)
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


def _pick_output_path(root: Path, args: argparse.Namespace, want_error_file: bool) -> Path:
    out_path_s = str(args.output_json or "").strip()
    if out_path_s:
        return Path(out_path_s).expanduser().resolve()
    if want_error_file:
        return root / VLM_EVALUATION_ERROR_FILENAME
    return root / VLM_EVALUATION_FILENAME


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
        help=f"Ruta fija del JSON (si se omite: {VLM_EVALUATION_FILENAME} o {VLM_EVALUATION_ERROR_FILENAME}).",
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
        help="Registrar advertencias fuertes en stderr si falta *_vlm.json (no cambia código de salida).",
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

    try:
        root = Path(args.chunk_main_dir).expanduser().resolve()
        if not root.is_dir():
            payload = evaluation_unavailable_payload(
                chunk_main_dir=root,
                pipeline_status="error",
                reason="chunk_main_dir no es un directorio",
                load_errors=[f"No existe o no es directorio: {root}"],
            )
            payload["prompt_variant_id_internal"] = PROMPT_VARIANT_ID
            if str(args.output_json or "").strip():
                out_path = Path(args.output_json).expanduser().resolve()
            else:
                out_path = Path.cwd() / VLM_EVALUATION_ERROR_FILENAME
            if not args.no_write_file:
                try:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as w:
                    _log(f"[vlm_rc1] No se pudo escribir {out_path}: {w}")
            write_json_stdout(payload, pretty=bool(args.pretty_json))
            return

        chunk_dirs = discover_chunk_dirs_under_parent(root)
        if not chunk_dirs:
            payload = evaluation_unavailable_payload(
                chunk_main_dir=root,
                pipeline_status="error",
                reason="no hay subcarpetas chunk con frames/ y frames_meta.json",
                chunks_discovered=0,
            )
            payload["prompt_variant_id_internal"] = PROMPT_VARIANT_ID
            out_path = _pick_output_path(root, args, want_error_file=True)
            if not args.no_write_file:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            write_json_stdout(payload, pretty=bool(args.pretty_json))
            return

        semantics_paths: list[Path] = []
        error_marker_paths: list[Path] = []
        missing: list[str] = []
        for ch in chunk_dirs:
            em = _resolve_error_marker(ch)
            if em is not None:
                error_marker_paths.append(em)
            cand = _resolve_semantics_json(ch)
            if cand is not None:
                semantics_paths.append(cand)
                continue
            missing.append(f"{ch} (esperado {vlm_semantics_basename(ch.name)})")

        if missing and args.strict:
            _log("[vlm_rc1] [strict] Faltan JSON de semántica:\n  " + "\n  ".join(missing))

        if not semantics_paths:
            payload = evaluation_unavailable_payload(
                chunk_main_dir=root,
                pipeline_status="error",
                reason="ningún *_vlm.json encontrado; revise *_vlm_error.json por chunk si run_semantics falló",
                chunks_discovered=len(chunk_dirs),
                semantics_paths_error=[str(p) for p in error_marker_paths],
                chunks_missing_json=missing,
            )
            payload["prompt_variant_id_internal"] = PROMPT_VARIANT_ID
            payload["semantics_error_markers"] = [str(p) for p in error_marker_paths]
            out_path = _pick_output_path(root, args, want_error_file=True)
            if not args.no_write_file:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            write_json_stdout(payload, pretty=bool(args.pretty_json))
            _log("[vlm_rc1] Sin datos de semántica; salida degradada (robbery_probability=-1).")
            return

        load_errors: list[str] = []
        frames, coll_err = _collect_frames_from_semantics_files(semantics_paths)
        load_errors.extend(coll_err)

        if not frames:
            payload = evaluation_unavailable_payload(
                chunk_main_dir=root,
                pipeline_status="error",
                reason="JSON presentes pero ningún frame usable para reglas (vacío o solo errores)",
                chunks_discovered=len(chunk_dirs),
                semantics_paths_ok=[str(p) for p in semantics_paths],
                load_errors=load_errors,
            )
            payload["prompt_variant_id_internal"] = PROMPT_VARIANT_ID
            payload["semantics_error_markers"] = [str(p) for p in error_marker_paths]
            out_path = _pick_output_path(root, args, want_error_file=True)
            if not args.no_write_file:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            write_json_stdout(payload, pretty=bool(args.pretty_json))
            return

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
        last_pct_out: float | int = SENTINEL_SCORE_UNAVAILABLE if last_pct is None else last_pct

        pipeline_status = "partial" if (missing or error_marker_paths) else "ok"

        payload = {
            "schema_version": "vlm_rc1_aggregate_rules_1.3",
            "vlm_rc1_pipeline_status": pipeline_status,
            "chunk_main_dir": str(root),
            "prompt_variant_id_internal": PROMPT_VARIANT_ID,
            "semantics_inputs": [str(p) for p in semantics_paths],
            "semantics_error_markers": [str(p) for p in error_marker_paths],
            "chunks_discovered": len(chunk_dirs),
            "chunks_with_semantics_json": len(semantics_paths),
            "chunks_missing_semantics_json": missing,
            "load_errors": load_errors,
            "last_chunk_object_visible_hands_pct": last_pct_out,
            "last_chunk_object_visible_hands": {
                "chunk_used_for_metric": last_name,
                "semantics_json": str(last_path) if last_path else None,
                "evaluable_frames": len(last_probs),
                "mean_softmax_object_in_hand_probability_pct": last_pct_out,
                "chunks_skipped_newer_no_evaluable_vlm": skipped_newer_empty_vlm,
                "selection_policy": (
                    "Último chunk en orden temporal con ≥1 frame VLM evaluable; si no, retrocede."
                ),
            },
            "aggregation_note": "Frames fusionados por (chunk, sample_idx). -1 = métrica no disponible.",
            **result_body,
        }

        out_path = _pick_output_path(root, args, want_error_file=False)
        if not args.no_write_file:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        write_json_stdout(payload, pretty=bool(args.pretty_json))

        _log(
            f"[vlm_rc1] Evaluación guardada: {out_path.resolve()}"
            if not args.no_write_file
            else "[vlm_rc1] Solo stdout (--no-write-file)."
        )
        _log(
            f"[vlm_rc1] robbery_probability={payload['robbery_probability']} "
            f"last_chunk_object_visible_hands_pct={payload['last_chunk_object_visible_hands_pct']} "
            f"status={pipeline_status}"
        )

    except Exception as e:
        _log(f"[vlm_rc1] ERROR interno (salida degradada, exit 0): {e}\n{traceback.format_exc()}")
        root_fallback = Path(args.chunk_main_dir).expanduser().resolve()
        payload = evaluation_unavailable_payload(
            chunk_main_dir=root_fallback,
            pipeline_status="error",
            reason=f"excepción en aggregate_rules_rc1: {e}",
            load_errors=[traceback.format_exc()[-8000:]],
        )
        payload["prompt_variant_id_internal"] = PROMPT_VARIANT_ID
        out_path = _pick_output_path(root_fallback, args, want_error_file=True)
        if not args.no_write_file:
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as w:
                _log(f"[vlm_rc1] No se pudo escribir {out_path}: {w}")
        write_json_stdout(payload, pretty=bool(args.pretty_json))


if __name__ == "__main__":
    main()
    sys.exit(0)
