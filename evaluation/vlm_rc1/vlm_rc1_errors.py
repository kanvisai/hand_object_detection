"""
Payloads de error / degradados para producción (sentinel -1 = métrica no disponible).
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from vlm_rc1_config import SENTINEL_SCORE_UNAVAILABLE


def semantics_load_failed_payload(
    *,
    chunk_dir: Path,
    chunk_stem: str,
    message: str,
    error_type: str = "validation_or_io",
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "vlm_rc1_semantics_error_1.0",
        "vlm_rc1_pipeline_status": "error",
        "chunk_dir": str(chunk_dir),
        "chunk_name": chunk_stem,
        "error_type": error_type,
        "error_message": message,
        "error_detail": detail,
        "robbery_probability": SENTINEL_SCORE_UNAVAILABLE,
        "note": "No hay frames semánticos válidos; usar aggregate solo como diagnóstico.",
    }


def semantics_runtime_error_payload(
    *,
    chunk_dir: Path,
    chunk_stem: str,
    exc: BaseException,
    stage: str,
) -> dict[str, Any]:
    tb = traceback.format_exc()
    return {
        "schema_version": "vlm_rc1_semantics_error_1.0",
        "vlm_rc1_pipeline_status": "error",
        "chunk_dir": str(chunk_dir),
        "chunk_name": chunk_stem,
        "error_type": "runtime",
        "error_stage": stage,
        "error_message": str(exc),
        "error_traceback": tb[-8000:] if len(tb) > 8000 else tb,
        "robbery_probability": SENTINEL_SCORE_UNAVAILABLE,
    }


def evaluation_unavailable_payload(
    *,
    chunk_main_dir: Path,
    pipeline_status: str,
    reason: str,
    chunks_discovered: int = 0,
    semantics_paths_ok: list[str] | None = None,
    semantics_paths_error: list[str] | None = None,
    chunks_missing_json: list[str] | None = None,
    load_errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "vlm_rc1_aggregate_rules_error_1.0",
        "vlm_rc1_pipeline_status": pipeline_status,
        "chunk_main_dir": str(chunk_main_dir),
        "reason": reason,
        "chunks_discovered": chunks_discovered,
        "semantics_inputs_ok": semantics_paths_ok or [],
        "semantics_inputs_error_markers": semantics_paths_error or [],
        "chunks_missing_semantics_json": chunks_missing_json or [],
        "load_errors": load_errors or [],
        "robbery_probability": float(SENTINEL_SCORE_UNAVAILABLE),
        "verdict": "unavailable",
        "last_chunk_object_visible_hands_pct": SENTINEL_SCORE_UNAVAILABLE,
        "last_chunk_object_visible_hands": {
            "chunk_used_for_metric": None,
            "mean_softmax_object_in_hand_probability_pct": SENTINEL_SCORE_UNAVAILABLE,
            "note": reason,
        },
        "counts": {
            "severe_events": 0,
            "suspicious_events": 0,
            "normal_events": 0,
        },
        "severe_events": [],
        "suspicious_events": [],
        "normal_events": [],
        "label_sequence_smoothed": [],
        "params": {},
    }
