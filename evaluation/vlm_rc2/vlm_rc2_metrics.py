"""
Métricas derivadas de los JSON de semántica rc2 (prob. clase `object_in_hand`, índice 0 del softmax).
"""

from __future__ import annotations

from typing import Any

OBJECT_IN_HAND_IDX = 0


def object_in_hand_probs_evaluable_frames(frames: list[dict[str, Any]]) -> list[float]:
    """Probabilidades softmax para la etiqueta object_in_hand solo en frames evaluables."""
    out: list[float] = []
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        if not bool(fr.get("evaluable", fr.get("vlm_applied", False))):
            continue
        v = fr.get("vlm_vector_prompt_probs")
        if isinstance(v, list) and len(v) > OBJECT_IN_HAND_IDX:
            try:
                out.append(float(v[OBJECT_IN_HAND_IDX]))
            except (TypeError, ValueError):
                continue
    return out


def mean_probability_pct(probs: list[float]) -> float | None:
    """Media en escala 0–100 %; None si no hay muestras."""
    if not probs:
        return None
    return round(100.0 * sum(probs) / len(probs), 2)


def enrich_semantics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("vlm_rc2_pipeline_status") == "error":
        return payload
    if str(payload.get("schema_version", "")).startswith("vlm_rc2_semantics_error"):
        return payload
    frames = payload.get("frames")
    if not isinstance(frames, list):
        frames = []
    probs = object_in_hand_probs_evaluable_frames([x for x in frames if isinstance(x, dict)])
    payload["vlm_rc2_chunk_metrics"] = {
        "object_in_hand_probability_mean_pct": mean_probability_pct(probs),
        "evaluable_frames": len(probs),
        "object_in_hand_softmax_index": OBJECT_IN_HAND_IDX,
        "note": "Media del softmax índice 0 (object_in_hand) en frames evaluables.",
    }
    return payload
