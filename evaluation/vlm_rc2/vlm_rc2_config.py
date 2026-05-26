"""
Parámetros del pipeline `vlm_rc2` (retail / chunks).

Los textos de prompt viven en `vlm_rc2_prompts.py`.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from vlm_rc2_prompts import PROMPT_VARIANT_ID

SEMANTICS_SCHEMA_VERSION_NOTE = "Misma estructura que session_semantics; prompts definidos en vlm_rc2_prompts.py."

VLM_SEMANTICS_FILENAME_SUFFIX = "_vlm.json"
VLM_SEMANTICS_ERROR_SUFFIX = "_vlm_error.json"

VLM_EVALUATION_FILENAME = "vlm_evaluation.json"
VLM_EVALUATION_ERROR_FILENAME = "vlm_evaluation_error.json"

SENTINEL_SCORE_UNAVAILABLE = -1


def vlm_semantics_basename(chunk_stem: str) -> str:
    """p. ej. chunk_001 → chunk_001_vlm.json"""
    return f"{chunk_stem}{VLM_SEMANTICS_FILENAME_SUFFIX}"


def vlm_error_semantics_basename(chunk_stem: str) -> str:
    """Marcador escrito si run_semantics_rc2 falla por chunk."""
    return f"{chunk_stem}{VLM_SEMANTICS_ERROR_SUFFIX}"


def write_json_stdout(payload: dict[str, Any], *, pretty: bool = False) -> None:
    if pretty:
        txt = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        txt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write(txt + "\n")
    sys.stdout.flush()
