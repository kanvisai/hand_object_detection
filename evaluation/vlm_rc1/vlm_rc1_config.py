"""
Parámetros del pipeline `vlm_rc1` (retail / chunks).

Los textos de prompt viven en `vlm_rc1_prompts.py` para poder copiar solo la carpeta
`vlm_rc1/` a otro repositorio sin depender de `retail_prompt_variants`.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from vlm_rc1_prompts import PROMPT_VARIANT_ID

SEMANTICS_SCHEMA_VERSION_NOTE = "Misma estructura que session_semantics; prompts definidos en vlm_rc1_prompts.py."

# Nombre de fichero por chunk (solo nombre de carpeta + sufijo; sin modelo ni variante).
VLM_SEMANTICS_FILENAME_SUFFIX = "_vlm.json"
VLM_SEMANTICS_ERROR_SUFFIX = "_vlm_error.json"

# Fichero agregado bajo chunk-main-dir.
VLM_EVALUATION_FILENAME = "vlm_evaluation.json"
# Si no hay datos suficientes / todo falló; evita confundir con “probabilidad 0 = no robo”.
VLM_EVALUATION_ERROR_FILENAME = "vlm_evaluation_error.json"

# Sentinel para métricas no calculables (no usar 0 como “sin robo”).
SENTINEL_SCORE_UNAVAILABLE = -1


def vlm_semantics_basename(chunk_stem: str) -> str:
    """p. ej. chunk_001 → chunk_001_vlm.json"""
    return f"{chunk_stem}{VLM_SEMANTICS_FILENAME_SUFFIX}"


def vlm_error_semantics_basename(chunk_stem: str) -> str:
    """Marcador escrito si run_semantics_rc1 falla por chunk (p. ej. chunk_001_vlm_error.json)."""
    return f"{chunk_stem}{VLM_SEMANTICS_ERROR_SUFFIX}"


def write_json_stdout(payload: dict[str, Any], *, pretty: bool = False) -> None:
    """
    Escribe un único objeto JSON en stdout (para piping / procesos que suben a MinIO).
    Los mensajes de log deben ir a stderr para no contaminar la salida.
    """
    if pretty:
        txt = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        txt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write(txt + "\n")
    sys.stdout.flush()
