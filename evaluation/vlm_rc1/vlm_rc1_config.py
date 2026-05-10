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


def _safe_variant_id(pv_id: str) -> str:
    """Igual que session_semantics._semantics_output_filename (sufijo de fichero)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in pv_id.strip())


def semantics_filename_for_chunk(chunk_stem: str, backend: str, variant_id: str | None = None) -> str:
    """Nombre esperado: `<chunk_stem>_<backend>_<variant>.json` (p. ej. chunk_001_siglip_frames_v2_probe.json)."""
    vid = variant_id if variant_id is not None else PROMPT_VARIANT_ID
    safe = _safe_variant_id(vid)
    return f"{chunk_stem}_{backend}_{safe}.json"
