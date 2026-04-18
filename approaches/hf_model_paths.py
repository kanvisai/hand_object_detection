"""
Mapeo opcional repo_id Hugging Face -> ruta local del snapshot (generado por check_models.py).
Permite cargar desde disco sin depender solo del cache implicito del Hub.
"""
from __future__ import annotations

import json
from pathlib import Path

_MANIFEST_NAME = "hf_model_paths.json"


def manifest_path() -> Path:
    return Path(__file__).resolve().parent / _MANIFEST_NAME


def resolve_hf_model_ref(model_ref: str) -> str:
    """
    Si existe entrada en hf_model_paths.json y la ruta es un directorio, devuelve esa ruta.
    En caso contrario devuelve model_ref (id Hub, hf-hub:..., o ruta ya local).
    """
    s = str(model_ref).strip()
    if not s:
        return s
    p = Path(s).expanduser()
    if p.is_dir():
        return str(p.resolve())

    mp = manifest_path()
    if not mp.is_file():
        return model_ref
    try:
        data: dict[str, str] = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return model_ref

    candidates: list[str] = [model_ref]
    if model_ref.startswith("hf-hub:"):
        candidates.append(model_ref.replace("hf-hub:", "").strip())
    low = model_ref.lower()
    if low == "microsoft/florence-2-base":
        candidates.append("florence-community/Florence-2-base")
    elif low == "microsoft/florence-2-large":
        candidates.append("florence-community/Florence-2-large")

    for key in candidates:
        if key in data:
            loc = Path(data[key]).expanduser()
            if loc.is_dir():
                return str(loc.resolve())
    return model_ref


def merge_manifest_entries(updates: dict[str, str]) -> None:
    """Fusiona updates en hf_model_paths.json (repo_id -> ruta snapshot)."""
    if not updates:
        return
    mp = manifest_path()
    cur: dict[str, str] = {}
    if mp.is_file():
        try:
            cur = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cur = {}
    cur.update(updates)
    mp.write_text(json.dumps(cur, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
