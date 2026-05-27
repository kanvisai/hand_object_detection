"""
Validación tolerante de carpetas chunk (no lanza excepciones por estructura incompleta).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def assess_chunk_dir(chunk_dir: Path) -> dict[str, Any]:
    """
    Inspecciona un chunk sin lanzar excepciones.

    Returns:
        ok: existe y es directorio
        processable: se puede intentar inferencia (aunque queden 0 frames evaluables)
        issues: problemas bloqueantes (p. ej. no es directorio)
        warnings: problemas no bloqueantes (sin meta, sin fotos, etc.)
    """
    p = chunk_dir.expanduser().resolve()
    issues: list[str] = []
    warnings: list[str] = []

    if not p.exists():
        issues.append("path_does_not_exist")
        return _result(p, issues=issues, warnings=warnings)
    if not p.is_dir():
        issues.append("not_a_directory")
        return _result(p, issues=issues, warnings=warnings)

    frames_dir = p / "frames"
    meta_path = p / "frames_meta.json"

    has_frames_dir = frames_dir.is_dir()
    has_meta = meta_path.is_file()

    if not has_frames_dir:
        warnings.append("missing_frames_directory")
    if not has_meta:
        warnings.append("missing_frames_meta_json")

    image_count = 0
    if has_frames_dir:
        try:
            image_count = sum(
                1
                for f in frames_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
        except OSError as e:
            warnings.append(f"frames_directory_unreadable:{e}")

    if has_frames_dir and image_count == 0:
        warnings.append("frames_directory_empty_or_no_images")

    meta_rows = 0
    if has_meta:
        meta_rows, meta_warn = _count_meta_rows_safe(meta_path)
        warnings.extend(meta_warn)

    processable = "not_a_directory" not in issues and "path_does_not_exist" not in issues
    return _result(
        p,
        issues=issues,
        warnings=warnings,
        has_frames_dir=has_frames_dir,
        has_meta=has_meta,
        image_count=image_count,
        meta_rows=meta_rows,
        processable=processable,
    )


def _result(
    p: Path,
    *,
    issues: list[str],
    warnings: list[str],
    has_frames_dir: bool = False,
    has_meta: bool = False,
    image_count: int = 0,
    meta_rows: int = 0,
    processable: bool = False,
) -> dict[str, Any]:
    ok = not issues and p.is_dir() if p.exists() else False
    return {
        "ok": ok,
        "processable": processable,
        "chunk_dir": str(p),
        "chunk_name": p.name or "chunk",
        "issues": issues,
        "warnings": warnings,
        "has_frames_dir": has_frames_dir,
        "has_frames_meta_json": has_meta,
        "frames_image_count": image_count,
        "frames_meta_row_count": meta_rows,
    }


def _count_meta_rows_safe(meta_path: Path) -> tuple[int, list[str]]:
    warnings: list[str] = []
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except OSError as e:
        warnings.append(f"frames_meta_unreadable:{e}")
        return 0, warnings
    except json.JSONDecodeError as e:
        warnings.append(f"frames_meta_invalid_json:{e}")
        return 0, warnings
    if not isinstance(raw, list):
        warnings.append("frames_meta_not_a_list")
        return 0, warnings
    n = sum(1 for r in raw if isinstance(r, dict) and str(r.get("image_key") or "").strip())
    return n, warnings
