"""Utilidades para recorrer frames_meta.json y barra de progreso."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def frame_has_wrist_visibility(meta_row: dict[str, Any]) -> bool:
    if bool(meta_row.get("left_wrist_visible")) or bool(meta_row.get("right_wrist_visible")):
        return True
    try:
        return int(meta_row.get("visible_wrists_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def load_frames_meta(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"frames_meta.json debe ser una lista: {path}")
    rows = [r for r in raw if isinstance(r, dict)]
    rows.sort(key=lambda r: int(r.get("sample_idx", 0)))
    return rows


def load_frames_meta_safe(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Carga frames_meta.json sin lanzar; devuelve (filas, avisos)."""
    warnings: list[str] = []
    if not path.is_file():
        warnings.append("missing_frames_meta_json")
        return [], warnings
    try:
        return load_frames_meta(path), warnings
    except OSError as e:
        warnings.append(f"frames_meta_unreadable:{e}")
        return [], warnings
    except (json.JSONDecodeError, RuntimeError, ValueError) as e:
        warnings.append(f"frames_meta_invalid:{e}")
        return [], warnings


def session_progress_line(
    session_label: str,
    cur: int,
    total: int,
    chunk_name: str,
    image_key: str,
    phase: str,
    *,
    elapsed_s: float,
    eta_s: float | None,
) -> None:
    if total > 0:
        pct = min(100.0, (100.0 * cur) / max(1, total))
    else:
        pct = 0.0
    bar_w = 28
    fill = int(round((pct / 100.0) * bar_w))
    bar = ("#" * fill) + ("-" * (bar_w - fill))
    eta_part = ""
    if eta_s is not None and eta_s > 0.5:
        eta_part = f" ETA ~{int(eta_s)}s"
    print(
        f"\r[session] {session_label} [{bar}] {cur}/{total} ({pct:5.1f}%) "
        f"{chunk_name}/{image_key} | {phase} | {elapsed_s:6.1f}s{eta_part}   ",
        end="",
        flush=True,
    )
