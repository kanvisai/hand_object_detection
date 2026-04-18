"""Descarga de pesos YOLO oficiales (GitHub Ultralytics assets) si faltan en disco."""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

# Mismos que usa Ultralytics al cargar por nombre de fichero.
YOLO_PT_DOWNLOAD_URL: dict[str, str] = {
    "yolo11n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
    ),
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}


def download_yolo_pt_if_missing(pt: Path) -> dict[str, Any]:
    """Descarga el .pt desde GitHub assets si falta."""
    info: dict[str, Any] = {"pt": str(pt), "downloaded": False}
    if pt.is_file():
        return info
    url = YOLO_PT_DOWNLOAD_URL.get(pt.name)
    if not url:
        info["error"] = f"No hay URL de descarga para {pt.name}"
        return info
    pt.parent.mkdir(parents=True, exist_ok=True)
    tmp = pt.with_suffix(pt.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(pt)
        info["downloaded"] = True
    except OSError as e:
        info["error"] = str(e)
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
    return info
