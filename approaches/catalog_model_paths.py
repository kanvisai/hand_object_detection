"""
Actualiza experiments_catalog.json con rutas resueltas tras check_models (HF + YOLO).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# id en experiments_catalog.approaches[] -> repo_id Hugging Face
APPROACH_ID_TO_REPO_ID: dict[str, str] = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
    "mobileclip": "apple/MobileCLIP-S1-OpenCLIP",
    "openvision": "UCSC-VLAA/openvision-vit-large-patch14-224",
    "qwen2vl": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen3vl": "Qwen/Qwen3-VL-2B-Instruct",
    "internvl2": "OpenGVLab/InternVL2-2B",
    "internvl3": "OpenGVLab/InternVL3-2B",
    "paligemma": "google/paligemma2-3b-pt-224",
    "moondream": "vikhyatk/moondream2",
    "florence2_base": "florence-community/Florence-2-base",
    "florence2_large": "florence-community/Florence-2-large",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTS_CATALOG = Path(__file__).resolve().parent / "experiments_catalog.json"


def _preferred_yolo_runtime_path(yentry: dict[str, Any]) -> str | None:
    """Prioriza .engine si engine_ok; si no, el .pt (según informe de check_models)."""
    if not yentry.get("ok"):
        return None
    eng = str(yentry.get("engine") or "").strip()
    pt = str(yentry.get("pt") or "").strip()
    if yentry.get("engine_ok") and eng:
        return str(Path(eng).expanduser().resolve())
    if pt:
        return str(Path(pt).expanduser().resolve())
    return None


def merge_check_report_into_experiments_catalog(
    data: dict[str, Any],
    report: dict[str, Any],
    *,
    yolo_weights_rel: tuple[str, ...],
) -> dict[str, Any]:
    """Devuelve copia del catálogo con defaults + vlm_model actualizados (rutas absolutas)."""
    out = json.loads(json.dumps(data))  # copia profunda barata

    yrows: list[dict[str, Any]] = list(report.get("yolo") or [])
    if len(yrows) >= 1 and len(yolo_weights_rel) >= 1:
        pose = _preferred_yolo_runtime_path(yrows[0])
        if pose:
            out.setdefault("defaults", {})["pose_weights"] = pose
    if len(yrows) >= 2 and len(yolo_weights_rel) >= 2:
        pers = _preferred_yolo_runtime_path(yrows[1])
        if pers:
            out.setdefault("defaults", {})["personal_weights"] = pers

    repo_to_path: dict[str, str] = {}
    for h in report.get("hub_models") or []:
        if not h.get("ok") or not h.get("cache_path"):
            continue
        rid = str(h.get("repo_id") or "").strip()
        cp = str(h.get("cache_path") or "").strip()
        if rid and cp:
            repo_to_path[rid] = str(Path(cp).resolve())

    approaches = out.get("approaches")
    if isinstance(approaches, list):
        for ap in approaches:
            if not isinstance(ap, dict):
                continue
            aid = str(ap.get("id") or "").strip()
            repo = APPROACH_ID_TO_REPO_ID.get(aid)
            if repo and repo in repo_to_path:
                ap["vlm_model"] = repo_to_path[repo]

    meta = out.setdefault("_paths_resolved_by", {})
    if isinstance(meta, dict):
        meta["check_models"] = True

    return out


def write_experiments_catalog_merged(
    catalog_path: Path,
    report: dict[str, Any],
    *,
    yolo_weights_rel: tuple[str, ...],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Lee JSON, fusiona rutas desde report, escribe si no dry_run. Devuelve {ok, path, error?}."""
    catalog_path = catalog_path.expanduser().resolve()
    if not catalog_path.is_file():
        return {"ok": False, "path": str(catalog_path), "error": "no existe el catálogo"}
    try:
        with open(catalog_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "path": str(catalog_path), "error": str(e)}

    merged = merge_check_report_into_experiments_catalog(data, report, yolo_weights_rel=yolo_weights_rel)
    if dry_run:
        return {"ok": True, "path": str(catalog_path), "dry_run": True}

    try:
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError as e:
        return {"ok": False, "path": str(catalog_path), "error": str(e)}

    return {"ok": True, "path": str(catalog_path)}
