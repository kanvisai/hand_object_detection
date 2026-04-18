#!/usr/bin/env python3
"""
Comprueba pesos YOLO (.pt + .engine opcional), cache/descarga de modelos HF/OpenCLIP,
y deja constancia de que los VLMs no usan el mismo flujo TensorRT que Ultralytics.

Uso: python check_models.py [--skip-yolo-engine] [--download-only] [--update-experiments-catalog [PATH]]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
APPROACHES_DIR = REPO_ROOT / "approaches"
if str(APPROACHES_DIR) not in sys.path:
    sys.path.insert(0, str(APPROACHES_DIR))

YOLO_WEIGHTS_REL = ("approaches/yolo11n-pose.pt", "approaches/yolo11n.pt")

# Pesos oficiales (GitHub Ultralytics assets); mismos que descarga Ultralytics por nombre.
YOLO_PT_DOWNLOAD_URL: dict[str, str] = {
    "yolo11n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
    ),
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}

# Fallos por repo gated / 401 no hacen fallar summary_ok si solo afectan a estos ids.
HUB_OPTIONAL_REPO_IDS = frozenset({"google/paligemma2-3b-pt-224"})

# Repositorios HF / Hub (open_clip usa ids sin prefijo hf-hub:)
HF_HUB_MODELS: list[tuple[str, str]] = [
    ("clip", "openai/clip-vit-base-patch32"),
    ("siglip", "google/siglip-base-patch16-224"),
    ("mobileclip", "apple/MobileCLIP-S1-OpenCLIP"),
    ("openvision", "UCSC-VLAA/openvision-vit-large-patch14-224"),
    ("qwen2vl", "Qwen/Qwen2-VL-2B-Instruct"),
    ("qwen3vl", "Qwen/Qwen3-VL-2B-Instruct"),
    ("internvl2", "OpenGVLab/InternVL2-2B"),
    ("internvl3", "OpenGVLab/InternVL3-2B"),
    ("paligemma", "google/paligemma2-3b-pt-224"),
    ("moondream", "vikhyatk/moondream2"),
    ("florence2_base", "florence-community/Florence-2-base"),
    ("florence2_large", "florence-community/Florence-2-large"),
]


def _hub_ensure(repo_id: str, token: bool = True) -> dict[str, Any]:
    """Descarga snapshot si falta en cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        return {"repo_id": repo_id, "ok": False, "error": f"huggingface_hub: {e}"}

    try:
        path = snapshot_download(repo_id=repo_id, local_files_only=False)
        return {"repo_id": repo_id, "ok": True, "cache_path": path}
    except Exception as e:
        return {"repo_id": repo_id, "ok": False, "error": str(e)}


def _yolo_paths() -> list[Path]:
    return [(REPO_ROOT / rel).resolve() for rel in YOLO_WEIGHTS_REL]


def _hf_hub_snapshots_parent(repo_id: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    sanitized = "models--" + repo_id.replace("/", "--")
    return hf_home / "hub" / sanitized / "snapshots"


def _hf_manual_local_dir(repo_id: str) -> Path:
    return APPROACHES_DIR / "hf_manual" / repo_id.replace("/", "__")


def _hub_row_extra(repo_id: str) -> dict[str, str]:
    manual = _hf_manual_local_dir(repo_id)
    snap_parent = _hf_hub_snapshots_parent(repo_id)
    return {
        "manual_wget_target_dir": str(manual.resolve()),
        "hf_standard_snapshots_parent": str(snap_parent.resolve()),
        "manual_wget_hint": (
            f"Descarga manual (sin Python): mkdir -p {manual} && "
            f"huggingface-cli download {repo_id} --local-dir {manual} "
            f"|| coloca ficheros desde https://huggingface.co/{repo_id}/tree/main . "
            f"Cache HF nativo (snapshots/<hash>/): {snap_parent}/"
        ),
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
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — URL fija autoría Ultralytics
        tmp.replace(pt)
        info["downloaded"] = True
    except OSError as e:
        info["error"] = str(e)
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
    return info


def ensure_yolo_engine(pt: Path, *, skip_engine: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"pt": str(pt), "engine": "", "engine_ok": False, "note": ""}
    wget_dir = str(APPROACHES_DIR.resolve())
    if not pt.is_file():
        dl = download_yolo_pt_if_missing(pt)
        out["download"] = dl
        if not pt.is_file():
            out["error"] = "Falta el fichero .pt (descarga automática falló)"
            out["hint"] = (
                f"Coloca el .pt en {wget_dir}/ (mismo nivel que los scripts approaches/) o wget: "
                f"wget -O {pt.name!r} {YOLO_PT_DOWNLOAD_URL.get(pt.name, '<URL>')}"
            )
            out["manual_wget_target_dir"] = wget_dir
            out["ok"] = False
            return out
        if dl.get("downloaded"):
            out["note"] = "Peso .pt descargado automáticamente desde GitHub assets."

    engine = pt.with_suffix(".engine")
    out["engine"] = str(engine)
    if engine.is_file():
        out["engine_ok"] = True
        out["ok"] = True
        out["note"] = "TensorRT .engine ya presente."
        return out

    if skip_engine:
        out["ok"] = True
        out["note"] = "Sin generar .engine (--skip-yolo-engine)."
        return out

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        out["ok"] = False
        out["error"] = str(e)
        out["note"] = "Ultralytics/torch no disponible para export."
        return out

    if not torch.cuda.is_available():
        out["ok"] = True
        out["note"] = (
            "Sin CUDA: no se genera .engine (TensorRT suele requerir GPU). "
            "Usa los .pt o ejecuta este script en máquina con GPU + TensorRT."
        )
        return out

    try:
        model = YOLO(str(pt))
        # half=True acelera export en GPU compatible
        model.export(format="engine", half=True, device=0)
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)
        out["note"] = "Fallo export TensorRT; revisa TensorRT/CUDA y ultralytics."
        return out

    if engine.is_file():
        out["engine_ok"] = True
        out["ok"] = True
        out["note"] = "TensorRT .engine generado."
    else:
        # Ultralytics puede volcar nombre ligeramente distinto
        candidates = list(pt.parent.glob(pt.stem + "*.engine"))
        if candidates:
            out["engine"] = str(candidates[0])
            out["engine_ok"] = True
            out["ok"] = True
            out["note"] = ".engine generado con nombre detectado por glob."
        else:
            out["ok"] = False
            out["note"] = "Export sin error pero .engine no encontrado junto al .pt."

    return out


def _run_checks_impl(*, skip_yolo_engine: bool, download_only: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "yolo": [],
        "hub_models": [],
        "vlm_tensorrt_note": (
            "Los modelos PaliGemma/Qwen/InternVL/Florence/Moondream (transformers) "
            "no se convierten aquí a .engine: no comparten el export de Ultralytics; "
            "un motor TensorRT por modelo implica ONNX/torch.export y reglas por arquitectura. "
            "Este script solo verifica cache/descarga HF. CLIP/SigLIP/MobileCLIP/OpenCLIP "
            "iguales: uso en inferencia PyTorch/OpenCLIP salvo pipeline TRT externo."
        ),
    }

    if download_only:
        for pt in _yolo_paths():
            dl = download_yolo_pt_if_missing(pt) if not pt.is_file() else {}
            eng = pt.with_suffix(".engine")
            report["yolo"].append(
                {
                    "pt": str(pt),
                    "pt_ok": pt.is_file(),
                    "engine": str(eng),
                    "engine_present": eng.is_file(),
                    "download": dl,
                    "ok": pt.is_file(),
                    "note": "Modo --download-only: .pt auto-descargado si faltaba; sin export .engine.",
                }
            )
    else:
        for pt in _yolo_paths():
            report["yolo"].append(ensure_yolo_engine(pt, skip_engine=skip_yolo_engine))

    for alias, repo in HF_HUB_MODELS:
        row = {"alias": alias, **_hub_ensure(repo), **_hub_row_extra(repo)}
        err_l = str(row.get("error", "")).lower()
        if (
            not row.get("ok")
            and repo in HUB_OPTIONAL_REPO_IDS
            and any(x in err_l for x in ("401", "gated", "restricted", "authenticated", "log in"))
        ):
            row["optional_gated_skip"] = True
            row["hint"] = (
                "Modelo gated en HF: entra en huggingface.co/google/paligemma2-3b-pt-224 "
                "y acepta términos; luego `huggingface-cli login` y vuelve a ejecutar."
            )
        report["hub_models"].append(row)

    ok_yolo = all(bool(y.get("ok")) for y in report["yolo"]) if report["yolo"] else True
    ok_hub = all(
        bool(h.get("ok")) or bool(h.get("optional_gated_skip")) for h in report["hub_models"]
    )

    manifest_updates: dict[str, str] = {}
    for h in report["hub_models"]:
        if h.get("ok") and h.get("cache_path"):
            manifest_updates[str(h["repo_id"])] = str(h["cache_path"])
    if manifest_updates:
        try:
            from hf_model_paths import manifest_path as _manifest_path_fn
            from hf_model_paths import merge_manifest_entries

            merge_manifest_entries(manifest_updates)
            report["hf_model_manifest_written"] = str(_manifest_path_fn())
        except Exception as e:
            report["hf_model_manifest_error"] = str(e)

    report["summary_ok"] = bool(ok_yolo and ok_hub)
    report["summary_notes"] = []
    if not ok_yolo:
        report["summary_notes"].append(
            "YOLO: sin .pt descargable en approaches/ el pipeline no arranca; ver hint en cada entrada."
        )
    if any(h.get("optional_gated_skip") for h in report["hub_models"]):
        report["summary_notes"].append(
            "PaliGemma omitido en el resumen (gated); resto de modelos HF siguen siendo obligatorios."
        )
    return report


def run_checks(
    *,
    skip_yolo_engine: bool,
    download_only: bool,
    update_experiments_catalog: Path | None = None,
) -> dict[str, Any]:
    """Tras _run_checks_impl, opcionalmente escribe experiments_catalog.json con rutas resueltas."""
    rep = _run_checks_impl(
        skip_yolo_engine=skip_yolo_engine,
        download_only=download_only,
    )
    if (
        update_experiments_catalog is not None
        and rep.get("summary_ok")
        and not download_only
    ):
        try:
            from catalog_model_paths import write_experiments_catalog_merged

            wr = write_experiments_catalog_merged(
                update_experiments_catalog,
                rep,
                yolo_weights_rel=YOLO_WEIGHTS_REL,
                dry_run=False,
            )
            rep["experiments_catalog_update"] = wr
        except Exception as e:
            rep["experiments_catalog_update"] = {"ok": False, "error": str(e)}
    elif update_experiments_catalog is not None:
        rep["experiments_catalog_update"] = {
            "ok": False,
            "skipped": True,
            "reason": "summary_ok false o modo --download-only",
        }
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Chequeo de pesos YOLO + modelos HF.")
    ap.add_argument("--skip-yolo-engine", action="store_true", help="No intentar generar .engine.")
    ap.add_argument("--download-only", action="store_true", help="Solo HF Hub (omitir bloque YOLO).")
    ap.add_argument(
        "--update-experiments-catalog",
        type=Path,
        nargs="?",
        const=REPO_ROOT / "approaches" / "experiments_catalog.json",
        default=None,
        metavar="PATH",
        help=(
            "Si el chequeo global OK, escribe rutas HF/YOLO en el catálogo (por defecto approaches/experiments_catalog.json). "
            "Usar sin PATH activa el valor por defecto."
        ),
    )
    ap.add_argument("--json", action="store_true", help="Salida solo JSON.")
    args = ap.parse_args()

    rep = run_checks(
        skip_yolo_engine=args.skip_yolo_engine,
        download_only=args.download_only,
        update_experiments_catalog=args.update_experiments_catalog,
    )

    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print("=== YOLO ===")
        for y in rep.get("yolo", []):
            print(json.dumps(y, indent=2, ensure_ascii=False))
        print("\n=== Hugging Face / Hub ===")
        for h in rep.get("hub_models", []):
            print(json.dumps({k: h[k] for k in h if k != "cache_path"}, indent=2, ensure_ascii=False))
            if h.get("cache_path"):
                print(f"  cache_path: {h['cache_path']}")
            if h.get("hint"):
                print(f"  hint: {h['hint']}")
            if h.get("manual_wget_hint"):
                print(f"  manual: {h['manual_wget_hint']}")
        print("\n=== TensorRT VLMs ===")
        print(rep.get("vlm_tensorrt_note"))
        if rep.get("hf_model_manifest_written"):
            print("\n=== Manifest HF ===")
            print(f"Actualizado: {rep['hf_model_manifest_written']}")
        if rep.get("hf_model_manifest_error"):
            print(f"Manifest error: {rep['hf_model_manifest_error']}")
        if rep.get("experiments_catalog_update"):
            print("\n=== Catálogo experimentos ===")
            print(json.dumps(rep["experiments_catalog_update"], indent=2, ensure_ascii=False))
        print("\n=== Resumen ===")
        print(f"OK global: {rep['summary_ok']}")
        for note in rep.get("summary_notes") or []:
            print(f"  · {note}")

    sys.exit(0 if rep["summary_ok"] else 1)


if __name__ == "__main__":
    main()
