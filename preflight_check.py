#!/usr/bin/env python3
"""
Pre-vuelo: ejecuta check_models, comprueba librerías, GPU, actualiza device y rutas de modelos en experiments_catalog.json,
y estima tiempos de campaña (heurísticas documentadas).

Uso: python preflight_check.py [--no-update-catalog] [--no-update-model-paths]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CATALOG = REPO_ROOT / "approaches" / "experiments_catalog.json"

# Paquete requirements.txt -> modulo import para probar
PACKAGE_IMPORT_CHECKS: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("Pillow", "PIL"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("open-clip-torch", "open_clip"),
    ("ultralytics", "ultralytics"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface_hub"),
    ("tokenizers", "tokenizers"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("sentencepiece", "sentencepiece"),
    ("protobuf", "google.protobuf"),
    ("psutil", "psutil"),
    ("einops", "einops"),
    ("timm", "timm"),
    ("peft", "peft"),
]

# Coste relativo ~ wall-clock por segundo de vídeo (orden de magnitud; calibrar con tus JSON reales).
# CPU / GPU separados; embedding vs generativo.
_REL_CPU_PER_VIDEO_SEC: dict[str, tuple[float, float]] = {
    "clip": (0.65, 0.12),
    "siglip": (0.65, 0.12),
    "mobileclip": (0.55, 0.10),
    "openvision": (0.70, 0.13),
    "qwen2vl": (18.0, 2.5),
    "qwen3vl": (18.0, 2.5),
    "internvl2": (35.0, 4.0),
    "internvl3": (35.0, 4.0),
    "paligemma": (28.0, 3.5),
    "moondream": (22.0, 3.0),
    "florence2_base": (12.0, 2.0),
    "florence2_large": (22.0, 3.5),
}

_PROFILE_STRIDE: dict[str, int] = {
    "baseline": 2,
    "stride_1": 1,
    "stride_3": 3,
    "stride_5": 5,
    "no_per_hand_fast": 2,
    "temporal_relaxed": 2,
}


def check_imports() -> tuple[bool, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    all_ok = True
    for pkg, mod in PACKAGE_IMPORT_CHECKS:
        try:
            importlib.import_module(mod)
            rows.append({"package": pkg, "import": mod, "ok": "yes"})
        except ImportError as e:
            all_ok = False
            rows.append({"package": pkg, "import": mod, "ok": "no", "error": str(e)})
    return all_ok, rows


def gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "device_str": "cpu", "details": []}
        n = torch.cuda.device_count()
        details = []
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            details.append(
                {
                    "index": i,
                    "name": name,
                    "total_memory_gib": round(props.total_memory / (1024**3), 3),
                }
            )
        return {
            "available": True,
            "device_count": n,
            "device_str": "cuda:0",
            "primary": details[0] if details else None,
            "details": details,
        }
    except Exception as e:
        return {"available": False, "error": str(e), "device_str": "cpu"}


def update_catalog_device(catalog_path: Path, device: str, *, dry: bool) -> bool:
    if not catalog_path.is_file():
        print(f"[catalog] No existe {catalog_path}")
        return False
    with open(catalog_path, encoding="utf-8") as f:
        data = json.load(f)
    defaults = data.setdefault("defaults", {})
    old = defaults.get("device")
    if old == device:
        print(f"[catalog] device ya es {device!r}")
        return True
    defaults["device"] = device
    if dry:
        print(f"[catalog] dry-run: device {old!r} -> {device!r}")
        return True
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[catalog] device actualizado: {old!r} -> {device!r} en {catalog_path}")
    return True


def scan_videos(paths: list[str]) -> list[dict[str, Any]]:
    try:
        import cv2
    except ImportError as e:
        return [{"path": "*", "ok": False, "error": f"opencv-python no instalado: {e}"}]

    out: list[dict[str, Any]] = []
    for p in paths:
        raw = str(p).strip()
        if not raw:
            continue
        vp = Path(raw).expanduser()
        if not vp.is_file():
            out.append({"path": raw, "ok": False, "error": "no existe"})
            continue
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            out.append({"path": str(vp.resolve()), "ok": False, "error": "no se abre"})
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (nframes / fps) if fps > 0 else None
        cap.release()
        out.append(
            {
                "path": str(vp.resolve()),
                "ok": True,
                "fps": fps,
                "frames": nframes,
                "duration_sec": round(dur, 4) if dur is not None else None,
            }
        )
    return out


def estimate_campaign_time(
    catalog: dict[str, Any],
    video_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    approaches = [a["id"] for a in catalog.get("approaches", []) if not str(a.get("id", "")).startswith("_")]
    profiles = [k for k in (catalog.get("profiles") or {}) if not k.startswith("_")]
    valid_videos = [v for v in video_rows if v.get("ok")]

    total_exp = len(approaches) * len(profiles) * len(valid_videos)
    sum_dur = sum(float(v["duration_sec"] or 0) for v in valid_videos)

    # Por cada triple: duracion_video * coste_relativo * factor stride (ref stride 2)
    cpu_sec = 0.0
    gpu_sec = 0.0
    for v in valid_videos:
        d = float(v.get("duration_sec") or 0.0)
        if d <= 0:
            continue
        for aid in approaches:
            cpu_r, gpu_r = _REL_CPU_PER_VIDEO_SEC.get(aid, (5.0, 1.0))
            for pid in profiles:
                stride = max(1, int(_PROFILE_STRIDE.get(pid, 2)))
                adj = 2.0 / float(stride)
                cpu_sec += d * cpu_r * adj
                gpu_sec += d * gpu_r * adj

    return {
        "experiment_count": total_exp,
        "videos_ok_count": len(valid_videos),
        "total_video_duration_sec": round(sum_dur, 2),
        "heuristic_note": (
            "Estimacion por tablas _REL_CPU_PER_VIDEO_SEC y stride por perfil; "
            "la escena real (ROI, manos, per_hand_fast) cambia llamadas VLM. "
            "Calibra con tus phase1_summary.json cuando tengas datos."
        ),
        "estimated_total_wall_sec_cpu": round(cpu_sec, 1),
        "estimated_total_wall_sec_gpu": round(gpu_sec, 1),
        "estimated_total_wall_hours_cpu": round(cpu_sec / 3600.0, 3),
        "estimated_total_wall_hours_gpu": round(gpu_sec / 3600.0, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-flight antes de experimentos.")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--no-update-catalog", action="store_true", help="No escribir device en experiments_catalog.json.")
    ap.add_argument(
        "--no-update-model-paths",
        action="store_true",
        help="No escribir rutas HF/YOLO resueltas en experiments_catalog (check_models).",
    )
    ap.add_argument("--skip-yolo-engine", action="store_true", help="Reenviado a check_models.")
    ap.add_argument("--download-only-check", action="store_true", help="Solo descarga HF en check_models.")
    args = ap.parse_args()

    print("=== 1. check_models ===")
    import check_models as cm

    catalog_path = args.catalog.expanduser().resolve()
    rep = cm.run_checks(
        skip_yolo_engine=args.skip_yolo_engine,
        download_only=args.download_only_check,
        update_experiments_catalog=None
        if args.no_update_model_paths
        else catalog_path,
    )
    print(
        json.dumps(
            {
                "summary_ok": rep.get("summary_ok"),
                "yolo_count": len(rep.get("yolo", [])),
                "hub_count": len(rep.get("hub_models", [])),
                "experiments_catalog_update": rep.get("experiments_catalog_update"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not rep.get("summary_ok"):
        print("[preflight] check_models fallo. Detalle:")
        for y in rep.get("yolo", []):
            if not y.get("ok"):
                print("  YOLO:", json.dumps(y, ensure_ascii=False))
        for h in rep.get("hub_models", []):
            if not h.get("ok"):
                print("  HF:", json.dumps({k: h[k] for k in h if k != "cache_path"}, ensure_ascii=False))
        sys.exit(1)

    print("\n=== 2. Librerías (requirements) ===")
    ok_lib, rows = check_imports()
    for r in rows:
        print(f"  {r['package']}: {r['ok']}")
    if not ok_lib:
        print("[preflight] Faltan librerías; pip install -r requirements.txt")

    print("\n=== 3. GPU ===")
    gi = gpu_info()
    print(json.dumps(gi, indent=2, ensure_ascii=False))
    device_str = "cpu"
    if gi.get("available") and gi.get("details"):
        device_str = f"cuda:{gi['details'][0]['index']}"

    if not args.no_update_catalog:
        update_catalog_device(catalog_path, device_str, dry=False)
    else:
        print(f"[catalog] sin cambios (--no-update-catalog); sugerido device={device_str!r}")

    print("\n=== 4. Vídeos del catálogo + estimación campaña ===")
    if catalog_path.is_file():
        with open(catalog_path, encoding="utf-8") as f:
            cat = json.load(f)
        vpaths = cat.get("videos") or []
        vrows = scan_videos([str(x) for x in vpaths])
        est = estimate_campaign_time(cat, vrows)
        print(json.dumps({"videos": vrows, "estimate": est}, indent=2, ensure_ascii=False))
    else:
        print(f"No hay catalogo en {catalog_path}")

    print("\n=== Resultado ===")
    if ok_lib:
        print("Librerías OK.")
    sys.exit(0 if ok_lib else 2)


if __name__ == "__main__":
    main()
