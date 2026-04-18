#!/usr/bin/env python3
"""
Pre-vuelo: ejecuta check_models, comprueba librerías, GPU, actualiza device y rutas de modelos en experiments_catalog.json,
y estima tiempos de campaña (heurísticas documentadas).

Uso: python preflight_check.py [opciones]

  --campaign NOMBRE_O_RUTA   (opcional) carpeta de una campaña ya ejecutada, bajo --output-root
                             o ruta absoluta. Si existe phase1_summary.json, ajusta la heurística
                             de tiempos con las medianas reales (derived_ranking_score por approach).
                             Muestra también cintas informativas si hay phase2/3 JSON.

Colores: desactiva con variable de entorno NO_COLOR. En salida no TTY se omite el color.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CATALOG = REPO_ROOT / "approaches" / "experiments_catalog.json"
# Mismo default que test_experiments_approaches.py --output-root
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output_results"

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

_W = 62


def _rule(char: str = "─") -> None:
    print(char * _W)


def _title(text: str) -> None:
    print(f"\n┌{'─' * (_W - 2)}┐")
    pad = max(0, _W - 4 - len(text))
    print(f"│ {text}{' ' * pad} │")
    print(f"└{'─' * (_W - 2)}┘")


def _section(n: str, title: str) -> None:
    print(f"\n  {n}  {title}")
    _rule("·")


def _ok_symbol(ok: bool) -> str:
    return "✓" if ok else "✗"


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR", "").strip()


def _c(code: str, text: str) -> str:
    if not _use_color():
        return text
    return f"{code}{text}\033[0m"


def _green(text: str) -> str:
    return _c("\033[1;32m", text)


def _red(text: str) -> str:
    return _c("\033[1;31m", text)


def _yellow(text: str) -> str:
    return _c("\033[1;33m", text)


def _dim(text: str) -> str:
    return _c("\033[0;2m", text)


def format_duration_hms(seconds: float) -> str:
    """Segundos de pared → cadena tipo 12:34:56 (horas sin límite de dígitos)."""
    t = max(0, int(round(float(seconds))))
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _fmt_gpu_human(gi: dict[str, Any]) -> None:
    if gi.get("error"):
        print(f"    {_ok_symbol(False)}  Error: {gi['error']}")
        return
    if not gi.get("available"):
        print("    CUDA no disponible → se usará CPU en el catálogo.")
        return
    prim = gi.get("primary") or {}
    mem = prim.get("total_memory_gib")
    mem_s = f"{mem} GiB" if mem is not None else "?"
    print(f"    {_ok_symbol(True)}  {prim.get('name', '?')}  ·  VRAM ~{mem_s}")
    if int(gi.get("device_count") or 0) > 1:
        print(f"       ({gi.get('device_count')} dispositivos)")


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


def update_catalog_device(
    catalog_path: Path, device: str, *, dry: bool, quiet: bool = False
) -> bool:
    if not catalog_path.is_file():
        if not quiet:
            print(f"    {_ok_symbol(False)}  No existe el catálogo: {catalog_path}")
        return False
    with open(catalog_path, encoding="utf-8") as f:
        data = json.load(f)
    defaults = data.setdefault("defaults", {})
    old = defaults.get("device")
    if old == device:
        if not quiet:
            print(f"    {_ok_symbol(True)}  device ya era {device!r}")
        return True
    defaults["device"] = device
    if dry:
        if not quiet:
            print(f"    (dry-run) device {old!r} → {device!r}")
        return True
    with open(catalog_path, encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if not quiet:
        print(f"    {_ok_symbol(True)}  device {old!r} → {device!r}")
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


def _effective_approach_profile_ids(catalog: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Igual que matrix en expand_run_specs (test_experiments_approaches)."""
    approaches = [a for a in catalog.get("approaches", []) if not str(a.get("id", "")).startswith("_")]
    profiles_raw = catalog.get("profiles") or {}
    profiles_map = {k: v for k, v in profiles_raw.items() if not k.startswith("_")}
    matrix = catalog.get("matrix") or {}
    mat_app = matrix.get("approaches", "all")
    mat_prof = matrix.get("profiles", "all")
    if mat_app != "all" and isinstance(mat_app, list):
        approaches = [a for a in approaches if a.get("id") in mat_app]
    if mat_prof != "all" and isinstance(mat_prof, list):
        profiles_map = {k: v for k, v in profiles_map.items() if k in mat_prof}
    aid_list = [str(a["id"]) for a in approaches]
    pid_list = list(profiles_map.keys())
    return aid_list, pid_list


def _effective_video_rows_for_estimate(
    catalog: dict[str, Any], video_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Videos que cuenta la fase 1 del orquestador (misma idea que expand_run_specs).
    - single_video_screening: solo el indice screening_video_index (una fila).
    - full_matrix + matrix.videos: filtra rutas resueltas.
    """
    strategy = str(catalog.get("experiment_strategy") or "").strip() or "full_matrix"
    if strategy == "single_video_screening":
        idx = int(catalog.get("screening_video_index", 0))
        if idx < 0 or idx >= len(video_rows):
            return []
        row = video_rows[idx]
        return [row] if row.get("ok") else []

    ok_rows = [v for v in video_rows if v.get("ok")]
    matrix = catalog.get("matrix") or {}
    mat_vid = matrix.get("videos", "all")
    if mat_vid != "all" and isinstance(mat_vid, list):
        sel = {Path(str(v)).expanduser().resolve() for v in mat_vid}
        ok_rows = [
            v for v in ok_rows if Path(str(v.get("path", ""))).expanduser().resolve() in sel
        ]
    return ok_rows


def _resolve_campaign_dir(output_root: Path, campaign: Path) -> Path:
    """Ruta a carpeta de campaña: absoluta o relativa a output_root (como orquestador)."""
    c = campaign.expanduser()
    if c.is_absolute():
        return c.resolve()
    return (output_root / c).resolve()


def calibrate_approach_scores_from_phase1(campaign_dir: Path) -> dict[str, float]:
    """
    Medianas de derived_ranking_score (= wall de pipeline / s de vídeo) por approach_id
    a partir de runs exitosos en phase1_summary.json. Vacío si no hay fichero o datos.
    """
    p1 = campaign_dir / "phase1_summary.json"
    if not p1.is_file():
        return {}
    try:
        with open(p1, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    runs = data.get("runs") or []
    by_aid: dict[str, list[float]] = {}
    for r in runs:
        if not r.get("success"):
            continue
        sc = r.get("derived_ranking_score")
        if sc is None:
            continue
        aid = str(r.get("approach_id") or "").strip()
        if not aid:
            continue
        by_aid.setdefault(aid, []).append(float(sc))
    out: dict[str, float] = {}
    for aid, scores in by_aid.items():
        if scores:
            out[aid] = float(statistics.median(scores))
    return out


def _rel_for_approach(
    aid: str, cal: dict[str, float] | None
) -> tuple[float, float, bool]:
    """
    (cpu_r, gpu_r) por segundo de vídeo fuente, y si se usó mediana phase1.
    Con datos de cal, gpu_r = observado; cpu_r = observado * (tabla_cpu/tabla_gpu) de ese approach.
    `cal` vacío {}: todo por tabla. `None`: sin intentar leer campaña.
    """
    base = _REL_CPU_PER_VIDEO_SEC.get(aid, (5.0, 1.0))
    t_cpu, t_gpu = float(base[0]), float(base[1])
    if t_gpu <= 0:
        t_gpu = 1.0
    if cal is not None and aid in cal and cal[aid] is not None:
        obs = float(cal[aid])
        return (obs * (t_cpu / t_gpu), obs, True)
    return (t_cpu, t_gpu, False)


def estimate_campaign_time(
    catalog: dict[str, Any],
    video_rows: list[dict[str, Any]],
    cal: dict[str, float] | None = None,
) -> dict[str, Any]:
    approaches, profiles = _effective_approach_profile_ids(catalog)
    valid_videos = _effective_video_rows_for_estimate(catalog, video_rows)
    videos_ok_all = [v for v in video_rows if v.get("ok")]

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
            cpu_r, gpu_r, _used_cal = _rel_for_approach(aid, cal)
            for pid in profiles:
                stride = max(1, int(_PROFILE_STRIDE.get(pid, 2)))
                adj = 2.0 / float(stride)
                cpu_sec += d * cpu_r * adj
                gpu_sec += d * gpu_r * adj

    strat = str(catalog.get("experiment_strategy") or "").strip() or "full_matrix"
    note_extra = ""
    if strat == "single_video_screening":
        note_extra = (
            " Fase 1 screening: solo UN video del catalogo (screening_video_index); "
            "la suma de duraciones ya no incluye el resto de clips."
        )
    cal_applied = [a for a in approaches if cal is not None and a in cal]
    cal_missing = [a for a in approaches if cal is None or a not in (cal or {})]
    if cal is not None:
        if cal_applied:
            cal_note = (
                f" Calibrado con phase1: {', '.join(sorted(cal_applied))} = mediana derived_ranking_score; "
                f"resto (tabla fija): {', '.join(sorted(cal_missing))}."
                if cal_missing
                else " Calibrado: todos los approaches de esta matriz con mediana observada en phase1."
            )
        else:
            cal_note = (
                " phase1_summary leído pero sin runs calibrables; heuristica fija (tabla _REL_*)."
            )
    else:
        cal_note = (
            " Heuristica fija. Pasa --campaign a una carpeta con phase1_summary.json "
            "para ajustar por mediana (misma máquina / entorno de la carrera de referencia)."
        )
    return {
        "experiment_count": total_exp,
        "videos_ok_count": len(videos_ok_all),
        "videos_used_in_estimate_count": len(valid_videos),
        "total_video_duration_sec": round(sum_dur, 2),
        "calibration_approach_ids": cal_applied,
        "unresolved_approach_ids": cal_missing,
        "heuristic_note": (
            "Estimacion por tablas _REL_CPU_PER_VIDEO_SEC y stride por perfil; "
            "la escena real (ROI, manos, per_hand_fast) cambia llamadas VLM. "
            + cal_note
            + " CPU vs GPU: en cuda, guia la fila GPU; la fila CPU reescala el ratio desde la misma mediana."
            + note_extra
        ),
        "estimated_total_wall_sec_cpu": round(cpu_sec, 1),
        "estimated_total_wall_sec_gpu": round(gpu_sec, 1),
        "estimated_total_wall_hours_cpu": round(cpu_sec / 3600.0, 3),
        "estimated_total_wall_hours_gpu": round(gpu_sec / 3600.0, 3),
        "estimated_duration_hms_cpu": format_duration_hms(cpu_sec),
        "estimated_duration_hms_gpu": format_duration_hms(gpu_sec),
    }


def print_phase23_reference(campaign_dir: Path) -> None:
    """Líneas informativas a partir de phase2_parallel_summary.json y phase3_parallel_sweep_summary.json."""
    p2 = campaign_dir / "phase2_parallel_summary.json"
    p3 = campaign_dir / "phase3_parallel_sweep_summary.json"
    if not p2.is_file() and not p3.is_file():
        return
    _title("Referencia fases 2 y 3 (misma campaña)")
    if p2.is_file():
        try:
            d2 = json.loads(p2.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            d2 = {}
        mlist = d2.get("models") or []
        for m in mlist[:8]:
            aid = m.get("approach_id", "?")
            nvid = m.get("videos_parallel", "?")
            bw = m.get("batch_wall_clock_sec")
            med = m.get("median_derived_ranking_score")
            print(
                f"    fase2  ·  {aid}  ·  {nvid} víd. en paral.  ·  lote pared {bw} s  ·  mediana score {med}"
            )
        if len(mlist) > 8:
            print(f"    …  +{len(mlist) - 8} entradas en models[]")
    if p3.is_file():
        try:
            d3 = json.loads(p3.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            d3 = {}
        nok = d3.get("last_fully_successful_n_parallel")
        why = d3.get("sweep_ended_because")
        mode = d3.get("sweep_mode")
        st = d3.get("n_steps_tried")
        if isinstance(st, list) and len(st) > 8:
            st_s = f"{st[:8]}… (total {len(st)})"
        else:
            st_s = st
        print(
            f"    fase3  ·  último N OK: {nok}  ·  fin: {why}  ·  modo: {mode}  ·  N probados: {st_s}"
        )


def campaign_metadata(
    catalog: dict[str, Any], video_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Cuenta experimentos = approaches × perfiles × vídeos (misma lógica que expand_run_specs / estimación)."""
    approaches, profiles = _effective_approach_profile_ids(catalog)
    valid_for_exp = _effective_video_rows_for_estimate(catalog, video_rows)
    videos_ok_all = [v for v in video_rows if v.get("ok")]
    n_exp = len(approaches) * len(profiles) * len(valid_for_exp) if valid_for_exp else 0
    return {
        "approach_ids": approaches,
        "approach_count": len(approaches),
        "profile_ids": profiles,
        "profile_count": len(profiles),
        "videos_listed": len(video_rows),
        "videos_ok": len(videos_ok_all),
        "videos_used_for_phase1_count": len(valid_for_exp),
        "experiment_count": n_exp,
        "experiment_strategy": str(catalog.get("experiment_strategy") or "").strip() or "(no definido)",
        "screening_video_index": catalog.get("screening_video_index"),
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
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Raíz de salida (igual que test_experiments_approaches --output-root); solo informativo.",
    )
    ap.add_argument(
        "--campaign",
        type=str,
        default="",
        metavar="CARPETA",
        help=(
            "Carpeta bajo --output-root (mismo --campaign del orquestador) o ruta absoluta. "
            "Si existe phase1_summary.json, la estimación de tiempos usa la mediana de derived_ranking_score "
            "por approach (fase1). Muestra referencia de fase2/3 si hay JSON de resumen."
        ),
    )
    args = ap.parse_args()
    out_root = args.output_root.expanduser().resolve()
    campaign_arg = (args.campaign or "").strip()
    campaign_resolved: Path | None = None
    if campaign_arg:
        campaign_resolved = _resolve_campaign_dir(out_root, Path(campaign_arg))
    cal_for_est: dict[str, float] | None = None
    if campaign_resolved is not None and campaign_resolved.is_dir():
        # {} = phase1 leído pero sin runs útiles; None = no carpeta o sin --campaign
        cal_for_est = calibrate_approach_scores_from_phase1(campaign_resolved)

    _title("Preflight · modelos, entorno y campaña estimada")
    if campaign_arg and campaign_resolved is not None:
        if campaign_resolved.is_dir():
            print(
                f"  Campaña (calibración / ref.): {campaign_resolved}  "
                f"(resuelta contra --output-root si el argumento era relativo)"
            )
        else:
            print(
                _yellow(
                    f"  --campaign: no existe {campaign_resolved} → sin calibración ni ref. fase2/3"
                )
            )
    print(f"  Catálogo: {args.catalog}")

    _section("1", "Modelos (check_models)")
    import check_models as cm

    catalog_path = args.catalog.expanduser().resolve()
    rep = cm.run_checks(
        skip_yolo_engine=args.skip_yolo_engine,
        download_only=args.download_only_check,
        update_experiments_catalog=None
        if args.no_update_model_paths
        else catalog_path,
    )

    y_ok = sum(1 for y in rep.get("yolo", []) if y.get("ok"))
    y_n = len(rep.get("yolo") or [])
    hub_ok = sum(
        1
        for h in rep.get("hub_models", [])
        if h.get("ok") or h.get("optional_gated_skip")
    )
    hub_n = len(rep.get("hub_models") or [])
    summ_ok = bool(rep.get("summary_ok"))

    print(f"    {_ok_symbol(summ_ok)}  Resumen global: {'OK' if summ_ok else 'FALLO'}")
    print(f"       YOLO ····· {y_ok}/{y_n}  ·  Hugging Face ····· {hub_ok}/{hub_n} modelos")

    ecu = rep.get("experiments_catalog_update") or {}
    if args.no_update_model_paths:
        print("       Rutas en experiments_catalog ····· omitidas (--no-update-model-paths)")
    elif ecu.get("ok"):
        print(f"       Rutas en experiments_catalog ····· {_ok_symbol(True)}  {ecu.get('path', '')}")
    elif ecu.get("skipped"):
        print(f"       Rutas en experiments_catalog ····· (sin escribir: {ecu.get('reason', '')})")
    elif ecu.get("error"):
        print(f"       Rutas en experiments_catalog ····· {_ok_symbol(False)}  {ecu.get('error')}")

    if not summ_ok:
        print()
        print(_red("    ✗  Modelos / Hub: corrige errores antes de lanzar la campaña."))
        print("\n    Detalle de fallos:")
        for y in rep.get("yolo", []):
            if not y.get("ok"):
                print("      YOLO:", json.dumps(y, ensure_ascii=False))
        for h in rep.get("hub_models", []):
            if not h.get("ok") and not h.get("optional_gated_skip"):
                print(
                    "      HF:",
                    json.dumps({k: h[k] for k in h if k != "cache_path"}, ensure_ascii=False),
                )
        sys.exit(1)

    _section("2", "Librerías (requirements)")
    ok_lib, rows = check_imports()
    ok_ct = sum(1 for r in rows if r["ok"] == "yes")
    print(f"    {_ok_symbol(ok_lib)}  {ok_ct}/{len(rows)} paquetes importables")
    if not ok_lib:
        for r in rows:
            if r["ok"] != "yes":
                print(f"       ✗  {r['package']}: {r.get('error', '')[:72]}")
        print("    → pip install -r requirements.txt")

    _section("3", "GPU y catálogo (device)")
    gi = gpu_info()
    _fmt_gpu_human(gi)
    device_str = "cpu"
    if gi.get("available") and gi.get("details"):
        device_str = f"cuda:{gi['details'][0]['index']}"

    print("    Catálogo experiments_catalog.json")
    if not args.no_update_catalog:
        update_catalog_device(catalog_path, device_str, dry=False)
    else:
        print(f"    …  sin escribir device (--no-update-catalog); sugerido: {device_str!r}")

    est: dict[str, Any] | None = None
    vrows: list[dict[str, Any]] = []
    cat_data: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None

    _section("4", "Vídeos del catálogo")
    if catalog_path.is_file():
        with open(catalog_path, encoding="utf-8") as f:
            cat_data = json.load(f)
        vpaths = cat_data.get("videos") or []
        vrows = scan_videos([str(x) for x in vpaths])
        v_ok = sum(1 for v in vrows if v.get("ok"))
        v_all = len(vrows) > 0 and v_ok == len(vrows)
        print(f"    {_ok_symbol(v_all)}  {v_ok}/{len(vrows)} vídeos accesibles")
        for v in vrows:
            if not v.get("ok"):
                short = str(v.get("path", ""))[-52:]
                print(f"       ✗  …{short}  ({v.get('error', '')})")
        est = estimate_campaign_time(cat_data, vrows, cal_for_est)
        meta = campaign_metadata(cat_data, vrows)
    else:
        print(f"    {_ok_symbol(False)}  No se encontró {catalog_path}")

    _title("Campaña · números y salida")
    issues: list[str] = []
    if not ok_lib:
        issues.append("Faltan paquetes Python (requirements).")
    if not catalog_path.is_file():
        issues.append("No existe el fichero de catálogo.")
    elif len(vrows) == 0:
        issues.append("El catálogo no define vídeos (lista vacía).")
    elif not all(v.get("ok") for v in vrows):
        issues.append("Hay rutas de vídeo inválidas o archivos inaccesibles.")
    elif meta and meta.get("experiment_count", 0) == 0:
        issues.append("Con los vídeos OK actuales el número de experimentos sería 0.")

    if meta:
        print(f"    Estrategia de experimentos · {meta['experiment_strategy']}")
        si = meta.get("screening_video_index")
        if si is not None and str(meta["experiment_strategy"]).startswith("single"):
            print(f"    Índice vídeo screening (fase 1) · {si}")
        print()
        print(f"    {_dim('Approaches (modelos)')}     {meta['approach_count']}")
        print(f"    {_dim('Perfiles')}               {meta['profile_count']}")
        print(f"    {_dim('Vídeos en catálogo')}      {meta['videos_listed']}")
        print(f"    {_dim('Vídeos OK (existen)')}    {meta['videos_ok']}")
        v_p1 = int(meta.get("videos_used_for_phase1_count") or 0)
        if v_p1 != int(meta.get("videos_ok") or 0):
            print(
                f"    {_dim('Vídeos en fase 1 (estim.)')}  {v_p1}  "
                f"{_dim('(p. ej. screening: un solo clip; el resto no cuenta en fase 1)')}"
            )
        print()
        print(_green(f"    Experimentos previstos · {meta['experiment_count']}") if not issues else _red(f"    Experimentos previstos · {meta['experiment_count']}"))
        print()
        print(f"    {_dim('Salida por defecto')}")
        print(f"       {out_root}/")
        print(f"       └── <nombre_campaña>/   ← usa --campaign en test_experiments_approaches.py")
        print(f"             ├── métricas JSON / vídeo procesado / resúmenes")
        print()
        print(f"    {_dim('Orquestador')}")
        print("       python approaches/test_experiments_approaches.py \\")
        print(f"         --catalog {catalog_path}")
        print(f"         --output-root {out_root} \\")
        print('         --campaign "mi_campaña"')
    else:
        print(_red("    No hay datos de campaña (catálogo ausente o ilegible)."))
        if catalog_path.is_file():
            issues.append("No se pudieron leer metadatos del catálogo.")

    _title("Tiempo estimado (heurística · pared)")
    if est:
        cpu_h = format_duration_hms(float(est["estimated_total_wall_sec_cpu"]))
        gpu_h = format_duration_hms(float(est["estimated_total_wall_sec_gpu"]))
        tw = max(len(cpu_h), len(gpu_h), 12)
        h_cpu = float(est["estimated_total_wall_hours_cpu"])
        h_gpu = float(est["estimated_total_wall_hours_gpu"])
        print(f"    {'CPU (aprox.)':<16} {cpu_h:>{tw}}    {_dim(f'(~{h_cpu:.2f} h)')}")
        print(f"    {'GPU (aprox.)':<16} {gpu_h:>{tw}}    {_dim(f'(~{h_gpu:.2f} h)')}")
        if gi.get("available"):
            print(
                f"    {_dim('Nota ·')} "
                f"{_dim('Si ejecutas en GPU (cuda), guía la fila GPU; CPU no es “peor caso” del mismo run, son heurísticas distintas.')}"
            )
        print()
        print(f"    {_dim('Suma duración vídeos OK ·')} {est['total_video_duration_sec']:.1f} s")
        wrap = est.get("heuristic_note", "")
        if wrap:
            chunk = 54
            print(f"    {_dim('Nota ·')} ", end="")
            print(_dim(wrap[:chunk]))
            for i in range(chunk, len(wrap), chunk):
                print(f"           {_dim(wrap[i : i + chunk])}")
        if cal_for_est is not None and not cal_for_est:
            print()
            print(
                f"    {_dim('phase1_summary.json leído sin medianas (ningún run exitoso con score).')}"
            )
        elif cal_for_est:
            print()
            print(f"    {_dim('Medianas phase1 (derived_ranking_score) aplicadas al coste por approach:')}")
            for aid, v in sorted(cal_for_est.items(), key=lambda x: x[0])[:24]:
                print(f"      {aid}  →  {v:.3f}  s pipeline / s vídeo")
            if len(cal_for_est) > 24:
                print(f"      …  +{len(cal_for_est) - 24} approaches")
    else:
        print(_yellow("    Sin estimación: falta catálogo o vídeos válidos."))

    if campaign_resolved is not None and campaign_resolved.is_dir():
        print_phase23_reference(campaign_resolved)

    print()
    _rule("═")
    launch_ready = (
        summ_ok
        and ok_lib
        and catalog_path.is_file()
        and len(issues) == 0
        and bool(meta)
        and int(meta.get("experiment_count") or 0) > 0
    )
    if launch_ready:
        print(_green(f"  {'✓ TODO OK · LISTO PARA LANZAR EXPERIMENTOS':^{_W}}"))
    else:
        print(_red(f"  {'✗ PREFLIGHT INCOMPLETO · REVISA ANTES DE LANZAR':^{_W}}"))
        for it in issues:
            print(_red(f"    • {it}"))
    _rule("═")
    print()

    sys.exit(0 if ok_lib else 2)


if __name__ == "__main__":
    main()
