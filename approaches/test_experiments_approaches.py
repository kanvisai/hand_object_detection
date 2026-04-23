#!/usr/bin/env python3
"""
Orquesta experimentos mano-objeto: recorre approaches (test_new_handobject_*.py) x perfiles x videos,
volcando JSON enriquecido + video bajo output_results/. phase1_summary.json incluye campaign_summary
(conteos, tasas de éxito por approach, medianas/p90). Cada ejecución escribe también
output_results/<campaña>/logs/orchestrator_<UTC>.log (salida del orquestador + stdout/stderr de cada hijo).

Tras la fase 1, opcionalmente lanza fase 2: por defecto los 2 mejores modelos; con --phase2-all-ranked todos los
del ranking (mejor perfil c/u), N videos en paralelo por modelo (cada modelo en su tanda).
phase1_summary/phase2_parallel incluyen `resource_summary` (CPU, RAM, VRAM) agregado desde metrics.json + experiment_summary.

Fase 3 opcional (--phase3-parallel-sweep o --only-phase3): el mismo approach y un vídeo, N subprocesos
en paralelo. Por defecto barrido adaptativo (dobla N y refina por búsqueda binaria) hasta fallo, OOM o
tope --phase3-max-parallel. Con --phase3-sweep-mode fixed se usa --phase3-steps. Ver phase3_parallel_sweep_summary.json.
Fases 3 y 4 fuerzan stride de pipeline PHASE34_PIPELINE_STRIDE (3), independiente del perfil del catálogo.

Fase 4 (--phase4-parallel-sweep o --only-phase4): repite el barrido de fase 3 para cada approach del
catálogo (mismo vídeo y perfil), estima N máximo por modelo y agrega min/mediana/max de latency_sec en
vlm_calls y tiempo pipeline vs duración del clip. Ver phase4_parallel_sweep_summary.json.

Modo sin ventana: fase1/2/3 no pasan --save (sin vídeo de salida; solo métricas; alineado a escenario
producción sin grabar). No se usa cv2.imshow. Entorno headless
MPLBACKEND=Agg. No forzar QT_QPA_PLATFORM=offscreen: muchos paquetes opencv solo traen el plugin
Qt xcb; forzar "offscreen" inexistente llena el log de errores. Sin DISPLAY, usar opencv-python-headless o xvfb.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import statistics
import socket
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from subprocess import PIPE, Popen, run as subprocess_run
from typing import Any, TextIO

# paquetes opcionales
try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
APPROACHES_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = APPROACHES_DIR / "experiments_catalog.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output_results"

# Fases 3 y 4: stride del pipeline (handobject_shared) fijo; no usa el stride del perfil de catálogo.
PHASE34_PIPELINE_STRIDE = 3


class _TeeIO:
    """Duplica escritura en terminal y fichero de log."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        return False


def _setup_session_logging(
    campaign_dir: Path,
    no_session_log: bool,
    orig_stdout: TextIO,
) -> tuple[tuple[TextIO, threading.Lock] | None, TextIO | None, Path | None]:
    if no_session_log:
        print(f"[campaña] salida en: {campaign_dir}")
        return None, None, None
    log_dir = campaign_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_log_path = log_dir / ("orchestrator_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".log")
    log_fp = open(session_log_path, "w", encoding="utf-8", buffering=1)
    log_lock = threading.Lock()
    sys.stdout = _TeeIO(orig_stdout, log_fp)
    print(f"[orchestrator] log de sesión: {session_log_path.resolve()}")
    print(f"[campaña] salida en: {campaign_dir}")
    return (log_fp, log_lock), log_fp, session_log_path


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)[:180]


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _strip_meta(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def merged_cli_to_argv(merged: dict[str, Any]) -> list[str]:
    """Convierte claves tipo argparse (snake_case) a argv para handobject_shared."""
    argv: list[str] = []
    bool_flags = {"per_hand_fast"}
    string_keys = {
        "video",
        "vlm_model",
        "vlm_prompt",
        "device",
        "display",
        "save",
        "output",
        "pose_weights",
        "personal_weights",
        "roi_region",
        "drop_zone_mode",
    }
    float_keys = {
        "fast_gray_zone",
        "wrist_conf_th",
        "elbow_conf_th",
        "yes_th",
        "force_drop_th",
        "raw_drop_th",
        "iou_track_th",
        "personal_conf",
        "personal_near_px",
    }
    int_keys = {
        "stride",
        "crop_size",
        "crop_min",
        "crop_max",
        "hold_frames",
        "drop_frames",
        "force_drop_frames",
        "raw_drop_frames",
        "no_hands_drop_steps",
        "drop_window_steps",
        "traj_smooth_len",
        "personal_stride",
        "max_track_lost",
    }
    for key, val in merged.items():
        if key.startswith("_"):
            continue
        if val is None or val == "":
            continue
        arg = "--" + key.replace("_", "-")
        if key in bool_flags:
            if val is True:
                argv.append(arg)
            continue
        if key in int_keys:
            argv.extend([arg, str(int(val))])
        elif key in float_keys:
            argv.extend([arg, str(float(val))])
        elif key in string_keys:
            argv.extend([arg, str(val)])
        else:
            argv.extend([arg, str(val)])
    return argv


@dataclass
class RunSpec:
    approach_id: str
    script: str
    profile_id: str
    video_path: Path
    merged_args: dict[str, Any]
    vlm_model_override: str | None = None
    # Por defecto no se escribe mp4 (--save vacío); poner True solo si hace falta depurar salida en disco
    write_output_video: bool = False


def load_catalog(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_run_specs(catalog: dict[str, Any]) -> list[RunSpec]:
    defaults = _strip_meta(catalog.get("defaults") or {})
    videos_raw = catalog.get("videos") or []
    approaches = catalog.get("approaches") or []
    profiles_raw = catalog.get("profiles") or {}
    profiles = {k: _strip_meta(v) for k, v in profiles_raw.items() if not k.startswith("_")}
    matrix = catalog.get("matrix") or {}

    vid_paths = [Path(v).expanduser() for v in videos_raw if str(v).strip()]
    if not vid_paths:
        raise RuntimeError("experiments_catalog.json: lista videos vacia o invalida.")

    strategy = catalog.get("experiment_strategy", "full_matrix")
    if strategy == "single_video_screening":
        idx = int(catalog.get("screening_video_index", 0))
        if idx < 0 or idx >= len(vid_paths):
            raise RuntimeError(
                f"screening_video_index={idx} fuera de rango "
                f"(videos disponibles tras filtros: {len(vid_paths)})."
            )
        vid_paths = [vid_paths[idx]]

    approach_objs = [a for a in approaches if not str(a.get("id", "")).startswith("_")]
    mat_app = matrix.get("approaches", "all")
    mat_vid = matrix.get("videos", "all")
    mat_prof = matrix.get("profiles", "all")

    if mat_app != "all" and isinstance(mat_app, list):
        approach_objs = [a for a in approach_objs if a.get("id") in mat_app]
    if mat_prof != "all" and isinstance(mat_prof, list):
        profiles = {k: v for k, v in profiles.items() if k in mat_prof}

    if not profiles:
        raise RuntimeError("No hay perfiles tras aplicar matrix.")

    runs: list[RunSpec] = []
    for ap in approach_objs:
        aid = str(ap["id"])
        script = str(ap["script"])
        vm = ap.get("vlm_model") or None
        for pi, prof_body in profiles.items():
            merged = _deep_merge(defaults, prof_body)
            if vm:
                merged["vlm_model"] = vm
            for vp in vid_paths:
                runs.append(
                    RunSpec(
                        approach_id=aid,
                        script=script,
                        profile_id=pi,
                        video_path=vp,
                        merged_args=dict(merged),
                        vlm_model_override=vm,
                    )
                )

    if mat_vid != "all" and isinstance(mat_vid, list):
        sel = {Path(v).expanduser().resolve() for v in mat_vid}
        runs = [r for r in runs if r.video_path.expanduser().resolve() in sel]

    return runs


def enrich_pipeline_report(report: dict[str, Any]) -> dict[str, Any]:
    """Anade metricas derivadas sobre el JSON que ya escribe run_pipeline."""
    calls = report.get("vlm_calls") or []
    latencies = [float(c["latency_sec"]) for c in calls if "latency_sec" in c]
    vlm_sum = float(sum(latencies))
    wall = report.get("wall_clock_processing_sec")
    dur = report.get("video_duration_sec_metadata")
    frames = report.get("frames_processed")
    inf = len(latencies)

    overhead = None
    if wall is not None and latencies:
        overhead = float(wall) - vlm_sum

    derived: dict[str, Any] = {
        "derived_experiment_version": 1,
        "derived_vlm_compute_sum_sec": round(vlm_sum, 6) if latencies else None,
        "derived_non_vlm_overhead_sec": round(overhead, 6) if overhead is not None else None,
        "derived_wall_clock_per_video_second": round(float(wall) / float(dur), 6)
        if wall and dur
        else None,
        "derived_vlm_compute_per_video_second": round(vlm_sum / float(dur), 6)
        if dur and latencies
        else None,
        "derived_mean_vlm_sec_per_call": report.get("vlm_latency_mean_sec"),
        "derived_vlm_calls_per_video_second": round(inf / float(dur), 6) if dur and inf else None,
        "derived_frames_per_wall_second": round(float(frames) / float(wall), 6)
        if frames and wall
        else None,
        "derived_ranking_score": None,
    }
    # Score unico para ordenar (menor = mejor coste tiempo vs duracion video)
    if wall and dur and float(dur) > 0:
        derived["derived_ranking_score"] = round(float(wall) / float(dur), 6)

    vlm_mean = report.get("vlm_latency_mean_sec")
    if vlm_mean is not None:
        vm = float(vlm_mean)
        derived["derived_vlm_mean_latency_ms"] = round(vm * 1000.0, 3)
        derived["derived_vlm_mean_latency_ns"] = int(round(vm * 1e9))
    if wall is not None and dur is not None:
        derived["derived_seconds_over_realtime"] = round(float(wall) - float(dur), 6)

    # Lectura rapida lado a lado en metrics.json (mismo clip):
    # video_duration_sec ≈ duracion del archivo; video_duration_vlm_application = tiempo total del pipeline.
    if dur is not None:
        derived["video_duration_sec"] = round(float(dur), 6)
    if wall is not None:
        derived["video_duration_vlm_application"] = round(float(wall), 6)

    out = dict(report)
    out.update(derived)
    return out


def _headless_env(base: dict[str, str]) -> dict[str, str]:
    """
    Entorno de hijos: matplotlib sin display.
    No fijar QT_QPA_PLATFORM=offscreen: la build típica de opencv-python solo incluye el plugin
    "xcb" bajo site-packages/cv2/qt; pedir "offscreen" hace que Qt falle aunque el run siga.
    Con DISPLAY, xcb es aceptable; el usuario puede exportar otra plataforma antes de invocar al orquestador.
    """
    env = dict(base)
    env.setdefault("MPLBACKEND", "Agg")
    if (
        "QT_QPA_PLATFORM" not in base
        and "QT_QPA_PLATFORM" not in env
        and (base.get("DISPLAY") or os.environ.get("DISPLAY"))
    ):
        env.setdefault("QT_QPA_PLATFORM", "xcb")
    return env


def run_single_experiment(
    spec: RunSpec,
    run_dir: Path,
    monitor_psutil: bool,
    *,
    stream_log: tuple[TextIO, threading.Lock] | None = None,
    run_label: str = "",
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    video_path = run_dir / "output_video.mp4"

    merged = dict(spec.merged_args)
    merged["video"] = str(spec.video_path.resolve())
    if spec.write_output_video:
        merged["save"] = str(video_path.resolve())
    else:
        merged["save"] = ""
    merged["output"] = str(metrics_path.resolve())

    argv = [sys.executable, str(APPROACHES_DIR / spec.script)] + merged_cli_to_argv(merged)

    env = _headless_env(dict(**os.environ))
    t0 = time.perf_counter()
    peak_rss = 0
    cpu_samples: list[float] = []

    proc = Popen(argv, cwd=str(APPROACHES_DIR), stdout=PIPE, stderr=PIPE, text=True, env=env)
    mon_stop = threading.Event()

    def _monitor() -> None:
        nonlocal peak_rss
        if not monitor_psutil or psutil is None:
            return
        pino = psutil.Process(proc.pid)
        while not mon_stop.wait(0.25):
            try:
                rss = int(pino.memory_info().rss)
                peak_rss = max(peak_rss, rss)
                cpu_samples.append(float(pino.cpu_percent(interval=None)))
            except (psutil.Error, OSError):
                break

    th = threading.Thread(target=_monitor, daemon=True)
    th.start()

    lb = run_label.strip()
    prefix = (lb + " ") if lb else ""

    if stream_log is None:
        out_b, err_b = proc.communicate()
        out_b = out_b or ""
        err_b = err_b or ""
    else:
        log_fp, log_lock = stream_log
        out_chunks: list[str] = []
        err_chunks: list[str] = []

        def _drain(pipe: Any, chunks: list[str], tag: str) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    chunks.append(line)
                    block = f"{prefix}[{tag}] {line}"
                    sys.stdout.write(block)
                    sys.stdout.flush()
                    # Si stdout ya está en tee (terminal + fichero), evitar doble volcado en log.
                    if not isinstance(sys.stdout, _TeeIO):
                        with log_lock:
                            log_fp.write(block)
                            log_fp.flush()
            finally:
                pipe.close()

        t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks, "stdout"))
        t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks, "stderr"))
        t_out.start()
        t_err.start()
        proc.wait()
        t_out.join()
        t_err.join()
        out_b = "".join(out_chunks)
        err_b = "".join(err_chunks)

    mon_stop.set()
    th.join(timeout=1.0)
    wall = time.perf_counter() - t0

    host_metrics: dict[str, Any] = {
        "subprocess_exit_code": proc.returncode,
        "subprocess_wall_sec": round(wall, 6),
        "host_peak_rss_bytes": peak_rss if peak_rss else None,
        "host_cpu_percent_mean": round(sum(cpu_samples) / len(cpu_samples), 4)
        if cpu_samples
        else None,
        "host_cpu_percent_max": round(max(cpu_samples), 4) if cpu_samples else None,
        "hostname": socket.gethostname(),
    }
    if psutil is None:
        host_metrics["host_monitor_note"] = "Instala psutil para RSS/CPU del proceso hijo."

    envelope: dict[str, Any] = {
        "run_id": run_dir.name,
        "approach_id": spec.approach_id,
        "profile_id": spec.profile_id,
        "video_input": str(spec.video_path.resolve()),
        "script": spec.script,
        "argv": argv,
        "started_utc": _utc_iso(),
        "host_process_metrics": host_metrics,
        "stdout_tail": (out_b or "")[-4000:],
        "stderr_tail": (err_b or "")[-8000:],
    }

    report: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as jf:
                raw = json.load(jf)
            report = enrich_pipeline_report(raw)
            with open(metrics_path, "w", encoding="utf-8") as jf:
                json.dump(report, jf, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, OSError) as e:
            envelope["metrics_load_error"] = str(e)
    else:
        envelope["metrics_missing"] = True

    success_flag = (
        proc.returncode == 0
        and metrics_path.exists()
        and not envelope.get("metrics_load_error")
        and pipeline_report_indicates_success(report)
    )
    envelope["ok"] = success_flag

    full = {"envelope": envelope, "pipeline": report}
    summary_path = run_dir / "experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as jf:
        json.dump(full, jf, indent=2, ensure_ascii=False)

    return {
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "metrics_json": str(metrics_path) if metrics_path.exists() else "",
        "video_out": str(video_path) if video_path.exists() else "",
        "approach_id": spec.approach_id,
        "profile_id": spec.profile_id,
        "video_path": str(spec.video_path.resolve()),
        "success": success_flag,
        "derived_ranking_score": report.get("derived_ranking_score"),
        "wall_clock_processing_sec": report.get("wall_clock_processing_sec"),
        "video_duration_sec": report.get("video_duration_sec")
        if report.get("video_duration_sec") is not None
        else report.get("video_duration_sec_metadata"),
        "vlm_inference_count": report.get("vlm_inference_count"),
        "vlm_latency_mean_sec": report.get("vlm_latency_mean_sec"),
        "derived_vlm_mean_latency_ms": report.get("derived_vlm_mean_latency_ms"),
        "derived_seconds_over_realtime": report.get("derived_seconds_over_realtime"),
        "video_duration_vlm_application": report.get("video_duration_vlm_application"),
    }


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    m = n // 2
    return float(s[m]) if n % 2 else (s[m - 1] + s[m]) / 2.0


def percentile(xs: list[float], pct: float) -> float | None:
    """Percentil lineal (pct en [0,100])."""
    if not xs or not (0 <= pct <= 100):
        return None
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return float(s[0])
    k = (n - 1) * (pct / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(s[f])
    return float(s[f] * (c - k) + s[c] * (k - f))


def pipeline_report_indicates_success(report: dict[str, Any]) -> bool:
    """Hay métricas mínimas del pipeline (evita {} o JSON vacío contando como OK)."""
    if not report:
        return False
    return (
        report.get("wall_clock_processing_sec") is not None
        or report.get("derived_ranking_score") is not None
    )


def _bytes_to_mib(b: int | None) -> float | None:
    if b is None or b <= 0:
        return None
    return round(float(b) / (1024.0**2), 4)


def extract_resource_row_from_run_dir(run_dir: str | Path) -> dict[str, Any]:
    """
    Lee metrics.json y experiment_summary.json de un run y devuelve subconjunto comparable.
    nvidia_smi en metrics refleja snapshots del GPU (puede incluir otros procesos en el instante del muestreo).
    """
    rd = Path(run_dir)
    out: dict[str, Any] = {
        "run_dir": str(rd),
        "host_peak_rss_mib": None,
        "subprocess_wall_sec": None,
        "host_cpu_percent_mean": None,
        "host_cpu_percent_max": None,
        "torch_cuda_peak_allocated_mib": None,
        "nvidia_gpu_utilization_mean_percent": None,
        "nvidia_gpu_memory_used_mean_mib": None,
        "nvidia_gpu_memory_total_mib_sample": None,
        "est_max_parallel_streams_by_memory_naive": None,
        "vlm_device": None,
    }
    sp = rd / "experiment_summary.json"
    if sp.is_file():
        try:
            with open(sp, encoding="utf-8") as f:
                ex = json.load(f)
            env = ex.get("envelope") or {}
            h = (env.get("host_process_metrics") or {}) if isinstance(env.get("host_process_metrics"), dict) else {}
            pr = h.get("host_peak_rss_bytes")
            if pr is not None:
                out["host_peak_rss_mib"] = _bytes_to_mib(int(pr))
            if h.get("subprocess_wall_sec") is not None:
                out["subprocess_wall_sec"] = float(h["subprocess_wall_sec"])
            if h.get("host_cpu_percent_mean") is not None:
                out["host_cpu_percent_mean"] = float(h["host_cpu_percent_mean"])
            if h.get("host_cpu_percent_max") is not None:
                out["host_cpu_percent_max"] = float(h["host_cpu_percent_max"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    mp = rd / "metrics.json"
    if mp.is_file():
        try:
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("torch_cuda_peak_memory_allocated_mib") is not None:
                out["torch_cuda_peak_allocated_mib"] = float(m["torch_cuda_peak_memory_allocated_mib"])
            if m.get("nvidia_gpu_utilization_mean_percent") is not None:
                out["nvidia_gpu_utilization_mean_percent"] = float(m["nvidia_gpu_utilization_mean_percent"])
            if m.get("nvidia_gpu_memory_used_mean_mib") is not None:
                out["nvidia_gpu_memory_used_mean_mib"] = float(m["nvidia_gpu_memory_used_mean_mib"])
            if m.get("nvidia_gpu_memory_total_mib_sample") is not None:
                out["nvidia_gpu_memory_total_mib_sample"] = float(m["nvidia_gpu_memory_total_mib_sample"])
            if m.get("est_max_parallel_streams_by_memory_naive") is not None:
                out["est_max_parallel_streams_by_memory_naive"] = int(m["est_max_parallel_streams_by_memory_naive"])
            if m.get("vlm_device") is not None:
                out["vlm_device"] = str(m["vlm_device"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return out


def _num_stats(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"min": None, "max": None, "median": None, "mean": None}
    m = median(xs)
    return {
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
        "median": round(m, 4) if m is not None else None,
        "mean": round(sum(xs) / len(xs), 4),
    }


def aggregate_resource_rows(
    rows: list[dict[str, Any]],
    *,
    include_per_run: bool = True,
) -> dict[str, Any]:
    """Agrega medianas/min/max de filas de extract_resource_row_from_run_dir."""
    if not rows:
        return {
            "runs_aggregated": 0,
            "numeric_summary": {},
            "per_run": [] if include_per_run else None,
        }
    keys_float = [
        "host_peak_rss_mib",
        "subprocess_wall_sec",
        "host_cpu_percent_mean",
        "host_cpu_percent_max",
        "torch_cuda_peak_allocated_mib",
        "nvidia_gpu_utilization_mean_percent",
        "nvidia_gpu_memory_used_mean_mib",
    ]
    per_key: dict[str, dict[str, float | None]] = {}
    for k in keys_float:
        vals: list[float] = []
        for r in rows:
            v = r.get(k)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if vals:
            per_key[k] = _num_stats(vals)  # type: ignore[assignment]
    out: dict[str, Any] = {
        "runs_aggregated": len(rows),
        "numeric_summary": per_key,
    }
    if include_per_run:
        out["per_run"] = rows
    return out


def build_best_profile_comparison(
    run_results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Una fila por approach usando el mejor perfil (ranked): tiempos medianos para comparar
    CLIP vs Qwen vs Florence en el mismo vídeo / misma campaña.
    """
    rows_out: list[dict[str, Any]] = []
    for entry in ranked:
        aid = str(entry.get("approach_id") or "")
        pid = str(entry.get("profile_id") or "")
        sub = [
            r
            for r in run_results
            if r.get("success")
            and str(r.get("approach_id")) == aid
            and str(r.get("profile_id")) == pid
        ]
        walls = [
            float(r["wall_clock_processing_sec"])
            for r in sub
            if r.get("wall_clock_processing_sec") is not None
        ]
        durs = [
            float(r["video_duration_sec"]) for r in sub if r.get("video_duration_sec") is not None
        ]
        vlms = [
            float(r["vlm_latency_mean_sec"])
            for r in sub
            if r.get("vlm_latency_mean_sec") is not None
        ]
        counts = [
            float(r["vlm_inference_count"])
            for r in sub
            if r.get("vlm_inference_count") is not None
        ]
        mw = median(walls)
        mdur = median(durs)
        mvlm = median(vlms)
        mcnt = median(counts)
        over = (mw - mdur) if mw is not None and mdur is not None else None
        rows_out.append(
            {
                "approach_id": aid,
                "profile_id": pid,
                "runs_used": len(sub),
                "median_wall_clock_processing_sec": round(mw, 6) if mw is not None else None,
                "median_video_duration_sec": round(mdur, 6) if mdur is not None else None,
                "median_seconds_over_realtime": round(over, 6) if over is not None else None,
                "median_vlm_latency_sec_per_query": round(mvlm, 6) if mvlm is not None else None,
                "median_vlm_latency_ms_per_query": round(mvlm * 1000.0, 3) if mvlm is not None else None,
                "median_vlm_inference_count": int(round(mcnt)) if mcnt is not None else None,
                "median_derived_ranking_score": entry.get("median_ranking_score"),
            }
        )

    rows_out.sort(
        key=lambda x: (
            x["median_wall_clock_processing_sec"] is None,
            x["median_wall_clock_processing_sec"] if x["median_wall_clock_processing_sec"] is not None else 1e30,
        )
    )
    return rows_out


def summarize_phase1_campaign(
    run_results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    top2: list[dict[str, Any]],
    *,
    phase1_wall_sec: float,
) -> dict[str, Any]:
    """Métricas agregadas de la fase 1 para lectura rápida sin recorrer todos los runs."""
    n = len(run_results)
    succeeded = [r for r in run_results if r.get("success")]
    failed = [r for r in run_results if not r.get("success")]
    scores = [float(r["derived_ranking_score"]) for r in run_results if r.get("derived_ranking_score") is not None]

    by_approach: dict[str, dict[str, Any]] = {}
    for r in run_results:
        aid = str(r.get("approach_id") or "")
        if aid not in by_approach:
            by_approach[aid] = {
                "runs": 0,
                "success_count": 0,
                "scores": [],
                "walls": [],
                "durs": [],
                "vlm_means": [],
                "vlm_counts": [],
                "overs": [],
            }
        by_approach[aid]["runs"] += 1
        if r.get("success"):
            by_approach[aid]["success_count"] += 1
            if r.get("wall_clock_processing_sec") is not None:
                by_approach[aid]["walls"].append(float(r["wall_clock_processing_sec"]))
            if r.get("video_duration_sec") is not None:
                by_approach[aid]["durs"].append(float(r["video_duration_sec"]))
            if r.get("vlm_latency_mean_sec") is not None:
                by_approach[aid]["vlm_means"].append(float(r["vlm_latency_mean_sec"]))
            if r.get("vlm_inference_count") is not None:
                by_approach[aid]["vlm_counts"].append(float(r["vlm_inference_count"]))
            if r.get("derived_seconds_over_realtime") is not None:
                by_approach[aid]["overs"].append(float(r["derived_seconds_over_realtime"]))
        sc = r.get("derived_ranking_score")
        if sc is not None:
            by_approach[aid]["scores"].append(float(sc))

    per_app: dict[str, Any] = {}
    for aid, d in by_approach.items():
        rc = int(d["runs"])
        scount = int(d["success_count"])
        sc_list = d["scores"]
        vm = median(d["vlm_means"])
        per_app[aid] = {
            "runs": rc,
            "success_count": scount,
            "success_rate": round(scount / rc, 4) if rc else 0.0,
            "median_derived_ranking_score": median(sc_list),
            "p90_derived_ranking_score": percentile(sc_list, 90),
            "median_wall_clock_processing_sec": median(d["walls"]),
            "median_video_duration_sec": median(d["durs"]),
            "median_seconds_over_realtime": median(d["overs"]),
            "median_vlm_latency_sec_per_query": round(vm, 6) if vm is not None else None,
            "median_vlm_latency_ms_per_query": round(vm * 1000.0, 3) if vm is not None else None,
            "median_vlm_inference_count": int(round(median(d["vlm_counts"])))
            if d["vlm_counts"]
            else None,
        }

    comparison_best_profile = build_best_profile_comparison(run_results, ranked)

    res_rows: list[dict[str, Any]] = []
    for r in succeeded:
        rd = r.get("run_dir")
        if rd and Path(str(rd)).exists():
            res_rows.append(extract_resource_row_from_run_dir(str(rd)))
    if res_rows and len(res_rows) > 20:
        resource_summary = aggregate_resource_rows(res_rows, include_per_run=False)
    else:
        resource_summary = aggregate_resource_rows(res_rows, include_per_run=True)

    return {
        "phase1_orchestrator_wall_sec": round(phase1_wall_sec, 3),
        "runs_planned": n,
        "runs_succeeded": len(succeeded),
        "runs_failed": len(failed),
        "runs_with_derived_ranking_score": len(scores),
        "ranked_approaches_count": len(ranked),
        "phase2_eligible_count": len(top2),
        "derived_ranking_score_global_median": median(scores),
        "derived_ranking_score_global_p90": percentile(scores, 90),
        "by_approach": per_app,
        "comparison_best_profile": comparison_best_profile,
        "resource_summary": resource_summary,
        "resource_fields_glossary": {
            "host_peak_rss_mib": "Pico RSS del proceso hijo (psutil) por run, en MiB; mediana/min/max en runs exitosos.",
            "subprocess_wall_sec": "Pared del subproceso (orquestador) incluye carga de modelos, etc.",
            "torch_cuda_peak_allocated_mib": "Pico VRAM reservada por asignación PyTorch del proceso; por run.",
            "nvidia_gpu_utilization_mean_percent": "Media muestreos nvidia-smi en el tramo con vídeo (puede mezclar con otros procesos en GPU).",
            "nvidia_gpu_memory_used_mean_mib": "Media memoria usada de la GPU (nvidia-smi, todo el dispositivo en el instante).",
            "host_cpu_percent_mean": "Media CPU% del subproceso mientras corre (hasta >100% con varios cores).",
        },
        "metrics_glossary": {
            "video_duration_sec": "Duración del clip en segundos (metadatos del vídeo).",
            "video_duration_vlm_application": (
                "Tiempo total en segundos del pipeline para ese clip (detección + consultas VLM + salida); "
                "equivale a wall_clock_processing_sec. Comparar con video_duration_sec."
            ),
            "derived_ranking_score": (
                "wall_clock_processing_sec / video_duration_sec_metadata; menor indica menos tiempo "
                "de pipeline respecto a la duración del clip."
            ),
            "wall_clock_processing_sec": (
                "Tiempo total del proceso (detección, consultas VLM cuando aplica, escritura de salida)."
            ),
            "median_seconds_over_realtime": (
                "Mediana de (wall − duración vídeo): cuántos segundos por encima del 'tiempo real' del clip."
            ),
            "vlm_latency_mean_sec": (
                "Media de latencias en vlm_calls: por cada consulta al clasificador/VLM, no por frame de vídeo "
                "(el stride puede hacer que solo algunos frames disparen inferencia)."
            ),
            "comparison_best_profile": (
                "Una fila por approach con su mejor perfil (misma métrica de ranking que phase1); "
                "ordenado por menor tiempo total de pipeline (carrera a ojo)."
            ),
        },
        "how_to_read": (
            "Para el mismo clip: median_wall_clock_processing_sec es el tiempo total del pipeline "
            "(YOLO + VLM + escritura). Si median_video_duration_sec es 10, un valor 10.05 significa ~5 ms "
            "por encima del tiempo real del vídeo. "
            "median_vlm_latency_ms_per_query es la media de latencia por llamada al clasificador (consulta VLM); "
            "no es por frame de vídeo, sino por cada inferencia registrada en vlm_calls. "
            "comparison_best_profile ordena approaches por menor wall (mejor perfil por approach según ranking)."
        ),
        "note": (
            "derived_ranking_score menor = mejor (wall_pipeline / duracion_video). "
            "success exige exit 0, metrics.json presente y métricas de pipeline."
        ),
    }


def phase1_models_coverage_report(
    run_results: list[dict[str, Any]],
    catalog_approaches: list[Any],
    *,
    phase1_plan_total: int,
    only_run_index: int | None,
) -> dict[str, Any]:
    """
    Por approach: cuántos runs OK vs fallidos; lista modelos totalmente OK, parciales o todos mal.
    """
    script_by_id: dict[str, str] = {}
    expected_ids: list[str] = []
    for a in catalog_approaches or []:
        if not isinstance(a, dict):
            continue
        aid = str(a.get("id") or "").strip()
        if not aid or aid.startswith("_"):
            continue
        expected_ids.append(aid)
        script_by_id[aid] = str(a.get("script") or "")

    grouped: dict[str, dict[str, int]] = {}
    for r in run_results:
        aid = str(r.get("approach_id") or "")
        if not aid:
            continue
        if aid not in grouped:
            grouped[aid] = {"runs": 0, "successes": 0, "failures": 0}
        grouped[aid]["runs"] += 1
        if r.get("success"):
            grouped[aid]["successes"] += 1
        else:
            grouped[aid]["failures"] += 1

    by_app: dict[str, Any] = {}
    full_success: list[str] = []
    partial: list[str] = []
    all_failed: list[str] = []

    for aid in sorted(grouped.keys()):
        g = grouped[aid]
        rc, sc, fc = int(g["runs"]), int(g["successes"]), int(g["failures"])
        if fc == 0 and sc > 0:
            st = "ok"
            full_success.append(aid)
        elif sc == 0 and fc > 0:
            st = "all_failed"
            all_failed.append(aid)
        elif sc > 0 and fc > 0:
            st = "partial"
            partial.append(aid)
        else:
            st = "unknown"
        by_app[aid] = {
            "script": script_by_id.get(aid, ""),
            "runs": rc,
            "successes": sc,
            "failures": fc,
            "status": st,
        }

    executed_ids = set(grouped.keys())
    missing = [e for e in expected_ids if e not in executed_ids]

    return {
        "by_approach": by_app,
        "approaches_full_success": sorted(full_success),
        "approaches_partial_failure": sorted(partial),
        "approaches_all_runs_failed": sorted(all_failed),
        "approaches_not_executed_this_phase": sorted(missing),
        "phase1_plan_total_runs": phase1_plan_total,
        "phase1_executed_run_records": len(run_results),
        "note_only_run_index": (
            f"Solo se ejecutó el run de plan #{only_run_index}; "
            "los demás approaches no corren en esta sesión."
            if only_run_index
            else ""
        ),
    }


def print_phase1_models_coverage(cov: dict[str, Any]) -> None:
    """Salida legible tras fase 1."""
    sep = "=" * 72
    print(sep)
    print("Fase 1 — resumen por modelo (approach)")
    print(sep)
    ok = cov.get("approaches_full_success") or []
    part = cov.get("approaches_partial_failure") or []
    bad = cov.get("approaches_all_runs_failed") or []
    skip = cov.get("approaches_not_executed_this_phase") or []
    by_app = cov.get("by_approach") or {}

    def _line(label: str, ids: list[str]) -> None:
        if not ids:
            print(f"  · {label}: (ninguno)")
            return
        parts: list[str] = []
        for aid in ids:
            ba = by_app.get(aid) if isinstance(by_app, dict) else None
            if isinstance(ba, dict) and ba.get("runs"):
                parts.append(f"{aid} ({ba.get('successes')}/{ba.get('runs')} OK)")
            else:
                parts.append(aid)
        print(f"  · {label}: {', '.join(parts)}")

    _line("Todos los runs OK", list(ok))
    _line("Algún run fallido (parcial)", list(part))
    _line("Todos los runs fallidos", list(bad))
    note_idx = str(cov.get("note_only_run_index") or "").strip()
    if note_idx:
        print(f"  · Nota: {note_idx}")
        if skip:
            print(
                "  · No ejecutados en esta sesión (esperado con --only-run-index): "
                f"{', '.join(skip)}"
            )
    elif skip:
        print(
            "  · Sin runs en esta fase (matriz/catálogo): "
            f"{', '.join(skip)}"
        )

    total_exec = int(cov.get("phase1_executed_run_records") or 0)
    plan_total = int(cov.get("phase1_plan_total_runs") or 0)
    print(f"  · Registros de run en esta fase: {total_exec} (plan fase 1: {plan_total} combinaciones).")
    if ok and not part and not bad and not skip:
        print("  · Resultado: todos los modelos ejecutados han completado sin fallos.")
    elif not ok and not part and bad and note_idx:
        pass
    elif bad or part:
        print(
            "  · Revisa stderr/metrics en phase1_runs/<run_id>/ por approach con fallos "
            "(phase1_summary.json → runs[] / campaign_summary)."
        )
    print(sep)


def rank_approaches_for_phase2(
    run_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Por cada (approach, profile) calcula mediana de derived_ranking_score en los videos.
    Elige el mejor perfil por approach; rankea approaches; devuelve top 2 metadatos.
    """
    # group[(approach, profile)] -> scores
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in run_results:
        if not r.get("success"):
            continue
        sc = r.get("derived_ranking_score")
        if sc is None:
            continue
        grouped[(r["approach_id"], r["profile_id"])].append(float(sc))

    best_per_approach: dict[str, tuple[str, float]] = {}
    for (aid, pid), scores in grouped.items():
        med = median(scores)
        if med is None:
            continue
        prev = best_per_approach.get(aid)
        if prev is None or med < prev[1]:
            best_per_approach[aid] = (pid, med)

    ranked = sorted(
        [{"approach_id": a, "profile_id": p, "median_ranking_score": m} for a, (p, m) in best_per_approach.items()],
        key=lambda x: x["median_ranking_score"],
    )

    top2 = ranked[:2]
    return ranked, top2


def build_merged_for_phase2(
    catalog: dict[str, Any], approach_id: str, profile_id: str
) -> dict[str, Any]:
    defaults = _strip_meta(catalog.get("defaults") or {})
    profiles = catalog.get("profiles") or {}
    prof = _strip_meta(profiles.get(profile_id) or {})
    return _deep_merge(defaults, prof)


def phase2_parallel_batch(
    catalog: dict[str, Any],
    campaign_dir: Path,
    ranked_entries: list[dict[str, Any]],
    videos: list[Path],
    workers: int,
    monitor_psutil: bool,
    *,
    stream_log: tuple[TextIO, threading.Lock] | None = None,
) -> dict[str, Any]:
    """Ejecuta cada entrada (approach + mejor perfil): todos los videos a la vez por modelo."""
    results_by_model: list[dict[str, Any]] = []
    for rank, entry in enumerate(ranked_entries, start=1):
        aid = entry["approach_id"]
        pid = entry["profile_id"]
        ap_obj = next((x for x in catalog["approaches"] if x["id"] == aid), None)
        if ap_obj is None:
            continue
        script = str(ap_obj["script"])
        merged_base = build_merged_for_phase2(catalog, aid, pid)
        if ap_obj.get("vlm_model"):
            merged_base["vlm_model"] = ap_obj["vlm_model"]

        phase_dir = campaign_dir / "phase2_parallel" / f"rank{rank}_{aid}_{pid}"
        phase_dir.mkdir(parents=True, exist_ok=True)

        def _one(vp: Path) -> dict[str, Any]:
            rd = phase_dir / _sanitize_id(vp.stem)
            spec = RunSpec(
                approach_id=aid,
                script=script,
                profile_id=pid,
                video_path=vp,
                merged_args=dict(merged_base),
                vlm_model_override=ap_obj.get("vlm_model"),
            )
            lbl = f"[fase2 rank{rank} {aid}/{pid} video={vp.name}]"
            return run_single_experiment(
                spec,
                rd,
                monitor_psutil=monitor_psutil,
                stream_log=stream_log,
                run_label=lbl,
            )

        batch_t0 = time.perf_counter()
        parallel_out: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(videos))) as ex:
            futs = {ex.submit(_one, vp): vp for vp in videos}
            for fu in as_completed(futs):
                parallel_out.append(fu.result())
        batch_wall = time.perf_counter() - batch_t0

        successes = [x for x in parallel_out if x.get("success")]
        scores = [float(x["derived_ranking_score"]) for x in successes if x.get("derived_ranking_score") is not None]

        res_rows_m: list[dict[str, Any]] = []
        per_video_res: list[dict[str, Any]] = []
        for r in parallel_out:
            rd = r.get("run_dir")
            if not rd:
                continue
            row = extract_resource_row_from_run_dir(str(rd))
            row["video_stem"] = Path(str(r.get("video_path", ""))).stem
            res_rows_m.append(row)
            per_video_res.append(
                {
                    "video_stem": row["video_stem"],
                    "host_peak_rss_mib": row.get("host_peak_rss_mib"),
                    "torch_cuda_peak_allocated_mib": row.get("torch_cuda_peak_allocated_mib"),
                    "nvidia_gpu_utilization_mean_percent": row.get("nvidia_gpu_utilization_mean_percent"),
                    "nvidia_gpu_memory_used_mean_mib": row.get("nvidia_gpu_memory_used_mean_mib"),
                }
            )
        resource_by_model = aggregate_resource_rows(res_rows_m, include_per_run=False)
        resource_by_model["per_video"] = per_video_res

        results_by_model.append(
            {
                "rank": rank,
                "approach_id": aid,
                "profile_id": pid,
                "script": script,
                "batch_wall_clock_sec": round(batch_wall, 6),
                "videos_parallel": len(videos),
                "runs": parallel_out,
                "median_derived_ranking_score": median(scores),
                "resource_summary": resource_by_model,
                "note": "batch_wall_clock_sec es hasta que terminan todas las corridas en paralelo (duracion del lote).",
            }
        )

    return {
        "generated_utc": _utc_iso(),
        "phase": 2,
        "description": (
            "Cada approach listado procesa todos los videos en paralelo; los modelos se ejecutan uno tras otro."
        ),
        "resource_fields_glossary": {
            "host_peak_rss_mib": "Pico RSS del subproceso (psutil) en MiB, por run (cada video).",
            "subprocess_wall_sec": "Pared del subproceso; en agregados, min/median/max entre videos del lote.",
            "torch_cuda_peak_allocated_mib": "Pico asignado PyTorch en GPU para ese run (metrics.json).",
            "nvidia_gpu_utilization_mean_percent": "Media muestreos nvidia-smi en el tramo con vídeo (puede mezclar con otros clientes de la GPU).",
            "nvidia_gpu_memory_used_mean_mib": "Memoria de GPU usada (nvidia-smi, dispositivo entero en el muestreo).",
            "est_max_parallel_streams_by_memory_naive": "Heurística VRAM_total / pico PyTorch del run; no es límite operativo real.",
        },
        "models": results_by_model,
    }


def _nvidia_gpu_mem_used_mib() -> int | None:
    try:
        p = subprocess_run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if p.returncode == 0 and p.stdout.strip():
            return int(float(p.stdout.strip().splitlines()[0].strip()))
    except (OSError, ValueError, FileNotFoundError):
        pass
    return None


def _mem_monitor_loop(stop: threading.Event, out: dict[str, Any]) -> None:
    if psutil is not None:
        try:
            psutil.cpu_percent(interval=0.05)
        except (OSError, RuntimeError, AttributeError):
            pass
    while not stop.wait(0.2):
        if psutil is not None:
            try:
                u = int(psutil.virtual_memory().used)
                out["host_used_max"] = max(int(out.get("host_used_max", 0)), u)
                cp = float(psutil.cpu_percent(interval=None))
                out["cpu_percent_max"] = max(float(out.get("cpu_percent_max", 0.0)), cp)
            except (OSError, RuntimeError, AttributeError):
                break
        nvm = _nvidia_gpu_mem_used_mib()
        if nvm is not None:
            if out.get("nv_mem_max_mib") is None:
                out["nv_mem_max_mib"] = nvm
            else:
                out["nv_mem_max_mib"] = max(int(out["nv_mem_max_mib"]), nvm)


def _stderr_all_runs(outs: list[dict[str, Any]]) -> str:
    s = []
    for r in outs:
        p = Path(str(r.get("run_dir", ""))) / "experiment_summary.json"
        if p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    ed = json.load(f)
                t = (ed.get("envelope", {}).get("stderr_tail", "")) or ""
            except (OSError, json.JSONDecodeError):
                t = ""
            s.append(t)
    return "\n".join(s).lower()


def _phase3_post_batch_hygiene(pause_sec: float, aggressive: bool = False) -> None:
    """Entre dosis: GC, liberar caches host/GPU (best-effort) y pausa."""
    gc.collect()
    if aggressive:
        gc.collect()
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        except Exception:
            pass
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if aggressive and hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if aggressive and hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    if pause_sec > 0:
        time.sleep(pause_sec)


def _parse_phase3_steps_str(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        t = part.strip()
        if t.isdigit() and int(t) > 0:
            out.append(int(t))
    return out


def _parse_csv_ids(s: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in (s or "").split(","):
        t = part.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _phase3_execute_n_batch(
    N: int,
    step_dir: Path,
    approach_id: str,
    profile_id: str,
    ap_obj: dict[str, Any],
    script: str,
    merged_base: dict[str, Any],
    video_path: Path,
    monitor_psutil: bool,
    stream_log: tuple[TextIO, threading.Lock] | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    """Un lote con N subprocesos. Devuelve (step_entry, all_success, oom_suspect)."""
    step_dir.mkdir(parents=True, exist_ok=True)
    peaks: dict[str, Any] = {"host_used_max": 0, "nv_mem_max_mib": None, "cpu_percent_max": 0.0}
    stop_m = threading.Event()
    t_mon = threading.Thread(target=_mem_monitor_loop, args=(stop_m, peaks), daemon=True)
    t_mon.start()
    t0 = time.perf_counter()

    def _one(i: int) -> dict[str, Any]:
        rd = step_dir / f"run_{i:02d}"
        sp = RunSpec(
                approach_id=approach_id,
                script=script,
                profile_id=profile_id,
                video_path=video_path,
                merged_args=dict(merged_base),
                vlm_model_override=ap_obj.get("vlm_model"),
            )
        return run_single_experiment(
            sp,
            rd,
            monitor_psutil,
            stream_log=stream_log,
            run_label=f"[fase3 N={N} run={i}]",
        )

    with ThreadPoolExecutor(max_workers=N) as ex:
        outs: list[dict[str, Any]] = list(ex.map(_one, list(range(N))))
    stop_m.set()
    t_mon.join(timeout=2.0)
    batch_wall = time.perf_counter() - t0
    all_ok = all(x.get("success") for x in outs)
    combined_err = _stderr_all_runs(outs)
    oom = "out of memory" in combined_err or "cuda out of memory" in combined_err

    walls: list[float] = []
    for r in outs:
        w = r.get("wall_clock_processing_sec")
        if w is not None:
            walls.append(float(w))
    sw: list[float] = []
    rss_sum_vals: list[float] = []
    for r in outs:
        rd0 = str(r.get("run_dir", ""))
        if rd0:
            row0 = extract_resource_row_from_run_dir(rd0)
            s0 = row0.get("subprocess_wall_sec")
            if s0 is not None:
                sw.append(float(s0))
            hpr = row0.get("host_peak_rss_mib")
            if hpr is not None:
                rss_sum_vals.append(float(hpr))

    step_entry: dict[str, Any] = {
        "n_parallel": N,
        "all_success": all_ok,
        "any_oom_suspect": oom,
        "batch_wall_clock_sec": round(batch_wall, 6),
        "mean_pipeline_wall_sec": (round(sum(walls) / len(walls), 6) if walls else None),
        "max_pipeline_wall_sec": (round(max(walls), 6) if walls else None),
        "min_pipeline_wall_sec": (round(min(walls), 6) if walls else None),
        "mean_subprocess_wall_sec": (round(sum(sw) / len(sw), 6) if sw else None),
        "peak_host_memory_used_bytes_during_batch": int(peaks["host_used_max"])
        if int(peaks.get("host_used_max", 0) or 0) > 0
        else None,
        "peak_host_cpu_percent_during_batch": round(float(peaks["cpu_percent_max"]), 2)
        if psutil is not None
        else None,
        "peak_nvidia_gpu_mem_used_mib_smi": int(peaks["nv_mem_max_mib"])
        if peaks.get("nv_mem_max_mib") is not None
        else None,
        "sum_host_peak_rss_mib_from_runs": (round(sum(rss_sum_vals), 2) if rss_sum_vals else None),
        "runs": [
            {
                "run_dir": r.get("run_dir"),
                "success": r.get("success"),
                "wall_clock_processing_sec": r.get("wall_clock_processing_sec"),
            }
            for r in outs
        ],
    }
    return step_entry, all_ok, oom


def run_phase3_parallel_sweep(
    catalog: dict[str, Any],
    campaign_dir: Path,
    approach_id: str,
    profile_id: str,
    video_path: Path,
    n_steps: list[int] | None,
    monitor_psutil: bool,
    *,
    sweep_mode: str = "adaptive",
    start_n: int = 1,
    max_parallel: int = 64,
    pause_sec: float = 2.0,
    min_free_ram_mib: int = 512,
    aggressive_cleanup: bool = False,
    stream_log: tuple[TextIO, threading.Lock] | None = None,
    sweep_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Fase 3: N subprocesos = N flujos. Modo `fixed`: N recorre n_steps. Modo `adaptive`: dobla N
    (desde start_n) hasta fallo, tope o cap; luego búsqueda binaria en (ultimo_ok, primer_fallo) para
    afinar el máximo N. Entre lotes: GC, empty_cache CUDA y pausa.
    """
    ap_obj = next((x for x in catalog.get("approaches", []) if x.get("id") == approach_id), None)
    if not ap_obj:
        raise RuntimeError(f"approach {approach_id!r} no en catalogo")
    script = str(ap_obj["script"])
    merged_base = build_merged_for_phase2(catalog, approach_id, profile_id)
    if ap_obj.get("vlm_model"):
        merged_base["vlm_model"] = ap_obj["vlm_model"]
    merged_base["stride"] = PHASE34_PIPELINE_STRIDE

    if sweep_mode not in ("adaptive", "fixed"):
        raise RuntimeError("sweep_mode debe ser 'adaptive' o 'fixed'")
    if sweep_mode == "fixed":
        n_sorted = sorted({int(n) for n in (n_steps or []) if n > 0})
        if not n_sorted:
            raise RuntimeError("modo fixed: n_steps vacia o no valida (enteros > 0).")
    else:
        n_sorted = []
    st0 = max(1, int(start_n))
    cap = max(1, int(max_parallel))

    sweep_root = Path(sweep_dir) if sweep_dir is not None else (campaign_dir / "phase3_parallel_sweep")
    sweep_root.mkdir(parents=True, exist_ok=True)

    step_results: list[dict[str, Any]] = []
    n_tried_order: list[int] = []
    last_fully_ok_n: int = 0
    stop_sweep: str | None = None
    adaptive_meta: dict[str, Any] = {
        "start_n": st0,
        "max_parallel_cap": cap,
        "exponential_last_ok": None,
        "exponential_first_fail": None,
        "binary_refinement_used": False,
    }

    def _low_ram_block(N: int) -> bool:
        if psutil is None or int(min_free_ram_mib) <= 0:
            return False
        av = int(psutil.virtual_memory().available)
        if av < int(min_free_ram_mib) * 1024 * 1024:
            step_results.append(
                {
                    "n_parallel": N,
                    "skipped": True,
                    "skip_reason": (
                        f"host_available_mib aprox: {av / 1024 / 1024:.0f} < {min_free_ram_mib} "
                        "(cota min_free_ram_mib)"
                    ),
                }
            )
            return True
        return False

    def _append_step(
        N: int,
        step_entry: dict[str, Any],
        all_ok: bool,
        oom: bool,
        *,
        on_fail: str = "incomplete_runs",
        set_stop: bool = True,
    ) -> None:
        nonlocal last_fully_ok_n, stop_sweep
        step_results.append(step_entry)
        n_tried_order.append(N)
        if all_ok and not oom:
            last_fully_ok_n = N
        if set_stop and (oom or not all_ok):
            stop_sweep = "cuda_or_host_oom" if oom else on_fail
        _phase3_post_batch_hygiene(pause_sec, aggressive=aggressive_cleanup)

    if sweep_mode == "fixed":
        for N in n_sorted:
            if stop_sweep is not None:
                break
            if _low_ram_block(N):
                stop_sweep = "low_host_ram"
                break
            sdir = sweep_root / f"n_parallel_{N:03d}"
            st, all_ok, oom = _phase3_execute_n_batch(
                N,
                sdir,
                approach_id,
                profile_id,
                ap_obj,
                script,
                merged_base,
                video_path,
                monitor_psutil,
                stream_log,
            )
            _append_step(N, st, all_ok, oom, set_stop=True, on_fail="incomplete_runs")
            if oom or not all_ok:
                break
    else:
        # Fase A: exponencial (start, 2*…, cap) hasta un fallo; Fase B: búsqueda en (últimoOK, fail)
        tried: set[int] = set()
        first_fail_n: int | None = None
        cur = min(st0, cap)
        if cur < 1:
            cur = 1

        while first_fail_n is None and stop_sweep is None and cur <= cap:
            N = cur
            if N in tried:
                break
            if _low_ram_block(N):
                stop_sweep = "low_host_ram"
                first_fail_n = N
                break
            sdir = sweep_root / f"n_parallel_{N:03d}"
            st, all_ok, oom = _phase3_execute_n_batch(
                N,
                sdir,
                approach_id,
                profile_id,
                ap_obj,
                script,
                merged_base,
                video_path,
                monitor_psutil,
                stream_log,
            )
            tried.add(N)
            if oom or not all_ok:
                first_fail_n = N
                _append_step(N, st, all_ok, oom, set_stop=False, on_fail="incomplete_runs")
                break
            _append_step(N, st, all_ok, oom, set_stop=False, on_fail="incomplete_runs")
            if N >= cap:
                stop_sweep = "hit_max_parallel_cap"
                break
            nxt = min(N * 2, cap) if N >= 1 else N + 1
            if nxt <= N:
                break
            cur = nxt

        lo_bracket = int(last_fully_ok_n)
        adaptive_meta["exponential_last_ok"] = int(last_fully_ok_n)
        adaptive_meta["exponential_first_fail"] = int(first_fail_n) if first_fail_n is not None else None

        cands_bin: list[int] = []
        if first_fail_n is not None and int(first_fail_n) > lo_bracket + 1 and stop_sweep is None:
            cands_bin = [c for c in range(lo_bracket + 1, int(first_fail_n)) if c not in tried]
        if cands_bin:
            adaptive_meta["binary_refinement_used"] = True
            l_i, r_i = 0, len(cands_bin) - 1
            best_ok = lo_bracket
            while l_i <= r_i and stop_sweep is None:
                mid_i = (l_i + r_i) // 2
                m = cands_bin[mid_i]
                if _low_ram_block(m):
                    stop_sweep = "low_host_ram"
                    break
                sdir2 = sweep_root / f"n_parallel_{m:03d}"
                st2, oka, oom2 = _phase3_execute_n_batch(
                    m,
                    sdir2,
                    approach_id,
                    profile_id,
                    ap_obj,
                    script,
                    merged_base,
                    video_path,
                    monitor_psutil,
                    stream_log,
                )
                tried.add(m)
                if oom2 or not oka:
                    r_i = mid_i - 1
                else:
                    best_ok = m
                    l_i = mid_i + 1
                _append_step(m, st2, oka, oom2, set_stop=False, on_fail="incomplete_runs")
            if stop_sweep is None:
                last_fully_ok_n = best_ok

        if stop_sweep is None and first_fail_n is not None:
            for se in reversed(step_results):
                if se.get("n_parallel") == first_fail_n and not se.get("skipped"):
                    if se.get("any_oom_suspect"):
                        stop_sweep = "cuda_or_host_oom"
                    else:
                        stop_sweep = "incomplete_runs"
                    break

    if stop_sweep is None and sweep_mode == "adaptive" and not n_tried_order:
        stop_sweep = "no_trials"

    return {
        "generated_utc": _utc_iso(),
        "phase": 3,
        "file_role": (
            "Resumen global fase3 (único JSON a pasar al intérprete). "
            "Mira: sweep_ended_because, last_fully_successful_n_parallel, steps[] (N, tiempos, picos), "
            "n_steps_tried, adaptive_search. Cada sub-run: metrics.json + experiment_summary.json "
            "bajo phase3_parallel_sweep/n_parallel_XXX/run_YY/ (mismo criterio que fase1/2: sin vídeo de salida)."
        ),
        "no_output_video": True,
        "description": (
            "Barrido: mismo vídeo, mismo approach. N subprocesos = N 'flujos' aislados. "
            "Modo adaptativo: exponencial + búsqueda binaria. Entre lotes: gc + empty_cache + pausa."
        ),
        "sweep_output_dir": str(sweep_root.resolve()),
        "sweep_mode": sweep_mode,
        "approach_id": approach_id,
        "profile_id": profile_id,
        "pipeline_stride": PHASE34_PIPELINE_STRIDE,
        "pipeline_stride_note": (
            "Fase 3/4 fuerzan este stride en CLI; el perfil del catálogo puede tener otro valor para fase 1/2."
        ),
        "video_path": str(video_path.resolve()),
        "n_steps_tried": n_tried_order if sweep_mode == "adaptive" else n_sorted,
        "adaptive_search": adaptive_meta if sweep_mode == "adaptive" else None,
        "sweep_ended_because": stop_sweep,
        "last_fully_successful_n_parallel": last_fully_ok_n,
        "steps": step_results,
        "recommendation_note": (
            "ultimo N con all_success (sin OOM) es last_fully_successful_n_parallel. "
            "Cada 'flujo' = proceso Python separado. Modo adaptativo: el tope se refina con búsqueda binaria "
            "entre el último lote bueno y el primero que falla (O/Mem). "
            "Para servicio 1-proceso multi-RTSP, el tope operativo suele ser distinto. "
            "Los runs de orquestación no usan --save; wall_clock de fase3 es comparable a fase1/2 (sin encod. a disco)."
        ),
        "resource_fields_glossary": {
            "batch_wall_clock_sec": "Hasta el último subproceso (max straggler).",
            "mean_pipeline_wall_sec": "Media de wall_clock_processing_sec entre los N (mismo mp4 en cada run).",
            "peak_host_memory_used_bytes_during_batch": "Máximo de psutil virtual_memory().used mientras dura el lote (con más procesos, host total).",
            "peak_nvidia_gpu_mem_used_mib_smi": "Máximo muestreo nvidia-smi memory.used (GPU 0) durante el lote; todo dispositivo.",
        },
    }


def _resolve_phase3_video_path(
    args: Any,
    catalog: dict[str, Any],
) -> Path | None:
    vids2 = [Path(v).expanduser() for v in (catalog.get("videos") or []) if Path(v).expanduser().exists()]
    if args.phase3_video and Path(str(args.phase3_video)).expanduser().is_file():
        return Path(str(args.phase3_video)).expanduser().resolve()
    idx0 = 0
    if catalog.get("experiment_strategy") == "single_video_screening":
        idx0 = int(catalog.get("screening_video_index", 0) or 0)
    if idx0 < len(vids2):
        return vids2[idx0]
    return vids2[0] if vids2 else None


def _phase3_run_write(
    catalog: dict[str, Any],
    campaign_dir: Path,
    video_path: Path,
    approach_id: str,
    profile_id: str,
    args: Any,
    monitor_psutil: bool,
    stream_log: tuple[TextIO, threading.Lock] | None = None,
) -> dict[str, Any] | None:
    """Lanza fase3 y escribe `phase3_parallel_sweep_summary.json`. Devuelve el dict o None si no aplica."""
    if not video_path or not video_path.is_file():
        return None
    sm = (getattr(args, "phase3_sweep_mode", "adaptive") or "adaptive").strip().lower()
    if sm not in ("adaptive", "fixed"):
        sm = "adaptive"
    n_steps: list[int] | None
    if sm == "fixed":
        n_steps = _parse_phase3_steps_str(str(args.phase3_steps or ""))
        if not n_steps:
            print(
                "[phase3] Modo fixed: no hay N válidos en --phase3-steps. Omito fase 3.",
                file=sys.stderr,
            )
            return None
    else:
        n_steps = None
    t0 = time.perf_counter()
    p3 = run_phase3_parallel_sweep(
        catalog,
        campaign_dir,
        approach_id,
        profile_id,
        video_path,
        n_steps,
        monitor_psutil,
        sweep_mode=sm,
        start_n=int(getattr(args, "phase3_start_n", 1) or 1),
        max_parallel=int(getattr(args, "phase3_max_parallel", 64) or 64),
        pause_sec=float(args.phase3_pause_sec),
        min_free_ram_mib=int(args.phase3_min_free_ram_mib),
        aggressive_cleanup=bool(getattr(args, "phase3_aggressive_cleanup", False)),
        stream_log=stream_log,
    )
    p3["phase3_orchestrator_wall_sec"] = round(time.perf_counter() - t0, 3)
    p3p = campaign_dir / "phase3_parallel_sweep_summary.json"
    with open(p3p, "w", encoding="utf-8") as jf:
        json.dump(p3, jf, indent=2, ensure_ascii=False)
    print(f"[ok] Fase 3: {p3p}")
    return p3


def _resolve_phase4_video_path(args: Any, catalog: dict[str, Any]) -> Path | None:
    pv = getattr(args, "phase4_video", None)
    if pv and Path(str(pv)).expanduser().is_file():
        return Path(str(pv)).expanduser().resolve()
    return _resolve_phase3_video_path(args, catalog)


def _resolve_phase4_profile_id(args: Any, campaign_dir: Path) -> str | None:
    p = (getattr(args, "phase4_profile", None) or "").strip()
    if p:
        return p
    p1 = campaign_dir / "phase1_summary.json"
    if p1.is_file():
        try:
            with open(p1, encoding="utf-8") as f:
                d = json.load(f)
            rows = d.get("ranking_median_by_approach_best_profile") or []
            if rows and rows[0].get("profile_id"):
                return str(rows[0]["profile_id"])
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _metrics_path_n1_first_ok(sweep_inner: dict[str, Any]) -> Path | None:
    """metrics.json del primer lote N=1 con todos los runs OK."""
    for step in sweep_inner.get("steps") or []:
        if step.get("skipped"):
            continue
        if int(step.get("n_parallel") or 0) != 1:
            continue
        if not step.get("all_success"):
            continue
        runs = step.get("runs") or []
        if not runs:
            continue
        rd = runs[0].get("run_dir")
        if not rd:
            continue
        mp = Path(str(rd)) / "metrics.json"
        if mp.is_file():
            return mp
    return None


def _metrics_paths_for_n_ok_runs(sweep_inner: dict[str, Any], n_parallel: int) -> list[Path]:
    """metrics.json de runs OK para un paso N concreto."""
    out: list[Path] = []
    for step in sweep_inner.get("steps") or []:
        if step.get("skipped"):
            continue
        if int(step.get("n_parallel") or 0) != int(n_parallel):
            continue
        if not step.get("all_success"):
            continue
        for r in step.get("runs") or []:
            if not r.get("success"):
                continue
            rd = r.get("run_dir")
            if not rd:
                continue
            mp = Path(str(rd)) / "metrics.json"
            if mp.is_file():
                out.append(mp)
        break
    return out


def _latency_and_pipeline_from_metrics(metrics_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Estadísticas vlm_calls + tiempos pipeline para contestar ~cuánto son 10s de vídeo."""
    try:
        with open(metrics_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return (
            {"error": str(e), "source_metrics_json": str(metrics_path)},
            {},
        )


def _aggregate_latency_pipeline_from_metrics_paths(paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Agrega latencias/pipeline de varios metrics.json (p. ej. runs de N máximo)."""
    lat_rows: list[dict[str, Any]] = []
    pipe_rows: list[dict[str, Any]] = []
    for p in paths:
        lat, pipe = _latency_and_pipeline_from_metrics(p)
        if not lat.get("error"):
            lat_rows.append(lat)
        if not pipe.get("error"):
            pipe_rows.append(pipe)
    if not lat_rows and not pipe_rows:
        return (
            {"note": "Sin metrics.json válidos para agregar.", "runs_sampled": 0},
            {"note": "Sin metrics.json válidos para agregar.", "runs_sampled": 0},
        )
    lat_med = _median_optional([x.get("median_sec") for x in lat_rows])
    lat_mean = _mean_optional([x.get("mean_sec") for x in lat_rows])
    lat_ms = (round(float(lat_med) * 1000.0, 3) if lat_med is not None else None)
    pipe_wall_per_10s = _median_optional([x.get("derived_wall_clock_per_10s_video_sec") for x in pipe_rows])
    pipe_wall = _median_optional([x.get("wall_clock_processing_sec") for x in pipe_rows])
    return (
        {
            "runs_sampled": len(lat_rows),
            "median_sec_per_vlm_call": round(float(lat_med), 6) if lat_med is not None else None,
            "mean_sec_per_vlm_call": round(float(lat_mean), 6) if lat_mean is not None else None,
            "median_ms_per_vlm_call": lat_ms,
        },
        {
            "runs_sampled": len(pipe_rows),
            "median_wall_clock_processing_sec": round(float(pipe_wall), 6) if pipe_wall is not None else None,
            "median_wall_clock_per_10s_video_sec": (
                round(float(pipe_wall_per_10s), 6) if pipe_wall_per_10s is not None else None
            ),
        },
    )
    report = enrich_pipeline_report(raw)
    calls = report.get("vlm_calls") or []
    lat = [float(c["latency_sec"]) for c in calls if c.get("latency_sec") is not None]
    latency_block: dict[str, Any] = {
        "source_metrics_json": str(metrics_path.resolve()),
        "note": "min/mediana/max sobre vlm_calls[].latency_sec del run N=1 (un proceso aislado).",
        "count": len(lat),
        "min_sec": round(min(lat), 6) if lat else None,
        "median_sec": round(statistics.median(lat), 6) if lat else None,
        "max_sec": round(max(lat), 6) if lat else None,
        "vlm_latency_mean_sec": report.get("vlm_latency_mean_sec"),
        "derived_vlm_mean_latency_ms": report.get("derived_vlm_mean_latency_ms"),
    }
    wall = report.get("wall_clock_processing_sec")
    dur = report.get("video_duration_sec") or report.get("video_duration_sec_metadata")
    pipe: dict[str, Any] = {
        "wall_clock_processing_sec": wall,
        "video_duration_sec": float(dur) if dur is not None else None,
        "derived_wall_clock_per_10s_video_sec": None,
        "derived_ranking_score": report.get("derived_ranking_score"),
    }
    if wall is not None and dur is not None and float(dur) > 0:
        pipe["derived_wall_clock_per_10s_video_sec"] = round(float(wall) / float(dur) * 10.0, 6)
    return latency_block, pipe


def _median_optional(xs: list[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def _phase4_capacity_cloud_metrics(p_inner: dict[str, Any]) -> dict[str, Any]:
    """
    Métricas extra para dimensionar cloud / coste: VRAM y RSS por proceso, curva N,
    estabilidad del barrido, proxies de throughput y arranque.
    """
    steps_raw = p_inner.get("steps") or []
    steps = [s for s in steps_raw if not s.get("skipped")]
    n_last = int(p_inner.get("last_fully_successful_n_parallel") or 0)

    out: dict[str, Any] = {
        "sweep_stability": {
            "steps_total": len(steps_raw),
            "steps_non_skipped": len(steps),
            "steps_all_success": sum(1 for s in steps if s.get("all_success")),
            "steps_any_failure": sum(1 for s in steps if not s.get("all_success")),
            "sweep_ended_because": p_inner.get("sweep_ended_because"),
        },
        "parallel_scaling_steps": [],
        "single_stream_process_resources": None,
        "at_last_successful_n_parallel": None,
        "cold_start_proxy_sec": None,
        "throughput_estimates": None,
        "video_duration_sec_from_n1_metrics": None,
    }

    mp_n1 = _metrics_path_n1_first_ok(p_inner)
    dur_clip: float | None = None
    est_naive_parallel_mem: int | None = None
    if mp_n1 and mp_n1.is_file():
        try:
            with open(mp_n1, encoding="utf-8") as f:
                raw_m = json.load(f)
            em = enrich_pipeline_report(raw_m)
            d0 = em.get("video_duration_sec") or em.get("video_duration_sec_metadata")
            if d0 is not None:
                dur_clip = float(d0)
                out["video_duration_sec_from_n1_metrics"] = round(dur_clip, 6)
            if raw_m.get("est_max_parallel_streams_by_memory_naive") is not None:
                est_naive_parallel_mem = int(raw_m["est_max_parallel_streams_by_memory_naive"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    for s in steps:
        nn = int(s.get("n_parallel") or 0)
        bw = s.get("batch_wall_clock_sec")
        mw = s.get("mean_pipeline_wall_sec")
        entry_sc: dict[str, Any] = {
            "n_parallel": nn,
            "all_success": bool(s.get("all_success")),
            "any_oom_suspect": bool(s.get("any_oom_suspect")),
            "batch_wall_clock_sec": bw,
            "mean_pipeline_wall_sec": mw,
            "mean_subprocess_wall_sec": s.get("mean_subprocess_wall_sec"),
            "peak_host_memory_used_bytes_during_batch": s.get("peak_host_memory_used_bytes_during_batch"),
            "peak_host_cpu_percent_during_batch": s.get("peak_host_cpu_percent_during_batch"),
            "peak_nvidia_gpu_mem_used_mib_smi": s.get("peak_nvidia_gpu_mem_used_mib_smi"),
            "sum_host_peak_rss_mib_from_runs": s.get("sum_host_peak_rss_mib_from_runs"),
        }
        if (
            dur_clip is not None
            and bw is not None
            and float(bw) > 0
            and nn > 0
            and s.get("all_success")
        ):
            # Streams completados en paralelo por segundo de reloj del lote (orden de magnitud).
            entry_sc["estimated_full_clip_streams_per_wall_clock_sec"] = round(
                float(nn) / float(bw), 6
            )
            entry_sc["estimated_video_seconds_processed_per_wall_clock_sec"] = round(
                float(nn) * float(dur_clip) / float(bw), 6
            )
        out["parallel_scaling_steps"].append(entry_sc)

    steps_ps = out["parallel_scaling_steps"]
    nv_list = [x.get("peak_nvidia_gpu_mem_used_mib_smi") for x in steps_ps if x.get("peak_nvidia_gpu_mem_used_mib_smi") is not None]
    hb_list = [x.get("peak_host_memory_used_bytes_during_batch") for x in steps_ps if x.get("peak_host_memory_used_bytes_during_batch") is not None]
    cpu_list = [x.get("peak_host_cpu_percent_during_batch") for x in steps_ps if x.get("peak_host_cpu_percent_during_batch") is not None]
    out["sweep_maxima_over_all_steps"] = {
        "note": (
            "Maximos entre todos los pasos del barrido (incluye lotes fallidos por OOM); "
            "VRAM = nvidia-smi memory.used del dispositivo; RAM host = psutil.virtual_memory().used del sistema; "
            "CPU = maximo de psutil.cpu_percent() (uso CPU global) durante el muestreo del lote."
        ),
        "peak_nvidia_gpu_mem_used_mib_smi_max": int(max(nv_list)) if nv_list else None,
        "peak_host_memory_used_bytes_max": int(max(hb_list)) if hb_list else None,
        "peak_host_memory_used_gib_max": round(float(max(hb_list)) / (1024.0**3), 4) if hb_list else None,
        "peak_host_cpu_percent_max": round(float(max(cpu_list)), 4) if cpu_list else None,
    }

    n1_step = next(
        (x for x in steps if int(x.get("n_parallel") or 0) == 1 and x.get("all_success")),
        None,
    )
    if n1_step:
        rows_r: list[dict[str, Any]] = []
        cold_deltas: list[float] = []
        naive_par: list[int] = []
        for r in n1_step.get("runs") or []:
            rd = r.get("run_dir")
            wc = r.get("wall_clock_processing_sec")
            if not rd:
                continue
            row = extract_resource_row_from_run_dir(str(rd))
            rows_r.append(row)
            try:
                mp_one = Path(str(rd)) / "metrics.json"
                if mp_one.is_file():
                    with open(mp_one, encoding="utf-8") as mf:
                        mj = json.load(mf)
                    if mj.get("est_max_parallel_streams_by_memory_naive") is not None:
                        naive_par.append(int(mj["est_max_parallel_streams_by_memory_naive"]))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
            if wc is not None and row.get("subprocess_wall_sec") is not None:
                cold_deltas.append(float(row["subprocess_wall_sec"]) - float(wc))

        if rows_r:
            out["single_stream_process_resources"] = {
                "median_torch_cuda_peak_allocated_mib": _median_optional(
                    [x.get("torch_cuda_peak_allocated_mib") for x in rows_r]
                ),
                "median_host_peak_rss_mib": _median_optional([x.get("host_peak_rss_mib") for x in rows_r]),
                "median_nvidia_gpu_utilization_mean_percent": _median_optional(
                    [x.get("nvidia_gpu_utilization_mean_percent") for x in rows_r]
                ),
                "median_nvidia_gpu_memory_used_mean_mib": _median_optional(
                    [x.get("nvidia_gpu_memory_used_mean_mib") for x in rows_r]
                ),
                "median_subprocess_wall_sec": _median_optional([x.get("subprocess_wall_sec") for x in rows_r]),
                "runs_sampled": len(rows_r),
                "est_max_parallel_streams_by_memory_naive_from_metrics": (
                    int(statistics.median(naive_par)) if naive_par else None
                ),
            }
        if cold_deltas:
            out["cold_start_proxy_sec"] = {
                "note": "subprocess_wall_sec − wall_clock_processing_sec por run en lote N=1 (carga modelo + overhead).",
                "median_subprocess_minus_pipeline_sec": round(float(statistics.median(cold_deltas)), 6),
                "max_subprocess_minus_pipeline_sec": round(max(cold_deltas), 6),
            }

    max_step = next(
        (
            x
            for x in steps
            if int(x.get("n_parallel") or 0) == n_last and x.get("all_success") and n_last > 0
        ),
        None,
    )
    if max_step and n_last > 0:
        rss_sum = max_step.get("sum_host_peak_rss_mib_from_runs")
        rss_per = round(float(rss_sum) / float(n_last), 4) if rss_sum is not None else None
        rows_mx: list[dict[str, Any]] = []
        for r in max_step.get("runs") or []:
            rd = r.get("run_dir")
            if rd and r.get("success"):
                rows_mx.append(extract_resource_row_from_run_dir(str(rd)))
        median_torch = _median_optional([x.get("torch_cuda_peak_allocated_mib") for x in rows_mx])
        median_util = _median_optional([x.get("nvidia_gpu_utilization_mean_percent") for x in rows_mx])
        median_gpu_mem_used = _median_optional([x.get("nvidia_gpu_memory_used_mean_mib") for x in rows_mx])
        gpu_tot = None
        for x in rows_mx:
            if x.get("nvidia_gpu_memory_total_mib_sample"):
                gpu_tot = float(x["nvidia_gpu_memory_total_mib_sample"])
                break
        headroom = None
        if gpu_tot and median_gpu_mem_used is not None:
            headroom = round(max(0.0, (gpu_tot - median_gpu_mem_used) / gpu_tot), 4)
        out["at_last_successful_n_parallel"] = {
            "n_parallel": n_last,
            "est_host_rss_mib_per_stream_sum_over_n": rss_per,
            "median_torch_cuda_peak_allocated_mib_per_process": median_torch,
            "median_nvidia_gpu_utilization_mean_percent_while_run": median_util,
            "median_nvidia_gpu_memory_used_mean_mib_while_run": median_gpu_mem_used,
            "nvidia_gpu_memory_total_mib_sample_if_present": gpu_tot,
            "est_gpu_utilization_headroom_vs_device_mean": headroom,
            "note_torch": (
                "Pico PyTorch por proceso (~VRAM de ese stream); no suma linealmente al usar varios procesos "
                "en una sola GPU."
            ),
            "runs_sampled": len(rows_mx),
        }

    n1_sc = next(
        (
            x
            for x in out["parallel_scaling_steps"]
            if int(x.get("n_parallel") or 0) == 1 and x.get("all_success")
        ),
        None,
    )
    nmax_sc = next(
        (
            x
            for x in out["parallel_scaling_steps"]
            if int(x.get("n_parallel") or 0) == n_last and x.get("all_success") and n_last >= 1
        ),
        None,
    )
    tp: dict[str, Any] = {}
    if n1_sc and nmax_sc and n_last > 1:
        bw1 = n1_sc.get("batch_wall_clock_sec")
        bwm = nmax_sc.get("batch_wall_clock_sec")
        sp1 = n1_sc.get("estimated_full_clip_streams_per_wall_clock_sec")
        spm = nmax_sc.get("estimated_full_clip_streams_per_wall_clock_sec")
        if sp1 and spm and float(sp1) > 0:
            tp["ratio_streams_per_sec_n_max_vs_n1"] = round(float(spm) / float(sp1), 6)
        mw1 = n1_sc.get("mean_pipeline_wall_sec")
        mwm = nmax_sc.get("mean_pipeline_wall_sec")
        if mw1 and mwm and float(mw1) > 0:
            tp["ratio_mean_pipeline_wall_n_max_vs_n1"] = round(float(mwm) / float(mw1), 6)
            tp["note_ratio_pipeline"] = (
                "Si >>1: mucha contienda al paralelizar (cada stream más lento). Si ~1: escalado eficiente en tiempo por stream."
            )
    if dur_clip and nmax_sc:
        bw = nmax_sc.get("batch_wall_clock_sec")
        if bw and float(bw) > 0 and n_last > 0:
            # Igual que segundos de vídeo por segundo de reloj del lote; equivale a horas/h en magnitud.
            tp["estimated_video_hours_per_gpu_wall_hour_at_n_max"] = round(
                float(n_last) * float(dur_clip) / float(bw),
                6,
            )
    if est_naive_parallel_mem is not None:
        tp["est_max_parallel_streams_by_memory_naive_snapshot"] = est_naive_parallel_mem
        tp["note_naive_parallel_mem"] = (
            "Heurística VRAM_total/torch_peak un solo proceso; contrastar con last_fully_successful_n_parallel."
        )
    if tp:
        out["throughput_estimates"] = tp

    return out


def run_phase4_multi_model_sweep(
    catalog: dict[str, Any],
    campaign_dir: Path,
    video_path: Path,
    profile_id: str,
    args: Any,
    monitor_psutil: bool,
    stream_log: tuple[TextIO, threading.Lock] | None = None,
) -> dict[str, Any]:
    """
    Para cada approach del catálogo: mismo barrido que fase 3 (subcarpeta phase4_parallel_sweep/<id>/).
    """
    sm = (getattr(args, "phase3_sweep_mode", "adaptive") or "adaptive").strip().lower()
    if sm not in ("adaptive", "fixed"):
        sm = "adaptive"
    n_steps: list[int] | None
    if sm == "fixed":
        n_steps = _parse_phase3_steps_str(str(args.phase3_steps or ""))
        if not n_steps:
            raise RuntimeError("fase 4 modo fixed: --phase3-steps sin enteros válidos.")
    else:
        n_steps = None

    approaches = catalog.get("approaches") or []
    only_ids_raw = str(getattr(args, "phase4_approaches", "") or "").strip()
    only_ids = {x.strip() for x in only_ids_raw.split(",") if x.strip()}
    models: list[dict[str, Any]] = []

    for ap_entry in approaches:
        aid = str(ap_entry.get("id") or "").strip()
        if not aid:
            continue
        if only_ids and aid not in only_ids:
            continue
        script_n = str(ap_entry.get("script") or "")
        sweep_sub = campaign_dir / "phase4_parallel_sweep" / _sanitize_id(aid)
        entry: dict[str, Any] = {
            "approach_id": aid,
            "script": script_n,
            "vlm_model": ap_entry.get("vlm_model"),
            "profile_id": profile_id,
            "sweep_dir": str(sweep_sub.resolve()),
        }
        print(f"[phase4] --- {aid} ({script_n}) perfil={profile_id} ---", flush=True)
        try:
            p_inner = run_phase3_parallel_sweep(
                catalog,
                campaign_dir,
                aid,
                profile_id,
                video_path,
                n_steps,
                monitor_psutil,
                sweep_mode=sm,
                start_n=int(getattr(args, "phase3_start_n", 1) or 1),
                max_parallel=int(getattr(args, "phase3_max_parallel", 64) or 64),
                pause_sec=float(args.phase3_pause_sec),
                min_free_ram_mib=int(args.phase3_min_free_ram_mib),
                aggressive_cleanup=bool(getattr(args, "phase3_aggressive_cleanup", False)),
                stream_log=stream_log,
                sweep_dir=sweep_sub,
            )
            frag = sweep_sub / "parallel_sweep_detail.json"
            with open(frag, "w", encoding="utf-8") as jf:
                json.dump(p_inner, jf, indent=2, ensure_ascii=False)

            mp = _metrics_path_n1_first_ok(p_inner)
            if mp is not None:
                lat_b, pipe_b = _latency_and_pipeline_from_metrics(mp)
                entry["latency_vlm_calls"] = lat_b
                entry["pipeline_single_stream"] = pipe_b
            else:
                entry["latency_vlm_calls"] = {
                    "note": "No hubo lote N=1 con todos los runs OK o falta metrics.json.",
                }
                entry["pipeline_single_stream"] = {}

            entry["capacity_cloud_metrics"] = _phase4_capacity_cloud_metrics(p_inner)

            n_ok = int(p_inner.get("last_fully_successful_n_parallel") or 0)
            entry["last_fully_successful_n_parallel"] = n_ok
            if n_ok > 0:
                mpaths_nmax = _metrics_paths_for_n_ok_runs(p_inner, n_ok)
                lat_nmax, pipe_nmax = _aggregate_latency_pipeline_from_metrics_paths(mpaths_nmax)
                entry["latency_vlm_calls_at_n_max"] = lat_nmax
                entry["pipeline_at_n_max"] = pipe_nmax
            entry["sweep_ended_because"] = p_inner.get("sweep_ended_because")
            entry["n_steps_tried"] = p_inner.get("n_steps_tried")
            entry["parallel_sweep_detail_json"] = str(frag.resolve())
            if n_ok == 0:
                print(
                    f"[phase4] Omitido del resumen (last_fully_successful_n_parallel=0): {aid}",
                    flush=True,
                )
                continue
            models.append(entry)
        except Exception as exc:  # pragma: no cover
            models.append(
                {
                    "approach_id": aid,
                    "script": script_n,
                    "profile_id": profile_id,
                    "error": repr(exc),
                }
            )

        _phase3_post_batch_hygiene(
            max(0.0, float(getattr(args, "phase3_pause_sec", 2.0))),
            aggressive=bool(getattr(args, "phase3_aggressive_cleanup", False)),
        )

    models.sort(
        key=lambda m: int(m.get("last_fully_successful_n_parallel") or 0),
        reverse=True,
    )

    return {
        "generated_utc": _utc_iso(),
        "phase": 4,
        "phase4_cloud_assumptions_note": (
            "Ratios €/hora o coste por stream no están en este JSON; combine manualmente con "
            "capacity_cloud_metrics y precios de instancia GPU."
        ),
        "file_role": (
            "Resumen fase 4: barrido paralelo N por cada approach (mismo vídeo y perfil). "
            "Models[] incluye latency_vlm_calls, pipeline_single_stream y capacity_cloud_metrics "
            "(VRAM/RSS/CPU/picos del barrido completo). "
            "Orden: last_fully_successful_n_parallel descendente. "
            "No se listan approaches con last_fully_successful_n_parallel=0 (p. ej. modelo que falla incluso en N=1). "
            "derived_wall_clock_per_10s_video_sec ≈ tiempo de pipeline para 10 s de vídeo real "
            "(mismo cociente wall/duración × 10)."
        ),
        "video_path": str(video_path.resolve()),
        "profile_id": profile_id,
        "phase3_sweep_params_reused": {
            "phase3_sweep_mode": sm,
            "phase3_steps": str(getattr(args, "phase3_steps", "")),
            "phase3_start_n": int(getattr(args, "phase3_start_n", 1) or 1),
            "phase3_max_parallel": int(getattr(args, "phase3_max_parallel", 64) or 64),
            "phase3_pause_sec": float(args.phase3_pause_sec),
            "phase3_min_free_ram_mib": int(args.phase3_min_free_ram_mib),
            "pipeline_stride": PHASE34_PIPELINE_STRIDE,
        },
        "metrics_glossary": {
            "latency_vlm_calls.min_sec": "Mínimo latency_sec entre llamadas al clasificador en el run N=1.",
            "latency_vlm_calls.median_sec": "Mediana de latency_sec (vlm_calls).",
            "pipeline_single_stream.derived_wall_clock_per_10s_video_sec": (
                "wall_clock_processing_sec / video_duration_sec × 10; orden de magnitud para procesar 10 s de clip."
            ),
            "last_fully_successful_n_parallel": "Igual que fase 3: máximo N con todos los subprocesos OK sin OOM.",
            "capacity_cloud_metrics.parallel_scaling_steps": (
                "Por cada N: tiempos, RSS sumada por proceso, pico RAM host (psutil used), CPU% max en el lote, "
                "pico VRAM nvidia-smi; estimated_* si el paso fue OK."
            ),
            "capacity_cloud_metrics.sweep_maxima_over_all_steps": (
                "Maximos entre todos los pasos (incluye lotes fallidos): VRAM dispositivo, RAM host, CPU% "
                "(ver nota interna)."
            ),
            "capacity_cloud_metrics.single_stream_process_resources": (
                "Medianas sobre runs del lote N=1 OK: pico PyTorch VRAM, RSS host, utilización GPU y memoria nvidia-smi."
            ),
            "capacity_cloud_metrics.at_last_successful_n_parallel": (
                "En el último N con todos OK: RSS/n como proxy por stream, medianas por proceso en ese lote, cabeza GPU."
            ),
            "capacity_cloud_metrics.cold_start_proxy_sec": (
                "Diferencia subprocess_wall − pipeline wall por run N=1 (modelo + arranque vs trabajo útil)."
            ),
            "capacity_cloud_metrics.throughput_estimates": (
                "Ratios N_max vs N=1 en streams/s y tiempo pipeline; horas de vídeo por hora de reloj GPU en N_max; heurística naive VRAM."
            ),
            "capacity_cloud_metrics.sweep_stability": "Conteos de pasos OK/fallo del barrido y razón de parada.",
        },
        "models": models,
    }


def _phase4_run_write(
    catalog: dict[str, Any],
    campaign_dir: Path,
    video_path: Path,
    profile_id: str,
    args: Any,
    monitor_psutil: bool,
    stream_log: tuple[TextIO, threading.Lock] | None = None,
) -> dict[str, Any] | None:
    """Ejecuta fase 4 y escribe phase4_parallel_sweep_summary.json."""
    if not video_path or not video_path.is_file():
        print("[phase4] Omitido: vídeo inválido.", file=sys.stderr)
        return None
    if not profile_id:
        print("[phase4] Omitido: falta profile_id.", file=sys.stderr)
        return None
    t0 = time.perf_counter()
    p4 = run_phase4_multi_model_sweep(
        catalog,
        campaign_dir,
        video_path,
        profile_id,
        args,
        monitor_psutil,
        stream_log=stream_log,
    )
    p4["phase4_orchestrator_wall_sec"] = round(time.perf_counter() - t0, 3)
    out_p = campaign_dir / "phase4_parallel_sweep_summary.json"
    with open(out_p, "w", encoding="utf-8") as jf:
        json.dump(p4, jf, indent=2, ensure_ascii=False)
    print(f"[ok] Fase 4: {out_p}")
    return p4


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimentos approaches mano-objeto.")
    ap.set_defaults(phase2=True)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="JSON de catalogo.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Raiz output_results.")
    ap.add_argument("--resume", action="store_true", help="Saltar run si experiment_summary.json existe.")
    ap.add_argument("--dry-run", action="store_true", help="Solo imprimir numero de runs y ejemplos.")
    ap.add_argument("--limit", type=int, default=0, help="Max runs (0=todos). Ignorado si --only-run-index.")
    ap.add_argument(
        "--only-run-index",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Ejecutar solo el run N del plan de fase 1 (mismo número que en [run] N/total), p. ej. 63. "
            "Útil para reprobar un approach/perfil sin esperar los anteriores. Usa --no-phase2 si solo quieres ese run."
        ),
    )
    ap.add_argument("--no-psutil", action="store_true", help="No muestrear RSS/CPU.")
    ap.add_argument("--no-phase2", action="store_false", dest="phase2", help="No ejecutar fase 2 (paralelo sobre catalogo).")
    ap.add_argument(
        "--phase2-all-ranked",
        action="store_true",
        help=(
            "Fase 2: ejecutar todos los approaches con ranking tras fase 1 (mejor perfil cada uno × todos los videos), "
            "no solo el top-2."
        ),
    )
    ap.add_argument("--phase2-workers", type=int, default=4, help="Max hilos paralelos por modelo en fase 2.")
    ap.add_argument(
        "--device",
        default="",
        help=(
            "Override global de device para todas las fases (p. ej. cpu, cuda, cuda:0). "
            "Si se omite, usa defaults.device del catálogo."
        ),
    )
    ap.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="Ignora entradas de videos que no existen en disco (recomendado si el catalogo tiene placeholders).",
    )
    ap.add_argument(
        "--campaign",
        default="",
        help=(
            "Subcarpeta bajo --output-root para esta campaña (ej. abril_cpu). "
            "Separa phase1_runs/, phase1_summary.json y fase 2 para no machacar ejecuciones anteriores."
        ),
    )
    ap.add_argument(
        "--unique-campaign",
        action="store_true",
        help="Crea una subcarpeta automática run_YYYYMMDDTHHMMSS (UTC) como --campaign.",
    )
    ap.add_argument(
        "--no-session-log",
        action="store_true",
        help="No crear output_results/.../logs/orchestrator_*.log ni duplicar salida.",
    )
    ap.add_argument(
        "--only-phase3",
        action="store_true",
        help="Solo fase 3: barrido de N procesos en paralelo; requiere --campaign (carpeta existente).",
    )
    ap.add_argument(
        "--phase3-parallel-sweep",
        action="store_true",
        help="Al terminar fase1/2, lanza fase3 (mismo vídeo repetido, rank-1 o --phase3-approach).",
    )
    ap.add_argument(
        "--phase3-sweep-mode",
        default="adaptive",
        choices=("adaptive", "fixed"),
        help="adaptive: dobla N y refina (sin lista manual); fixed: recorre --phase3-steps.",
    )
    ap.add_argument(
        "--phase3-steps",
        default="3,6,8,10,15,20,30,40,50",
        help="Solo con --phase3-sweep-mode fixed: N separados por comas, orden creciente recomendado.",
    )
    ap.add_argument(
        "--phase3-start-n",
        type=int,
        default=1,
        metavar="N",
        help="Solo adaptativo: primer lote (>=1) antes de ir doblando N.",
    )
    ap.add_argument(
        "--phase3-max-parallel",
        type=int,
        default=64,
        metavar="N",
        help="Cota superior de N en el barrido adaptativo; evita crecer sin límite teórico.",
    )
    ap.add_argument(
        "--phase3-approach", default="", help="Approach; si vacío se usa el rank 1 (phase1 o phase2)."
    )
    ap.add_argument(
        "--phase3-profile", default="", help="Perfil; si vacío se usa el del rank 1 (phase1 o phase2)."
    )
    ap.add_argument(
        "--phase3-video", type=Path, default=None, help="Ruta a un mp4; sino, vídeo de screening o primero del catálogo."
    )
    ap.add_argument("--phase3-pause-sec", type=float, default=2.0, help="Pausa entre dosis (liberar carga suave).")
    ap.add_argument(
        "--phase3-aggressive-cleanup",
        action="store_true",
        help=(
            "Limpieza reforzada entre lotes/enfoques: gc extra + malloc_trim + CUDA empty_cache/ipc_collect "
            "(best-effort; no equivale a reiniciar el sistema)."
        ),
    )
    ap.add_argument(
        "--phase3-min-free-ram-mib",
        type=int,
        default=512,
        metavar="M",
        help="No lanzar un lote N si available RAM < M MiB (host). 0=desactivar comprobación.",
    )
    ap.add_argument(
        "--only-phase4",
        action="store_true",
        help="Solo fase 4: barrido N por cada approach del catálogo; requiere --campaign y vídeo válido.",
    )
    ap.add_argument(
        "--only-phase34",
        action="store_true",
        help=(
            "Solo fases 3 y 4 en cadena, sin fase 1/2. "
            "Fase 3 acepta varios IDs en --phase3-approach separados por coma; luego fase 4 usa ese mismo conjunto."
        ),
    )
    ap.add_argument(
        "--phase4-parallel-sweep",
        action="store_true",
        help="Tras fase 1/2/(3), ejecuta fase 4 (todos los modelos × mismo vídeo/perfil).",
    )
    ap.add_argument(
        "--phase4-profile",
        default="",
        help="Perfil catálogo para todos los approaches en fase 4; si vacío: primero del ranking en phase1_summary o stride_5.",
    )
    ap.add_argument(
        "--phase4-video",
        type=Path,
        default=None,
        help="Mp4 para fase 4; si vacío se usa --phase3-video o el vídeo de screening del catálogo.",
    )
    ap.add_argument(
        "--phase4-approaches",
        default="siglip_v2,siglip,clip,mobileclip",
        help=(
            "IDs de approach para fase 4 separados por coma. "
            "Por defecto limita a siglip_v2,siglip,clip,mobileclip."
        ),
    )
    args = ap.parse_args()

    only_modes = int(bool(getattr(args, "only_phase3", False))) + int(bool(getattr(args, "only_phase4", False))) + int(
        bool(getattr(args, "only_phase34", False))
    )
    if only_modes > 1:
        print("[error] Elige solo uno: --only-phase3, --only-phase4 o --only-phase34.", file=sys.stderr)
        raise SystemExit(2)

    catalog_path = args.catalog.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    campaign_name = (args.campaign or "").strip()
    if args.unique_campaign:
        campaign_name = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if campaign_name:
        campaign_name = _sanitize_id(campaign_name)
    campaign_dir = (output_root / campaign_name) if campaign_name else output_root

    catalog = load_catalog(catalog_path)
    device_override = str(getattr(args, "device", "") or "").strip()
    if device_override:
        catalog.setdefault("defaults", {})
        catalog["defaults"]["device"] = device_override
        print(f"[catalog] override defaults.device={device_override!r}")
    if args.skip_missing_videos:
        vids = catalog.get("videos") or []
        kept = [v for v in vids if Path(str(v)).expanduser().exists()]
        dropped = [v for v in vids if not Path(str(v)).expanduser().exists()]
        if dropped:
            print(f"[catalog] omitidos {len(dropped)} videos inexistentes: {dropped[:3]!r}...")
        catalog["videos"] = kept
        if not kept:
            raise RuntimeError("Tras --skip-missing-videos no queda ningun video en el catalogo.")

    if args.only_phase4:
        if not args.campaign.strip() and not args.unique_campaign:
            print(
                "[error] --only-phase4 requiere --campaign (carpeta existente bajo --output-root).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not campaign_dir.is_dir():
            print(f"[error] no existe la campaña: {campaign_dir}", file=sys.stderr)
            raise SystemExit(1)
        pid4 = _resolve_phase4_profile_id(args, campaign_dir) or "stride_5"
        if not (args.phase4_profile or "").strip():
            print(f"[phase4] --phase4-profile vacío; usando {pid4!r}.")
        vpick = _resolve_phase4_video_path(args, catalog)
        if vpick is None or not vpick.is_file():
            print("[error] Ningún vídeo válido para fase 4. Usa --phase4-video o --phase3-video.", file=sys.stderr)
            raise SystemExit(1)
        if (args.phase3_sweep_mode or "adaptive") == "fixed" and not _parse_phase3_steps_str(
            str(args.phase3_steps or "")
        ):
            print(
                "[error] --phase3-sweep-mode fixed requiere --phase3-steps con al menos un entero > 0.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        p4m = not args.no_psutil and psutil is not None
        _orig_stdout = sys.stdout
        session_log_tuple, _log_fp, _ = _setup_session_logging(campaign_dir, bool(args.no_session_log), _orig_stdout)
        try:
            if (
                _phase4_run_write(
                    catalog,
                    campaign_dir,
                    vpick,
                    pid4,
                    args,
                    p4m,
                    session_log_tuple,
                )
                is None
            ):
                raise SystemExit(1)
        finally:
            sys.stdout = _orig_stdout
            if _log_fp is not None:
                _log_fp.close()
        return

    if args.only_phase3 or args.only_phase34:
        if not args.campaign.strip() and not args.unique_campaign:
            print(
                "[error] --only-phase3/--only-phase34 requiere --campaign (carpeta existente bajo --output-root).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not campaign_dir.is_dir():
            print(f"[error] no existe la campaña: {campaign_dir}", file=sys.stderr)
            raise SystemExit(1)
        p2f = campaign_dir / "phase2_parallel_summary.json"
        p1f = campaign_dir / "phase1_summary.json"
        aid_list = _parse_csv_ids(str(args.phase3_approach or ""))
        aid0 = aid_list[0] if aid_list else None
        pid0 = (args.phase3_profile or "").strip() or None
        if aid0 is None or pid0 is None:
            if p2f.is_file():
                with open(p2f, encoding="utf-8") as f:
                    p2d = json.load(f)
                m0 = (p2d.get("models") or [None])[0] or {}
                aid0 = aid0 or m0.get("approach_id")
                pid0 = pid0 or m0.get("profile_id")
        if aid0 is None or pid0 is None:
            if p1f.is_file():
                with open(p1f, encoding="utf-8") as f:
                    p1d = json.load(f)
                rrow = (p1d.get("ranking_median_by_approach_best_profile") or [None])[0] or {}
                aid0 = aid0 or rrow.get("approach_id")
                pid0 = pid0 or rrow.get("profile_id")
        if not aid0 or not pid0:
            print(
                "[error] No se pudo leer approach/perfil. Usa --phase3-approach y --phase3-profile "
                "o deja phase1/phase2 summary en la campaña.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        vpick = _resolve_phase3_video_path(args, catalog)
        if vpick is None or not vpick.is_file():
            print("[error] Ningun video valido. Usa --phase3-video.", file=sys.stderr)
            raise SystemExit(1)
        if (args.phase3_sweep_mode or "adaptive") == "fixed" and not _parse_phase3_steps_str(
            str(args.phase3_steps or "")
        ):
            print(
                "[error] --phase3-sweep-mode fixed requiere --phase3-steps con al menos un entero > 0.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        p3m = not args.no_psutil and psutil is not None
        _orig_stdout = sys.stdout
        session_log_tuple, _log_fp, _ = _setup_session_logging(campaign_dir, bool(args.no_session_log), _orig_stdout)
        try:
            if not aid_list:
                aid_list = [str(aid0)]
            for aid in aid_list:
                print(f"[phase3] --- {aid} perfil={pid0} ---", flush=True)
                if _phase3_run_write(
                    catalog,
                    campaign_dir,
                    vpick,
                    str(aid),
                    str(pid0),
                    args,
                    p3m,
                    session_log_tuple,
                ) is None:
                    raise SystemExit(1)
            if args.only_phase34:
                if not (args.phase4_approaches or "").strip():
                    args.phase4_approaches = ",".join(aid_list)
                pid4 = (args.phase4_profile or "").strip() or str(pid0) or "stride_5"
                if _phase4_run_write(
                    catalog,
                    campaign_dir,
                    vpick,
                    pid4,
                    args,
                    p3m,
                    session_log_tuple,
                ) is None:
                    raise SystemExit(1)
        finally:
            sys.stdout = _orig_stdout
            if _log_fp is not None:
                _log_fp.close()
        return

    runs = expand_run_specs(catalog)
    phase1_plan_total = len(runs)
    if args.only_run_index:
        if args.only_run_index < 1:
            raise RuntimeError("--only-run-index debe ser >= 1 (coincide con [run] N/total).")
        if args.only_run_index > phase1_plan_total:
            raise RuntimeError(
                f"--only-run-index={args.only_run_index} fuera de rango: el plan tiene {phase1_plan_total} runs."
            )
        runs = [runs[args.only_run_index - 1]]
    elif args.limit and args.limit > 0:
        runs = runs[: args.limit]

    if args.dry_run:
        print(f"Catalogo: {catalog_path}")
        print(f"Carpeta campaña: {campaign_dir}")
        strat = catalog.get("experiment_strategy", "full_matrix")
        print(f"Estrategia: {strat}")
        if strat == "single_video_screening":
            print(
                f"  (fase 1 solo video índice {catalog.get('screening_video_index', 0)}; "
                "fase 2 = top-2 o --phase2-all-ranked × todos los videos del catalogo)"
            )
        print(f"Runs totales fase 1: {len(runs)}")
        if args.only_run_index:
            print(
                f"  (modo --only-run-index {args.only_run_index} de {phase1_plan_total}; "
                "fase 1 ejecutará 1 run)"
            )
        if runs:
            r0 = runs[0]
            print(f"Ejemplo: {r0.approach_id} / {r0.profile_id} / {r0.video_path.name}")
        return

    campaign_dir.mkdir(parents=True, exist_ok=True)

    _orig_stdout = sys.stdout
    _log_fp: TextIO | None = None
    _log_lock = threading.Lock()
    session_log_tuple: tuple[TextIO, threading.Lock] | None = None
    session_log_path: Path | None = None

    if not args.no_session_log:
        log_dir = campaign_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        session_log_path = log_dir / (
            "orchestrator_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".log"
        )
        _log_fp = open(session_log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _TeeIO(_orig_stdout, _log_fp)
        session_log_tuple = (_log_fp, _log_lock)
        print(f"[orchestrator] log de sesión: {session_log_path.resolve()}")
        print(f"[campaña] salida en: {campaign_dir}")
    else:
        print(f"[campaña] salida en: {campaign_dir}")

    orchestrator_started_utc: str | None = None
    t_orchestrator_0: float | None = None

    try:
        orchestrator_started_utc = _utc_iso()
        t_orchestrator_0 = time.perf_counter()
        phase1_dir = campaign_dir / "phase1_runs"
        phase1_dir.mkdir(parents=True, exist_ok=True)
        if args.only_run_index:
            print(
                f"[phase1] ejecutando solo el run de plan {args.only_run_index}/{phase1_plan_total} "
                "(--only-run-index; mismo run_id que campaña completa)."
            )

        monitor = not args.no_psutil and psutil is not None

        run_results: list[dict[str, Any]] = []
        t_phase1_0 = time.perf_counter()
        for i, spec in enumerate(runs, start=1):
            disp_i = args.only_run_index if args.only_run_index else i
            disp_tot = phase1_plan_total if args.only_run_index else len(runs)
            run_id = _sanitize_id(f"{spec.approach_id}__{spec.profile_id}__{spec.video_path.stem}")
            run_dir = phase1_dir / run_id
            summary_file = run_dir / "experiment_summary.json"
            if args.resume and summary_file.exists():
                try:
                    with open(summary_file, encoding="utf-8") as jf:
                        cached = json.load(jf)
                    env = cached.get("envelope") or {}
                    metrics_path_resume = run_dir / "metrics.json"
                    metrics_exists = metrics_path_resume.exists()
                    pl: dict[str, Any]
                    if metrics_exists:
                        try:
                            with open(metrics_path_resume, encoding="utf-8") as mf:
                                pl = enrich_pipeline_report(json.load(mf))
                        except (json.JSONDecodeError, OSError):
                            pl = enrich_pipeline_report(dict(cached.get("pipeline") or {}))
                    else:
                        pl = enrich_pipeline_report(dict(cached.get("pipeline") or {}))
                    success_resume = (
                        bool(env.get("ok"))
                        and metrics_exists
                        and pipeline_report_indicates_success(pl)
                    )
                    vid_out = run_dir / "output_video.mp4"
                    run_results.append(
                        {
                            "run_dir": str(run_dir),
                            "summary_json": str(summary_file),
                            "metrics_json": str(metrics_path_resume) if metrics_exists else "",
                            "video_out": str(vid_out) if vid_out.exists() else "",
                            "approach_id": spec.approach_id,
                            "profile_id": spec.profile_id,
                            "video_path": str(spec.video_path.resolve()),
                            "success": success_resume,
                            "derived_ranking_score": pl.get("derived_ranking_score"),
                            "wall_clock_processing_sec": pl.get("wall_clock_processing_sec"),
                            "video_duration_sec": pl.get("video_duration_sec")
                            if pl.get("video_duration_sec") is not None
                            else pl.get("video_duration_sec_metadata"),
                            "vlm_inference_count": pl.get("vlm_inference_count"),
                            "vlm_latency_mean_sec": pl.get("vlm_latency_mean_sec"),
                            "derived_vlm_mean_latency_ms": pl.get("derived_vlm_mean_latency_ms"),
                            "derived_seconds_over_realtime": pl.get("derived_seconds_over_realtime"),
                            "video_duration_vlm_application": pl.get("video_duration_vlm_application"),
                        }
                    )
                    print(f"[resume] {disp_i}/{disp_tot} {run_id}")
                    continue
                except (json.JSONDecodeError, OSError):
                    pass

            print(f"[run] {disp_i}/{disp_tot} {run_id}")
            lbl = f"[fase1 {disp_i}/{disp_tot} {run_id}]"
            res = run_single_experiment(
                spec,
                run_dir,
                monitor_psutil=monitor,
                stream_log=session_log_tuple,
                run_label=lbl,
            )
            run_results.append(res)

        t_phase1_1 = time.perf_counter()
        ranked, top2 = rank_approaches_for_phase2(run_results)
        campaign_summary = summarize_phase1_campaign(
            run_results,
            ranked,
            top2,
            phase1_wall_sec=t_phase1_1 - t_phase1_0,
        )
        phase1_models_cov = phase1_models_coverage_report(
            run_results,
            catalog.get("approaches") or [],
            phase1_plan_total=int(phase1_plan_total),
            only_run_index=int(args.only_run_index) if getattr(args, "only_run_index", None) else None,
        )

        phase1_summary = {
            "generated_utc": _utc_iso(),
            "file_role": (
                "Resumen global fase1 (JSON principal). Lee campaign_summary (recursos, medianas) y "
                "ranking_median_by_approach_best_profile. Detalle: runs[] o phase1_runs/<id>/{metrics,experiment_summary}.json. "
                "Orquestación sin --save (no output_video; solo pipeline + métricas)."
            ),
            "catalog_path": str(catalog_path),
            "output_root": str(output_root),
            "campaign_dir": str(campaign_dir),
            "campaign_name": campaign_name or None,
            "orchestrator_session_log": str(session_log_path.resolve())
            if session_log_path
            else None,
            "experiment_strategy": catalog.get("experiment_strategy", "full_matrix"),
            "screening_video_index": catalog.get("screening_video_index"),
            "phase1_screening_note": (
                "single_video_screening: fase 1 solo barrido approaches×perfiles sobre un clip; "
                "fase 2 valida los top modelos en todos los videos listados."
                if catalog.get("experiment_strategy") == "single_video_screening"
                else ""
            ),
            "total_runs": len(runs),
            "phase1_plan_total_runs": phase1_plan_total,
            "phase1_only_run_index": int(args.only_run_index) if args.only_run_index else None,
            "campaign_summary": campaign_summary,
            "phase1_models_coverage": phase1_models_cov,
            "runs": run_results,
            "ranking_median_by_approach_best_profile": ranked,
            "phase2_top2": top2,
            "metrics_glossary": {
                "video_duration_sec": "Duración del clip (s); mismo valor que video_duration_sec_metadata tras enriquecer.",
                "video_duration_vlm_application": "Tiempo total del pipeline para ese clip (s); igual que wall_clock_processing_sec; comparar con video_duration_sec.",
                "derived_ranking_score": "wall_clock_processing_sec / video_duration_sec_metadata (menor = menos tiempo de proceso por segundo de video).",
                "derived_vlm_compute_sum_sec": "Suma de latency_sec de todas las llamadas al clasificador (tiempo efectivo modelo).",
                "derived_non_vlm_overhead_sec": "wall_clock - suma VLM (pose, decodificacion, escritura, etc.).",
                "derived_wall_clock_per_video_second": "Igual que derived_ranking_score si el video se procesa entero.",
                "vlm_inference_count": "Numero de inferencias del clasificador en el clip.",
                "processing_fps_effective": "frames_processed / wall_clock (pipeline).",
                "realtime_factor_vs_video_fps": "video_fps / processing_fps_effective (<1 suele significar mas rapido que tiempo real).",
                "torch_cuda_peak_memory_allocated_mib": "Pico VRAM PyTorch si device=cuda.",
                "nvidia_gpu_utilization_mean_percent": "Media muestreos nvidia-smi durante el run (solo CUDA).",
                "host_peak_rss_bytes": "Pico RSS del proceso hijo (psutil, si disponible).",
                "campaign_summary": "Bloque resumido: conteos exito/fallo, tiempos orquestador fase 1, medianas/p90 globales y por approach.",
                "phase1_models_coverage": (
                    "Por approach: runs OK vs fallidos, listas approaches_full_success / partial / all_failed; "
                    "approaches_not_executed_this_phase si matriz no los incluyo o --only-run-index."
                ),
                "resource_summary": "Solo en runs con metrics completos: mediana de picos host/GPU; ver resource_fields_glossary en campaign_summary.",
            },
        }
        summary_path = campaign_dir / "phase1_summary.json"
        with open(summary_path, "w", encoding="utf-8") as jf:
            json.dump(phase1_summary, jf, indent=2, ensure_ascii=False)
        print(f"[ok] Resumen fase 1: {summary_path}")
        print_phase1_models_coverage(phase1_models_cov)

        do_phase2 = bool(args.phase2)
        phase2_entries = list(ranked) if args.phase2_all_ranked else top2
        sel_note = "todos los rankeados" if args.phase2_all_ranked else "top-2"
        if do_phase2 and len(phase2_entries) >= 1:
            videos = [Path(v).expanduser() for v in catalog.get("videos") or []]
            videos = [v for v in videos if v.exists()]
            if not videos:
                print("[phase2] Omitido: no hay videos existentes en el catalogo.")
            else:
                print(
                    f"[phase2] Modo: {sel_note} ({len(phase2_entries)} modelo(s)); "
                    f"{len(videos)} video(s) en paralelo por modelo."
                )
                t_p2_0 = time.perf_counter()
                p2 = phase2_parallel_batch(
                    catalog,
                    campaign_dir,
                    phase2_entries,
                    videos,
                    workers=min(args.phase2_workers, len(videos)),
                    monitor_psutil=monitor,
                    stream_log=session_log_tuple,
                )
                p2["file_role"] = (
                    "Resumen global fase2 (JSON principal). models[]: un lote por approach con todos los "
                    "videos en paralelo; resource_summary y runs[] en phase2_runs. Compara con phase1 vía approach_id. "
                    "Sin --save en orquestación (no output_video por run)."
                )
                p2["phase2_orchestrator_wall_sec"] = round(time.perf_counter() - t_p2_0, 3)
                p2["phase2_selection"] = "all_ranked" if args.phase2_all_ranked else "top2"
                p2["phase2_models_count"] = len(phase2_entries)
                if session_log_path:
                    p2["orchestrator_session_log"] = str(session_log_path.resolve())
                p2_path = campaign_dir / "phase2_parallel_summary.json"
                with open(p2_path, "w", encoding="utf-8") as jf:
                    json.dump(p2, jf, indent=2, ensure_ascii=False)
                print(f"[ok] Resumen fase 2: {p2_path}")
        elif do_phase2:
            print(
                "[phase2] No hay modelos en el ranking (ningún run con derived_ranking_score válido). "
                "Revisa fallos en fase 1 y campaign_summary en phase1_summary.json."
            )

        if args.phase3_parallel_sweep and bool(ranked) and ranked[0].get("approach_id"):
            v3b = _resolve_phase3_video_path(args, catalog)
            if (args.phase3_sweep_mode or "adaptive") == "fixed" and not _parse_phase3_steps_str(
                str(args.phase3_steps or "")
            ):
                print(
                    "[phase3] Modo fixed: omiso (--phase3-steps vacío o sin enteros). "
                    "Usa adaptivo o fija N en --phase3-steps.",
                    file=sys.stderr,
                )
            elif v3b and v3b.is_file():
                _phase3_run_write(
                    catalog,
                    campaign_dir,
                    v3b,
                    str((args.phase3_approach or "").strip() or str(ranked[0]["approach_id"])),
                    str((args.phase3_profile or "").strip() or str(ranked[0]["profile_id"])),
                    args,
                    monitor,
                    session_log_tuple,
                )

        if getattr(args, "phase4_parallel_sweep", False):
            v4b = _resolve_phase4_video_path(args, catalog)
            pid4 = (args.phase4_profile or "").strip() or (
                str(ranked[0]["profile_id"]) if ranked and ranked[0].get("profile_id") else ""
            )
            if not pid4:
                pid4 = _resolve_phase4_profile_id(args, campaign_dir) or "stride_5"
                print(f"[phase4] perfil desde defecto del catálogo/ranking: {pid4!r}")
            if (args.phase3_sweep_mode or "adaptive") == "fixed" and not _parse_phase3_steps_str(
                str(args.phase3_steps or "")
            ):
                print(
                    "[phase4] Modo fixed omitido (--phase3-steps vacío). No ejecuto fase 4.",
                    file=sys.stderr,
                )
            elif v4b and v4b.is_file():
                _phase4_run_write(
                    catalog,
                    campaign_dir,
                    v4b,
                    pid4,
                    args,
                    monitor,
                    session_log_tuple,
                )
            else:
                print("[phase4] Omitido: vídeo no válido.", file=sys.stderr)

        # Si hubo excepcion antes, este bloque no corre (no aparece FINALIZADO en el log).
        orch_end = _utc_iso()
        orch_wall = time.perf_counter() - t_orchestrator_0 if t_orchestrator_0 is not None else 0.0
        print("")
        print("========== FINALIZADO ==========")
        print(f"  inicio_utc:      {orchestrator_started_utc}")
        print(f"  fin_utc:         {orch_end}")
        print(f"  tiempo_total_s:  {orch_wall:.3f}")
        print("================================")

    finally:
        sys.stdout = _orig_stdout
        if _log_fp is not None:
            _log_fp.close()


if __name__ == "__main__":
    main()
