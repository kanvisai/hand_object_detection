#!/usr/bin/env python3
"""
Orquesta experimentos mano-objeto: recorre approaches (test_new_handobject_*.py) x perfiles x videos,
volcando JSON enriquecido + video bajo output_results/. Cada ejecución escribe también
output_results/<campaña>/logs/orchestrator_<UTC>.log (salida del orquestador + stdout/stderr de cada hijo).

Tras la fase 1, opcionalmente lanza fase 2: 2 mejores modelos, N videos en paralelo (cada modelo en su tanda).

Modo sin ventana: cada run pasa --save (video en disco); no se usa cv2.imshow. Entorno headless
(QT_QPA_PLATFORM=offscreen, MPLBACKEND=Agg) para evitar GUI accidental en segundo plano.
"""
from __future__ import annotations

import argparse
import json
import re
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
from subprocess import PIPE, Popen
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

    out = dict(report)
    out.update(derived)
    return out


def _headless_env(base: dict[str, str]) -> dict[str, str]:
    """Evita backends GUI (Qt/mpl) cuando se ejecuta sin pantalla / en batch."""
    env = dict(base)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLBACKEND", "Agg")
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
    merged["save"] = str(video_path.resolve())
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
        "ok": proc.returncode == 0,
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
        "success": bool(proc.returncode == 0 and report),
        "derived_ranking_score": report.get("derived_ranking_score"),
        "wall_clock_processing_sec": report.get("wall_clock_processing_sec"),
        "video_duration_sec": report.get("video_duration_sec_metadata"),
    }


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    m = n // 2
    return float(s[m]) if n % 2 else (s[m - 1] + s[m]) / 2.0


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
    top2: list[dict[str, Any]],
    videos: list[Path],
    workers: int,
    monitor_psutil: bool,
    *,
    stream_log: tuple[TextIO, threading.Lock] | None = None,
) -> dict[str, Any]:
    """Ejecuta cada modelo del top2: todos los videos a la vez (workers = len(videos))."""
    results_by_model: list[dict[str, Any]] = []
    for rank, entry in enumerate(top2, start=1):
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
                "note": "batch_wall_clock_sec es hasta que terminan todas las corridas en paralelo (duracion del lote).",
            }
        )

    return {
        "generated_utc": _utc_iso(),
        "phase": 2,
        "description": "Dos modelos top; cada uno procesa todos los videos en paralelo (secuencialmente entre modelos).",
        "models": results_by_model,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimentos approaches mano-objeto.")
    ap.set_defaults(phase2=True)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="JSON de catalogo.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Raiz output_results.")
    ap.add_argument("--resume", action="store_true", help="Saltar run si experiment_summary.json existe.")
    ap.add_argument("--dry-run", action="store_true", help="Solo imprimir numero de runs y ejemplos.")
    ap.add_argument("--limit", type=int, default=0, help="Max runs (0=todos).")
    ap.add_argument("--no-psutil", action="store_true", help="No muestrear RSS/CPU.")
    ap.add_argument("--no-phase2", action="store_false", dest="phase2", help="No ejecutar fase 2 (paralelo top-2).")
    ap.add_argument("--phase2-workers", type=int, default=4, help="Max hilos paralelos por modelo en fase 2.")
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
    args = ap.parse_args()

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
    if args.skip_missing_videos:
        vids = catalog.get("videos") or []
        kept = [v for v in vids if Path(str(v)).expanduser().exists()]
        dropped = [v for v in vids if not Path(str(v)).expanduser().exists()]
        if dropped:
            print(f"[catalog] omitidos {len(dropped)} videos inexistentes: {dropped[:3]!r}...")
        catalog["videos"] = kept
        if not kept:
            raise RuntimeError("Tras --skip-missing-videos no queda ningun video en el catalogo.")
    runs = expand_run_specs(catalog)
    if args.limit and args.limit > 0:
        runs = runs[: args.limit]

    if args.dry_run:
        print(f"Catalogo: {catalog_path}")
        print(f"Carpeta campaña: {campaign_dir}")
        strat = catalog.get("experiment_strategy", "full_matrix")
        print(f"Estrategia: {strat}")
        if strat == "single_video_screening":
            print(
                f"  (fase 1 solo video índice {catalog.get('screening_video_index', 0)}; "
                "fase 2 = top modelos × todos los videos del catalogo)"
            )
        print(f"Runs totales fase 1: {len(runs)}")
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

    try:
        phase1_dir = campaign_dir / "phase1_runs"
        phase1_dir.mkdir(parents=True, exist_ok=True)

        monitor = not args.no_psutil and psutil is not None

        run_results: list[dict[str, Any]] = []
        for i, spec in enumerate(runs, start=1):
            run_id = _sanitize_id(f"{spec.approach_id}__{spec.profile_id}__{spec.video_path.stem}")
            run_dir = phase1_dir / run_id
            summary_file = run_dir / "experiment_summary.json"
            if args.resume and summary_file.exists():
                try:
                    with open(summary_file, encoding="utf-8") as jf:
                        cached = json.load(jf)
                    env = cached.get("envelope") or {}
                    pl = cached.get("pipeline") or {}
                    run_results.append(
                        {
                            "run_dir": str(run_dir),
                            "summary_json": str(summary_file),
                            "approach_id": spec.approach_id,
                            "profile_id": spec.profile_id,
                            "video_path": str(spec.video_path.resolve()),
                            "success": bool(env.get("ok")),
                            "derived_ranking_score": pl.get("derived_ranking_score"),
                            "wall_clock_processing_sec": pl.get("wall_clock_processing_sec"),
                            "video_duration_sec": pl.get("video_duration_sec_metadata"),
                        }
                    )
                    print(f"[resume] {i}/{len(runs)} {run_id}")
                    continue
                except (json.JSONDecodeError, OSError):
                    pass

            print(f"[run] {i}/{len(runs)} {run_id}")
            lbl = f"[fase1 {i}/{len(runs)} {run_id}]"
            res = run_single_experiment(
                spec,
                run_dir,
                monitor_psutil=monitor,
                stream_log=session_log_tuple,
                run_label=lbl,
            )
            run_results.append(res)

        ranked, top2 = rank_approaches_for_phase2(run_results)

        phase1_summary = {
            "generated_utc": _utc_iso(),
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
            "runs": run_results,
            "ranking_median_by_approach_best_profile": ranked,
            "phase2_top2": top2,
            "metrics_glossary": {
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
            },
        }
        summary_path = campaign_dir / "phase1_summary.json"
        with open(summary_path, "w", encoding="utf-8") as jf:
            json.dump(phase1_summary, jf, indent=2, ensure_ascii=False)
        print(f"[ok] Resumen fase 1: {summary_path}")

        do_phase2 = bool(args.phase2)
        if do_phase2 and len(top2) >= 1:
            videos = [Path(v).expanduser() for v in catalog.get("videos") or []]
            videos = [v for v in videos if v.exists()]
            if not videos:
                print("[phase2] Omitido: no hay videos existentes en el catalogo.")
            else:
                p2 = phase2_parallel_batch(
                    catalog,
                    campaign_dir,
                    top2,
                    videos,
                    workers=min(args.phase2_workers, len(videos)),
                    monitor_psutil=monitor,
                    stream_log=session_log_tuple,
                )
                if session_log_path:
                    p2["orchestrator_session_log"] = str(session_log_path.resolve())
                p2_path = campaign_dir / "phase2_parallel_summary.json"
                with open(p2_path, "w", encoding="utf-8") as jf:
                    json.dump(p2, jf, indent=2, ensure_ascii=False)
                print(f"[ok] Resumen fase 2: {p2_path}")
        elif do_phase2:
            print("[phase2] No hay al menos 2 modelos rankeables (revisa runs fallidos).")

    finally:
        sys.stdout = _orig_stdout
        if _log_fp is not None:
            _log_fp.close()


if __name__ == "__main__":
    main()
