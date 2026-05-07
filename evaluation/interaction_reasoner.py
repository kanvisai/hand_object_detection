#!/usr/bin/env python3
"""
Lee JSON de `session_semantics.py` (un chunk) y produce un veredicto.

Puedes pasar **varios** `--input-json` (uno por chunk); se concatenan `frames` y se ordenan
por `(chunk, sample_idx)` para reconstruir la línea temporal entre chunks.

Con `--chunks-root-dir`, si el directorio contiene varias carpetas de chunk (chunk_001, …),
cada una con JSON por modelo (`*_siglip.json`, `*_openclip.json`, …), se ejecuta el razonador
por cada VLM y se imprime el resumen de acciones para comparar backends.

Reglas v1 (parametrizables): ventana de suavizado, persistencia de etiquetas, heurística objeto / depósito.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from retail_semantic_prompts import CANONICAL_LABELS, LABELS_ES  # noqa: E402

# Nombre del JSON por carpeta chunk (stem = nombre de la carpeta): stem + sufijo
_VLM_SUFFIX_BY_BACKEND: dict[str, str] = {
    "siglip": "_siglip.json",
    "openclip": "_openclip.json",
    "mobileclip": "_mobileclip.json",
    "semantics": "_semantics.json",
}
_BACKEND_DISPLAY_ORDER: tuple[str, ...] = ("siglip", "openclip", "mobileclip", "semantics")


def _natural_chunk_sort_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name)
    key: list[Any] = []
    for t in parts:
        if not t:
            continue
        if t.isdigit():
            key.append(int(t))
        else:
            key.append(t.lower())
    return key


def _discover_chunk_subdirs(root: Path) -> list[Path]:
    subs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(subs, key=_natural_chunk_sort_key)


def _collect_semantics_paths_per_backend(root: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    chunk_dirs = _discover_chunk_subdirs(root)
    per_backend: dict[str, list[Path]] = defaultdict(list)
    for cd in chunk_dirs:
        stem = cd.name
        for backend, tail in _VLM_SUFFIX_BY_BACKEND.items():
            candidate = cd / f"{stem}{tail}"
            if candidate.is_file():
                per_backend[backend].append(candidate)
    filtered = {k: v for k, v in per_backend.items() if v}
    return filtered, chunk_dirs


def _ordered_backend_keys(per_backend: dict[str, list[Path]]) -> list[str]:
    known = [b for b in _BACKEND_DISPLAY_ORDER if b in per_backend]
    rest = sorted(k for k in per_backend if k not in _BACKEND_DISPLAY_ORDER)
    return known + rest


def _load_semantics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("JSON raíz debe ser un objeto.")
    return data


def _sorted_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Orden estable: nombre de chunk, luego sample_idx."""

    def key(fr: dict[str, Any]) -> tuple[str, int]:
        ch = str(fr.get("chunk") or "")
        try:
            si = int(fr.get("sample_idx") or 0)
        except (TypeError, ValueError):
            si = 0
        return (ch, si)

    return sorted(frames, key=key)


def _smooth_labels(
    labels: list[str | None],
    window: int,
    *,
    treat_unknown_as_gap: bool,
) -> list[str | None]:
    if window <= 1:
        return list(labels)
    half = window // 2
    out: list[str | None] = []
    n = len(labels)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk_vals: list[str] = []
        for j in range(lo, hi):
            lab = labels[j]
            if lab is None:
                continue
            if treat_unknown_as_gap and lab == "unknown":
                continue
            chunk_vals.append(lab)
        if not chunk_vals:
            out.append(labels[i])
            continue
        try:
            mode = statistics.mode(chunk_vals)
        except statistics.StatisticsError:
            mode = max(set(chunk_vals), key=chunk_vals.count)
        out.append(mode)
    return out


def _max_run_length(seq: list[str | None], target: str) -> int:
    best = cur = 0
    for x in seq:
        if x == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _count_evaluable(frames: list[dict[str, Any]]) -> int:
    return sum(1 for f in frames if f.get("evaluable"))


_LABEL_ES_BY_CANON: dict[str, str] = {
    c: LABELS_ES[i] for i, c in enumerate(CANONICAL_LABELS) if i < len(LABELS_ES)
}


def _label_to_es(label: str | None) -> str:
    if label is None:
        return "sin clasificar"
    if label == "unknown":
        return "señal poco clara"
    return _LABEL_ES_BY_CANON.get(label, label)


def _extract_temporal_runs(
    ordered: list[dict[str, Any]],
    labels_smooth: list[str | None],
) -> list[dict[str, Any]]:
    """Segmentos consecutivos con la misma etiqueta suavizada (None se omite)."""
    n = len(ordered)
    runs: list[dict[str, Any]] = []
    i = 0
    while i < n:
        lab = labels_smooth[i]
        if lab is None:
            i += 1
            continue
        j = i
        while j < n and labels_smooth[j] == lab:
            j += 1
        slice_f = ordered[i:j]
        probs: list[float] = []
        for f in slice_f:
            if f.get("evaluable"):
                mp = f.get("max_prob")
                try:
                    probs.append(float(mp) if mp is not None else 0.0)
                except (TypeError, ValueError):
                    probs.append(0.0)
        n_ev = len(probs)
        mean_p = float(sum(probs) / n_ev) if n_ev else 0.0
        score = float(n_ev * mean_p)
        chunks_order: list[str] = []
        seen: set[str] = set()
        for f in slice_f:
            ch = str(f.get("chunk") or "").strip()
            if ch and ch not in seen:
                seen.add(ch)
                chunks_order.append(ch)
        primary = str(slice_f[0].get("chunk") or "").strip() if slice_f else ""
        runs.append(
            {
                "label": lab,
                "label_es": _label_to_es(lab),
                "start_index": i,
                "end_index": j - 1,
                "frame_count": j - i,
                "evaluable_count": n_ev,
                "mean_confidence": round(mean_p, 4),
                "weight_score": round(score, 4),
                "chunks": chunks_order,
                "primary_chunk": primary,
            }
        )
        i = j
    return runs


def _merge_adjacent_same_label(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not actions:
        return []
    out: list[dict[str, Any]] = [dict(actions[0])]
    out[0]["chunks"] = list(actions[0].get("chunks") or [])
    for a in actions[1:]:
        if a.get("label") == out[-1].get("label"):
            prev = out[-1]
            ne = int(prev["evaluable_count"]) + int(a["evaluable_count"])
            nc = int(prev["frame_count"]) + int(a["frame_count"])
            m = (
                float(prev["mean_confidence"]) * int(prev["evaluable_count"])
                + float(a["mean_confidence"]) * int(a["evaluable_count"])
            ) / max(1, ne)
            prev["evaluable_count"] = ne
            prev["frame_count"] = nc
            prev["mean_confidence"] = round(m, 4)
            prev["weight_score"] = round(ne * m, 4)
            prev["end_index"] = a["end_index"]
            for c in a.get("chunks") or []:
                if c not in prev["chunks"]:
                    prev["chunks"].append(c)
        else:
            x = dict(a)
            x["chunks"] = list(a.get("chunks") or [])
            out.append(x)
    return out


def _select_top_runs_per_chunk(
    runs: list[dict[str, Any]],
    ordered: list[dict[str, Any]],
    *,
    per_chunk_max: int,
) -> list[dict[str, Any]]:
    """Por cada chunk, hasta `per_chunk_max` runs con mayor weight_score (solo runs que empiezan en ese chunk)."""
    by_start: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        i0 = int(r["start_index"])
        ch = str(ordered[i0].get("chunk") or "").strip() if 0 <= i0 < len(ordered) else ""
        by_start[ch or "_"].append(r)
    selected: list[dict[str, Any]] = []
    for ch in sorted(k for k in by_start if k != "_"):
        rs = sorted(by_start[ch], key=lambda x: float(x["weight_score"]), reverse=True)[
            : max(0, per_chunk_max)
        ]
        selected.extend(rs)
    if "_" in by_start:
        rs = sorted(by_start["_"], key=lambda x: float(x["weight_score"]), reverse=True)[
            : max(0, per_chunk_max)
        ]
        selected.extend(rs)
    selected.sort(key=lambda x: int(x["start_index"]))
    return selected


def _trim_to_max_actions(actions: list[dict[str, Any]], max_actions: int) -> list[dict[str, Any]]:
    if len(actions) <= max_actions:
        return actions
    acts = sorted(actions, key=lambda a: int(a["start_index"]))
    while len(acts) > max_actions:
        worst = min(range(len(acts)), key=lambda i: float(acts[i]["weight_score"]))
        del acts[worst]
    acts.sort(key=lambda a: int(a["start_index"]))
    return acts


def build_session_summary_es(
    ordered: list[dict[str, Any]],
    labels_smooth: list[str | None],
    *,
    per_chunk_max: int,
    global_max: int,
    verdict: str | None = None,
) -> dict[str, Any]:
    """
    Resumen legible: prioriza runs con más frames y mayor confianza media; acota por chunk y globalmente.
    """
    all_runs = _extract_temporal_runs(ordered, labels_smooth)
    picked = _select_top_runs_per_chunk(all_runs, ordered, per_chunk_max=per_chunk_max)
    merged = _merge_adjacent_same_label(picked)
    final_actions = _trim_to_max_actions(merged, global_max)

    lines: list[str] = []
    for k, a in enumerate(final_actions, start=1):
        ch_s = ", ".join(a.get("chunks") or []) or "—"
        lines.append(
            f"{k}. {a['label_es']} — {a['evaluable_count']} frames eval., "
            f"conf. media {float(a['mean_confidence']):.2f} (chunks: {ch_s})"
        )
    one_liner = " → ".join(str(a["label_es"]) for a in final_actions) if final_actions else "(sin segmentos con etiqueta)"

    note = ""
    if verdict == "insufficient_data":
        note = "Pocos frames evaluables: el veredicto es débil; el resumen es orientativo."

    return {
        "one_liner_es": one_liner,
        "lines_es": lines,
        "actions": final_actions,
        "runs_detected": len(all_runs),
        "params": {"per_chunk_max": per_chunk_max, "global_max": global_max},
        "note": note,
    }


def merge_chunk_semantics_payloads(paths: list[Path]) -> dict[str, Any]:
    """Une varios JSON de chunk en un único payload (misma convención que un solo archivo)."""
    if not paths:
        raise RuntimeError("Se necesita al menos un JSON.")
    merged_frames: list[dict[str, Any]] = []
    base: dict[str, Any] | None = None
    sources: list[str] = []
    for p in paths:
        data = _load_semantics(p)
        sources.append(str(p.resolve()))
        if base is None:
            base = {k: v for k, v in data.items() if k != "frames"}
        frs = data.get("frames")
        if isinstance(frs, list):
            for item in frs:
                if isinstance(item, dict):
                    merged_frames.append(item)
    out = dict(base) if base else {}
    out["frames"] = merged_frames
    out["merged_chunk_semantics_sources"] = sources
    out["merge_note"] = "Varias salidas de session_semantics concatenadas; campo chunk en cada frame preserva origen."
    return out


def reason_interaction(
    payload: dict[str, Any],
    *,
    smooth_window: int,
    min_evaluable_frames: int,
    min_run_object: int,
    min_run_deposit: int,
    treat_unknown_as_gap: bool,
    summary_per_chunk_max: int = 2,
    summary_global_max: int = 5,
) -> dict[str, Any]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("payload.frames debe ser una lista.")
    ordered = _sorted_frames([f for f in frames if isinstance(f, dict)])
    labels_raw: list[str | None] = []
    for fr in ordered:
        if not fr.get("evaluable"):
            labels_raw.append(None)
            continue
        lab = fr.get("semantic_label")
        labels_raw.append(str(lab) if lab else None)

    labels_smooth = _smooth_labels(
        [x if x else None for x in labels_raw],
        smooth_window,
        treat_unknown_as_gap=treat_unknown_as_gap,
    )

    n_eval = _count_evaluable(ordered)
    run_obj = _max_run_length(labels_smooth, "object_in_hand")
    run_basket = _max_run_length(labels_smooth, "shopping_basket")
    run_cart = _max_run_length(labels_smooth, "shopping_cart")
    run_bag = _max_run_length(labels_smooth, "personal_bag_deposit")
    run_deposit = max(run_basket, run_cart, run_bag)

    evidence: dict[str, Any] = {
        "frames_total": len(ordered),
        "frames_evaluable": n_eval,
        "max_run_object_in_hand": run_obj,
        "max_run_shopping_basket": run_basket,
        "max_run_shopping_cart": run_cart,
        "max_run_personal_bag_deposit": run_bag,
        "smooth_window": smooth_window,
    }

    if n_eval < min_evaluable_frames:
        conf = float(n_eval) / max(1, min_evaluable_frames)
        return {
            "verdict": "insufficient_data",
            "confidence": round(min(1.0, conf), 4),
            "evidence": evidence,
            "label_sequence_raw": labels_raw,
            "label_sequence_smoothed": labels_smooth,
            "reason": "few_evaluable_frames",
            "session_summary_es": build_session_summary_es(
                ordered,
                labels_smooth,
                per_chunk_max=summary_per_chunk_max,
                global_max=summary_global_max,
                verdict="insufficient_data",
            ),
        }

    # Heurística v1
    had_object = run_obj >= min_run_object
    had_deposit = run_deposit >= min_run_deposit

    if had_object and had_deposit:
        verdict = "likely_normal_purchase_or_deposit"
        conf = min(1.0, (run_obj / max(1, min_run_object)) * 0.5 + (run_deposit / max(1, min_run_deposit)) * 0.5)
    elif had_object and not had_deposit:
        verdict = "object_signal_without_container_signal"
        conf = min(1.0, run_obj / max(1, min_run_object))
    elif not had_object and had_deposit:
        verdict = "container_signal_without_object_signal"
        conf = min(1.0, run_deposit / max(1, min_run_deposit))
    else:
        verdict = "no_clear_interaction_pattern"
        conf = 0.4

    return {
        "verdict": verdict,
        "confidence": round(float(conf), 4),
        "evidence": evidence,
        "label_sequence_raw": labels_raw,
        "label_sequence_smoothed": labels_smooth,
        "reason": "heuristic_v1",
        "session_summary_es": build_session_summary_es(
            ordered,
            labels_smooth,
            per_chunk_max=summary_per_chunk_max,
            global_max=summary_global_max,
            verdict=None,
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Razonador sobre JSON de session_semantics (uno o más; p. ej. chunk_001_semantics.json).",
    )
    p.add_argument(
        "--chunk-dir",
        default="",
        help=(
            "Carpeta de un chunk: busca el JSON de session_semantics (nombre típico <carpeta>_semantics.json; "
            "también chunk_semantics.json o un único *_semantics.json). Sin archivo → mensaje y cómo generarlo."
        ),
    )
    p.add_argument(
        "--chunks-root-dir",
        default="",
        metavar="DIR",
        help=(
            "Carpeta padre con subcarpetas de chunk (chunk_001, …). En cada una se buscan "
            "<nombre>_siglip.json, _openclip.json, _mobileclip.json o _semantics.json; "
            "por cada VLM encontrado se fusionan los JSON en orden de carpeta y se muestra el resumen comparativo."
        ),
    )
    p.add_argument(
        "--input-json",
        nargs="*",
        default=[],
        help="Uno o más JSON generados por session_semantics.py. Si usas --chunk-dir, no pases esto.",
    )
    p.add_argument(
        "--output-json",
        default="",
        help="Si se indica, escribe el resultado aquí (si no, solo stdout).",
    )
    p.add_argument("--smooth-window", type=int, default=3, help="Ventana impar recomendada (>=1).")
    p.add_argument(
        "--min-evaluable-frames",
        type=int,
        default=4,
        help="Mínimo de frames evaluables para no declarar insufficient_data.",
    )
    p.add_argument(
        "--min-run-object",
        type=int,
        default=2,
        help="Frames seguidos (tras suavizado) mínimos para contar 'object_in_hand' persistente.",
    )
    p.add_argument(
        "--min-run-deposit",
        type=int,
        default=2,
        help="Run mínimo en shopping_basket o shopping_cart.",
    )
    p.add_argument(
        "--treat-unknown-as-gap",
        action="store_true",
        help="Al suavizar, ignorar etiquetas 'unknown' en la ventana.",
    )
    p.add_argument(
        "--summary-per-chunk-max",
        type=int,
        default=2,
        metavar="N",
        help="Resumen: como mucho N segmentos destacados por chunk (por peso frames×confianza).",
    )
    p.add_argument(
        "--summary-global-max",
        type=int,
        default=5,
        metavar="N",
        help="Resumen: como mucho N acciones en total tras fusionar y acotar.",
    )
    p.add_argument(
        "--quiet-summary",
        action="store_true",
        help="No imprimir el resumen legible en stderr (solo JSON en stdout / mensaje de fichero).",
    )
    return p


def _resolve_semantics_json_for_chunk(chunk_dir: Path) -> Path:
    """
    Busca JSON de session_semantics (orden de preferencia si hay varios nombres).
    """
    stem = chunk_dir.name or "chunk"
    candidates = [
        chunk_dir / f"{stem}_siglip.json",
        chunk_dir / f"{stem}_semantics.json",
        chunk_dir / f"{stem}_openclip.json",
        chunk_dir / f"{stem}_mobileclip.json",
        chunk_dir / "chunk_semantics.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    matches = sorted(chunk_dir.glob("*_semantics.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        lst = "\n  ".join(str(p) for p in matches)
        raise SystemExit(
            f"Hay varios *_semantics.json en:\n  {chunk_dir}\nElige uno con --input-json:\n  {lst}"
        )
    # chunk_001_siglip.json, chunk_001_siglip_default.json, chunk_001_openclip_foo.json, …
    for backend in ("siglip", "openclip", "mobileclip"):
        globs = sorted(chunk_dir.glob(f"{stem}_{backend}*.json"))
        if len(globs) == 1:
            return globs[0]
        if len(globs) > 1:
            lst = "\n  ".join(str(p) for p in globs)
            raise SystemExit(
                f"Hay varios JSON `{stem}_{backend}*.json` en:\n  {chunk_dir}\n"
                f"Elige uno con --input-json:\n  {lst}"
            )
    raise SystemExit(
        "No se encontró ningún JSON de session_semantics en esa carpeta.\n\n"
        "Genera uno antes, p. ej.:\n"
        f"  python3 session_semantics.py --chunk-dir {chunk_dir}\n\n"
        "O indica el fichero:\n"
        "  python3 interaction_reasoner.py --input-json /ruta/al/archivo.json\n"
    )


def _reason_interaction_cli(payload: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    sw = max(1, int(args.smooth_window))
    if sw % 2 == 0:
        sw += 1
    result = reason_interaction(
        payload,
        smooth_window=sw,
        min_evaluable_frames=int(args.min_evaluable_frames),
        min_run_object=int(args.min_run_object),
        min_run_deposit=int(args.min_run_deposit),
        treat_unknown_as_gap=bool(args.treat_unknown_as_gap),
        summary_per_chunk_max=max(1, int(args.summary_per_chunk_max)),
        summary_global_max=max(1, int(args.summary_global_max)),
    )
    return result, sw


def _reasoner_params_dict(sw: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "smooth_window": sw,
        "min_evaluable_frames": int(args.min_evaluable_frames),
        "min_run_object": int(args.min_run_object),
        "min_run_deposit": int(args.min_run_deposit),
        "treat_unknown_as_gap": bool(args.treat_unknown_as_gap),
        "summary_per_chunk_max": max(1, int(args.summary_per_chunk_max)),
        "summary_global_max": max(1, int(args.summary_global_max)),
    }


def _emit_summary_block_stderr(summ: dict[str, Any]) -> None:
    note = str(summ.get("note") or "").strip()
    lines = summ.get("lines_es")
    print("\n--- Resumen de acciones (ES) ---", file=sys.stderr)
    if note:
        print(note, file=sys.stderr)
    if isinstance(lines, list):
        for line in lines:
            print(line, file=sys.stderr)
    ol = summ.get("one_liner_es")
    if ol:
        print(f"Secuencia compacta: {ol}", file=sys.stderr)


def _emit_compare_summaries_stderr(by_backend: dict[str, Any], order: list[str]) -> None:
    for backend in order:
        r = by_backend.get(backend) or {}
        summ = r.get("session_summary_es")
        print(file=sys.stderr)
        print(f"{'=' * 12} VLM / backend: {backend} {'=' * 12}", file=sys.stderr)
        print(f"verdict={r.get('verdict')}  confidence={r.get('confidence')}", file=sys.stderr)
        src = r.get("source_semantics")
        if isinstance(src, list) and src:
            print(f"fuentes ({len(src)} JSON):", file=sys.stderr)
            for sp in src[:12]:
                print(f"  {sp}", file=sys.stderr)
            if len(src) > 12:
                print(f"  … (+{len(src) - 12} más)", file=sys.stderr)
        elif isinstance(src, str):
            print(f"fuente: {src}", file=sys.stderr)
        if isinstance(summ, dict):
            _emit_summary_block_stderr(summ)
        print(file=sys.stderr)


def main() -> None:
    args = build_arg_parser().parse_args()
    chunk_s = str(getattr(args, "chunk_dir", "") or "").strip()
    chunks_root_s = str(getattr(args, "chunks_root_dir", "") or "").strip()
    raw_inputs: list[str] = list(getattr(args, "input_json", None) or [])

    mode_ct = sum(
        1 for x in (chunks_root_s, chunk_s, bool(raw_inputs)) if x
    )
    if mode_ct != 1:
        raise SystemExit(
            "Indica exactamente uno de:\n"
            "  --chunks-root-dir DIR   (varios chunks bajo DIR, comparar JSON por VLM)\n"
            "  --chunk-dir DIR         (un solo chunk)\n"
            "  --input-json F [F2 …]   (uno o más JSON)"
        )
    if chunk_s and raw_inputs:
        raise SystemExit("Usa solo --chunk-dir o --input-json, no los dos a la vez.")

    if chunks_root_s:
        root = Path(chunks_root_s).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"No es un directorio: {root}")
        per_backend, chunk_dirs = _collect_semantics_paths_per_backend(root)
        if not per_backend:
            raise SystemExit(
                f"No se encontró ningún JSON de session_semantics en subcarpetas de:\n  {root}\n\n"
                "Por cada carpeta chunk esperado, p. ej. chunk_001/chunk_001_siglip.json, "
                "chunk_001_openclip.json, chunk_001_mobileclip.json o chunk_001_semantics.json\n"
                "(el prefijo del fichero debe coincidir con el nombre de la carpeta)."
            )
        order = _ordered_backend_keys(per_backend)
        by_backend: dict[str, Any] = {}
        coverage: dict[str, Any] = {}
        n_chunks = len(chunk_dirs)
        for backend in order:
            paths = per_backend[backend]
            coverage[backend] = {
                "json_files": len(paths),
                "chunks_under_root": n_chunks,
                "chunks_missing_for_this_backend": max(0, n_chunks - len(paths)),
            }
            for p in paths:
                if not p.is_file():
                    raise SystemExit(f"No existe el fichero: {p}")
            payload = merge_chunk_semantics_payloads(paths)
            result, sw = _reason_interaction_cli(payload, args)
            result["source_semantics"] = [str(p) for p in paths]
            result["reasoner_params"] = _reasoner_params_dict(sw, args)
            result["vlm_backend_source"] = backend
            by_backend[backend] = result

        out_obj: dict[str, Any] = {
            "mode": "chunks_root_compare_vlm",
            "chunks_root": str(root),
            "chunk_subdirs_scanned": [str(p) for p in chunk_dirs],
            "coverage": coverage,
            "by_backend": by_backend,
        }
        text = json.dumps(out_obj, ensure_ascii=False, indent=2)
        out_s = str(args.output_json or "").strip()
        if out_s:
            outp = Path(out_s).expanduser().resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(text, encoding="utf-8")
            print(
                f"[reasoner] compare_vlm backends={','.join(order)} chunks={n_chunks} -> {outp}",
                flush=True,
            )
        else:
            print(text)

        if not bool(args.quiet_summary):
            print("\n--- Comparativa por VLM (stderr) ---", file=sys.stderr)
            _emit_compare_summaries_stderr(by_backend, order)
        return

    if chunk_s:
        cd = Path(chunk_s).expanduser().resolve()
        paths = [_resolve_semantics_json_for_chunk(cd)]
    else:
        paths = [Path(p).expanduser().resolve() for p in raw_inputs]

    for p in paths:
        if not p.is_file():
            raise SystemExit(f"No existe el fichero: {p}")
    if len(paths) == 1:
        payload = _load_semantics(paths[0])
    else:
        payload = merge_chunk_semantics_payloads(paths)
    result, sw = _reason_interaction_cli(payload, args)
    result["source_semantics"] = str(paths[0]) if len(paths) == 1 else [str(p) for p in paths]
    result["reasoner_params"] = _reasoner_params_dict(sw, args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    out_s = str(args.output_json or "").strip()
    if out_s:
        outp = Path(out_s).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        print(
            f"[reasoner] {result.get('verdict')} conf={result.get('confidence')} -> {outp}",
            flush=True,
        )
    else:
        print(text)

    if not bool(args.quiet_summary):
        summ = result.get("session_summary_es")
        if isinstance(summ, dict):
            _emit_summary_block_stderr(summ)
            print(file=sys.stderr)


if __name__ == "__main__":
    main()
