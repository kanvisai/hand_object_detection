#!/usr/bin/env python3
"""
Fusiona señales de session_semantics (+ opcional modelo de movimiento) para inferir:
  - normal_purchase
  - suspicious
  - uncertain

Diseñado para reducir falsos positivos: exige patrón temporal, no picos aislados.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(fr: dict[str, Any]) -> tuple[str, int]:
        ch = str(fr.get("chunk") or "")
        try:
            si = int(fr.get("sample_idx") or 0)
        except (TypeError, ValueError):
            si = 0
        return (ch, si)

    return sorted(frames, key=key)


def _extract_motion_map(data: Any) -> dict[tuple[str, int], float]:
    """
    Formatos tolerados:
    - {"frames":[{"chunk","sample_idx","robbery_prob":0.6}, ...]}
    - [{"chunk","sample_idx","score":0.7}, ...]
    """
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        cand = data.get("frames")
        if isinstance(cand, list):
            rows = [x for x in cand if isinstance(x, dict)]
    elif isinstance(data, list):
        rows = [x for x in data if isinstance(x, dict)]

    out: dict[tuple[str, int], float] = {}
    for r in rows:
        ch = str(r.get("chunk") or "")
        try:
            si = int(r.get("sample_idx") or 0)
        except (TypeError, ValueError):
            continue
        val = None
        for k in ("robbery_prob", "robbery_score", "score", "prob", "risk"):
            if k in r:
                val = r.get(k)
                break
        if val is None:
            continue
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue
        out[(ch, si)] = max(0.0, min(1.0, fv))
    return out


def _presence_state(fr: dict[str, Any], *, obj_on: float, empty_on: float) -> str:
    probs = fr.get("vlm_vector_prompt_probs")
    if not isinstance(probs, list) or len(probs) < 2:
        return "uncertain"
    try:
        p_obj = float(probs[0])
        p_empty = float(probs[1])
    except (TypeError, ValueError):
        return "uncertain"
    if p_obj >= obj_on and p_obj > p_empty:
        return "object"
    if p_empty >= empty_on and p_empty > p_obj:
        return "empty"
    return "uncertain"


def _risk_frame(fr: dict[str, Any], motion_prob: float, *, w_motion: float, w_object_drop: float, w_container: float) -> tuple[float, dict[str, float]]:
    probs = fr.get("vlm_vector_prompt_probs")
    p_obj = p_basket = p_cart = 0.0
    if isinstance(probs, list) and len(probs) >= 6:
        try:
            p_obj = float(probs[0])
            p_basket = float(probs[4])
            p_cart = float(probs[5])
        except (TypeError, ValueError):
            pass
    p_container = max(0.0, min(1.0, p_basket + p_cart))
    # Base: movimiento sospechoso + evidencia de objeto; resta contexto compra en contenedor.
    score = (w_motion * motion_prob) + (w_object_drop * p_obj) - (w_container * p_container)
    score = max(0.0, min(1.0, score))
    return score, {"motion": motion_prob, "p_obj": p_obj, "p_container": p_container}


def _compress_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    cur = dict(rows[0])
    cur["start"] = 0
    cur["end"] = 0
    for i, r in enumerate(rows[1:], start=1):
        if r["state"] == cur["state"]:
            cur["end"] = i
            continue
        out.append(cur)
        cur = dict(r)
        cur["start"] = i
        cur["end"] = i
    out.append(cur)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fusiona VLM + movimiento para riesgo de robo.")
    p.add_argument("--semantics-json", nargs="+", required=True, metavar="JSON")
    p.add_argument("--motion-json", default="", help="Opcional: JSON de score de movimiento por frame.")
    p.add_argument("--output-json", default="")
    p.add_argument("--obj-on", type=float, default=0.55)
    p.add_argument("--empty-on", type=float, default=0.55)
    p.add_argument("--risk-th", type=float, default=0.60, help="Umbral para frame suspicious.")
    p.add_argument("--risk-run-min", type=int, default=2, help="Frames consecutivos suspicious para evento.")
    p.add_argument("--w-motion", type=float, default=0.60)
    p.add_argument("--w-object", type=float, default=0.35)
    p.add_argument("--w-container", type=float, default=0.45)
    args = p.parse_args()

    frames: list[dict[str, Any]] = []
    for sj in args.semantics_json:
        data = _load_json(Path(sj).expanduser().resolve())
        if isinstance(data, dict):
            frs = data.get("frames")
            if isinstance(frs, list):
                frames.extend([x for x in frs if isinstance(x, dict)])
    ordered = _sorted_frames(frames)

    motion_map: dict[tuple[str, int], float] = {}
    if str(args.motion_json).strip():
        motion_map = _extract_motion_map(_load_json(Path(args.motion_json).expanduser().resolve()))

    out_rows: list[dict[str, Any]] = []
    cur_run = 0
    best_run = 0
    for fr in ordered:
        ch = str(fr.get("chunk") or "")
        try:
            si = int(fr.get("sample_idx") or 0)
        except (TypeError, ValueError):
            si = 0
        motion = float(motion_map.get((ch, si), 0.0))
        risk, comp = _risk_frame(
            fr,
            motion,
            w_motion=float(args.w_motion),
            w_object_drop=float(args.w_object),
            w_container=float(args.w_container),
        )
        state = "suspicious" if risk >= float(args.risk_th) else "normal_purchase"
        if state == "suspicious":
            cur_run += 1
            best_run = max(best_run, cur_run)
        else:
            cur_run = 0
        pres = _presence_state(fr, obj_on=float(args.obj_on), empty_on=float(args.empty_on))
        if pres == "uncertain" and state == "normal_purchase":
            state = "uncertain"
        out_rows.append(
            {
                "chunk": ch,
                "sample_idx": si,
                "image_key": fr.get("image_key"),
                "presence": pres,
                "state": state,
                "risk_score": round(risk, 4),
                "components": {k: round(v, 4) for k, v in comp.items()},
            }
        )

    # Decisión sesión: suspicious solo si hay run suficiente.
    if best_run >= int(args.risk_run_min):
        session_verdict = "suspicious"
    else:
        session_verdict = "uncertain" if any(r["state"] == "uncertain" for r in out_rows) else "normal_purchase"

    counts = {"suspicious": 0, "normal_purchase": 0, "uncertain": 0}
    for r in out_rows:
        st = str(r["state"])
        if st in counts:
            counts[st] += 1

    out = {
        "inputs": {"semantics_json": args.semantics_json, "motion_json": args.motion_json or None},
        "params": {
            "obj_on": args.obj_on,
            "empty_on": args.empty_on,
            "risk_th": args.risk_th,
            "risk_run_min": args.risk_run_min,
            "w_motion": args.w_motion,
            "w_object": args.w_object,
            "w_container": args.w_container,
        },
        "session_verdict": session_verdict,
        "best_suspicious_run": best_run,
        "counts": counts,
        "runs": _compress_runs(out_rows),
        "frames": out_rows,
    }

    text = json.dumps(out, ensure_ascii=False, indent=2)
    out_s = str(args.output_json or "").strip()
    if out_s:
        outp = Path(out_s).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        print(f"[risk_fusion] Guardado: {outp}")
    else:
        print(text)


if __name__ == "__main__":
    main()
