#!/usr/bin/env python3
"""
Score de robo basado en reglas temporales sobre etiquetas semánticas.

Entrada: JSON(s) de session_semantics (uno o varios chunks).
Salida: robbery_probability (0..1) + razones y patrones detectados.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
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


def _smooth_mode(labels: list[str | None], window: int) -> list[str | None]:
    if window <= 1:
        return list(labels)
    half = window // 2
    out: list[str | None] = []
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        vals = [x for x in labels[lo:hi] if x is not None]
        if not vals:
            out.append(labels[i])
            continue
        best = max(set(vals), key=vals.count)
        out.append(best)
    return out


def _runs(labels: list[str | None]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0
    n = len(labels)
    while i < n:
        lab = labels[i]
        j = i + 1
        while j < n and labels[j] == lab:
            j += 1
        out.append({"label": lab, "start": i, "end": j - 1, "length": j - i})
        i = j
    return out


def _is_container(label: str | None) -> bool:
    return label in {"shopping_basket", "shopping_cart"}


def _is_suspicious_followup(label: str | None) -> bool:
    return label in {"gesture_no_object", "pockets_hidden", "unknown"}


def compute_robbery_rules_score(
    frames: list[dict[str, Any]],
    *,
    smooth_window: int = 3,
    min_object_run: int = 2,
    lookahead_runs: int = 4,
    w_severe: float = 0.55,
    w_suspicious: float = 0.30,
    w_normal: float = 0.30,
    base: float = 0.20,
) -> dict[str, Any]:
    """
    Misma lógica que la CLI: una lista plana de dicts `frames` (como en JSON de session_semantics).
    Devuelve el objeto de resultado sin campo `inputs` (añádelo en el llamador si hace falta).
    """
    ordered = _sorted_frames(list(frames))

    labels_raw: list[str | None] = []
    for fr in ordered:
        if not bool(fr.get("evaluable", fr.get("vlm_applied", False))):
            labels_raw.append(None)
            continue
        lab = fr.get("semantic_label")
        labels_raw.append(str(lab) if lab else None)

    sw = max(1, int(smooth_window))
    if sw % 2 == 0:
        sw += 1
    labels = _smooth_mode(labels_raw, sw)
    rr = _runs(labels)

    severe_events: list[dict[str, Any]] = []
    suspicious_events: list[dict[str, Any]] = []
    normal_events: list[dict[str, Any]] = []

    for i, r in enumerate(rr):
        lab = r["label"]
        if lab != "object_in_hand" or int(r["length"]) < int(min_object_run):
            continue

        tail = rr[i + 1 : i + 1 + max(1, int(lookahead_runs))]
        has_container = any(_is_container(t["label"]) for t in tail)
        has_susp_follow = any(_is_suspicious_followup(t["label"]) for t in tail)
        has_object_recovery = any(t["label"] == "object_in_hand" for t in tail)

        if has_container:
            normal_events.append(
                {
                    "type": "object_then_container",
                    "object_run": r,
                    "tail_labels": [t["label"] for t in tail],
                }
            )
        elif has_susp_follow and (not has_object_recovery):
            severe_events.append(
                {
                    "type": "object_then_uncertain_or_gesture_without_container_and_without_recovery",
                    "object_run": r,
                    "tail_labels": [t["label"] for t in tail],
                }
            )
        elif has_susp_follow:
            suspicious_events.append(
                {
                    "type": "object_disappears_then_gesture_or_unknown",
                    "object_run": r,
                    "tail_labels": [t["label"] for t in tail],
                }
            )
        else:
            suspicious_events.append(
                {
                    "type": "object_then_nonobject_without_container",
                    "object_run": r,
                    "tail_labels": [t["label"] for t in tail],
                }
            )

    for r in rr:
        if _is_container(r["label"]) and int(r["length"]) >= 2:
            normal_events.append({"type": "container_run", "run": r})

    score = float(base)
    score += float(w_severe) * float(len(severe_events))
    score += float(w_suspicious) * float(len(suspicious_events))
    score -= float(w_normal) * float(len(normal_events))
    robbery_probability = max(0.0, min(1.0, score))

    if robbery_probability >= 0.65:
        verdict = "suspicious"
    elif robbery_probability <= 0.35:
        verdict = "low_risk_signal"
    else:
        verdict = "uncertain"

    return {
        "params": {
            "smooth_window": sw,
            "min_object_run": int(min_object_run),
            "lookahead_runs": int(lookahead_runs),
            "base": float(base),
            "w_severe": float(w_severe),
            "w_suspicious": float(w_suspicious),
            "w_normal": float(w_normal),
        },
        "robbery_probability": round(float(robbery_probability), 4),
        "verdict": verdict,
        "counts": {
            "severe_events": len(severe_events),
            "suspicious_events": len(suspicious_events),
            "normal_events": len(normal_events),
        },
        "severe_events": severe_events,
        "suspicious_events": suspicious_events,
        "normal_events": normal_events,
        "label_sequence_smoothed": labels,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Probabilidad de robo por reglas de secuencia.")
    p.add_argument("--input-json", nargs="+", required=True, metavar="JSON")
    p.add_argument("--output-json", default="")
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--min-object-run", type=int, default=2)
    p.add_argument("--lookahead-runs", type=int, default=4, help="Cuántos runs mirar tras object_in_hand.")
    p.add_argument("--w-severe", type=float, default=0.55, help="Peso patrón severo: object->(gesture/unknown) sin contenedor ni recuperación.")
    p.add_argument("--w-suspicious", type=float, default=0.30, help="Peso patrón sospechoso moderado.")
    p.add_argument("--w-normal", type=float, default=0.30, help="Resta por patrón normal (contenedor).")
    p.add_argument("--base", type=float, default=0.20)
    args = p.parse_args()

    frames: list[dict[str, Any]] = []
    for pj in args.input_json:
        data = _load(Path(pj).expanduser().resolve())
        if isinstance(data, dict):
            frs = data.get("frames")
            if isinstance(frs, list):
                frames.extend([x for x in frs if isinstance(x, dict)])
    out = {
        "inputs": [str(Path(p).expanduser().resolve()) for p in args.input_json],
        **compute_robbery_rules_score(
            frames,
            smooth_window=int(args.smooth_window),
            min_object_run=int(args.min_object_run),
            lookahead_runs=int(args.lookahead_runs),
            w_severe=float(args.w_severe),
            w_suspicious=float(args.w_suspicious),
            w_normal=float(args.w_normal),
            base=float(args.base),
        ),
    }

    txt = json.dumps(out, ensure_ascii=False, indent=2)
    out_s = str(args.output_json or "").strip()
    if out_s:
        op = Path(out_s).expanduser().resolve()
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(txt, encoding="utf-8")
        print(f"[robbery_rules] Guardado: {op}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
