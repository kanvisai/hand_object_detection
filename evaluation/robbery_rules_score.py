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


def main() -> None:
    p = argparse.ArgumentParser(description="Probabilidad de robo por reglas de secuencia.")
    p.add_argument("--input-json", nargs="+", required=True, metavar="JSON")
    p.add_argument("--output-json", default="")
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--min-object-run", type=int, default=2)
    p.add_argument("--lookahead-runs", type=int, default=3, help="Cuántos runs mirar tras object_in_hand.")
    p.add_argument("--w-suspicious", type=float, default=0.22)
    p.add_argument("--w-normal", type=float, default=0.18)
    p.add_argument("--base", type=float, default=0.35)
    args = p.parse_args()

    frames: list[dict[str, Any]] = []
    for pj in args.input_json:
        data = _load(Path(pj).expanduser().resolve())
        if isinstance(data, dict):
            frs = data.get("frames")
            if isinstance(frs, list):
                frames.extend([x for x in frs if isinstance(x, dict)])
    ordered = _sorted_frames(frames)

    labels_raw: list[str | None] = []
    for fr in ordered:
        if not bool(fr.get("evaluable", fr.get("vlm_applied", False))):
            labels_raw.append(None)
            continue
        lab = fr.get("semantic_label")
        labels_raw.append(str(lab) if lab else None)

    sw = max(1, int(args.smooth_window))
    if sw % 2 == 0:
        sw += 1
    labels = _smooth_mode(labels_raw, sw)
    rr = _runs(labels)

    suspicious_events: list[dict[str, Any]] = []
    normal_events: list[dict[str, Any]] = []

    for i, r in enumerate(rr):
        lab = r["label"]
        if lab != "object_in_hand" or int(r["length"]) < int(args.min_object_run):
            continue

        tail = rr[i + 1 : i + 1 + max(1, int(args.lookahead_runs))]
        has_container = any(_is_container(t["label"]) for t in tail)
        has_susp_follow = any(_is_suspicious_followup(t["label"]) for t in tail)

        if has_container:
            normal_events.append(
                {
                    "type": "object_then_container",
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
            # objeto -> no objeto sin explicación clara de contenedor
            suspicious_events.append(
                {
                    "type": "object_then_nonobject_without_container",
                    "object_run": r,
                    "tail_labels": [t["label"] for t in tail],
                }
            )

    # Contenedor prolongado sin objeto previo suele ser compra normal/contexto normal.
    for r in rr:
        if _is_container(r["label"]) and int(r["length"]) >= 2:
            normal_events.append({"type": "container_run", "run": r})

    score = float(args.base)
    score += float(args.w_suspicious) * float(len(suspicious_events))
    score -= float(args.w_normal) * float(len(normal_events))
    robbery_probability = max(0.0, min(1.0, score))

    if robbery_probability >= 0.65:
        verdict = "suspicious"
    elif robbery_probability <= 0.35:
        verdict = "low_risk_signal"
    else:
        verdict = "uncertain"

    out = {
        "inputs": [str(Path(p).expanduser().resolve()) for p in args.input_json],
        "params": {
            "smooth_window": sw,
            "min_object_run": args.min_object_run,
            "lookahead_runs": args.lookahead_runs,
            "base": args.base,
            "w_suspicious": args.w_suspicious,
            "w_normal": args.w_normal,
        },
        "robbery_probability": round(float(robbery_probability), 4),
        "verdict": verdict,
        "counts": {
            "suspicious_events": len(suspicious_events),
            "normal_events": len(normal_events),
        },
        "suspicious_events": suspicious_events,
        "normal_events": normal_events,
        "label_sequence_smoothed": labels,
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
