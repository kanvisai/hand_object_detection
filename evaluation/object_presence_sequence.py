#!/usr/bin/env python3
"""
Deriva secuencia por frame: object | empty | uncertain
desde JSON(s) de session_semantics.

Objetivo: apoyar decisión con un canal robusto "hay objeto en mano o no".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON inválido: {path}")
    return data


def _sorted_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(fr: dict[str, Any]) -> tuple[str, int]:
        ch = str(fr.get("chunk") or "")
        try:
            si = int(fr.get("sample_idx") or 0)
        except (TypeError, ValueError):
            si = 0
        return (ch, si)

    return sorted(frames, key=key)


def _extract_probs(fr: dict[str, Any], *, obj_idx: int, empty_idx: int) -> tuple[float, float] | None:
    probs = fr.get("vlm_vector_prompt_probs")
    need = max(obj_idx, empty_idx) + 1
    if not isinstance(probs, list) or len(probs) < need:
        return None
    try:
        p_obj = float(probs[obj_idx])
        p_empty = float(probs[empty_idx])
    except (TypeError, ValueError):
        return None
    return p_obj, p_empty


def _label_frame(
    p_obj: float,
    p_empty: float,
    prev_state: str | None,
    *,
    obj_on: float,
    obj_off: float,
    net_on: float,
    net_off: float,
    margin: float,
    empty_on: float,
) -> str:
    net = p_obj - p_empty
    is_obj_strong = (p_obj >= obj_on) and (net >= net_on) and (p_obj >= (p_empty + margin))
    is_obj_weak = (p_obj >= obj_off) and (net >= net_off)
    is_empty_strong = (p_empty >= empty_on) and (p_empty >= (p_obj + margin))

    # Histeresis: mantener "object" ante pequeñas caídas.
    if prev_state == "object":
        if is_obj_weak:
            return "object"
        if is_empty_strong:
            return "empty"
        return "uncertain"

    if is_obj_strong:
        return "object"
    if is_empty_strong:
        return "empty"
    return "uncertain"


def _compress_runs(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not seq:
        return []
    out: list[dict[str, Any]] = []
    cur = dict(seq[0])
    cur["start_idx"] = 0
    cur["end_idx"] = 0
    for i, row in enumerate(seq[1:], start=1):
        if row["state"] == cur["state"]:
            cur["end_idx"] = i
            continue
        out.append(cur)
        cur = dict(row)
        cur["start_idx"] = i
        cur["end_idx"] = i
    out.append(cur)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Secuencia object/empty/uncertain desde session_semantics.")
    p.add_argument("--input-json", nargs="+", required=True, metavar="JSON")
    p.add_argument("--output-json", default="", help="Opcional: guarda resultado en JSON.")
    p.add_argument("--obj-on", type=float, default=0.55, help="Entrada a estado object.")
    p.add_argument("--obj-off", type=float, default=0.40, help="Salida suave de estado object (histeresis).")
    p.add_argument("--net-on", type=float, default=0.20, help="(p_obj - p_empty) para entrar en object.")
    p.add_argument("--net-off", type=float, default=0.05, help="(p_obj - p_empty) para mantener object.")
    p.add_argument("--margin", type=float, default=0.10, help="Margen mínimo entre p_obj y p_empty.")
    p.add_argument("--empty-on", type=float, default=0.55, help="Umbral para marcar empty.")
    p.add_argument("--obj-idx", type=int, default=0, help="Índice del prompt 'objeto en mano' en vlm_vector_prompt_probs.")
    p.add_argument("--empty-idx", type=int, default=1, help="Índice del prompt 'manos vacías' en vlm_vector_prompt_probs.")
    args = p.parse_args()

    payloads = [_load_payload(Path(x).expanduser().resolve()) for x in args.input_json]
    frames: list[dict[str, Any]] = []
    for data in payloads:
        frs = data.get("frames")
        if isinstance(frs, list):
            frames.extend([f for f in frs if isinstance(f, dict)])
    ordered = _sorted_frames(frames)

    seq: list[dict[str, Any]] = []
    prev_state: str | None = None
    for fr in ordered:
        base = {
            "chunk": fr.get("chunk"),
            "sample_idx": fr.get("sample_idx"),
            "image_key": fr.get("image_key"),
        }
        # Compatibilidad:
        # - session_semantics.py usa "evaluable"
        # - test_new_handobject_siglip_v2_frames.py (session vectors) usa "vlm_applied"
        is_evaluable = bool(fr.get("evaluable")) if ("evaluable" in fr) else bool(fr.get("vlm_applied"))
        if not is_evaluable:
            row = {**base, "state": "uncertain", "reason": "not_evaluable", "p_obj": None, "p_empty": None}
            seq.append(row)
            prev_state = "uncertain"
            continue
        pp = _extract_probs(fr, obj_idx=int(args.obj_idx), empty_idx=int(args.empty_idx))
        if pp is None:
            row = {**base, "state": "uncertain", "reason": "missing_probs", "p_obj": None, "p_empty": None}
            seq.append(row)
            prev_state = "uncertain"
            continue
        p_obj, p_empty = pp
        state = _label_frame(
            p_obj,
            p_empty,
            prev_state,
            obj_on=float(args.obj_on),
            obj_off=float(args.obj_off),
            net_on=float(args.net_on),
            net_off=float(args.net_off),
            margin=float(args.margin),
            empty_on=float(args.empty_on),
        )
        row = {**base, "state": state, "p_obj": round(p_obj, 4), "p_empty": round(p_empty, 4)}
        seq.append(row)
        prev_state = state

    runs = _compress_runs(seq)
    one_liner = ", ".join(str(x["state"]) for x in seq)
    counts = {"object": 0, "empty": 0, "uncertain": 0}
    for x in seq:
        st = str(x["state"])
        if st in counts:
            counts[st] += 1

    out = {
        "inputs": [str(Path(x).expanduser().resolve()) for x in args.input_json],
        "params": {
            "obj_on": args.obj_on,
            "obj_off": args.obj_off,
            "net_on": args.net_on,
            "net_off": args.net_off,
            "margin": args.margin,
            "empty_on": args.empty_on,
        },
        "counts": counts,
        "one_liner": one_liner,
        "runs": runs,
        "frames": seq,
    }

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if str(args.output_json).strip():
        outp = Path(str(args.output_json)).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        print(f"[presence] Guardado: {outp}")
    else:
        print(text)


if __name__ == "__main__":
    main()
