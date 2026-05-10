#!/usr/bin/env python3
"""
Pipeline rc1: semántica para **una carpeta de chunk** (`frames/` + `frames_meta.json`)
con prompts definidos en `vlm_rc1_prompts.py` y **SigLIP** únicamente.

Escribe el mismo JSON en **stdout** (una línea, UTF-8) para que otro proceso lo capture (p. ej. MinIO).
Los mensajes `[vlm_rc1]` van a **stderr**. Por defecto también guarda fichero junto a `frames_meta.json`:
`<chunk_stem>_siglip_<variant>.json` (`PROMPT_VARIANT_ID` en `vlm_rc1_prompts.py`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent.parent
_RCDIR = Path(__file__).resolve().parent
for _p in (_EVAL, _RCDIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from session_semantics import _bootstrap_hf_auth, _semantics_output_filename, run_chunk_semantics  # noqa: E402
from vlm_factory import create_retail_vlm  # noqa: E402

from vlm_rc1_config import write_json_stdout  # noqa: E402
from vlm_rc1_prompts import PROMPT_VARIANT_ID, RC1_PROMPT_TEXTS_EN, assert_prompts_ok  # noqa: E402

_BACKEND = "siglip"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="vlm_rc1: un chunk → JSON semántico SigLIP (prompts en vlm_rc1_prompts.py).",
    )
    p.add_argument(
        "--chunk-dir",
        "--chunk_dir",
        required=True,
        metavar="DIR",
        help="Carpeta del chunk con frames/ + frames_meta.json.",
    )
    p.add_argument(
        "--output-json",
        default="",
        help="Ruta opcional del JSON; si se omite: <chunk-dir>/<stem>_siglip_<variant>.json",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Dispositivo PyTorch (por defecto cuda:0). Usa cpu si no hay GPU.",
    )
    p.add_argument(
        "--hf-token",
        "--hf_token",
        dest="hf_token",
        default="",
        metavar="TOKEN",
        help="Token Hugging Face para SigLIP (alternativa a HF_TOKEN en el entorno).",
    )
    p.add_argument(
        "--vlm-model",
        default="google/siglip-so400m-patch14-384",
        help="Id del modelo SigLIP en Hugging Face.",
    )
    p.add_argument("--multicrop-mode", default="batch3", choices=["off", "light", "full", "batch3"])
    p.add_argument("--net-th", type=float, default=0.35)
    p.add_argument("--net-margin-th", type=float, default=0.30)
    p.add_argument("--decision-mode", default="weighted", choices=["weighted", "majority"])
    p.add_argument("--emit-image-embedding", action="store_true")
    p.add_argument("--tau-max-prob", type=float, default=0.35)
    p.add_argument("--min-margin", type=float, default=0.08)
    p.add_argument(
        "--max-entropy",
        type=float,
        default=-1.0,
        help="Si >= 0, rechazar si entropía > valor.",
    )
    p.add_argument(
        "--pretty-json",
        action="store_true",
        help="JSON formateado en stdout (por defecto: una línea compacta).",
    )
    p.add_argument(
        "--no-write-file",
        action="store_true",
        help="No guardar JSON en disco; solo stdout (la inferencia sigue igual).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="No imprimir mensajes en stderr (stdout sigue siendo solo el JSON).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    _bootstrap_hf_auth(args)

    assert_prompts_ok()
    pv_id = PROMPT_VARIANT_ID

    chunk_dir = Path(args.chunk_dir).expanduser().resolve()
    out_s = str(args.output_json or "").strip()

    max_ent = float(args.max_entropy)
    max_entropy = None if max_ent < 0 else max_ent

    variant_texts = list(RC1_PROMPT_TEXTS_EN)
    all_pv_ids = [pv_id]

    def _log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    _log(f"[vlm_rc1] SigLIP | prompt={pv_id} — cargando…")
    clf = create_retail_vlm(
        _BACKEND,
        device=str(args.device),
        hf_model=str(args.vlm_model),
        openclip_model="ViT-B-32",
        openclip_pretrained="laion2b_s34b_b79k",
        mobileclip_model="MobileCLIP2-S0",
        mobileclip_pretrained="dfndr2b",
        net_th=float(args.net_th),
        net_margin_th=float(args.net_margin_th),
        decision_mode=str(args.decision_mode),
        multicrop_mode=str(args.multicrop_mode),
        prompt_texts_en=variant_texts,
    )
    _log(f"[vlm_rc1] SigLIP listo en {clf.device} ({clf.model_name})")

    chunk_stem = chunk_dir.name or "chunk"
    if args.no_write_file:
        output_json = Path(os.devnull)
    elif out_s:
        output_json = Path(out_s).expanduser().resolve()
    else:
        fname = _semantics_output_filename(chunk_stem, _BACKEND, pv_id, all_pv_ids=all_pv_ids)
        output_json = chunk_dir / fname

    payload = run_chunk_semantics(
        chunk_dir,
        clf,
        vlm_backend=_BACKEND,
        prompt_variant_id=pv_id,
        output_json=output_json,
        emit_image_embedding=bool(args.emit_image_embedding),
        tau_max_prob=float(args.tau_max_prob),
        min_margin=float(args.min_margin),
        max_entropy=max_entropy,
        quiet=True,
    )

    write_json_stdout(payload, pretty=bool(args.pretty_json))
    if not args.quiet:
        if args.no_write_file:
            _log("[vlm_rc1] Sin escritura en disco (--no-write-file); JSON solo en stdout.")
        else:
            _log(f"[vlm_rc1] JSON escrito: {output_json.resolve()}")


if __name__ == "__main__":
    main()
