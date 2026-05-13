#!/usr/bin/env python3
"""
Pipeline rc1: semántica para **una carpeta de chunk** (`frames/` + `frames_meta.json`)
con prompts definidos en `vlm_rc1_prompts.py` y **SigLIP** únicamente.

En producción: fallos por frame no abortan el chunk; fallo total escribe `*_vlm_error.json`
y por defecto **un único JSON en stdout** (el proceso padre puede capturarlo con `subprocess.run(..., text=True).stdout`).
Usa `--no-stdout-json` si no quieres JSON en stdout. Salida del proceso: código 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

_EVAL = Path(__file__).resolve().parent.parent
_RCDIR = Path(__file__).resolve().parent
for _p in (_EVAL, _RCDIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from session_semantics import _bootstrap_hf_auth, run_chunk_semantics  # noqa: E402
from vlm_factory import create_retail_vlm  # noqa: E402

from vlm_rc1_config import (  # noqa: E402
    vlm_error_semantics_basename,
    vlm_semantics_basename,
    write_json_stdout,
)
from vlm_rc1_errors import semantics_runtime_error_payload  # noqa: E402
from vlm_rc1_metrics import enrich_semantics_payload  # noqa: E402
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
        help="Ruta opcional del JSON; si se omite: <chunk-dir>/<stem>_vlm.json",
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
        "--no-stdout-json",
        action="store_true",
        help="No escribir el JSON en stdout (solo fichero / marcador de error). Por defecto sí: el proceso padre puede capturar stdout.",
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

    def _log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    chunk_dir = Path(args.chunk_dir).expanduser().resolve()
    chunk_stem = chunk_dir.name or "chunk"
    out_s = str(args.output_json or "").strip()

    max_ent = float(args.max_entropy)
    max_entropy = None if max_ent < 0 else max_ent

    if args.no_write_file:
        output_json = Path(os.devnull)
    elif out_s:
        output_json = Path(out_s).expanduser().resolve()
    else:
        output_json = chunk_dir / vlm_semantics_basename(chunk_stem)

    error_marker_path = chunk_dir / vlm_error_semantics_basename(chunk_stem)

    try:
        _bootstrap_hf_auth(args)
        assert_prompts_ok()
        pv_id = PROMPT_VARIANT_ID
        variant_texts = list(RC1_PROMPT_TEXTS_EN)

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
            persist_json=False,
            continue_on_inference_error=True,
        )
        enrich_semantics_payload(payload)
        payload["vlm_rc1_pipeline_status"] = "ok"

        n_fail = int(payload.get("vlm_inference_failure_count") or 0)
        if n_fail and not args.quiet:
            _log(f"[vlm_rc1] Aviso: {n_fail} frame(s) con fallo de inferencia VLM (se siguieron procesando el resto).")

        if not args.no_write_file:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                if error_marker_path.is_file():
                    error_marker_path.unlink()
            except OSError:
                pass

        if not args.no_stdout_json:
            write_json_stdout(payload, pretty=bool(args.pretty_json))
        if not args.quiet:
            if args.no_write_file:
                if args.no_stdout_json:
                    _log("[vlm_rc1] Sin escritura en disco ni JSON en stdout.")
                else:
                    _log("[vlm_rc1] Sin escritura en disco (--no-write-file); JSON en stdout.")
            else:
                _log(f"[vlm_rc1] JSON escrito: {output_json.resolve()}")
                if args.no_stdout_json:
                    _log("[vlm_rc1] Sin JSON en stdout (--no-stdout-json).")

    except Exception as e:
        err_payload = semantics_runtime_error_payload(
            chunk_dir=chunk_dir,
            chunk_stem=chunk_stem,
            exc=e,
            stage="run_semantics_rc1",
        )
        _log(f"[vlm_rc1] ERROR (no aborta proceso): {e}\n{traceback.format_exc()}")

        if not args.no_write_file:
            try:
                error_marker_path.parent.mkdir(parents=True, exist_ok=True)
                error_marker_path.write_text(
                    json.dumps(err_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _log(f"[vlm_rc1] Marcador de error escrito: {error_marker_path.resolve()}")
            except Exception as w:
                _log(f"[vlm_rc1] No se pudo escribir {error_marker_path}: {w}")

        if not args.no_stdout_json:
            write_json_stdout(err_payload, pretty=bool(args.pretty_json))


if __name__ == "__main__":
    main()
    sys.exit(0)
