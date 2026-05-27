"""
Lógica compartida: CLI local, cliente HTTP y servicio persistente.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _session_semantics import _bootstrap_hf_auth, run_chunk_semantics
from vlm_factory import create_retail_vlm
from vlm_rc2_config import vlm_error_semantics_basename, vlm_semantics_basename
from vlm_rc2_errors import semantics_runtime_error_payload
from vlm_rc2_metrics import enrich_semantics_payload
from rc2_settings import VlmRc2Settings, effective_string, get_settings
from vlm_rc2_prompts import PROMPT_VARIANT_ID, RC2_PROMPT_TEXTS_EN, assert_prompts_ok

BACKEND = "siglip"


@dataclass
class SemanticsRunOptions:
    device: str = "cuda:0"
    hf_token: str = ""
    vlm_model: str = "google/siglip-so400m-patch14-384"
    multicrop_mode: str = "batch3"
    net_th: float = 0.35
    net_margin_th: float = 0.30
    decision_mode: str = "weighted"
    emit_image_embedding: bool = False
    tau_max_prob: float = 0.35
    min_margin: float = 0.08
    max_entropy: float | None = None
    quiet: bool = True
    persist_json: bool = False
    write_file: bool = True
    output_json: Path | None = None
    continue_on_inference_error: bool = True


def options_from_namespace(args: Any, settings: VlmRc2Settings | None = None) -> SemanticsRunOptions:
    s = settings or get_settings()
    max_ent = float(getattr(args, "max_entropy", s.max_entropy))
    hf_cli = str(getattr(args, "hf_token", "") or "")
    return SemanticsRunOptions(
        device=effective_string(cli=str(getattr(args, "device", "")), config=s.device),
        hf_token=effective_string(cli=hf_cli, config=s.hf_token, env_key="HF_TOKEN"),
        vlm_model=effective_string(cli=str(getattr(args, "vlm_model", "")), config=s.vlm_model),
        multicrop_mode=effective_string(
            cli=str(getattr(args, "multicrop_mode", "")), config=s.multicrop_mode
        ),
        net_th=float(getattr(args, "net_th", s.net_th)),
        net_margin_th=float(getattr(args, "net_margin_th", s.net_margin_th)),
        decision_mode=effective_string(
            cli=str(getattr(args, "decision_mode", "")), config=s.decision_mode
        ),
        emit_image_embedding=bool(getattr(args, "emit_image_embedding", s.emit_image_embedding)),
        tau_max_prob=float(getattr(args, "tau_max_prob", s.tau_max_prob)),
        min_margin=float(getattr(args, "min_margin", s.min_margin)),
        max_entropy=None if max_ent < 0 else max_ent,
        quiet=bool(getattr(args, "quiet", True)),
        persist_json=False,
        write_file=not bool(getattr(args, "no_write_file", False)),
    )


def build_classifier(opts: SemanticsRunOptions) -> Any:
    assert_prompts_ok()
    _bootstrap_hf_auth(opts)
    return create_retail_vlm(
        BACKEND,
        device=opts.device,
        hf_model=opts.vlm_model,
        openclip_model="ViT-B-32",
        openclip_pretrained="laion2b_s34b_b79k",
        mobileclip_model="MobileCLIP2-S0",
        mobileclip_pretrained="dfndr2b",
        net_th=opts.net_th,
        net_margin_th=opts.net_margin_th,
        decision_mode=opts.decision_mode,
        multicrop_mode=opts.multicrop_mode,
        prompt_texts_en=list(RC2_PROMPT_TEXTS_EN),
    )


def resolve_output_json(chunk_dir: Path, chunk_stem: str, opts: SemanticsRunOptions) -> Path:
    if opts.output_json is not None:
        return opts.output_json
    if not opts.write_file:
        return Path(os.devnull)
    return chunk_dir / vlm_semantics_basename(chunk_stem)


def process_chunk_directory(
    chunk_dir: Path,
    clf: Any,
    opts: SemanticsRunOptions,
    *,
    stage: str = "process_chunk_directory",
) -> dict[str, Any]:
    """Ejecuta semántica sobre un chunk; no propaga excepciones de estructura de datos."""
    chunk_dir = chunk_dir.expanduser().resolve()
    chunk_stem = chunk_dir.name or "chunk"
    output_json = resolve_output_json(chunk_dir, chunk_stem, opts)

    try:
        payload = run_chunk_semantics(
            chunk_dir,
            clf,
            vlm_backend=BACKEND,
            prompt_variant_id=PROMPT_VARIANT_ID,
            output_json=output_json,
            emit_image_embedding=opts.emit_image_embedding,
            tau_max_prob=opts.tau_max_prob,
            min_margin=opts.min_margin,
            max_entropy=opts.max_entropy,
            quiet=opts.quiet,
            persist_json=opts.persist_json,
            continue_on_inference_error=opts.continue_on_inference_error,
        )
        status = str(payload.get("vlm_rc2_pipeline_status") or "")
        if status != "skipped":
            enrich_semantics_payload(payload)
            payload["vlm_rc2_pipeline_status"] = "ok"
        return payload
    except Exception as e:
        return semantics_runtime_error_payload(
            chunk_dir=chunk_dir,
            chunk_stem=chunk_stem,
            exc=e,
            stage=stage,
        )


def persist_chunk_result(
    chunk_dir: Path,
    chunk_stem: str,
    payload: dict[str, Any],
    *,
    write_file: bool,
) -> None:
    if not write_file:
        return
    status = str(payload.get("vlm_rc2_pipeline_status") or "")
    if status == "error":
        path = chunk_dir / vlm_error_semantics_basename(chunk_stem)
    elif status == "ok":
        path = chunk_dir / vlm_semantics_basename(chunk_stem)
        err_path = chunk_dir / vlm_error_semantics_basename(chunk_stem)
        try:
            if err_path.is_file():
                err_path.unlink()
        except OSError:
            pass
    else:
        path = chunk_dir / vlm_semantics_basename(chunk_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
