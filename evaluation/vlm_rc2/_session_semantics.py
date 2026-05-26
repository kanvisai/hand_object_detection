"""
Funciones extraídas de session_semantics.py — autosuficiente dentro de vlm_rc2.

Exporta:
  - _bootstrap_hf_auth(args)
  - run_chunk_semantics(chunk_dir, clf, ...)
  - discover_chunk_dirs_under_parent(parent)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from _chunk_helpers import (
    frame_has_wrist_visibility as _frame_has_wrist_visibility,
    load_frames_meta as _load_frames_meta,
    session_progress_line as _session_progress_line,
)
from retail_semantic_prompts import CANONICAL_LABELS, CODE_UNKNOWN, LABELS_ES

SCHEMA_VERSION = "1.5"
CODE_NOT_EVALUABLE = -1


# ---------------------------------------------------------------------------
#  Validación de carpeta de chunk
# ---------------------------------------------------------------------------

def _validate_chunk_dir(chunk_dir: Path) -> Path:
    p = chunk_dir.expanduser().resolve()
    if not p.is_dir():
        raise RuntimeError(f"No es un directorio: {p}")
    if not (p / "frames").is_dir():
        raise RuntimeError(f"Falta carpeta frames/ en: {p}")
    if not (p / "frames_meta.json").is_file():
        raise RuntimeError(f"Falta frames_meta.json en: {p}")
    return p


# ---------------------------------------------------------------------------
#  Descubrimiento de subcarpetas chunk
# ---------------------------------------------------------------------------

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


def discover_chunk_dirs_under_parent(parent: Path) -> list[Path]:
    """
    Lista subcarpetas directas que son chunks válidos (`frames/` + `frames_meta.json`).
    Orden "natural" por nombre (chunk_002 antes que chunk_010).
    """
    root = parent.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"No es un directorio: {root}")
    found: list[Path] = []
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        try:
            _validate_chunk_dir(sub)
        except RuntimeError:
            continue
        found.append(sub)
    return sorted(found, key=_natural_chunk_sort_key)


# ---------------------------------------------------------------------------
#  Desambiguación de probabilidades
# ---------------------------------------------------------------------------

def _shannon_entropy_natural(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _disambiguate(
    probs: np.ndarray,
    *,
    tau_max: float,
    min_margin: float,
    max_entropy: float | None,
) -> dict[str, Any]:
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return {
            "semantic_label": None,
            "semantic_code": CODE_UNKNOWN,
            "top_prob": None,
            "second_prob": None,
            "margin": None,
            "entropy": None,
            "disambiguation_passed": False,
            "disambiguation_reason": "empty_probs",
        }
    order = np.argsort(-p)
    top_i = int(order[0])
    second_p = float(p[order[1]]) if p.size > 1 else 0.0
    top_p = float(p[top_i])
    margin = top_p - second_p
    ent = _shannon_entropy_natural(p)
    reasons: list[str] = []
    if top_p < tau_max:
        reasons.append(f"max_prob<{tau_max}")
    if margin < min_margin:
        reasons.append(f"margin<{min_margin}")
    if max_entropy is not None and ent > max_entropy:
        reasons.append(f"entropy>{max_entropy}")
    ok = not reasons
    if ok and not (0 <= top_i < len(CANONICAL_LABELS)):
        ok = False
        reasons.append("top_idx_oob")
    return {
        "semantic_label": (CANONICAL_LABELS[top_i] if ok else "unknown"),
        "semantic_code": (top_i if ok else CODE_UNKNOWN),
        "top_prob": top_p,
        "second_prob": second_p,
        "margin": margin,
        "entropy": ent,
        "disambiguation_passed": ok,
        "disambiguation_reason": ("ok" if ok else "; ".join(reasons)),
    }


# ---------------------------------------------------------------------------
#  Auth Hugging Face
# ---------------------------------------------------------------------------

def _bootstrap_hf_auth(args: object) -> None:
    """Evita prompts interactivos del Hub cuando el token va por CLI o env."""
    cli = str(getattr(args, "hf_token", "") or "").strip()
    if cli:
        os.environ["HF_TOKEN"] = cli
        os.environ["HUGGING_FACE_HUB_TOKEN"] = cli
        os.environ.setdefault("HF_HUB_DISABLE_INTERACTIVE_PROMPTS", "1")

    tok = cli or str(os.environ.get("HF_TOKEN") or "").strip()
    tok = tok or str(os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not tok:
        return
    try:
        from huggingface_hub import login as hf_login

        hf_login(token=tok, add_to_git_credential=False)
    except Exception as e:
        print(
            f"[semantics] Aviso: huggingface_hub.login falló ({e}); se sigue con token en env.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
#  Semántica por chunk
# ---------------------------------------------------------------------------

def run_chunk_semantics(
    chunk_dir: Path,
    clf: Any,
    *,
    vlm_backend: str,
    prompt_variant_id: str,
    output_json: Path,
    emit_image_embedding: bool,
    tau_max_prob: float,
    min_margin: float,
    max_entropy: float | None,
    quiet: bool,
    persist_json: bool = True,
    continue_on_inference_error: bool = False,
) -> dict[str, Any]:
    chunk_dir = _validate_chunk_dir(chunk_dir)
    chunk_name = chunk_dir.name
    progress_label = chunk_name or str(chunk_dir)
    texts = list(clf.texts)
    frames_dir = chunk_dir / "frames"

    plan: list[dict[str, Any]] = []
    for row in _load_frames_meta(chunk_dir / "frames_meta.json"):
        if not str(row.get("image_key") or "").strip():
            continue
        plan.append(row)

    total = len(plan)
    if not quiet:
        print(
            f"[semantics] Chunk: {chunk_dir} | variant={prompt_variant_id} | frames en meta: {total}",
            flush=True,
        )

    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    inference_failure_count = 0

    for i, row in enumerate(plan, start=1):
        image_key = str(row.get("image_key") or "").strip()
        img_path = frames_dir / image_key
        base: dict[str, Any] = {
            "chunk": chunk_name,
            "sample_idx": row.get("sample_idx"),
            "image_key": image_key,
            "image_path": str(img_path.resolve()),
            "left_wrist_visible": bool(row.get("left_wrist_visible")),
            "right_wrist_visible": bool(row.get("right_wrist_visible")),
            "visible_wrists_count": row.get("visible_wrists_count"),
        }

        def _progress(phase: str) -> None:
            if quiet:
                return
            elapsed = time.perf_counter() - t0
            eta = (elapsed / i) * (total - i) if i > 0 else None
            _session_progress_line(
                progress_label, i, total, chunk_name, image_key, phase, elapsed_s=elapsed, eta_s=eta
            )

        if not _frame_has_wrist_visibility(row):
            records.append(
                {
                    **base,
                    "evaluable": False,
                    "vlm_applied": False,
                    "skip_reason": "no_wrist_visibility",
                    "semantic_label": None,
                    "semantic_code": CODE_NOT_EVALUABLE,
                    "vlm_vector_prompt_probs": None,
                    "vlm_vector_prompt_logits": None,
                    "gated_yes_prob": None,
                    "latency_sec": None,
                    "hands_in_shopping_basket_prob": None,
                    "hands_in_shopping_cart_prob": None,
                    "hands_in_personal_bag_prob": None,
                    "hands_in_shopping_basket_or_cart_prob": None,
                    "hands_basket_cart_or_personal_bag_prob": None,
                    "max_prob": None,
                    "second_prob": None,
                    "margin": None,
                    "entropy": None,
                    "disambiguation_passed": None,
                    "disambiguation_reason": "no_wrist_visibility",
                    "fused_image_embedding": None,
                }
            )
            _progress("omitir (sin muñeca)")
            continue

        skip_tail = {
            "semantic_label": None,
            "semantic_code": CODE_NOT_EVALUABLE,
            "vlm_vector_prompt_probs": None,
            "vlm_vector_prompt_logits": None,
            "gated_yes_prob": None,
            "latency_sec": None,
            "hands_in_shopping_basket_prob": None,
            "hands_in_shopping_cart_prob": None,
            "hands_in_personal_bag_prob": None,
            "hands_in_shopping_basket_or_cart_prob": None,
            "hands_basket_cart_or_personal_bag_prob": None,
            "max_prob": None,
            "second_prob": None,
            "margin": None,
            "entropy": None,
            "disambiguation_passed": None,
            "disambiguation_reason": None,
            "fused_image_embedding": None,
        }

        if not img_path.is_file():
            records.append(
                {
                    **base,
                    "evaluable": False,
                    "vlm_applied": False,
                    "skip_reason": "missing_image_file",
                    **skip_tail,
                    "disambiguation_reason": "missing_image_file",
                }
            )
            _progress("omitir (sin archivo)")
            continue

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            records.append(
                {
                    **base,
                    "evaluable": False,
                    "vlm_applied": False,
                    "skip_reason": "imread_failed",
                    **skip_tail,
                    "disambiguation_reason": "imread_failed",
                }
            )
            _progress("omitir (lectura)")
            continue

        if not quiet:
            elapsed_pre = time.perf_counter() - t0
            eta_pre = (elapsed_pre / (i - 1)) * (total - (i - 1)) if i > 1 and total > 0 else None
            lab = f"{vlm_backend}/{prompt_variant_id}"
            _session_progress_line(
                progress_label, i, total, chunk_name, image_key, f"{lab}…", elapsed_s=elapsed_pre, eta_s=eta_pre
            )

        try:
            out = clf.encode_frame_vectors(bgr)
        except Exception as e:
            if not continue_on_inference_error:
                raise
            inference_failure_count += 1
            records.append(
                {
                    **base,
                    "evaluable": False,
                    "vlm_applied": False,
                    "skip_reason": "vlm_inference_error",
                    "semantic_label": None,
                    "semantic_code": CODE_NOT_EVALUABLE,
                    "vlm_vector_prompt_probs": None,
                    "vlm_vector_prompt_logits": None,
                    "gated_yes_prob": None,
                    "latency_sec": None,
                    "hands_in_shopping_basket_prob": None,
                    "hands_in_shopping_cart_prob": None,
                    "hands_in_personal_bag_prob": None,
                    "hands_in_shopping_basket_or_cart_prob": None,
                    "hands_basket_cart_or_personal_bag_prob": None,
                    "max_prob": None,
                    "second_prob": None,
                    "margin": None,
                    "entropy": None,
                    "disambiguation_passed": None,
                    "disambiguation_reason": "vlm_inference_error",
                    "vlm_inference_error_message": str(e)[:500],
                    "fused_image_embedding": None,
                }
            )
            _progress("omitir (error inferencia VLM)")
            continue

        probs = np.asarray(out["fused_prompt_probs"], dtype=np.float64)
        disc = _disambiguate(
            probs,
            tau_max=tau_max_prob,
            min_margin=min_margin,
            max_entropy=max_entropy,
        )

        rec = {
            **base,
            "evaluable": True,
            "vlm_applied": True,
            "skip_reason": None,
            "semantic_label": disc["semantic_label"],
            "semantic_code": disc["semantic_code"],
            "vlm_vector_prompt_probs": probs.tolist(),
            "vlm_vector_prompt_logits": np.asarray(out["fused_prompt_logits"], dtype=np.float64).tolist(),
            "gated_yes_prob": float(out["gated_yes_prob"]),
            "latency_sec": round(float(out["latency_sec"]), 6),
            "hands_in_shopping_basket_prob": float(out["hands_in_shopping_basket_prob"]),
            "hands_in_shopping_cart_prob": float(out["hands_in_shopping_cart_prob"]),
            "hands_in_personal_bag_prob": float(out["hands_in_personal_bag_prob"]),
            "hands_in_shopping_basket_or_cart_prob": float(out["hands_in_shopping_basket_or_cart_prob"]),
            "hands_basket_cart_or_personal_bag_prob": float(out["hands_basket_cart_or_personal_bag_prob"]),
            "max_prob": disc["top_prob"],
            "second_prob": disc["second_prob"],
            "margin": disc["margin"],
            "entropy": disc["entropy"],
            "disambiguation_passed": disc["disambiguation_passed"],
            "disambiguation_reason": disc["disambiguation_reason"],
        }
        if emit_image_embedding:
            rec["fused_image_embedding"] = np.asarray(out["fused_image_embedding"], dtype=np.float64).tolist()
        else:
            rec["fused_image_embedding"] = None
        records.append(rec)

        if not quiet:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / i) * (total - i) if i > 0 and total > 0 else None
            ms = float(out["latency_sec"]) * 1000.0
            _session_progress_line(
                progress_label,
                i,
                total,
                chunk_name,
                image_key,
                f"{vlm_backend}/{prompt_variant_id} ok ({ms:.0f}ms)",
                elapsed_s=elapsed,
                eta_s=eta,
            )

    if total > 0 and not quiet:
        print(flush=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_dir": str(chunk_dir),
        "chunk_name": chunk_name,
        "vlm_backend": vlm_backend,
        "prompt_variant_id": prompt_variant_id,
        "vlm_model": clf.model_name,
        "prompt_texts_en": texts,
        "siglip_prompt_texts": texts,
        "canonical_labels": CANONICAL_LABELS,
        "siglip_prompt_labels_es": LABELS_ES,
        "semantic_code_map": {
            "not_evaluable": CODE_NOT_EVALUABLE,
            "unknown": CODE_UNKNOWN,
            **{name: i for i, name in enumerate(CANONICAL_LABELS)},
        },
        "decision_thresholds": {
            "tau_max_prob": tau_max_prob,
            "min_margin_top1_minus_top2": min_margin,
            "max_entropy": max_entropy,
            "note": "Etiqueta unknown si falla tau, margen o entropía (cuando max_entropy no es null).",
        },
        "frames": records,
    }

    if continue_on_inference_error:
        payload["vlm_inference_failure_count"] = inference_failure_count

    if persist_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    wall = time.perf_counter() - t0
    if not quiet:
        print(f"[semantics] Listo en {wall:.1f}s -> {output_json}", flush=True)
    return payload
