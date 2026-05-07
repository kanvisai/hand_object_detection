#!/usr/bin/env python3
"""
Semántica por chunk (`frames/` + `frames_meta.json`) con **varios backends VLM**: siglip, openclip, mobileclip.

Salida por defecto: `<nombre_chunk>_<backend>.json` o, con varias variantes de prompts,
`<nombre_chunk>_<backend>_<prompt_variant>.json` (p. ej. chunk_002_siglip_empty_hands_minimal.json).

Variantes de texto (`retail_prompt_variants.py`): mismo orden de 7 clases que CANONICAL_LABELS para experimentos A/B.

Por defecto SigLIP usa **SO400M 384** (`google/siglip-so400m-patch14-384`, alineado con test_new_handobject_siglip_v2_frames).
Para varios chunks: `--chunk-parent-dir` sobre la carpeta que contiene `chunk_001`, `chunk_002`, …
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from chunk_session_helpers import (  # noqa: E402
    frame_has_wrist_visibility as _frame_has_wrist_visibility,
    load_frames_meta as _load_frames_meta,
    session_progress_line as _session_progress_line,
)
from retail_semantic_prompts import CANONICAL_LABELS, CODE_UNKNOWN, LABELS_ES  # noqa: E402
from retail_prompt_variants import get_variant_texts, list_experiment_variant_ids, list_variant_ids  # noqa: E402
from vlm_factory import create_retail_vlm  # noqa: E402

SCHEMA_VERSION = "1.5"

CODE_NOT_EVALUABLE = -1


def _validate_chunk_dir(chunk_dir: Path) -> Path:
    p = chunk_dir.expanduser().resolve()
    if not p.is_dir():
        raise RuntimeError(f"No es un directorio: {p}")
    if not (p / "frames").is_dir():
        raise RuntimeError(f"Falta carpeta frames/ en: {p}")
    if not (p / "frames_meta.json").is_file():
        raise RuntimeError(f"Falta frames_meta.json en: {p}")
    return p


def _semantics_output_filename(stem: str, backend: str, pv_id: str, *, all_pv_ids: list[str]) -> str:
    """Nombre de fichero JSON bajo la carpeta del chunk."""
    safe_pv = "".join(c if c.isalnum() or c in "-_" else "_" for c in pv_id.strip())
    if len(all_pv_ids) == 1 and safe_pv == "default":
        return f"{stem}_{backend}.json"
    return f"{stem}_{backend}_{safe_pv}.json"


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
    Orden “natural” por nombre (chunk_002 antes que chunk_010).
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

        out = clf.encode_frame_vectors(bgr)
        probs = np.asarray(out["fused_prompt_probs"], dtype=np.float64)
        disc = _disambiguate(
            probs,
            tau_max=tau_max_prob,
            min_margin=min_margin,
            max_entropy=max_entropy,
        )

        rec: dict[str, Any] = {
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

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    wall = time.perf_counter() - t0
    if not quiet:
        print(f"[semantics] Listo en {wall:.1f}s -> {output_json}", flush=True)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Chunk(s) con frames/ + frames_meta.json → JSON semántico (siglip / openclip / mobileclip).",
    )
    p.add_argument(
        "--prompt-variant-all",
        action="store_true",
        help=(
            "Ejecuta todas las variantes únicas (retail_prompt_variants.list_experiment_variant_ids): "
            "omite duplicados con la misma lista de 7 prompts. Incompatible con --prompt-variant."
        ),
    )
    p.add_argument(
        "--prompt-variant",
        nargs="*",
        default=None,
        metavar="ID",
        help=(
            "Conjunto(s) de prompts retail (7 clases, ver retail_prompt_variants). "
            "Varios ids = varios JSON por chunk/backend. "
            f"Ids: {', '.join(list_variant_ids())}"
        ),
    )
    p.add_argument(
        "--chunk-dir",
        "--chunk_dir",
        default="",
        help="Una carpeta de chunk (frames/ + frames_meta.json). Alternativas: --chunk-dirs o --chunk-parent-dir.",
    )
    p.add_argument(
        "--chunk-parent-dir",
        "--chunk_parent_dir",
        default="",
        metavar="DIR",
        help=(
            "Carpeta que contiene subcarpetas chunk (chunk_001, chunk_002, …), cada una con frames/ + frames_meta.json. "
            "Alternativa a listar todas con --chunk-dirs."
        ),
    )
    p.add_argument(
        "--chunk-dirs",
        "--chunk_dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="Varias carpetas de chunk explícitas. Alternativa: --chunk-parent-dir.",
    )
    p.add_argument(
        "--output-json",
        default="",
        help=(
            "Solo si hay un chunk y un solo --vlm-backend: ruta del JSON. "
            "Si no, salida <chunk>/<nombre>_<backend>.json por cada combinación."
        ),
    )
    p.add_argument(
        "--vlm-backend",
        nargs="*",
        default=["siglip"],
        choices=["siglip", "openclip", "mobileclip"],
        metavar="NAME",
        help="siglip (HF), openclip (open_clip_torch), mobileclip (MobileCLIP en open_clip). Varias = varios JSON por chunk.",
    )
    p.add_argument("--device", default="cpu", help="cpu o cuda:0.")
    p.add_argument(
        "--hf-token",
        "--hf_token",
        dest="hf_token",
        default="",
        metavar="TOKEN",
        help=(
            "Token Hugging Face (solo SigLIP). Equivale a HF_TOKEN; útil si lanzas el script desde el IDE "
            "y no hereda el export del terminal."
        ),
    )
    p.add_argument(
        "--vlm-model",
        default="google/siglip-so400m-patch14-384",
        help=(
            "Solo backend siglip: id HF del modelo. Por defecto SO400M 384 (como test_new_handobject_siglip_v2_frames). "
            "Alternativa más rápida: google/siglip2-base-patch16-224. "
            "Sin prefijo 'google/' se añade si el nombre empieza por siglip o siglip2."
        ),
    )
    p.add_argument(
        "--openclip-model",
        default="ViT-B-32",
        help="Backend openclip: nombre modelo open_clip (ej. ViT-B-32).",
    )
    p.add_argument(
        "--openclip-pretrained",
        default="laion2b_s34b_b79k",
        help="Backend openclip: checkpoint pretrained open_clip.",
    )
    p.add_argument(
        "--mobileclip-model",
        default="MobileCLIP2-S0",
        help="Backend mobileclip: nombre en open_clip (ajusta si tu versión usa otro id).",
    )
    p.add_argument(
        "--mobileclip-pretrained",
        default="dfndr2b",
        help="Backend mobileclip: tag pretrained open_clip (p. ej. dfndr2b; si falla, el error suele listar tags disponibles).",
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
        help="Si >= 0, rechazar si entropía natural > este valor.",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def _bootstrap_hf_auth(args: argparse.Namespace) -> None:
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
        print(f"[semantics] Aviso: huggingface_hub.login falló ({e}); se sigue con token en env.", file=sys.stderr)


def main() -> None:
    args = build_arg_parser().parse_args()
    _bootstrap_hf_auth(args)
    single = str(args.chunk_dir or "").strip()
    multi = args.chunk_dirs
    parent_s = str(getattr(args, "chunk_parent_dir", "") or "").strip()
    out_s = str(args.output_json or "").strip()
    backends: list[str] = list(args.vlm_backend or [])
    if not backends:
        backends = ["siglip"]

    mode_n = sum(1 for x in (single, multi is not None, parent_s) if x)
    if mode_n != 1:
        raise SystemExit(
            "Indica exactamente uno de:\n"
            "  --chunk-dir DIR\n"
            "  --chunk-dirs DIR [DIR2 …]\n"
            "  --chunk-parent-dir DIR   (carpeta que contiene chunk_001, chunk_002, …)"
        )
    if multi is not None:
        chunk_dirs = [Path(p).expanduser().resolve() for p in multi]
        if out_s:
            raise SystemExit("Con --chunk-dirs no uses --output-json; cada chunk escribe sus JSON por backend.")
    elif parent_s:
        try:
            chunk_dirs = discover_chunk_dirs_under_parent(Path(parent_s))
        except RuntimeError as e:
            raise SystemExit(str(e)) from e
        if not chunk_dirs:
            raise SystemExit(
                f"No hay subcarpetas válidas (frames/ + frames_meta.json) bajo:\n  {Path(parent_s).resolve()}"
            )
        if out_s:
            raise SystemExit("Con --chunk-parent-dir no uses --output-json; cada chunk escribe su JSON en su carpeta.")
        if not args.quiet:
            print(
                f"[semantics] --chunk-parent-dir → {len(chunk_dirs)} chunk(s): "
                f"{', '.join(p.name for p in chunk_dirs[:8])}"
                f"{'…' if len(chunk_dirs) > 8 else ''}",
                flush=True,
            )
    else:
        chunk_dirs = [Path(single).expanduser().resolve()]

    if len(backends) > 1 and out_s:
        raise SystemExit("Con varios --vlm-backend no uses --output-json (se genera un archivo por backend).")
    if len(chunk_dirs) > 1 and out_s:
        raise SystemExit("Con varios --chunk-dirs no uses --output-json.")

    all_pv_flag = bool(getattr(args, "prompt_variant_all", False))
    pv_arg = getattr(args, "prompt_variant", None)
    if all_pv_flag and pv_arg:
        raise SystemExit("No combines --prompt-variant-all con --prompt-variant.")
    if all_pv_flag:
        prompt_variants = list_experiment_variant_ids()
        if not args.quiet:
            print(
                f"[semantics] --prompt-variant-all → {len(prompt_variants)} variantes únicas: "
                f"{', '.join(prompt_variants)}",
                flush=True,
            )
    else:
        prompt_variants = list(pv_arg) if pv_arg else ["default"]
        if not prompt_variants:
            prompt_variants = ["default"]

    if len(prompt_variants) > 1 and out_s:
        raise SystemExit("Con varios --prompt-variant no uses --output-json.")

    max_ent = float(args.max_entropy)
    max_entropy = None if max_ent < 0 else max_ent

    use_custom_output_json = (
        len(backends) == 1 and len(chunk_dirs) == 1 and len(prompt_variants) == 1 and bool(out_s)
    )

    for pvi, pv_id in enumerate(prompt_variants):
        variant_texts = get_variant_texts(pv_id)
        for bi, backend in enumerate(backends):
            print(
                f"[semantics] Prompt variant {pvi + 1}/{len(prompt_variants)} ({pv_id}) | "
                f"backend {bi + 1}/{len(backends)}: {backend} — cargando modelo…",
                flush=True,
            )
            clf = create_retail_vlm(
                backend,
                device=str(args.device),
                hf_model=str(args.vlm_model),
                openclip_model=str(args.openclip_model),
                openclip_pretrained=str(args.openclip_pretrained),
                mobileclip_model=str(args.mobileclip_model),
                mobileclip_pretrained=str(args.mobileclip_pretrained),
                net_th=float(args.net_th),
                net_margin_th=float(args.net_margin_th),
                decision_mode=str(args.decision_mode),
                multicrop_mode=str(args.multicrop_mode),
                prompt_texts_en=variant_texts,
            )
            print(f"[semantics] {pv_id} / {backend} listo en {clf.device} ({clf.model_name})", flush=True)

            for idx, chunk_dir in enumerate(chunk_dirs):
                if len(chunk_dirs) > 1 and not args.quiet:
                    print(f"[semantics] Chunk {idx + 1}/{len(chunk_dirs)}: {chunk_dir}", flush=True)
                chunk_stem = chunk_dir.name or "chunk"
                if use_custom_output_json:
                    output_json = Path(out_s).expanduser().resolve()
                else:
                    fname = _semantics_output_filename(
                        chunk_stem, backend, pv_id, all_pv_ids=prompt_variants
                    )
                    output_json = chunk_dir / fname
                run_chunk_semantics(
                    chunk_dir,
                    clf,
                    vlm_backend=backend,
                    prompt_variant_id=pv_id,
                    output_json=output_json,
                    emit_image_embedding=bool(args.emit_image_embedding),
                    tau_max_prob=float(args.tau_max_prob),
                    min_margin=float(args.min_margin),
                    max_entropy=max_entropy,
                    quiet=bool(args.quiet),
                )


if __name__ == "__main__":
    main()
