"""Construye el clasificador retail según backend (siglip / openclip / mobileclip)."""

from __future__ import annotations

from typing import Any

from openclip_retail_classifier import OpenClipRetailClassifier
from siglip_retail_classifier import SiglipRetailClassifier

# OpenCLIP “clásico” (ViT-B-32 LAION)
DEFAULT_OPENCLIP_MODEL = "ViT-B-32"
DEFAULT_OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"

# MobileCLIP vía open_clip (tags válidos dependen de la versión; ver lista en error de create_model)
DEFAULT_MOBILECLIP_MODEL = "MobileCLIP2-S0"
DEFAULT_MOBILECLIP_PRETRAINED = "dfndr2b"


def normalize_hf_siglip_model_id(model_id: str) -> str:
    """
    HF exige id con organización, p. ej. google/siglip2-base-patch16-224 (SigLIP 2) o
    google/siglip-so400m-patch14-384 (SigLIP 1). Sin 'google/' el Hub no resuelve el repo.
    Si el nombre empieza por siglip / siglip2 y no lleva '/', se antepone google/.
    """
    mid = str(model_id).strip()
    if not mid or "/" in mid:
        return mid
    lo = mid.lower()
    if lo.startswith("siglip2") or lo.startswith("siglip"):
        return f"google/{mid}"
    return mid


def create_retail_vlm(
    backend: str,
    *,
    device: str,
    hf_model: str,
    openclip_model: str,
    openclip_pretrained: str,
    mobileclip_model: str,
    mobileclip_pretrained: str,
    net_th: float,
    net_margin_th: float,
    decision_mode: str,
    multicrop_mode: str,
    prompt_texts_en: list[str] | None = None,
) -> Any:
    b = str(backend).strip().lower()
    if b == "siglip":
        hf_resolved = normalize_hf_siglip_model_id(hf_model)
        return SiglipRetailClassifier(
            hf_resolved,
            device,
            prompt_texts_en=prompt_texts_en,
            net_th=net_th,
            net_margin_th=net_margin_th,
            decision_mode=decision_mode,
            multicrop_mode=multicrop_mode,
        )
    if b == "openclip":
        return OpenClipRetailClassifier(
            openclip_model or DEFAULT_OPENCLIP_MODEL,
            openclip_pretrained or DEFAULT_OPENCLIP_PRETRAINED,
            device,
            prompt_texts_en=prompt_texts_en,
            net_th=net_th,
            net_margin_th=net_margin_th,
            decision_mode=decision_mode,
            multicrop_mode=multicrop_mode,
        )
    if b == "mobileclip":
        try:
            return OpenClipRetailClassifier(
                mobileclip_model or DEFAULT_MOBILECLIP_MODEL,
                mobileclip_pretrained or DEFAULT_MOBILECLIP_PRETRAINED,
                device,
                prompt_texts_en=prompt_texts_en,
                net_th=net_th,
                net_margin_th=net_margin_th,
                decision_mode=decision_mode,
                multicrop_mode=multicrop_mode,
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                "No se pudo cargar MobileCLIP. Prueba: pip install -U open_clip_torch\n"
                "Si el nombre del modelo cambia en tu versión, usa --mobileclip-model / --mobileclip-pretrained.\n"
                f"Detalle: {e}"
            ) from e
    raise ValueError(f"Backend desconocido: {backend!r}. Usa siglip | openclip | mobileclip.")
