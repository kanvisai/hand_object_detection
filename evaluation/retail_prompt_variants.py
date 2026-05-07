"""
Conjuntos alternativos de prompts retail para experimentación.

Cada variante es una lista de **7** textos en inglés en el **mismo orden** que
`CANONICAL_LABELS` / `RETAIL_PROMPT_TEXTS_EN` (softmax coherente con session_semantics).

Claves usadas en `--prompt-variant` y en sufijos de fichero JSON (solo [a-z0-9_]).
"""

from __future__ import annotations

from retail_semantic_prompts import CANONICAL_LABELS, RETAIL_PROMPT_TEXTS_EN

_EXPECTED = len(CANONICAL_LABELS)

# --- Índice semántico por slot (referencia) ---
# 0 object_in_hand, 1 empty_hands, 2 pockets_hidden, 3 gesture_no_object,
# 4 shopping_basket, 5 shopping_cart, 6 personal_bag_deposit

_BASKET = RETAIL_PROMPT_TEXTS_EN[4]
_CART = RETAIL_PROMPT_TEXTS_EN[5]
# Política actual: evitar falsos positivos de "guardar en bolso personal".
# Dejamos el slot 6 como clase negativa/neutral para mantener compatibilidad de 7 clases.
_BAG_DISABLED = "No clear evidence of storing an item in a personal bag in this frame."


def _validate(variant_id: str, texts: list[str]) -> None:
    if len(texts) != _EXPECTED:
        raise ValueError(
            f"Variante {variant_id!r}: se esperan {_EXPECTED} prompts; hay {len(texts)} "
            f"(orden = CANONICAL_LABELS)."
        )


PROMPT_VARIANTS: dict[str, list[str]] = {
    # Producción / línea base actual (retail_semantic_prompts).
    "default": list(RETAIL_PROMPT_TEXTS_EN),
    # Lista recomendada para retail (escenario tienda) con foco en mano vacía/objeto/depósito.
    "retail_store_core_v1": list(RETAIL_PROMPT_TEXTS_EN),
    # Igual que `default` (nombre explícito por si migráis la línea base).
    "retail_current": list(RETAIL_PROMPT_TEXTS_EN),
    # Misma línea base que test_new_handobject_siglip_v2_frames (6 primeros) + bolso como en retail.
    "frames_v2_probe": [
        "A shopper holding a packaged grocery item or product box with their hands in a store aisle.",
        "Empty hands with palms open, not grasping any package, bottle or box.",
        "Hands in pockets or fully hidden, not visible holding anything.",
        "Hands gesturing or touching clothing with no product, bottle or box in hand.",
        "Hands inside a shopping basket among food or products.",
        "Hands inside a shopping cart or store trolley.",
        _BAG_DISABLED,
    ],
    # Refinamiento de frames_v2_probe: más contraste entre "objeto en mano" y contenedores.
    "frames_v2_probe_v2_fine": [
        "A shopper clearly holding a product, package, bottle, or boxed item in hand.",
        "Both hands visibly empty, not holding any item.",
        "Hands in pockets or not visible, with no visible item in hand.",
        "Hands touching clothes or making neutral gestures, with no item in hand.",
        "Hands clearly inside a visible shopping basket, interacting with items in the basket.",
        "Hands clearly inside a visible shopping cart/trolley, interacting with items in the cart.",
        "Hands putting or hiding an item into a personal handbag, backpack, tote, or personal plastic bag.",
    ],
    # Solo cambia la etiqueta "manos vacías" (experimento tipo usuario).
    "empty_hands_minimal": [
        RETAIL_PROMPT_TEXTS_EN[0],
        "Empty hands with no object at all.",
        RETAIL_PROMPT_TEXTS_EN[2],
        RETAIL_PROMPT_TEXTS_EN[3],
        RETAIL_PROMPT_TEXTS_EN[4],
        RETAIL_PROMPT_TEXTS_EN[5],
        RETAIL_PROMPT_TEXTS_EN[6],
    ],
    # Cuatro anclas de persona como test_new_handobject_siglip_v2.py + tres contenedores retail.
    "v2_person_four_slots": [
        "A person holding a stolen object in their hands.",
        "A person with empty hands walking.",
        "A person with hands in their pockets.",
        "A person clapping or rubbing their hands together.",
        _BASKET,
        _CART,
        _BAG_DISABLED,
    ],
    # Igual que arriba pero sin "stolen" (más adecuado a tienda).
    "v2_person_four_retail_product": [
        "A person holding a product or packaged item in their hands.",
        "A person with empty hands walking.",
        "A person with hands in their pockets.",
        "A person clapping or rubbing their hands together.",
        _BASKET,
        _CART,
        _BAG_DISABLED,
    ],
    # Frases más cortas en slots persona + mismos contenedores.
    "compact_person_slots": [
        "Hands holding a product or package.",
        "Hands clearly empty, no product.",
        "Hands hidden in pockets or out of view.",
        "Hands touching clothes or gesturing, no product in hand.",
        _BASKET,
        _CART,
        _BAG_DISABLED,
    ],
    # Variante "anti-falsos positivos": fuerza negativos ambiguos en slots no-objeto.
    # No añade una clase 8; mantiene 7 clases para no romper mapping.
    "hard_negative_ambiguous": [
        "Hands clearly holding a product, package, bottle, or boxed item.",
        "Empty hands, no object held at all.",
        "Hands not visible or hidden in pockets; no clear object in hands.",
        "Ambiguous hand motion or touching clothes, with no clear evidence of holding a product.",
        "Hands reaching into or inside a shopping basket.",
        "Hands reaching into or inside a shopping cart or trolley.",
        "Hands placing something into a handbag, backpack, tote, or personal plastic bag.",
    ],
    # Variante con lenguaje explícito tipo 'unknown' para reducir positivos espurios.
    "unknown_like_negatives": [
        "A shopper clearly carrying an item in hand (package, box, bottle, or product).",
        "No object in hands; hands are empty.",
        "Hands hidden or out of view, action uncertain.",
        "Hands visible but interaction is unclear or non-object-related.",
        _BASKET,
        _CART,
        _BAG_DISABLED,
    ],
    # Etapa A: estado de manos (evita que cesta/carro secuestren la clasificación).
    # Mantiene 7 slots por compatibilidad, pero basket/cart/bag son hard-negatives explícitos.
    "hand_state_only_v1": [
        "Hands clearly holding an item or product.",
        "Hands clearly empty, no item in either hand.",
        "Hands hidden in pockets, behind body, or not visible.",
        "Hands visible but not manipulating any item (neutral gesture or resting posture).",
        "No visible shopping basket interaction in this frame.",
        "No visible shopping cart interaction in this frame.",
        "No visible personal bag storing action in this frame.",
    ],
    # Etapa B: contexto de contenedor/depósito (para correr en segunda pasada).
    "container_context_v1": [
        "Hands holding an item, not depositing it into a container.",
        "Hands empty, no container interaction.",
        "Hands hidden or uncertain, no clear container interaction.",
        "Hands near body or shelves, but not clearly depositing into a container.",
        "Hands placing or dropping an item into a shopping basket, with basket clearly visible.",
        "Hands placing or dropping an item into a shopping cart/trolley, with cart clearly visible.",
        _BAG_DISABLED,
    ],
    # Etapa A v2 (más minimalista): objetivo principal = objeto en mano / no objeto / no claro.
    # Se mantiene formato de 7 slots por compatibilidad con CANONICAL_LABELS.
    "hand_state_only_v2": [
        "A person clearly holding any item in at least one hand.",
        "A person with both hands clearly empty, holding nothing.",
        "Hands not visible, partially occluded, or hidden in pockets.",
        "Hands visible but action ambiguous, with no clear evidence of holding an item.",
        "No clear evidence of basket interaction in this frame.",
        "No clear evidence of shopping cart interaction in this frame.",
        "No clear evidence of personal bag storing action in this frame.",
    ],
}

for _vid, _txts in PROMPT_VARIANTS.items():
    _validate(_vid, _txts)


def list_variant_ids() -> list[str]:
    return sorted(PROMPT_VARIANTS.keys())


def list_experiment_variant_ids() -> list[str]:
    """
    Variantes para corridas batch (`--prompt-variant-all`): mismas claves ordenadas,
    pero **sin repetir** listas de 7 prompts idénticas (p. ej. default ≈ retail_current ≈ frames_v2_probe).
    """
    seen: set[tuple[str, ...]] = set()
    out: list[str] = []
    for vid in sorted(PROMPT_VARIANTS.keys()):
        tup = tuple(PROMPT_VARIANTS[vid])
        if tup in seen:
            continue
        seen.add(tup)
        out.append(vid)
    return out


def get_variant_texts(variant_id: str) -> list[str]:
    """
    Resuelve el id de variante (tolerante a mayúsculas y a guiones como underscore).
    """
    key = str(variant_id or "").strip().lower().replace("-", "_")
    if not key:
        key = "default"
    if key not in PROMPT_VARIANTS:
        raise KeyError(
            f"Prompt variant desconocida: {variant_id!r}. "
            f"Disponibles: {', '.join(list_variant_ids())}"
        )
    return list(PROMPT_VARIANTS[key])
