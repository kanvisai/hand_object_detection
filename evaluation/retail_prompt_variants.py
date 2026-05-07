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
_BAG = RETAIL_PROMPT_TEXTS_EN[6]


def _validate(variant_id: str, texts: list[str]) -> None:
    if len(texts) != _EXPECTED:
        raise ValueError(
            f"Variante {variant_id!r}: se esperan {_EXPECTED} prompts; hay {len(texts)} "
            f"(orden = CANONICAL_LABELS)."
        )


PROMPT_VARIANTS: dict[str, list[str]] = {
    # Producción / línea base actual (retail_semantic_prompts).
    "default": list(RETAIL_PROMPT_TEXTS_EN),
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
        _BAG,
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
        _BAG,
    ],
    # Igual que arriba pero sin "stolen" (más adecuado a tienda).
    "v2_person_four_retail_product": [
        "A person holding a product or packaged item in their hands.",
        "A person with empty hands walking.",
        "A person with hands in their pockets.",
        "A person clapping or rubbing their hands together.",
        _BASKET,
        _CART,
        _BAG,
    ],
    # Frases más cortas en slots persona + mismos contenedores.
    "compact_person_slots": [
        "Hands holding a product or package.",
        "Hands clearly empty, no product.",
        "Hands hidden in pockets or out of view.",
        "Hands touching clothes or gesturing, no product in hand.",
        _BASKET,
        _CART,
        _BAG,
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
        _BAG,
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
