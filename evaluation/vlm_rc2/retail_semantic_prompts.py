"""
Textos SigLIP (fuente única). Pensados para **recorte bbox de persona**: sin lugar concreto
(supermercado, farmacia, etc.) ni tipo de artículo concreto (botella, etc.).

Orden fijo = índices en softmax, CANONICAL_LABELS y JSON de salida.
Inglés para SigLIP SO400M; LABELS_ES solo para documentación / UI.
"""

from __future__ import annotations

# --- Índices (no cambiar orden sin actualizar CANONICAL_LABELS y razonador) ---
IDX_OBJECT_IN_HAND = 0
IDX_EMPTY_HANDS = 1
IDX_POCKETS_HIDDEN = 2
IDX_GESTURE_NO_OBJECT = 3
IDX_SHOPPING_BASKET = 4
IDX_SHOPPING_CART = 5
IDX_PERSONAL_BAG = 6

RETAIL_PROMPT_TEXTS_EN: list[str] = [
    "A shopper holding any item or product in hand, regardless of item type.",
    "Hands clearly empty, person not holding any item.",
    "Hands moving toward or into pockets/waistband, as if hiding or storing an item on the body.",
    "Person standing or moving casually with no item in hand and no clear container interaction.",
    "Person placing or dropping an item into a shopping basket, or hands clearly inside the basket.",
    "Person carrying, pushing, or interacting with a shopping cart/trolley, including placing an item into it.",
    "Person placing or hiding an item inside a personal bag, handbag, backpack, tote, or personal plastic bag.",
]

CANONICAL_LABELS: list[str] = [
    "object_in_hand",
    "empty_hands",
    "pockets_hidden",
    "gesture_no_object",
    "shopping_basket",
    "shopping_cart",
    "personal_bag_deposit",
]

LABELS_ES: list[str] = [
    "producto en las manos",
    "manos vacías (sin producto)",
    "manos en bolsillos u ocultas",
    "ropa o gestos, sin producto en la mano",
    "manos en cesta portátil",
    "manos en carrito",
    "guardando algo en bolso, mochila o bolsa personal",
]

CODE_UNKNOWN = 7
