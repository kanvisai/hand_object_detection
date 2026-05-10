"""
Prompts SigLIP para el pipeline rc1 (copiable a otro proyecto tal cual).

Lista **fija de 7** textos en inglés; el orden debe coincidir con las etiquetas canónicas
del clasificador (`object_in_hand` … `personal_bag_deposit`) — mismo contrato que
`session_semantics` / `robbery_rules_score`.

Índice → etiqueta:
  0 object_in_hand
  1 empty_hands
  2 pockets_hidden
  3 gesture_no_object
  4 shopping_basket
  5 shopping_cart
  6 personal_bag_deposit
"""

from __future__ import annotations

# Sufijo en nombres de JSON: `<chunk>_siglip_<id>.json`
PROMPT_VARIANT_ID: str = "frames_v2_probe"

EXPECTED_PROMPTS: int = 7

RC1_PROMPT_TEXTS_EN: list[str] = [
    "A shopper holding a packaged grocery item or product box with their hands in a store aisle.",
    "Empty hands with palms open, not grasping any package, bottle or box.",
    "Hands in pockets or fully hidden, not visible holding anything.",
    "Hands gesturing or touching clothing with no product, bottle or box in hand.",
    "Hands inside a shopping basket among food or products.",
    "Hands inside a shopping cart or store trolley.",
    "Person placing or hiding an item inside a personal bag, handbag, backpack, tote, or personal plastic bag.",
]


def assert_prompts_ok() -> None:
    if len(RC1_PROMPT_TEXTS_EN) != EXPECTED_PROMPTS:
        raise RuntimeError(
            f"vlm_rc1_prompts: se esperan {EXPECTED_PROMPTS} prompts; hay {len(RC1_PROMPT_TEXTS_EN)}."
        )
