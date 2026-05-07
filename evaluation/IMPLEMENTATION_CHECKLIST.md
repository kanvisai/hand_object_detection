# Plan de implementación: sesión → semántica SigLIP → razonador

Objetivo: pipeline desde carpeta de sesión (chunks + `frames_meta`) hasta veredicto de interacción (compra, suelta, depósito en carro, etc.), con **umbrales configurables** y trazabilidad.

---

## Fase A — Contrato de datos y extractor (`session_semantics.py`)

- Entrada: **`--chunk-dir`** = una carpeta de chunk (`frames/` + `frames_meta.json`). El orquestador llama una vez por chunk; no hay carpeta “sesión” con muchos chunks dentro.
- Salida por defecto: `<chunk-dir>/<nombre-carpeta>_semantics.json` (p. ej. `chunk_001/chunk_001_semantics.json`; `schema_version` 1.3, campos `chunk_dir` / `chunk_name`).

- [x] **A.1** Esquema JSON versionado (`schema_version`) con lista `frames` ordenada en el tiempo (chunk + `sample_idx`).
- [x] **A.2** Por frame evaluable: vectores `vlm_vector_prompt_probs` / logits, métricas `max_prob`, `second_prob`, `margin`, `entropy`.
- [x] **A.3** Etiqueta canónica por prompt (`semantic_label` + `semantic_code`) y reglas **τ (max prob) + margen** + opcional **entropía máxima** → `unknown` si no pasan.
- [x] **A.4** Frames sin muñeca visible: `evaluable: false`, código **-1** (sin inferencia SigLIP).
- [x] **A.5** CLI parametrizada (`--tau-max-prob`, `--min-margin`, `--max-entropy`, modelo, dispositivo, multicrop, salida).
- [x] **A.6** Barra de progreso / modo silencioso para sesiones largas (`--quiet`).

**Estado:** `session_semantics.py` listo para iterar umbrales con datos reales.

---

## Fase B — Razonador (`interaction_reasoner.py`)

- **Varios chunks:** `--input-json a.json b.json …` fusiona `frames` y ordena por `(chunk, sample_idx)`.

- [x] **B.1** Leer JSON de Fase A; ordenar frames de forma estable.
- [x] **B.2** Secuencia de etiquetas + suavizado temporal (ventana `--smooth-window`, ignorar `unknown` opcional).
- [x] **B.3** Detección de eventos: persistencia mínima de `object_in_hand`, de `shopping_basket` / `shopping_cart`, transiciones (runs).
- [x] **B.4** Veredictos v1 (reglas explicables): `likely_normal_purchase_or_deposit`, `object_signal_without_container_signal`, etc.
- [x] **B.5** Salida: `verdict`, `confidence`, `evidence`, secuencias raw/smoothed.
- [x] **B.6** CLI parametrizada (`--min-evaluable-frames`, `--min-run-object`, `--min-run-deposit`, …).

**Estado:** heurística v1; refinar B.4 con etiquetas reales y más estados si hace falta.

---

## Fase C — Validación y producto

- [ ] **C.1** Sesiones de prueba anotadas manualmente; tabla precisión/recall por veredicto.
- [ ] **C.2** Ajuste de τ, margen y entropía sobre validación (no fijos “a ojo”).
- [ ] **C.3** Integración con orquestador externo (entrada/salida estable, logs).
- [ ] **C.4** Tests unitarios mínimos: entropía, disambiguation, ordenación de chunks.

---

## Fase D — Mejoras opcionales

- [ ] **D.1** Prompt extra “ninguna de las anteriores” / OOD.
- [ ] **D.2** Export CSV ligero además de JSON para etiquetas por frame.
- [ ] **D.3** Unificar duplicación de bucle sesión entre `test_new_handobject_siglip_v2_frames.py` y `session_semantics` (módulo común).

---

## Orden recomendado de trabajo en sesiones de implementación

1. Cerrar **A** en una sesión de código (extractor + JSON rico).
2. Probar **A** con `--session-dir` real y revisar `unknown` vs sobre-etiquetado.
3. Implementar **B** con veredictos mínimos y `--smooth-window`.
4. Iterar **C.2** con tus vídeos.
5. Opcional **D**.

Marcar casillas arriba conforme se complete cada punto.
