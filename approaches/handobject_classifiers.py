"""
Clasificadores mano-con-objeto (SÍ/NO) para distintos VLMs / CLIP.
Usados por test_new_handobject_*.py junto con handobject_shared.run_pipeline.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import transformers
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError("Instala transformers: pip install transformers pillow") from e

AutoModelForVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
AutoModelForImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM", None)

try:
    from transformers.models.florence2.modeling_florence2 import Florence2ForConditionalGeneration
except ImportError:  # pragma: no cover
    Florence2ForConditionalGeneration = None  # type: ignore[misc, assignment]

from hf_model_paths import resolve_hf_model_ref


def _try_remove_broken_transformers_remote_code_cache(exc: BaseException) -> bool:
    """
    trust_remote_code descarga modulos Python bajo ~/.cache/huggingface/modules/transformers_modules/<hash>/.
    Si la descarga queda a medias, falta p.ej. rope.py y from_pretrained falla con FileNotFoundError.
    Borramos solo ese hash y el siguiente intento vuelve a bajar el codigo desde el Hub.
    """
    if not isinstance(exc, FileNotFoundError):
        return False
    msg = str(exc)
    if "transformers_modules" not in msg:
        return False
    m = re.search(r"transformers_modules[/\\]([^/\\]+)[/\\]", msg)
    if not m:
        return False
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    bad_dir = hf_home / "modules" / "transformers_modules" / m.group(1)
    if bad_dir.is_dir():
        shutil.rmtree(bad_dir)
        return True
    return False


def _openclip_arch_for_hf_snapshot(model_ref: str) -> str | None:
    """
    Si model_ref describe un modelo MobileCLIP en Hugging Face, devuelve el nombre de arquitectura
    esperado por open_clip (p. ej. MobileCLIP-S1). None si no es MobileCLIP (p. ej. otro Hub OpenCLIP).
    """
    s = str(model_ref).replace("hf-hub:", "").strip().lower()
    if "mobileclip" not in s:
        return None
    if "mobileclip-b" in s or "mobileclip_b" in s:
        return "MobileCLIP-B"
    for i in (0, 1, 2):
        tok = f"s{i}"
        if (
            f"mobileclip-{tok}" in s
            or f"s{i}-openclip" in s
            or f"/mobileclips{tok}" in s
            or f"-s{i}-" in s
        ):
            return f"MobileCLIP-S{i}"
    return "MobileCLIP-S1"


# Punteros Git LFS ~134 B; pesos reales MobileCLIP ~340 MB.
_MIN_OPENCLIP_WEIGHT_BYTES = 512 * 1024


def _open_clip_weights_file_from_hub_snapshot(snapshot_dir: Path) -> Path | None:
    """
    Snapshots HF (apple/MobileCLIP-*-OpenCLIP): pesos en archivos sueltos.

    Preferir **open_clip_pytorch_model.bin** antes que `.safetensors`:
    versiones antiguas de open_clip cargan cualquier ruta con `torch.load`; los safetensors
    no son pickle y provocan `invalid load key`. El .bin va con `_torch_load_open_clip_bin_checkpoints_ok`
    (weights_only=False, PyTorch 2.6+).

    Si solo existe .safetensors, hace falta open_clip reciente (rama safetensors en load_state_dict)
    o actualizar open-clip-torch.
    """
    snap = snapshot_dir.expanduser().resolve()
    if not snap.is_dir():
        return None
    m = _MIN_OPENCLIP_WEIGHT_BYTES

    named_bin = snap / "open_clip_pytorch_model.bin"
    if named_bin.is_file() and named_bin.stat().st_size >= m:
        return named_bin

    bin_ok = [p for p in snap.glob("*.bin") if p.is_file() and p.stat().st_size >= m]
    if bin_ok:
        return max(bin_ok, key=lambda p: p.stat().st_size)

    named_st = snap / "open_clip_model.safetensors"
    if named_st.is_file() and named_st.stat().st_size >= m:
        return named_st

    st_ok = [p for p in snap.glob("*.safetensors") if p.is_file() and p.stat().st_size >= m]
    if st_ok:
        return max(st_ok, key=lambda p: p.stat().st_size)

    return None


@contextmanager
def _torch_load_open_clip_bin_checkpoints_ok():
    """
    PyTorch >= 2.6 usa weights_only=True por defecto en torch.load; los .bin antiguos
    de open_clip fallan (pickle). Checkpoints del Hub son de fuente fija (HF/Apple).
    """
    orig = torch.load

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return orig(*args, **kwargs)

    torch.load = _wrapped  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.load = orig  # type: ignore[method-assign]


def _internlm_past_seq_length(past_key_values: Any) -> int:
    """Longitud de secuencia en cache; tuple legacy o DynamicCache (K/V por capa pueden ser None al inicio)."""
    if past_key_values is None:
        return 0
    gs = getattr(past_key_values, "get_seq_length", None)
    if callable(gs):
        try:
            return int(gs())
        except Exception:
            pass
    try:
        layer0 = past_key_values[0]
        k0 = layer0[0] if isinstance(layer0, (tuple, list)) else layer0
        if k0 is not None and hasattr(k0, "shape") and len(k0.shape) > 2:
            return int(k0.shape[2])
    except Exception:
        pass
    return 0


def _patch_internlm_prepare_inputs_for_generation_compat(model: Any) -> None:
    """
    InternLM2 remoto indexa past_key_values[0][0].shape; con Cache HF puede ser None.
    """
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    cls = lm.__class__
    if getattr(cls, "_kanvis_internlm_prepare_patch", False):
        return

    def _wrapped(
        self: Any,
        input_ids: Any,
        past_key_values: Any = None,
        attention_mask: Any = None,
        inputs_embeds: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if past_key_values is not None and _internlm_past_seq_length(past_key_values) == 0:
            past_key_values = None

        past_length = _internlm_past_seq_length(past_key_values)
        if past_length > 0:
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_length > 0:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    cls.prepare_inputs_for_generation = _wrapped  # type: ignore[method-assign]
    cls._kanvis_internlm_prepare_patch = True  # type: ignore[attr-defined]


def _patch_internlm2_model_forward_normalize_empty_cache(model: Any) -> None:
    inner = getattr(getattr(model, "language_model", None), "model", None)
    if inner is None:
        return
    cls = inner.__class__
    if getattr(cls, "_kanvis_internlm2_forward_patch", False):
        return
    _orig = cls.forward

    def _wrapped(
        self: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        **kwargs: Any,
    ) -> Any:
        if past_key_values is not None and _internlm_past_seq_length(past_key_values) == 0:
            past_key_values = None
        return _orig(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    cls.forward = _wrapped  # type: ignore[method-assign]
    cls._kanvis_internlm2_forward_patch = True  # type: ignore[attr-defined]


def _patch_internvl_language_model_generate(model: Any) -> None:
    """
    Transformers 5.x separa GenerationMixin del PreTrainedModel; el InternLM2 remoto del Hub solo hereda PreTrainedModel.
    Sin mezcla con GenerationMixin, generate() ausente y chat() falla. Re-clasamos la instancia del LLM.
    MiniCPM-V usa el submodulo llm en lugar de language_model.
    """
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    lm = getattr(model, "language_model", None) or getattr(model, "llm", None)
    if lm is None or lm.__class__.__name__.endswith("__GenFix"):
        return
    cls = lm.__class__
    if issubclass(cls, GenerationMixin):
        return
    new_cls = type(
        f"{cls.__name__}__GenFix",
        (cls, GenerationMixin),
        {"__module__": getattr(cls, "__module__", None) or "transformers"},
    )
    lm.__class__ = new_cls
    if getattr(lm, "generation_config", None) is None:
        try:
            lm.generation_config = GenerationConfig.from_model_config(lm.config)
        except Exception:
            lm.generation_config = GenerationConfig()


def _patch_minicpm_cache_get_usable_length_shims() -> None:
    """
    Codigo remoto (MiniCPM-V, Phi-3.5-V) usa get_usable_length / seen_tokens / get_max_length en Cache;
    transformers 4.5x usa otra API en DynamicCache.
    """
    try:
        from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
    except Exception:
        return

    def cache_get_usable_length(self: Any, kv_seq_len: int = 0, layer_idx: int = 0) -> int:  # noqa: ARG002
        # MiniCPMDecoderLayer recibe el Cache completo como past_key_value por capa (Hub).
        return int(self.get_seq_length(layer_idx))

    def layer_get_usable_length(self: Any, kv_seq_len: int = 0, layer_idx: int = 0) -> int:  # noqa: ARG002
        return int(self.get_seq_length())

    if not getattr(DynamicCache, "_kanvis_usable_len_shim", False):
        DynamicCache.get_usable_length = cache_get_usable_length  # type: ignore[misc]
        DynamicCache._kanvis_usable_len_shim = True  # type: ignore[attr-defined]

    # microsoft/Phi-3.5-vision (Hub): prepare_inputs_for_generation usa .seen_tokens y .get_max_length().
    if not getattr(DynamicCache, "_kanvis_phi_cache_shim", False):

        def _dyn_seen_tokens(self: Any) -> int:
            return int(self.get_seq_length())

        DynamicCache.seen_tokens = property(_dyn_seen_tokens)  # type: ignore[misc]

        def _dyn_get_max_length(self: Any, layer_idx: int = 0) -> int:  # noqa: ARG002
            m = int(self.get_max_cache_shape(layer_idx))
            return m if m > 0 else -1

        DynamicCache.get_max_length = _dyn_get_max_length  # type: ignore[misc]
        DynamicCache._kanvis_phi_cache_shim = True  # type: ignore[attr-defined]

    for cls in (DynamicLayer, DynamicSlidingWindowLayer):
        if getattr(cls, "_kanvis_layer_usable_len_shim", False):
            continue
        cls.get_usable_length = layer_get_usable_length  # type: ignore[misc]
        cls._kanvis_layer_usable_len_shim = True  # type: ignore[attr-defined]


def _minicpm_fix_past_key_values(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None
    pk = past_key_values
    if hasattr(pk, "to_legacy_cache"):
        pk = pk.to_legacy_cache()
    try:
        if pk[0][0] is None:
            return None
    except (TypeError, IndexError, AttributeError):
        return None
    return pk


def _patch_minicpm_llm_prepare_legacy_cache(model: Any) -> None:
    """modeling_minicpm: Cache.seen_tokens (obsoleto) y K/V None al inicio; alinear con legacy + caché vacia = None."""
    lm = getattr(model, "llm", None)
    if lm is None:
        return
    cls = lm.__class__
    if getattr(cls, "_kanvis_minicpm_prepatch", False):
        return
    _orig = cls.prepare_inputs_for_generation

    def _wrapped(
        self: Any,
        input_ids: Any,
        past_key_values: Any = None,
        attention_mask: Any = None,
        inputs_embeds: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        pk = _minicpm_fix_past_key_values(past_key_values)
        return _orig(
            self,
            input_ids,
            past_key_values=pk,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    cls.prepare_inputs_for_generation = _wrapped  # type: ignore[method-assign]
    cls._kanvis_minicpm_prepatch = True  # type: ignore[attr-defined]


def _patch_minicpmv_decode_no_cache(model: Any) -> None:
    """Evita estados KV inconsistentes en bucle de generacion (inputs_embeds + beam)."""
    cls = model.__class__
    if getattr(cls, "_kanvis_minicpm_decode_patch", False):
        return
    _orig = cls._decode

    def _wrapped(self: Any, inputs_embeds: Any, tokenizer: Any, **kwargs: Any) -> Any:
        kwargs["use_cache"] = False
        return _orig(self, inputs_embeds, tokenizer, **kwargs)

    cls._decode = _wrapped  # type: ignore[method-assign]
    cls._kanvis_minicpm_decode_patch = True  # type: ignore[attr-defined]


def _patch_internvl_missing_tied_keys_shim() -> None:
    """
    InternVL en el Hub no llama a post_init(); en transformers 5.5+ falta all_tied_weights_keys
    y _finalize_model_loading falla. Un diccionario vacio basta para el bucle que filtra tied.
    En 4.x el hook se llama _move_missing_keys_from_meta_to_cpu; en 5.x, *_to_device.
    """
    mu = transformers.modeling_utils
    cls = mu.PreTrainedModel
    if getattr(cls, "_kanvis_internvl_tied_shim", False):
        return
    hook_name = "_move_missing_keys_from_meta_to_device"
    if not hasattr(cls, hook_name):
        hook_name = "_move_missing_keys_from_meta_to_cpu"
    if not hasattr(cls, hook_name):
        return
    _orig = getattr(cls, hook_name)

    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig(self, *args, **kwargs)

    setattr(cls, hook_name, _wrapped)  # type: ignore[method-assign]
    cls._kanvis_internvl_tied_shim = True


class YesNoTextMixin:
    @staticmethod
    def _yes_no_from_text(text: str) -> float | None:
        t = text.strip().lower()
        tokens = [x.strip(".,:;!?()[]{}\"'") for x in t.split()]
        if "no" in tokens:
            return 0.0
        if "yes" in tokens or "sí" in tokens or "si" in tokens:
            return 1.0
        return None

    @staticmethod
    def _extract_yes_no_anywhere(text: str) -> float | None:
        t = text.lower()
        if re.search(r"\bno\b", t):
            return 0.0
        if re.search(r"\byes\b", t) or re.search(r"\bs[ií]\b", t):
            return 1.0
        return None

    @staticmethod
    def _yes_no_from_answer_tail(text: str) -> float | None:
        """Ultimo YES/NO al final (p. ej. el modelo repite la pregunta y cierra en YES/NO)."""
        m = re.search(r"(?i)\b(yes|no)\s*\.?\s*$", text.strip())
        if m:
            return 1.0 if m.group(1).lower() == "yes" else 0.0
        return None

    @staticmethod
    def _yes_no_leading_letter_yn(text: str) -> float | None:
        """Primera palabra Y o N (prompts tipo 'reply with Y or N'; no confunde con 'YES')."""
        m = re.match(r"(?i)^\s*([yn])\b", text.strip())
        if not m:
            return None
        return 1.0 if m.group(1).lower() == "y" else 0.0

    @staticmethod
    def _yes_no_first_word(text: str) -> float | None:
        """Primera palabra Yes/No/Y/N (prioridad sobre una sola letra)."""
        s = text.strip()
        if not s:
            return None
        raw = s.split()[0]
        w = re.sub(r"^[^\w]+|[^\w]+$", "", raw).lower()
        if w in ("yes", "y"):
            return 1.0
        if w in ("no", "n"):
            return 0.0
        return None


class Qwen2VLHandClassifier(YesNoTextMixin):
    """Qwen2-VL (chat template + vision2seq)."""

    experiment_backend = "qwen2_vl"
    _name_tokens: tuple[str, ...] = ("qwen2-vl",)

    def __init__(self, model_name: str, device: str, prompt: str) -> None:
        low = model_name.lower()
        if not any(t in low for t in self._name_tokens):
            raise RuntimeError(
                f"Modelo no reconocido para esta clase; se esperaba alguno de: {self._name_tokens}"
            )
        self.prompt = prompt
        self.prompt_fallback = "Answer with one word only: YES or NO."
        self.last_answer_text = ""
        self.last_prompt_used = ""
        self.model_name = model_name
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        load_id = resolve_hf_model_ref(model_name)
        self.processor = AutoProcessor.from_pretrained(load_id, trust_remote_code=True)
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
        }
        vlm_auto_cls = AutoModelForVision2Seq or AutoModelForImageTextToText
        if vlm_auto_cls is None:
            raise RuntimeError("Actualiza transformers (AutoModelForVision2Seq).")
        self.model = vlm_auto_cls.from_pretrained(load_id, **model_kwargs).to(self.device)
        self.model.eval()
        self.last_debug = ""

    def _generate_timed(
        self,
        inputs: Any,
        max_new_tokens: int,
        frame_index: int | None,
        vlm_calls: list[dict[str, Any]] | None,
        stage: str,
    ) -> Any:
        t0 = time.perf_counter()
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        latency = time.perf_counter() - t0
        if frame_index is not None and vlm_calls is not None:
            vlm_calls.append(
                {
                    "frame_prompt": frame_index,
                    "frame_response": frame_index,
                    "latency_sec": round(latency, 6),
                    "stage": stage,
                    "note": "Mismo frame: inferencia sincrona (no avanza el video hasta terminar).",
                }
            )
        return out_ids

    def predict_yes_prob(
        self,
        bgr: np.ndarray,
        frame_index: int | None = None,
        vlm_calls: list[dict[str, Any]] | None = None,
    ) -> float:
        if not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError("El processor debe exponer apply_chat_template.")
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        prompt_main = self.prompt
        self.last_prompt_used = prompt_main
        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_main},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            out_ids = self._generate_timed(
                inputs, max_new_tokens=8, frame_index=frame_index, vlm_calls=vlm_calls, stage="primary"
            )
            in_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
            gen_ids = out_ids[:, in_len:] if in_len > 0 else out_ids
            out_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            score = self._yes_no_from_text(out_text)

            if score is None:
                full_text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                out_text = full_text
                score = self._extract_yes_no_anywhere(out_text)

            if score is None:
                prompt2 = self.prompt_fallback
                messages2 = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt2},
                        ],
                    }
                ]
                text2 = self.processor.apply_chat_template(
                    messages2, tokenize=False, add_generation_prompt=True
                )
                inputs2 = self.processor(
                    text=[text2],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                out_ids2 = self._generate_timed(
                    inputs2,
                    max_new_tokens=4,
                    frame_index=frame_index,
                    vlm_calls=vlm_calls,
                    stage="fallback_prompt",
                )
                in_len2 = int(inputs2["input_ids"].shape[1]) if "input_ids" in inputs2 else 0
                gen_ids2 = out_ids2[:, in_len2:] if in_len2 > 0 else out_ids2
                out_text = self.processor.batch_decode(gen_ids2, skip_special_tokens=True)[0].strip()
                score = self._extract_yes_no_anywhere(out_text)

            self.last_answer_text = out_text.strip()
        return float(score) if score is not None else 0.0


class Qwen3VLHandClassifier(Qwen2VLHandClassifier):
    """Qwen3-VL: misma ruta que Qwen2 si el chat template es compatible."""

    experiment_backend = "qwen3_vl"
    _name_tokens = ("qwen3-vl", "qwen3_vl")


class GenericChatVLMClassifier(YesNoTextMixin):
    """VLM con chat tipo Qwen2 (imagen + texto) y generate; para InternVL, SmolVLM, PaliGemma, etc."""

    _INTERNVL_BACKENDS = frozenset({"internvl2", "internvl3"})
    # openbmb/MiniCPM-V-*: tokenizer remoto importa peft; inferencia oficial = model.chat(image, msgs, context, tokenizer).
    _MINICPM_BACKENDS = frozenset({"minicpm_v20", "minicpm_v26"})
    # microsoft/Phi-3.5-vision-instruct: Phi3VConfig no esta en AutoModelForVision2Seq; el Hub pide flash_attn si el config usa flash.
    _PHI35_VISION_BACKENDS = frozenset({"phi35_vision"})

    def __init__(
        self,
        model_name: str,
        device: str,
        prompt: str,
        *,
        backend_name: str = "generic_chat_vlm",
    ) -> None:
        self.experiment_backend = backend_name
        self.prompt = prompt
        self.prompt_fallback = "Answer with one word only: YES or NO."
        self.last_answer_text = ""
        self.last_prompt_used = ""
        self.model_name = model_name
        load_id = resolve_hf_model_ref(model_name)
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        dt = torch.float16 if self.device.type == "cuda" else torch.float32
        model_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dt}
        # InternVL2/3: tokenizer + CLIPImageProcessor + model.chat(); chat_template solo admite content str.
        if backend_name in self._INTERNVL_BACKENDS:
            _patch_internvl_missing_tied_keys_shim()
            model_kwargs["low_cpu_mem_usage"] = False
            self.processor = AutoTokenizer.from_pretrained(load_id, trust_remote_code=True)
            self._internvl_image_processor = AutoImageProcessor.from_pretrained(
                load_id, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(load_id, **model_kwargs).to(self.device)
            _patch_internvl_language_model_generate(self.model)
            _patch_internlm_prepare_inputs_for_generation_compat(self.model)
            _patch_internlm2_model_forward_normalize_empty_cache(self.model)
        elif backend_name in self._MINICPM_BACKENDS:
            _patch_minicpm_cache_get_usable_length_shims()
            self.processor = AutoProcessor.from_pretrained(load_id, trust_remote_code=True)
            if AutoModelForCausalLM is None:
                raise RuntimeError("Actualiza transformers (AutoModelForCausalLM).")
            self.model = AutoModelForCausalLM.from_pretrained(load_id, **model_kwargs).to(self.device)
            _patch_internvl_language_model_generate(self.model)
            _patch_minicpm_llm_prepare_legacy_cache(self.model)
            _patch_minicpmv_decode_no_cache(self.model)
        elif backend_name in self._PHI35_VISION_BACKENDS:
            _patch_minicpm_cache_get_usable_length_shims()
            self.processor = AutoProcessor.from_pretrained(load_id, trust_remote_code=True)
            if AutoModelForCausalLM is None:
                raise RuntimeError("Actualiza transformers (AutoModelForCausalLM).")
            # El config del Hub suele traer _attn_implementation=flash_attention_2 sin flash_attn instalado.
            phi_cfg = AutoConfig.from_pretrained(load_id, trust_remote_code=True)
            phi_cfg._attn_implementation = "eager"
            self.model = AutoModelForCausalLM.from_pretrained(
                load_id,
                config=phi_cfg,
                trust_remote_code=True,
                torch_dtype=model_kwargs.get("torch_dtype"),
            ).to(self.device)
        else:
            self.processor = AutoProcessor.from_pretrained(load_id, trust_remote_code=True)
            vlm_cls = AutoModelForVision2Seq or AutoModelForImageTextToText
            if vlm_cls is None:
                raise RuntimeError("Actualiza transformers.")
            try:
                self.model = vlm_cls.from_pretrained(load_id, **model_kwargs).to(self.device)
            except Exception:
                if AutoModelForCausalLM is None:
                    raise
                self.model = AutoModelForCausalLM.from_pretrained(load_id, **model_kwargs).to(self.device)
        self.model.eval()
        self.last_debug = ""

    def _generate_timed(
        self,
        inputs: Any,
        max_new_tokens: int,
        frame_index: int | None,
        vlm_calls: list[dict[str, Any]] | None,
        stage: str,
    ) -> Any:
        t0 = time.perf_counter()
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        latency = time.perf_counter() - t0
        if frame_index is not None and vlm_calls is not None:
            vlm_calls.append(
                {
                    "frame_prompt": frame_index,
                    "frame_response": frame_index,
                    "latency_sec": round(latency, 6),
                    "stage": stage,
                    "note": "Mismo frame: inferencia sincrona (no avanza el video hasta terminar).",
                }
            )
        return out_ids

    def predict_yes_prob(
        self,
        bgr: np.ndarray,
        frame_index: int | None = None,
        vlm_calls: list[dict[str, Any]] | None = None,
    ) -> float:
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        prompt_main = self.prompt
        self.last_prompt_used = prompt_main
        with torch.no_grad():
            if self.experiment_backend in self._INTERNVL_BACKENDS:
                md = getattr(self.model, "dtype", None) or next(self.model.parameters()).dtype
                pv = self._internvl_image_processor(images=image, return_tensors="pt")["pixel_values"]
                pv = pv.to(device=self.device, dtype=md)
                gen_cfg: dict[str, Any] = {"max_new_tokens": 16, "do_sample": False}
                t0 = time.perf_counter()
                out_text = self.model.chat(self.processor, pv, prompt_main, gen_cfg)
                latency = time.perf_counter() - t0
                if frame_index is not None and vlm_calls is not None:
                    vlm_calls.append(
                        {
                            "frame_prompt": frame_index,
                            "frame_response": frame_index,
                            "latency_sec": round(latency, 6),
                            "stage": "primary",
                            "note": "InternVL model.chat (no apply_chat_template multimodal).",
                        }
                    )
                score = self._yes_no_from_text(out_text) or self._extract_yes_no_anywhere(out_text)
                if score is None:
                    t1 = time.perf_counter()
                    out_text = self.model.chat(self.processor, pv, self.prompt_fallback, gen_cfg)
                    latency2 = time.perf_counter() - t1
                    if frame_index is not None and vlm_calls is not None:
                        vlm_calls.append(
                            {
                                "frame_prompt": frame_index,
                                "frame_response": frame_index,
                                "latency_sec": round(latency2, 6),
                                "stage": "fallback_prompt",
                                "note": "InternVL segundo intento con prompt_fallback.",
                            }
                        )
                    score = self._yes_no_from_text(out_text) or self._extract_yes_no_anywhere(out_text)
                self.last_answer_text = out_text.strip()
                return float(score) if score is not None else 0.0

            if self.experiment_backend in self._MINICPM_BACKENDS:
                msgs: list[dict[str, str]] = [{"role": "user", "content": prompt_main}]
                t0 = time.perf_counter()
                out_text, _, _ = self.model.chat(
                    image,
                    msgs,
                    [],
                    self.processor,
                    sampling=False,
                    max_new_tokens=24,
                )
                latency = time.perf_counter() - t0
                if frame_index is not None and vlm_calls is not None:
                    vlm_calls.append(
                        {
                            "frame_prompt": frame_index,
                            "frame_response": frame_index,
                            "latency_sec": round(latency, 6),
                            "stage": "primary",
                            "note": "MiniCPM-V model.chat (API openbmb).",
                        }
                    )
                score = self._yes_no_from_text(out_text) or self._extract_yes_no_anywhere(out_text)
                if score is None:
                    t1 = time.perf_counter()
                    out_text, _, _ = self.model.chat(
                        image,
                        [{"role": "user", "content": self.prompt_fallback}],
                        [],
                        self.processor,
                        sampling=False,
                        max_new_tokens=24,
                    )
                    latency2 = time.perf_counter() - t1
                    if frame_index is not None and vlm_calls is not None:
                        vlm_calls.append(
                            {
                                "frame_prompt": frame_index,
                                "frame_response": frame_index,
                                "latency_sec": round(latency2, 6),
                                "stage": "fallback_prompt",
                                "note": "MiniCPM-V segundo intento prompt_fallback.",
                            }
                        )
                    score = self._yes_no_from_text(out_text) or self._extract_yes_no_anywhere(out_text)
                self.last_answer_text = (out_text or "").strip()
                return float(score) if score is not None else 0.0

            if self.experiment_backend in self._PHI35_VISION_BACKENDS:
                # Phi3VProcessor: sin chat_template; el Hub exige <|image_1|> en el texto (ver processing_phi3_v.py).
                def _phi35_inputs(prompt: str) -> Any:
                    return self.processor(
                        text=f"<|image_1|>\n{prompt}",
                        images=image,
                        return_tensors="pt",
                    ).to(self.device)

                inputs = _phi35_inputs(prompt_main)
                out_ids = self._generate_timed(
                    inputs, max_new_tokens=16, frame_index=frame_index, vlm_calls=vlm_calls, stage="primary"
                )
                in_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
                gen_ids = out_ids[:, in_len:] if in_len > 0 else out_ids
                out_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                score = self._yes_no_from_text(out_text) or self._extract_yes_no_anywhere(out_text)
                if score is None:
                    inputs2 = _phi35_inputs(self.prompt_fallback)
                    out_ids2 = self._generate_timed(
                        inputs2,
                        max_new_tokens=8,
                        frame_index=frame_index,
                        vlm_calls=vlm_calls,
                        stage="fallback_prompt",
                    )
                    in_len2 = int(inputs2["input_ids"].shape[1]) if "input_ids" in inputs2 else 0
                    gen_ids2 = out_ids2[:, in_len2:] if in_len2 > 0 else out_ids2
                    out_text = self.processor.batch_decode(gen_ids2, skip_special_tokens=True)[0].strip()
                    score = self._yes_no_from_text(out_text) or self._extract_yes_no_anywhere(out_text)
                self.last_answer_text = out_text.strip()
                return float(score) if score is not None else 0.0

            _tok = getattr(self.processor, "tokenizer", None)
            _has_chat_tpl = getattr(self.processor, "chat_template", None) is not None or (
                _tok is not None and getattr(_tok, "chat_template", None) is not None
            )
            if hasattr(self.processor, "apply_chat_template") and _has_chat_tpl:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_main},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
            else:
                inputs = self.processor(
                    text=prompt_main,
                    images=image,
                    return_tensors="pt",
                ).to(self.device)

            out_ids = self._generate_timed(
                inputs, max_new_tokens=16, frame_index=frame_index, vlm_calls=vlm_calls, stage="primary"
            )
            in_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
            gen_ids = out_ids[:, in_len:] if in_len > 0 else out_ids
            out_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            score = self._yes_no_from_text(out_text) or self._extract_yes_no_anywhere(out_text)

            if score is None:
                prompt2 = self.prompt_fallback
                if hasattr(self.processor, "apply_chat_template"):
                    messages2 = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt2},
                            ],
                        }
                    ]
                    text2 = self.processor.apply_chat_template(
                        messages2, tokenize=False, add_generation_prompt=True
                    )
                    inputs2 = self.processor(
                        text=[text2],
                        images=[image],
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)
                else:
                    inputs2 = self.processor(
                        text=prompt2,
                        images=image,
                        return_tensors="pt",
                    ).to(self.device)
                out_ids2 = self._generate_timed(
                    inputs2,
                    max_new_tokens=8,
                    frame_index=frame_index,
                    vlm_calls=vlm_calls,
                    stage="fallback_prompt",
                )
                in_len2 = int(inputs2["input_ids"].shape[1]) if "input_ids" in inputs2 else 0
                gen_ids2 = out_ids2[:, in_len2:] if in_len2 > 0 else out_ids2
                out_text = self.processor.batch_decode(gen_ids2, skip_special_tokens=True)[0].strip()
                score = self._extract_yes_no_anywhere(out_text)

            self.last_answer_text = out_text.strip()
        return float(score) if score is not None else 0.0


class MoondreamHandClassifier(YesNoTextMixin):
    """Moondream2 (vikhyatk/moondream2): usa answer_question si existe."""

    experiment_backend = "moondream2"

    def __init__(self, model_name: str, device: str, prompt: str) -> None:
        self.prompt = prompt
        self.last_answer_text = ""
        self.last_prompt_used = ""
        self.model_name = model_name
        load_id = resolve_hf_model_ref(model_name)
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        dt = torch.float16 if self.device.type == "cuda" else torch.float32
        kw = {"trust_remote_code": True, "torch_dtype": dt}
        self.model = None
        for attempt in range(2):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(load_id, **kw).to(self.device)
                break
            except FileNotFoundError as e:
                if attempt == 0 and _try_remove_broken_transformers_remote_code_cache(e):
                    continue
                raise RuntimeError(
                    "Moondream: cache de codigo remoto (transformers_modules) incompleto o corrupto. "
                    f"Intento de reparacion automatica fallo. Detalle: {e!s}. "
                    "Prueba: rm -rf ~/.cache/huggingface/modules/transformers_modules/* "
                    "y vuelve a ejecutar."
                ) from e
        assert self.model is not None
        self.model.eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_id, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoProcessor.from_pretrained(load_id, trust_remote_code=True)
        self.last_debug = ""

    def _generate_timed(
        self,
        fn: Any,
        frame_index: int | None,
        vlm_calls: list[dict[str, Any]] | None,
        stage: str,
    ) -> str:
        t0 = time.perf_counter()
        out = fn()
        latency = time.perf_counter() - t0
        if frame_index is not None and vlm_calls is not None:
            vlm_calls.append(
                {
                    "frame_prompt": frame_index,
                    "frame_response": frame_index,
                    "latency_sec": round(latency, 6),
                    "stage": stage,
                    "note": "Mismo frame: inferencia sincrona (no avanza el video hasta terminar).",
                }
            )
        return str(out).strip()

    def predict_yes_prob(
        self,
        bgr: np.ndarray,
        frame_index: int | None = None,
        vlm_calls: list[dict[str, Any]] | None = None,
    ) -> float:
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.last_prompt_used = self.prompt
        with torch.no_grad():
            if hasattr(self.model, "answer_question"):
                out_text = self._generate_timed(
                    lambda: self.model.answer_question(image, self.prompt, self.tokenizer),
                    frame_index,
                    vlm_calls,
                    "primary",
                )
            else:
                raise RuntimeError(
                    "Este checkpoint de Moondream no expone answer_question; revisa la documentacion del modelo."
                )
        self.last_answer_text = out_text
        score = self._yes_no_from_text(out_text) or self._extract_yes_no_anywhere(out_text)
        return float(score) if score is not None else 0.0


# Florence-2 a veces solo continua el texto del prompt (eco); una segunda pasada muy corta suele devolver Y/N.
_FLORENCE_ECHO_FALLBACK_PROMPT = "Y/N — grasped?"


def _resolve_florence2_hub_id(model_name: str) -> str:
    """
    microsoft/Florence-2-* en el Hub usa codigo remoto incompatible con transformers 5.x.
    El checkpoint equivalente mantenido en la libreria es florence-community/Florence-2-*.
    """
    p = Path(model_name)
    if p.exists() and p.is_dir():
        return model_name
    low = model_name.strip().lower()
    if low == "microsoft/florence-2-base":
        return "florence-community/Florence-2-base"
    if low == "microsoft/florence-2-large":
        return "florence-community/Florence-2-large"
    return model_name


class Florence2HandClassifier(YesNoTextMixin):
    """Florence-2 (HF nativo en transformers 4.46+ / 5.x; sin trust_remote_code legacy)."""

    experiment_backend = "florence2"

    @staticmethod
    def _florence_output_echoes_prompt(out_text: str, prompt: str) -> bool:
        """True si la salida es sobre todo una repeticion del enunciado (sin respuesta Y/N)."""
        o = out_text.strip().lower()
        p = prompt.strip().lower()
        if len(o) < 14 or not p:
            return False
        if o in p:
            return True
        # Continuacion desde mitad del prompt (subcadena larga compartida)
        best = 0
        for i in range(len(o)):
            for j in range(i + 14, len(o) + 1):
                chunk = o[i:j]
                if chunk in p:
                    best = max(best, len(chunk))
        return best >= max(24, len(o) * 0.55)

    def _florence_score_text(self, out_text: str) -> float | None:
        return (
            self._yes_no_first_word(out_text)
            or self._yes_no_from_answer_tail(out_text)
            or self._yes_no_leading_letter_yn(out_text)
            or self._yes_no_from_text(out_text)
            or self._extract_yes_no_anywhere(out_text)
        )

    def __init__(self, model_name: str, device: str, prompt: str) -> None:
        if Florence2ForConditionalGeneration is None:
            raise RuntimeError(
                "Tu version de transformers no incluye Florence2ForConditionalGeneration. Actualiza transformers."
            )
        self.prompt = prompt
        self.last_answer_text = ""
        self.last_prompt_used = ""
        self.model_name = model_name
        self.resolved_hub_id = _resolve_florence2_hub_id(model_name)
        load_from = resolve_hf_model_ref(self.resolved_hub_id)
        if load_from == self.resolved_hub_id:
            load_from = resolve_hf_model_ref(model_name)
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        dt = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(load_from, trust_remote_code=False)
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            load_from,
            trust_remote_code=False,
            torch_dtype=dt,
        ).to(self.device)
        self.model.eval()
        self.last_debug = ""

    def predict_yes_prob(
        self,
        bgr: np.ndarray,
        frame_index: int | None = None,
        vlm_calls: list[dict[str, Any]] | None = None,
    ) -> float:
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.last_prompt_used = self.prompt
        out_text = ""
        score: float | None = None
        with torch.no_grad():
            inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            t0 = time.perf_counter()
            # Respuestas cortas tipo Yes/No; ~12 tokens bastan sin invitar tanto a repetir la pregunta como con 32.
            out_ids = self.model.generate(**inputs, max_new_tokens=12, do_sample=False, num_beams=1)
            latency = time.perf_counter() - t0
            if frame_index is not None and vlm_calls is not None:
                vlm_calls.append(
                    {
                        "frame_prompt": frame_index,
                        "frame_response": frame_index,
                        "latency_sec": round(latency, 6),
                        "stage": "primary",
                        "note": "Mismo frame: inferencia sincrona (no avanza el video hasta terminar).",
                    }
                )
            # Seq2seq: generate() devuelve solo la secuencia del decoder (no concatena el prompt).
            out_text = self.processor.batch_decode(
                out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            score = self._florence_score_text(out_text)

            retry = self._florence_output_echoes_prompt(out_text, self.prompt) or score is None
            if retry and self.prompt.strip() != _FLORENCE_ECHO_FALLBACK_PROMPT:
                t1 = time.perf_counter()
                inputs2 = self.processor(
                    images=image, text=_FLORENCE_ECHO_FALLBACK_PROMPT, return_tensors="pt"
                )
                inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
                out_ids2 = self.model.generate(**inputs2, max_new_tokens=12, do_sample=False, num_beams=1)
                latency2 = time.perf_counter() - t1
                out2 = self.processor.batch_decode(
                    out_ids2, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                score2 = self._florence_score_text(out2)
                if frame_index is not None and vlm_calls is not None:
                    vlm_calls.append(
                        {
                            "frame_prompt": frame_index,
                            "frame_response": frame_index,
                            "latency_sec": round(latency2, 6),
                            "stage": "florence_echo_fallback",
                            "note": f"Reintento por eco o sin Y/N; prompt={_FLORENCE_ECHO_FALLBACK_PROMPT!r}",
                        }
                    )
                out_text = f"{out_text}  [retry→ {out2}]" if out_text else out2
                self.last_prompt_used = f"{self.prompt}  || retry: {_FLORENCE_ECHO_FALLBACK_PROMPT}"
                score = score2 if score2 is not None else score
                if score2 is not None:
                    extra = "florence_echo_retry_ok"
                else:
                    extra = "florence_echo_retry_still_unparsed"
                self.last_debug = (
                    f"{self.last_debug}; {extra}" if self.last_debug else extra
                )
        if self.model_name != self.resolved_hub_id:
            map_note = f"cargado como {self.resolved_hub_id} (mapeo desde {self.model_name})"
            self.last_debug = f"{self.last_debug}; {map_note}" if self.last_debug else map_note
        self.last_answer_text = out_text
        return float(score) if score is not None else 0.0


class ClipLikeClassifier(YesNoTextMixin):
    """CLIP o SigLIP: embeddings imagen vs texto de positivos/negativos."""

    def __init__(
        self,
        model_name: str,
        device: str,
        prompt: str,
        *,
        backend_name: str = "clip",
    ) -> None:
        self.experiment_backend = backend_name
        self.prompt = prompt
        self.last_answer_text = ""
        self.last_debug = ""
        self.last_prompt_used = prompt
        self.model_name = model_name
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        load_id = resolve_hf_model_ref(model_name)
        try:
            self.processor = AutoProcessor.from_pretrained(load_id, use_fast=True)
        except (TypeError, ValueError, OSError):
            self.processor = AutoProcessor.from_pretrained(load_id)
        self.model = AutoModel.from_pretrained(load_id).to(self.device)
        self.model.eval()
        self.positive_texts = [
            "a hand holding an object",
            "a person holding an item in hand",
            "hand grasping an object",
            "hand carrying a small object",
        ]
        self.negative_texts = [
            "an empty hand",
            "a hand with no object",
            "open hand with nothing",
            "person hand resting without object",
            "hand touching clothes on the body",
            "hand resting on hip or leg",
            "hand touching a worn handbag on the body",
            "hand touching shoulder bag strap",
            "hand touching backpack worn by person",
            "hand touching body only",
            "empty hand resting on thigh or knee",
            "hand placed on leg with no object",
            "relaxed hand on pants without grasping",
        ]
        self.texts = self.positive_texts + self.negative_texts
        self.n_pos = len(self.positive_texts)
        with torch.no_grad():
            txt = self.processor(text=self.texts, return_tensors="pt", padding=True).to(self.device)
            txt_feat = self._encode_text(txt)
            self.text_features = self._l2_normalize(txt_feat)

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    @staticmethod
    def _extract_tensor(out: Any) -> torch.Tensor | None:
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "text_embeds") and out.text_embeds is not None:
            return out.text_embeds
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            return out.image_embeds
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state[:, 0, :]
        return None

    def _encode_text(self, txt_inputs: Any) -> torch.Tensor:
        if hasattr(self.model, "get_text_features"):
            out = self.model.get_text_features(**txt_inputs)
            feat = self._extract_tensor(out)
            if feat is not None:
                return feat
        if hasattr(self.model, "text_model"):
            out = self.model.text_model(
                input_ids=txt_inputs.get("input_ids"),
                attention_mask=txt_inputs.get("attention_mask"),
            )
            feat = self._extract_tensor(out)
            if feat is not None:
                return feat
        raise RuntimeError("No se pudieron extraer embeddings de texto.")

    def _encode_image(self, img_inputs: Any) -> torch.Tensor:
        if hasattr(self.model, "get_image_features"):
            out = self.model.get_image_features(**img_inputs)
            feat = self._extract_tensor(out)
            if feat is not None:
                return feat
        if hasattr(self.model, "vision_model"):
            out = self.model.vision_model(pixel_values=img_inputs.get("pixel_values"))
            feat = self._extract_tensor(out)
            if feat is not None:
                return feat
        raise RuntimeError("No se pudieron extraer embeddings de imagen.")

    def predict_yes_prob(
        self,
        bgr: np.ndarray,
        frame_index: int | None = None,
        vlm_calls: list[dict[str, Any]] | None = None,
    ) -> float:
        # CLIP/SigLIP: no usa --vlm-prompt en la inferencia; contrasta anclas fijas pos/neg en ingles.
        self.last_prompt_used = (
            "CLIP/SigLIP: embeddings vs anclas texto (positivas/negativas); "
            "el --vlm-prompt no modifica esos anclas."
        )
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            inp = self.processor(images=img, return_tensors="pt").to(self.device)
            t0 = time.perf_counter()
            img_feat = self._encode_image(inp)
            img_feat = self._l2_normalize(img_feat)
            logits = (100.0 * img_feat @ self.text_features.T)[0]
            latency = time.perf_counter() - t0
            if frame_index is not None and vlm_calls is not None:
                vlm_calls.append(
                    {
                        "frame_prompt": frame_index,
                        "frame_response": frame_index,
                        "latency_sec": round(latency, 6),
                        "stage": "clip_embedding",
                        "note": "Mismo frame: inferencia sincrona (no avanza el video hasta terminar).",
                    }
                )
            pos_score = torch.mean(logits[: self.n_pos])
            neg_score = torch.mean(logits[self.n_pos :])
            margin = float(pos_score - neg_score)
            if margin < 0.0:
                pos_score = pos_score + (1.5 * margin)
            pair = torch.stack([pos_score, neg_score], dim=0)
            probs = torch.softmax(pair, dim=0)
            p_yes = float(probs[0].item())
            self.last_answer_text = "YES" if p_yes >= 0.5 else "NO"
            self.last_debug = (
                f"sim_pos={float(pos_score):.3f} sim_neg={float(neg_score):.3f} p_yes={p_yes:.3f}"
            )
        return p_yes


class OpenClipLikeClassifier(YesNoTextMixin):
    """MobileCLIP, OpenVision (Hub OpenCLIP) y similares: embeddings via paquete open_clip."""

    def __init__(
        self,
        model_name: str,
        device: str,
        prompt: str,
        *,
        backend_name: str = "open_clip",
    ) -> None:
        try:
            import open_clip
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Se requiere open_clip para este backend (pip install open-clip-torch)."
            ) from e

        self.experiment_backend = backend_name
        self.prompt = prompt
        self.last_answer_text = ""
        self.last_debug = ""
        self.last_prompt_used = prompt
        self.model_name = model_name
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        load_ref = resolve_hf_model_ref(model_name)
        root = Path(load_ref).expanduser()
        # Snapshot HF local: open_clip espera nombre de arquitectura + pretrained=ruta (no solo la ruta).
        if root.is_dir():
            weights_path = _open_clip_weights_file_from_hub_snapshot(root)
            arch = _openclip_arch_for_hf_snapshot(model_name)
            if arch is None:
                raise RuntimeError(
                    "Snapshot OpenCLIP local sin mapeo de arquitectura conocido; "
                    f"path={root!s} model_ref={model_name!r}. "
                    "Para MobileCLIP usa p. ej. hf-hub:apple/MobileCLIP-S1-OpenCLIP."
                )
            if weights_path is None:
                raise RuntimeError(
                    "En el snapshot HF no hay pesos open_clip "
                    f"(archivos >= {_MIN_OPENCLIP_WEIGHT_BYTES // 1024} KiB: "
                    "open_clip_pytorch_model.bin preferido, open_clip_model.safetensors). "
                    f"Directorio: {root.resolve()!s}"
                )
            with _torch_load_open_clip_bin_checkpoints_ok():
                self.model, self.preprocess = open_clip.create_model_from_pretrained(
                    arch,
                    pretrained=str(weights_path.resolve()),
                    device=self.device,
                )
            self.tokenizer = open_clip.get_tokenizer(arch)
        else:
            with _torch_load_open_clip_bin_checkpoints_ok():
                self.model, self.preprocess = open_clip.create_model_from_pretrained(
                    model_name, device=self.device
                )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        self.positive_texts = [
            "a hand holding an object",
            "a person holding an item in hand",
            "hand grasping an object",
            "hand carrying a small object",
        ]
        self.negative_texts = [
            "an empty hand",
            "a hand with no object",
            "open hand with nothing",
            "person hand resting without object",
            "hand touching clothes on the body",
            "hand resting on hip or leg",
            "hand touching a worn handbag on the body",
            "hand touching shoulder bag strap",
            "hand touching backpack worn by person",
            "hand touching body only",
            "empty hand resting on thigh or knee",
            "hand placed on leg with no object",
            "relaxed hand on pants without grasping",
        ]
        self.texts = self.positive_texts + self.negative_texts
        self.n_pos = len(self.positive_texts)
        with torch.no_grad():
            txt = self.tokenizer(self.texts).to(self.device)
            txt_feat = self.model.encode_text(txt)
            self.text_features = self._l2_normalize(txt_feat)

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def predict_yes_prob(
        self,
        bgr: np.ndarray,
        frame_index: int | None = None,
        vlm_calls: list[dict[str, Any]] | None = None,
    ) -> float:
        self.last_prompt_used = (
            "open_clip (Hub): embeddings vs anclas texto (positivas/negativas); "
            "el --vlm-prompt no modifica esos anclas."
        )
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            pv = self.preprocess(img).unsqueeze(0).to(self.device)
            t0 = time.perf_counter()
            img_feat = self.model.encode_image(pv)
            img_feat = self._l2_normalize(img_feat)
            logits = (100.0 * img_feat @ self.text_features.T)[0]
            latency = time.perf_counter() - t0
            if frame_index is not None and vlm_calls is not None:
                vlm_calls.append(
                    {
                        "frame_prompt": frame_index,
                        "frame_response": frame_index,
                        "latency_sec": round(latency, 6),
                        "stage": "open_clip_embedding",
                        "note": "Mismo frame: inferencia sincrona (no avanza el video hasta terminar).",
                    }
                )
            pos_score = torch.mean(logits[: self.n_pos])
            neg_score = torch.mean(logits[self.n_pos :])
            margin = float(pos_score - neg_score)
            if margin < 0.0:
                pos_score = pos_score + (1.5 * margin)
            pair = torch.stack([pos_score, neg_score], dim=0)
            probs = torch.softmax(pair, dim=0)
            p_yes = float(probs[0].item())
            self.last_answer_text = "YES" if p_yes >= 0.5 else "NO"
            self.last_debug = (
                f"sim_pos={float(pos_score):.3f} sim_neg={float(neg_score):.3f} p_yes={p_yes:.3f}"
            )
        return p_yes
