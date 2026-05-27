"""
Carga de configuración desde `vlm_rc2.settings.json`.

Prioridad por parámetro: CLI > variable de entorno > fichero > valores por defecto.

Ubicación del fichero (primera que exista):
  1. Ruta explícita (--config o init_settings(path=...))
  2. VLM_RC2_CONFIG_PATH
  3. <carpeta vlm_rc2>/vlm_rc2.settings.json
  4. ./vlm_rc2.settings.json (directorio de trabajo actual)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_BASENAME = "vlm_rc2.settings.json"
_EXAMPLE_BASENAME = "vlm_rc2.settings.example.json"

_cached: "VlmRc2Settings | None" = None
_cached_path: Path | None = None


@dataclass
class VlmRc2Settings:
    """Configuración efectiva del pipeline vlm_rc2."""

    config_path: Path | None = None

    # Servicio HTTP (semantics_service.py)
    service_host: str = "0.0.0.0"
    service_port: int = 6161
    service_log_level: str = "info"
    service_skip_model_load: bool = False

    # Cliente (run_semantics_rc2.py → servicio)
    client_semantics_service_url: str = ""
    client_timeout_sec: float = 900.0

    # Modelo / inferencia
    device: str = "cuda:0"
    hf_token: str = ""
    vlm_model: str = "google/siglip-so400m-patch14-384"
    multicrop_mode: str = "batch3"
    net_th: float = 0.35
    net_margin_th: float = 0.30
    decision_mode: str = "weighted"
    tau_max_prob: float = 0.35
    min_margin: float = 0.08
    max_entropy: float = -1.0
    emit_image_embedding: bool = False

    # Rutas (documentación / despliegue; el servicio recibe chunk_dir en cada petición)
    chunks_root: str = "/data/chunks"

    def max_entropy_or_none(self) -> float | None:
        return None if self.max_entropy < 0 else float(self.max_entropy)

    def semantics_service_url(self) -> str:
        return str(self.client_semantics_service_url or "").strip().rstrip("/")

    def apply_hf_token_to_environ(self) -> None:
        tok = str(self.hf_token or "").strip()
        if not tok:
            return
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        os.environ.setdefault("HF_HUB_DISABLE_INTERACTIVE_PROMPTS", "1")

def default_settings() -> VlmRc2Settings:
    return VlmRc2Settings()


def resolve_config_path(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_file() else None

    env_path = str(os.environ.get("VLM_RC2_CONFIG_PATH") or "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p

    for candidate in (
        _PKG_DIR / _DEFAULT_CONFIG_BASENAME,
        Path.cwd() / _DEFAULT_CONFIG_BASENAME,
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _deep_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _parse_settings_file(path: Path) -> VlmRc2Settings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuración raíz debe ser un objeto JSON: {path}")

    s = default_settings()
    s.config_path = path

    # Sección "service"
    s.service_host = str(_deep_get(raw, "service", "host", default=s.service_host))
    s.service_port = int(_deep_get(raw, "service", "port", default=s.service_port))
    s.service_log_level = str(_deep_get(raw, "service", "log_level", default=s.service_log_level))
    s.service_skip_model_load = bool(
        _deep_get(raw, "service", "skip_model_load", default=s.service_skip_model_load)
    )

    # Sección "client"
    s.client_semantics_service_url = str(
        _deep_get(raw, "client", "semantics_service_url", default=s.client_semantics_service_url)
    )
    s.client_timeout_sec = float(
        _deep_get(raw, "client", "timeout_sec", default=s.client_timeout_sec)
    )

    # Sección "model"
    s.device = str(_deep_get(raw, "model", "device", default=s.device))
    s.hf_token = str(_deep_get(raw, "model", "hf_token", default=s.hf_token))
    s.vlm_model = str(_deep_get(raw, "model", "vlm_model", default=s.vlm_model))
    s.multicrop_mode = str(_deep_get(raw, "model", "multicrop_mode", default=s.multicrop_mode))
    s.net_th = float(_deep_get(raw, "model", "net_th", default=s.net_th))
    s.net_margin_th = float(_deep_get(raw, "model", "net_margin_th", default=s.net_margin_th))
    s.decision_mode = str(_deep_get(raw, "model", "decision_mode", default=s.decision_mode))
    s.tau_max_prob = float(_deep_get(raw, "model", "tau_max_prob", default=s.tau_max_prob))
    s.min_margin = float(_deep_get(raw, "model", "min_margin", default=s.min_margin))
    s.max_entropy = float(_deep_get(raw, "model", "max_entropy", default=s.max_entropy))
    s.emit_image_embedding = bool(
        _deep_get(raw, "model", "emit_image_embedding", default=s.emit_image_embedding)
    )

    # Sección "paths"
    s.chunks_root = str(_deep_get(raw, "paths", "chunks_root", default=s.chunks_root))

    return s


def load_settings(explicit_path: str | Path | None = None) -> VlmRc2Settings:
    path = resolve_config_path(explicit_path)
    if path is None:
        return default_settings()
    return _parse_settings_file(path)


def init_settings(explicit_path: str | Path | None = None, *, force_reload: bool = False) -> VlmRc2Settings:
    """Carga (o recarga) configuración y la deja en caché para get_settings()."""
    global _cached, _cached_path
    if _cached is not None and not force_reload:
        if explicit_path is None or _cached_path == Path(explicit_path).expanduser().resolve():
            return _cached

    _cached = load_settings(explicit_path)
    _cached_path = _cached.config_path
    _cached.apply_hf_token_to_environ()
    return _cached


def get_settings() -> VlmRc2Settings:
    global _cached
    if _cached is None:
        return init_settings()
    return _cached


def effective_string(*, cli: str = "", config: str = "", env_key: str | None = None) -> str:
    """CLI > env > config."""
    if str(cli or "").strip():
        return str(cli).strip()
    if env_key:
        ev = str(os.environ.get(env_key) or "").strip()
        if ev:
            return ev
    return str(config or "").strip()


def effective_float(
    *,
    cli: float | None,
    config: float,
    env_key: str | None = None,
    default: float,
) -> float:
    if cli is not None:
        return float(cli)
    if env_key:
        ev = str(os.environ.get(env_key) or "").strip()
        if ev:
            try:
                return float(ev)
            except ValueError:
                pass
    if config != default or get_settings().config_path is not None:
        return float(config)
    return float(default)


def example_config_path() -> Path:
    return _PKG_DIR / _EXAMPLE_BASENAME
