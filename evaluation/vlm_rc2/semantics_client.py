"""
Cliente HTTP para el servicio persistente vlm_rc2 (stdlib: urllib).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from rc2_settings import effective_string, get_settings
from semantics_core import SemanticsRunOptions


class SemanticsServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def service_url_from_settings(*, cli: str = "") -> str:
    """URL del servicio: CLI > env VLM_RC2_SEMANTICS_SERVICE_URL > vlm_rc2.settings.json."""
    s = get_settings()
    return effective_string(
        cli=cli,
        config=s.semantics_service_url(),
        env_key="VLM_RC2_SEMANTICS_SERVICE_URL",
    ).rstrip("/")


def client_timeout_sec(*, cli: float | None = None) -> float:
    s = get_settings()
    if cli is not None and cli > 0:
        return float(cli)
    return float(s.client_timeout_sec)


def check_service_health(base_url: str, *, timeout_sec: float = 5.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_service_ready(base_url: str, *, timeout_sec: float = 5.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/ready"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_chunk_semantics(
    base_url: str,
    chunk_dir: str,
    opts: SemanticsRunOptions,
    *,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    """
    POST /v1/semantics/chunk — el servicio lee chunk_dir en su filesystem (volumen compartido).
    """
    if timeout_sec is None:
        timeout_sec = client_timeout_sec()

    body = {
        "chunk_dir": str(chunk_dir),
        "write_file": bool(opts.write_file),
        "emit_image_embedding": opts.emit_image_embedding,
        "tau_max_prob": opts.tau_max_prob,
        "min_margin": opts.min_margin,
        "max_entropy": -1.0 if opts.max_entropy is None else opts.max_entropy,
        "quiet": opts.quiet,
    }
    data = json.dumps(body).encode("utf-8")
    url = f"{base_url.rstrip('/')}/v1/semantics/chunk"
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise SemanticsServiceError(
            f"Servicio semántica HTTP {e.code}: {detail[:500]}",
            status_code=e.code,
            body=detail,
        ) from e
    except urllib.error.URLError as e:
        raise SemanticsServiceError(f"No se pudo conectar al servicio {base_url}: {e}") from e
    except json.JSONDecodeError as e:
        raise SemanticsServiceError(f"Respuesta JSON inválida del servicio: {e}") from e


# Alias retrocompatible
service_url_from_env = service_url_from_settings
