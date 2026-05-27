#!/usr/bin/env python3
"""
Servicio HTTP persistente: carga SigLIP una vez y procesa chunks bajo demanda.

Uso local:
  pip install fastapi uvicorn
  export VLM_RC2_DEVICE=cuda:0
  python semantics_service.py

Endpoints:
  GET  /health
  GET  /ready
  POST /v1/semantics/chunk   body: {"chunk_dir": "/data/chunks/chunk_001", ...}

Variables de entorno (ver INTEGRATION.md en esta carpeta).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_RCDIR = Path(__file__).resolve().parent
if str(_RCDIR) not in sys.path:
    sys.path.insert(0, str(_RCDIR))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(
        "semantics_service requiere: pip install fastapi uvicorn pydantic\n"
        f"Detalle: {e}"
    ) from e

from rc2_settings import (
    _DEFAULT_CONFIG_BASENAME,
    _EXAMPLE_BASENAME,
    get_settings,
    init_settings,
)
from semantics_core import SemanticsRunOptions, build_classifier, persist_chunk_result, process_chunk_directory

_inference_lock = threading.Lock()
_state: dict[str, Any] = {
    "model_loaded": False,
    "model_loading": False,
    "load_error": None,
    "load_started_at": None,
    "load_finished_at": None,
    "requests_total": 0,
    "requests_failed": 0,
    "classifier": None,
    "default_opts": None,
}


class ChunkSemanticsRequest(BaseModel):
    chunk_dir: str = Field(..., description="Ruta absoluta o relativa al chunk (frames/ + frames_meta.json).")
    write_file: bool = Field(
        default=False,
        description="Si true, escribe <chunk>_vlm.json en la carpeta del chunk.",
    )
    emit_image_embedding: bool = False
    tau_max_prob: float = 0.35
    min_margin: float = 0.08
    max_entropy: float = Field(default=-1.0, description="Si >= 0, umbral de entropía.")
    quiet: bool = True


def _settings_to_opts() -> SemanticsRunOptions:
    s = get_settings()
    return SemanticsRunOptions(
        device=s.device,
        hf_token=s.hf_token,
        vlm_model=s.vlm_model,
        multicrop_mode=s.multicrop_mode,
        net_th=s.net_th,
        net_margin_th=s.net_margin_th,
        decision_mode=s.decision_mode,
        quiet=True,
        write_file=False,
        continue_on_inference_error=True,
        max_entropy=s.max_entropy_or_none(),
    )


def _load_model_sync() -> None:
    if _state["model_loaded"] or _state["model_loading"]:
        return
    _state["model_loading"] = True
    _state["load_started_at"] = time.time()
    try:
        opts = _settings_to_opts()
        _state["default_opts"] = opts
        clf = build_classifier(opts)
        _state["classifier"] = clf
        _state["model_loaded"] = True
        _state["load_error"] = None
        print(
            f"[vlm_rc2-service] Modelo listo: {clf.model_name} en {clf.device}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as e:
        _state["load_error"] = f"{e}\n{traceback.format_exc()}"
        print(f"[vlm_rc2-service] ERROR cargando modelo: {e}", file=sys.stderr, flush=True)
    finally:
        _state["model_loading"] = False
        _state["load_finished_at"] = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if get_settings().service_skip_model_load:
        print("[vlm_rc2-service] service.skip_model_load=true — modelo no cargado (solo pruebas HTTP).", file=sys.stderr)
    else:
        thread = threading.Thread(target=_load_model_sync, name="vlm-rc2-model-load", daemon=False)
        thread.start()
        thread.join()
    yield
    _state["classifier"] = None
    _state["model_loaded"] = False


# Carga automática si existe vlm_rc2.settings.json (uvicorn semantics_service:app)
init_settings()

app = FastAPI(
    title="vlm_rc2 semantics service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "vlm_rc2_semantics",
        "model_loaded": bool(_state["model_loaded"]),
        "model_loading": bool(_state["model_loading"]),
        "requests_total": int(_state["requests_total"]),
        "requests_failed": int(_state["requests_failed"]),
    }


@app.get("/ready")
def ready() -> JSONResponse:
    if _state["model_loading"]:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "service": "vlm_rc2_semantics"},
        )
    if not _state["model_loaded"]:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "vlm_rc2_semantics",
                "load_error": _state.get("load_error"),
            },
        )
    return JSONResponse(content={"status": "ready", "service": "vlm_rc2_semantics"})


@app.post("/v1/semantics/chunk")
def semantics_chunk(req: ChunkSemanticsRequest) -> dict[str, Any]:
    if not _state["model_loaded"] or _state["classifier"] is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "load_error": _state.get("load_error"),
            },
        )

    _state["requests_total"] += 1
    chunk_dir = Path(req.chunk_dir).expanduser().resolve()
    chunk_stem = chunk_dir.name or "chunk"

    base_opts: SemanticsRunOptions = _state["default_opts"] or _settings_to_opts()
    opts = SemanticsRunOptions(
        device=base_opts.device,
        hf_token=base_opts.hf_token,
        vlm_model=base_opts.vlm_model,
        multicrop_mode=base_opts.multicrop_mode,
        net_th=base_opts.net_th,
        net_margin_th=base_opts.net_margin_th,
        decision_mode=base_opts.decision_mode,
        emit_image_embedding=req.emit_image_embedding,
        tau_max_prob=req.tau_max_prob,
        min_margin=req.min_margin,
        max_entropy=None if req.max_entropy < 0 else req.max_entropy,
        quiet=req.quiet,
        write_file=req.write_file,
        persist_json=False,
        continue_on_inference_error=True,
    )

    t0 = time.perf_counter()
    try:
        with _inference_lock:
            payload = process_chunk_directory(
                chunk_dir,
                _state["classifier"],
                opts,
                stage="semantics_service",
            )
        if req.write_file:
            persist_chunk_result(chunk_dir, chunk_stem, payload, write_file=True)
        payload["service_elapsed_sec"] = round(time.perf_counter() - t0, 3)
        if str(payload.get("vlm_rc2_pipeline_status")) == "error":
            _state["requests_failed"] += 1
        return payload
    except Exception as e:
        _state["requests_failed"] += 1
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal",
                "message": str(e),
                "traceback": traceback.format_exc()[-4000:],
            },
        ) from e


def main() -> None:
    import argparse

    import uvicorn

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        default="",
        help=f"Ruta a {_DEFAULT_CONFIG_BASENAME} (por defecto: carpeta vlm_rc2/ o cwd).",
    )
    pre_args, _ = pre.parse_known_args()

    init_settings(pre_args.config or None, force_reload=True)
    s = get_settings()
    if s.config_path:
        print(f"[vlm_rc2-service] Config: {s.config_path}", file=sys.stderr, flush=True)
    else:
        print(
            f"[vlm_rc2-service] Sin { _DEFAULT_CONFIG_BASENAME }; valores por defecto. "
            f"Copia { _EXAMPLE_BASENAME } → { _DEFAULT_CONFIG_BASENAME }",
            file=sys.stderr,
            flush=True,
        )

    uvicorn.run(
        app,
        host=s.service_host,
        port=int(s.service_port),
        log_level=s.service_log_level,
    )


if __name__ == "__main__":
    main()
