#!/usr/bin/env python3
"""
Pipeline rc2: semántica para **una carpeta de chunk** (`frames/` + `frames_meta.json`).

Configuración: `vlm_rc2.settings.json` (copia desde vlm_rc2.settings.example.json).
Modos:
  - **Local**: carga SigLIP en este proceso (client.semantics_service_url vacío).
  - **Servicio**: delega en semantics_service.py (URL en el JSON o --semantics-service-url).

100 % autosuficiente dentro de vlm_rc2/.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_RCDIR = Path(__file__).resolve().parent
if str(_RCDIR) not in sys.path:
    sys.path.insert(0, str(_RCDIR))

from rc2_settings import (  # noqa: E402
    _DEFAULT_CONFIG_BASENAME,
    _EXAMPLE_BASENAME,
    VlmRc2Settings,
    init_settings,
    get_settings,
)
from semantics_client import (  # noqa: E402
    SemanticsServiceError,
    client_timeout_sec,
    request_chunk_semantics,
    service_url_from_settings,
)
from semantics_core import (  # noqa: E402
    build_classifier,
    options_from_namespace,
    persist_chunk_result,
    process_chunk_directory,
)
from vlm_rc2_config import write_json_stdout  # noqa: E402
from vlm_rc2_errors import semantics_runtime_error_payload  # noqa: E402


def _build_parser(s: VlmRc2Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="vlm_rc2: un chunk → JSON semántico SigLIP.",
    )
    p.add_argument(
        "--config",
        default="",
        help=f"Ruta a {_DEFAULT_CONFIG_BASENAME} (si no está en carpeta vlm_rc2/ ni cwd).",
    )
    p.add_argument(
        "--chunk-dir",
        "--chunk_dir",
        required=True,
        metavar="DIR",
        help="Carpeta del chunk con frames/ + frames_meta.json.",
    )
    p.add_argument(
        "--semantics-service-url",
        "--semantics_service_url",
        default="",
        help="URL del servicio (vacío = usar client.semantics_service_url del JSON o modo local).",
    )
    p.add_argument(
        "--service-timeout-sec",
        type=float,
        default=0.0,
        help=f"Timeout HTTP (0 = client.timeout_sec del JSON, default {s.client_timeout_sec}).",
    )
    p.add_argument(
        "--output-json",
        default="",
        help="Ruta opcional del JSON; si se omite: <chunk-dir>/<stem>_vlm.json",
    )
    p.add_argument("--device", default=s.device, help="Dispositivo PyTorch (solo modo local).")
    p.add_argument(
        "--hf-token",
        "--hf_token",
        dest="hf_token",
        default=s.hf_token,
        metavar="TOKEN",
        help="Token Hugging Face (también en model.hf_token del JSON).",
    )
    p.add_argument("--vlm-model", default=s.vlm_model, help="Id del modelo SigLIP en Hugging Face.")
    p.add_argument(
        "--multicrop-mode",
        default=s.multicrop_mode,
        choices=["off", "light", "full", "batch3"],
    )
    p.add_argument("--net-th", type=float, default=s.net_th)
    p.add_argument("--net-margin-th", type=float, default=s.net_margin_th)
    p.add_argument(
        "--decision-mode",
        default=s.decision_mode,
        choices=["weighted", "majority"],
    )
    p.add_argument("--emit-image-embedding", action="store_true", default=s.emit_image_embedding)
    p.add_argument("--tau-max-prob", type=float, default=s.tau_max_prob)
    p.add_argument("--min-margin", type=float, default=s.min_margin)
    p.add_argument(
        "--max-entropy",
        type=float,
        default=s.max_entropy,
        help="Si >= 0, rechazar si entropía > valor.",
    )
    p.add_argument("--pretty-json", action="store_true")
    p.add_argument("--no-stdout-json", action="store_true")
    p.add_argument("--no-write-file", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    pre_args, argv_rest = pre.parse_known_args()

    init_settings(pre_args.config or None, force_reload=True)
    s = get_settings()
    if s.config_path:
        print(f"[vlm_rc2] Config: {s.config_path}", file=sys.stderr, flush=True)
    elif not (pre_args.config or "").strip():
        print(
            f"[vlm_rc2] Aviso: sin {_DEFAULT_CONFIG_BASENAME}; "
            f"copia {_EXAMPLE_BASENAME} y edítalo.",
            file=sys.stderr,
            flush=True,
        )

    args = _build_parser(s).parse_args(argv_rest)

    def _log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    chunk_dir = Path(args.chunk_dir).expanduser().resolve()
    chunk_stem = chunk_dir.name or "chunk"
    opts = options_from_namespace(args, s)
    opts.write_file = not args.no_write_file

    out_s = str(args.output_json or "").strip()
    if out_s:
        opts.output_json = Path(out_s).expanduser().resolve()
    elif args.no_write_file:
        opts.output_json = None

    service_url = service_url_from_settings(cli=str(args.semantics_service_url or ""))
    timeout = client_timeout_sec(
        cli=float(args.service_timeout_sec) if float(args.service_timeout_sec) > 0 else None
    )

    try:
        if service_url:
            _log(f"[vlm_rc2] Modo servicio → {service_url}")
            opts_service = opts
            opts_service.write_file = False
            payload = request_chunk_semantics(
                service_url,
                str(chunk_dir),
                opts_service,
                timeout_sec=timeout,
            )
            if not args.no_write_file:
                persist_chunk_result(chunk_dir, chunk_stem, payload, write_file=True)
        else:
            _log("[vlm_rc2] Modo local — cargando SigLIP…")
            clf = build_classifier(opts)
            _log(f"[vlm_rc2] SigLIP listo en {clf.device} ({clf.model_name})")
            payload = process_chunk_directory(
                chunk_dir,
                clf,
                opts,
                stage="run_semantics_rc2",
            )
            if not args.no_write_file:
                persist_chunk_result(chunk_dir, chunk_stem, payload, write_file=True)

        status = str(payload.get("vlm_rc2_pipeline_status") or "")
        n_fail = int(payload.get("vlm_inference_failure_count") or 0)
        if n_fail and not args.quiet:
            _log(f"[vlm_rc2] Aviso: {n_fail} frame(s) con fallo de inferencia VLM.")
        if status == "skipped" and not args.quiet:
            warns = payload.get("chunk_validation_warnings") or payload.get("chunk_validation_issues")
            _log(f"[vlm_rc2] Chunk omitido (skipped): {warns}")

        if not args.no_stdout_json:
            write_json_stdout(payload, pretty=bool(args.pretty_json))
        if not args.quiet and not args.no_write_file and status in ("ok", "skipped"):
            _log(f"[vlm_rc2] Resultado: status={status}")

    except SemanticsServiceError as e:
        err_payload = semantics_runtime_error_payload(
            chunk_dir=chunk_dir,
            chunk_stem=chunk_stem,
            exc=e,
            stage="run_semantics_rc2_service_client",
        )
        err_payload["service_url"] = service_url
        err_payload["service_response"] = (e.body or "")[:2000]
        _log(f"[vlm_rc2] ERROR servicio: {e}\n{traceback.format_exc()}")
        if not args.no_write_file:
            persist_chunk_result(chunk_dir, chunk_stem, err_payload, write_file=True)
        if not args.no_stdout_json:
            write_json_stdout(err_payload, pretty=bool(args.pretty_json))

    except Exception as e:
        err_payload = semantics_runtime_error_payload(
            chunk_dir=chunk_dir,
            chunk_stem=chunk_stem,
            exc=e,
            stage="run_semantics_rc2",
        )
        _log(f"[vlm_rc2] ERROR: {e}\n{traceback.format_exc()}")
        if not args.no_write_file:
            persist_chunk_result(chunk_dir, chunk_stem, err_payload, write_file=True)
        if not args.no_stdout_json:
            write_json_stdout(err_payload, pretty=bool(args.pretty_json))


if __name__ == "__main__":
    main()
    sys.exit(0)
