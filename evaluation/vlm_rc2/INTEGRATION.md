# Integración vlm_rc2 — servicio persistente + Triton

Copia la carpeta **`vlm_rc2/`** completa al proyecto de despliegue (junto al contenedor Triton o en la misma imagen).

## Arquitectura

```
Orquestador Kanvis
    │  escribe chunk en volumen: /data/chunks/chunk_XXX/
    │  POST o CLI con --semantics-service-url
    ▼
vlm-rc2-semantics:8080  (semantics_service.py, modelo cargado 1 vez)
    │  lee /data/chunks/chunk_XXX/frames + frames_meta.json
    ▼
JSON semántica (stdout o chunk_XXX_vlm.json)
    ▼
aggregate_rules_rc2.py  (sin GPU; puede ser otro proceso/CLI)
```

**Triton**: este worker es el componente de inferencia retail SigLIP con lógica de chunk. Triton puede convivir en el mismo host/contenedor para otros modelos; este servicio no usa la API HTTP de Triton por defecto.

## 0. Configuración (`vlm_rc2.settings.json`)

```bash
cd vlm_rc2
cp vlm_rc2.settings.example.json vlm_rc2.settings.json
# Edita device, hf_token, client.semantics_service_url, etc.
```

El fichero se busca automáticamente en:

1. Ruta de `--config` (si la pasas al CLI o al servicio)
2. `VLM_RC2_CONFIG_PATH` (solo para indicar ruta alternativa)
3. `vlm_rc2/vlm_rc2.settings.json`
4. `./vlm_rc2.settings.json` (directorio de trabajo)

**Prioridad por valor:** argumento CLI > variable de entorno > JSON > defecto interno.

Ejemplo mínimo:

```json
{
  "service": { "host": "0.0.0.0", "port": 8080 },
  "client": { "semantics_service_url": "http://127.0.0.1:8080", "timeout_sec": 900 },
  "model": {
    "device": "cuda:0",
    "hf_token": "hf_xxxx",
    "vlm_model": "google/siglip-so400m-patch14-384"
  },
  "paths": { "chunks_root": "/data/chunks" }
}
```

No subas `vlm_rc2.settings.json` con tokens a git (está en `.gitignore`).

## 1. Levantar el servicio (local con GPU)

```bash
cd vlm_rc2
pip install -r requirements-service.txt
# + dependencias del proyecto (torch, transformers, cv2, …)

cp vlm_rc2.settings.example.json vlm_rc2.settings.json
# Edita model.device y model.hf_token

python semantics_service.py
# Lee host/port/model del JSON → http://0.0.0.0:8080 por defecto
```

Comprobar:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/ready
```

`/ready` devuelve 503 hasta que el modelo terminó de cargar.

## 2. Procesar un chunk vía CLI (cliente → servicio)

En `vlm_rc2.settings.json` deja `client.semantics_service_url` apuntando al servicio.
Misma ruta de chunk **visible para el servicio** (volumen compartido):

```bash
python run_semantics_rc2.py \
  --chunk-dir /data/chunks/chunk_001 \
  --no-stdout-json
```

Si el servicio está en otra URL puntual:

```bash
python run_semantics_rc2.py \
  --chunk-dir /data/chunks/chunk_001 \
  --semantics-service-url http://otro-host:8080
```

## 3. API HTTP directa

```bash
curl -s -X POST http://127.0.0.1:8080/v1/semantics/chunk \
  -H 'Content-Type: application/json' \
  -d '{
    "chunk_dir": "/data/chunks/chunk_001",
    "write_file": true,
    "tau_max_prob": 0.35,
    "min_margin": 0.08
  }'
```

Respuesta: mismo JSON que `run_semantics_rc2` (campos `frames`, `vlm_rc2_pipeline_status`, etc.).

## 4. Docker + reinicio automático

```bash
cd vlm_rc2
cp vlm_rc2.settings.example.json vlm_rc2.settings.json
# Edita model.hf_token, paths.chunks_root, service.port si hace falta

export CHUNKS_HOST_PATH=/ruta/en/host/a/chunks
docker compose -f docker-compose.semantics-service.yml up -d --build
```

`restart: unless-stopped` + `healthcheck` → si el proceso muere, Docker lo reinicia (el modelo se vuelve a cargar al arranque).

En **Kubernetes**: Deployment + `livenessProbe` → `/health`, `readinessProbe` → `/ready`.

## 5. Variables de entorno (opcionales)

La configuración principal va en **`vlm_rc2.settings.json`**. Solo usa env para sobreescribir puntualmente:

| Variable | Uso |
|----------|-----|
| `VLM_RC2_CONFIG_PATH` | Ruta alternativa al JSON de configuración |
| `VLM_RC2_SEMANTICS_SERVICE_URL` | Sobreescribe `client.semantics_service_url` |
| `HF_TOKEN` | Sobreescribe `model.hf_token` (si no está en el JSON) |

## 6. Tolerancia a errores de chunk

No se cae el servicio si:

- No existe `frames/` → aviso, 0 frames evaluables o `missing_image_file` por fila
- No existe `frames_meta.json` → aviso, plan vacío
- `frames_meta.json` inválido → aviso, plan vacío
- Carpeta `frames/` sin imágenes → aviso
- Imagen corrupta / `imread` falla → frame omitido (`imread_failed` / `imread_exception`)
- Error VLM en un frame → omitido si `continue_on_inference_error` (activo en servicio)

Estados en JSON:

- `vlm_rc2_pipeline_status`: `ok` | `skipped` | `error`
- `chunk_validation_warnings`: lista de avisos no bloqueantes

## 7. Agregar reglas (sin modelo)

```bash
python aggregate_rules_rc2.py \
  --chunk-main-dir /data/chunks/session_xyz \
  --pretty-json
```

## 8. Integrar en el orquestador (pseudocódigo)

```python
import subprocess

# Opción A: CLI (lee client.semantics_service_url de /app/vlm_rc2/vlm_rc2.settings.json)
subprocess.run([
    "python", "/app/vlm_rc2/run_semantics_rc2.py",
    "--config", "/app/vlm_rc2/vlm_rc2.settings.json",
    "--chunk-dir", chunk_path,
    "--no-stdout-json",
], check=False)

# Opción B: HTTP POST /v1/semantics/chunk (mismo contrato)
```

**Importante**: `chunk_path` debe ser la ruta **dentro del contenedor del servicio** (p. ej. `/data/chunks/chunk_001`), no la del host si difieren.

## 9. Un solo worker en GPU

Un solo proceso del servicio por GPU. Las peticiones se serializan con un lock (una inferencia a la vez).

## 10. Ficheros a copiar al otro proyecto

Copia **toda** la carpeta `vlm_rc2/` (todos los `.py`, `Dockerfile.semantics-service`, `docker-compose.semantics-service.yml`, este `INTEGRATION.md`).

No hace falta `evaluation/session_semantics.py` ni la carpeta `vlm_rc1/`.
