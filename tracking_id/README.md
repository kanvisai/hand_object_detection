# Tracking ID benchmark (YOLO11n)

Scripts para comparar `ByteTrack`, `BoT-SORT` y `DeepSORT` sobre videos de personas (`class 0`), con:

- visualizacion en ventana `1280x720`.
- cajas y IDs grandes.
- capa adicional de estabilizacion para mitigar cambios de ID.
- opcion de guardado de video.

## Requisitos

Desde `hand_object_detection/.venv` o tu entorno Python:

- `pip install -r hand_object_detection/requirements.txt`
- Para DeepSORT: `pip install deep-sort-realtime`

## Uso rapido

### ByteTrack

```bash
python tracking_id/run_bytetrack.py --video /ruta/video.mp4 --show --save-video
```

### BoT-SORT

```bash
python tracking_id/run_botsort.py --video /ruta/video.mp4 --show --save-video
```

Con ReID de BoT-SORT (si tu version de ultralytics lo soporta en tu entorno):

```bash
python tracking_id/run_botsort.py --video /ruta/video.mp4 --show --save-video --botsort-with-reid
```

### DeepSORT

```bash
python tracking_id/run_deepsort.py --video /ruta/video.mp4 --show --save-video
```

### OC-SORT (nuevo)

```bash
python tracking_id/run_ocsort.py --video /ruta/video.mp4 --show --save-video
```

### StrongSORT (nuevo)

```bash
python tracking_id/run_strongsort.py --video /ruta/video.mp4 --show --save-video
```

### Norfair (nuevo)

```bash
python tracking_id/run_norfair.py --video /ruta/video.mp4 --show --save-video
```

## Argumentos utiles

- `--output /ruta/salida.mp4` salida de video.
- `--output-fps 25` fps del video guardado (0 = fps original).
- `--max-absence-sec 900` intentar mantener/reasignar ID hasta 15 min de ausencia (ya viene por defecto).
- `--similarity-threshold 0.76` umbral de matching de identidad (default mas tolerante para reducir saltos de ID).
- `--yolo-model yolo11n.pt` modelo YOLO.
- `--device auto` (default, usa CUDA si existe; si no, CPU). Tambien puedes forzar `--device cpu`.
- `--imgsz 640` (default actual), `--conf 0.25`, `--iou 0.45`.

## Nota importante

Mantener exactamente el mismo ID tras una ausencia muy larga (por ejemplo 5 minutos) depende mucho del aspecto visual, la camara y la calidad del tracker/ReID.  
Este pipeline añade una capa de reidentificacion por apariencia (histograma HSV) para mejorar estabilidad, pero no garantiza identidad perfecta en todos los casos.

## Dependencias adicionales para nuevos trackers

- `pip install boxmot` (para `ocsort` y `strongsort`)
- `pip install norfair` (para `norfair`)
