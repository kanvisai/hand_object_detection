#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tracking_common import IdentityStabilizer, color_for_id, draw_box_with_big_id


def _resolve_device(device_arg: str) -> str:
    if str(device_arg).lower() == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device_arg


def _detect_people(
    model: YOLO,
    frame: np.ndarray,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, device=device, classes=[0], verbose=False)
    result = results[0]
    if result.boxes is None or result.boxes.xyxy is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.zeros((len(boxes),), dtype=np.float32)
    out: list[tuple[float, tuple[int, int, int, int]]] = []
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in b]
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((float(s), (x1, y1, x2, y2)))
    return out


def _to_track_pairs_flexible(tracks: Any) -> list[tuple[int, tuple[int, int, int, int]]]:
    out: list[tuple[int, tuple[int, int, int, int]]] = []
    if tracks is None:
        return out
    arr = np.asarray(tracks)
    if arr.size == 0:
        return out
    arr = np.atleast_2d(arr)
    for row in arr:
        r = row.tolist()
        if len(r) < 5:
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in r[:4]]
        track_id = None
        # Formatos tipicos:
        # [x1,y1,x2,y2,id,conf,cls]
        # [x1,y1,x2,y2,conf,cls,id]
        idx_candidates = [4, 6, 5]
        for idx in idx_candidates:
            if idx < len(r):
                val = r[idx]
                if float(val).is_integer() and int(val) >= 0:
                    track_id = int(val)
                    break
        if track_id is None:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((track_id, (x1, y1, x2, y2)))
    return out


def _run_ocsort(dets: list[tuple[float, tuple[int, int, int, int]]], tracker: Any, frame: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    if not dets:
        tracks = tracker.update(np.empty((0, 6), dtype=np.float32), frame)
        return _to_track_pairs_flexible(tracks)
    arr = []
    for score, (x1, y1, x2, y2) in dets:
        arr.append([x1, y1, x2, y2, score, 0.0])
    tracks = tracker.update(np.asarray(arr, dtype=np.float32), frame)
    return _to_track_pairs_flexible(tracks)


def _run_strongsort(
    dets: list[tuple[float, tuple[int, int, int, int]]],
    tracker: Any,
    frame: np.ndarray,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    if not dets:
        tracks = tracker.update(np.empty((0, 6), dtype=np.float32), frame)
        return _to_track_pairs_flexible(tracks)
    arr = []
    for score, (x1, y1, x2, y2) in dets:
        arr.append([x1, y1, x2, y2, score, 0.0])
    tracks = tracker.update(np.asarray(arr, dtype=np.float32), frame)
    return _to_track_pairs_flexible(tracks)


def _run_norfair(dets: list[tuple[float, tuple[int, int, int, int]]], tracker: Any) -> list[tuple[int, tuple[int, int, int, int]]]:
    from norfair import Detection

    nf_dets = []
    for score, (x1, y1, x2, y2) in dets:
        points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        nf_dets.append(Detection(points=points, scores=np.array([score, score], dtype=np.float32)))

    tracked = tracker.update(nf_dets)
    out: list[tuple[int, tuple[int, int, int, int]]] = []
    for t in tracked:
        if t.id is None:
            continue
        est = np.asarray(t.estimate)
        if est.shape[0] < 2:
            continue
        x1, y1 = [int(v) for v in est[0]]
        x2, y2 = [int(v) for v in est[1]]
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((int(t.id), (x1, y1, x2, y2)))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trackers extra: ocsort / strongsort / norfair")
    p.add_argument("--video", required=True)
    p.add_argument("--algorithm", choices=["ocsort", "strongsort", "norfair"], required=True)
    p.add_argument("--yolo-model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="auto")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--output", default="")
    p.add_argument("--window-width", type=int, default=1280)
    p.add_argument("--window-height", type=int, default=720)
    p.add_argument("--max-absence-sec", type=float, default=900.0)
    p.add_argument("--similarity-threshold", type=float, default=0.76)
    return p


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.video).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el video: {input_path}")

    device = _resolve_device(args.device)
    model = YOLO(args.yolo_model)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if (not fps or math.isnan(fps) or fps <= 0) else float(fps)
    max_absence_frames = int(max(1, args.max_absence_sec * fps))
    stabilizer = IdentityStabilizer(
        max_absence_frames=max_absence_frames,
        similarity_threshold=float(args.similarity_threshold),
    )

    algo = args.algorithm
    tracker = None
    if algo in {"ocsort", "strongsort"}:
        try:
            from boxmot import OCSORT, StrongSORT
        except ImportError as exc:
            raise ImportError(
                "Para usar ocsort/strongsort instala: pip install boxmot"
            ) from exc
        if algo == "ocsort":
            tracker = OCSORT(
                det_thresh=float(args.conf),
                max_age=120,
                min_hits=3,
                iou_threshold=0.3,
            )
        else:
            tracker = StrongSORT(
                model_weights=Path("osnet_x0_25_msmt17.pt"),
                device=device,
                fp16=False if str(device).lower() == "cpu" else True,
                max_age=120,
                nn_budget=100,
                max_iou_dist=0.7,
                max_cos_dist=0.2,
            )
    else:
        try:
            from norfair import Tracker
        except ImportError as exc:
            raise ImportError("Para usar norfair instala: pip install norfair") from exc
        tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=45,
            hit_counter_max=30,
            initialization_delay=2,
        )

    writer = None
    out_path = None
    if args.save_video:
        out_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_name(
            f"{input_path.stem}_{algo}_stable.mp4"
        )
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (args.window_width, args.window_height),
        )

    if args.show:
        cv2.namedWindow("tracking_id_extra", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tracking_id_extra", args.window_width, args.window_height)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        dets = _detect_people(model, frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=device)
        if algo == "ocsort":
            tracks = _run_ocsort(dets, tracker, frame)
        elif algo == "strongsort":
            tracks = _run_strongsort(dets, tracker, frame)
        else:
            tracks = _run_norfair(dets, tracker)

        stabilizer.step_frame(frame_idx)
        assignments = stabilizer.assign_batch(detections=tracks, frame_bgr=frame, frame_idx=frame_idx)
        for raw_id, cid, bbox in assignments:
            draw_box_with_big_id(frame, bbox, cid, raw_id, color_for_id(cid))

        info = f"Alg: {algo} | frame {frame_idx} | tracks: {len(tracks)}"
        cv2.putText(frame, info, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        out_frame = cv2.resize(frame, (args.window_width, args.window_height), interpolation=cv2.INTER_LINEAR)

        if args.show:
            cv2.imshow("tracking_id_extra", out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
        if writer is not None:
            writer.write(out_frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if out_path is not None:
        print(f"Video guardado en: {out_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
