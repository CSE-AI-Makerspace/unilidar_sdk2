#!/usr/bin/env python3
"""Multi-model YOLO detector for the Amcrest fisheye camera.

Runs any combination of three ultralytics YOLO models on the latest RTSP frame and
publishes a single merged, labeled 2D detection stream as JSON UDP datagrams:

  - pose  : YOLO-pose, people only, with 17 COCO keypoints (skeletons).
  - det   : YOLO detection, 80 fixed COCO classes (chair, backpack, laptop, ...).
  - world : YOLO-World open-vocabulary, detects arbitrary classes you name in text.

Overlapping boxes of the same label are de-duplicated with IoU, preferring the source with
the richest output (pose > det > world) so people keep their skeleton. Same container, GPU,
RTSP and UDP path regardless of which models are enabled. All config via env vars.

Datagram schema:
  {"ts": float, "frame_w": int, "frame_h": int,
   "persons": [{"box":[x1,y1,x2,y2], "conf":float, "label":str, "source":str,
                "kpts":[[x,y,c], ...]}]}   # kpts empty for non-pose detections
"""
from __future__ import annotations

import json
import os
import socket
import time

import cv2
import numpy as np
from ultralytics import YOLO

SOURCE_PRIORITY = {"pose": 0, "det": 1, "world": 2}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def build_rtsp_url() -> str:
    explicit = _env("RTSP_URL", "")
    if explicit:
        return explicit
    user = _env("RTSP_USERNAME", "admin")
    password = _env("RTSP_PASSWORD", "")
    host = _env("RTSP_HOST", "10.0.0.24")
    subtype = _env("RTSP_SUBTYPE", "0")
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"rtsp://{auth}{host}:554/cam/realmonitor?channel=1&subtype={subtype}"


def open_capture(url: str) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def load_models(enabled: list[str], paths: dict[str, str], world_classes: list[str]) -> dict[str, YOLO]:
    models: dict[str, YOLO] = {}
    for name in enabled:
        path = paths.get(name)
        if not path:
            continue
        model = YOLO(path)
        if name == "world" and world_classes:
            model.set_classes(world_classes)
        models[name] = model
        print(f"[detect] loaded {name}: {path}", flush=True)
    return models


def run_model(source: str, model: YOLO, frame: np.ndarray, imgsz: int, conf: float, device: str) -> list[dict]:
    r = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names
    kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

    out = []
    for i in range(len(boxes)):
        label = names[clss[i]] if isinstance(names, (list, tuple)) else names.get(clss[i], str(clss[i]))
        out.append(
            {
                "box": [round(float(v), 1) for v in boxes[i]],
                "conf": round(float(confs[i]), 3),
                "label": str(label),
                "source": source,
                "kpts": (
                    [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in kpts[i]]
                    if kpts is not None
                    else []
                ),
            }
        )
    return out


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def merge(detections: list[dict], iou_thr: float = 0.6) -> list[dict]:
    """De-dup same-label overlapping boxes; keep the highest-priority/conf instance."""
    ordered = sorted(detections, key=lambda d: (SOURCE_PRIORITY.get(d["source"], 9), -d["conf"]))
    kept: list[dict] = []
    for d in ordered:
        if any(k["label"] == d["label"] and _iou(d["box"], k["box"]) > iou_thr for k in kept):
            continue
        kept.append(d)
    return kept


def main() -> None:
    url = build_rtsp_url()
    enabled = [m.strip() for m in _env("MODELS", "pose,det").split(",") if m.strip()]
    paths = {
        "pose": _env("POSE_MODEL", "yolo11n-pose.pt"),
        "det": _env("DET_MODEL", "yolo11n.pt"),
        "world": _env("WORLD_MODEL", "yolov8s-worldv2.pt"),
    }
    world_classes = [c.strip() for c in _env("WORLD_CLASSES", "").split(",") if c.strip()]
    pub_host = _env("PUB_HOST", "127.0.0.1")
    pub_port = int(_env("PUB_PORT", "7700"))
    conf = float(_env("CONF", "0.25"))
    imgsz = int(_env("IMGSZ", "1280"))
    device = _env("DEVICE", "0")
    iou_thr = float(_env("MERGE_IOU", "0.6"))
    overlay_path = _env("SAVE_OVERLAY", "")
    overlay_every = float(_env("OVERLAY_EVERY", "2.0"))

    print(f"[detect] models={enabled} imgsz={imgsz} conf={conf} device={device}", flush=True)
    if "world" in enabled:
        print(f"[detect] world classes={world_classes or '(none -> world disabled at runtime)'}", flush=True)
    print(f"[detect] publishing UDP JSON -> {pub_host}:{pub_port}", flush=True)

    models = load_models(enabled, paths, world_classes)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (pub_host, pub_port)

    frames = published = 0
    last_stats = time.time()
    last_overlay = 0.0

    while True:
        cap = open_capture(url)
        if not cap.isOpened():
            print("[detect] cannot open RTSP, retrying in 3s", flush=True)
            time.sleep(3)
            continue
        print("[detect] stream opened", flush=True)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[detect] frame read failed, reconnecting", flush=True)
                break
            ts = time.time()
            h, w = frame.shape[:2]

            raw: list[dict] = []
            for name, model in models.items():
                raw.extend(run_model(name, model, frame, imgsz, conf, device))
            persons = merge(raw, iou_thr)

            msg = {"ts": ts, "frame_w": w, "frame_h": h, "persons": persons}
            try:
                sock.sendto(json.dumps(msg).encode("utf-8"), dst)
                published += 1
            except OSError as exc:
                print(f"[detect] publish failed: {exc}", flush=True)

            frames += 1
            if overlay_path and (ts - last_overlay) >= overlay_every:
                _save_overlay(frame, persons, overlay_path)
                last_overlay = ts

            if ts - last_stats >= 5.0:
                fps = frames / (ts - last_stats)
                counts: dict[str, int] = {}
                for p in persons:
                    counts[p["label"]] = counts.get(p["label"], 0) + 1
                print(f"[detect] {fps:.1f} fps | {len(persons)} objs {counts} | published={published}", flush=True)
                frames = 0
                last_stats = ts

        cap.release()
        time.sleep(1)


def _save_overlay(frame: np.ndarray, persons: list[dict], path: str) -> None:
    img = frame.copy()
    for p in persons:
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        color = (0, 255, 0) if p["source"] == "pose" else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{p['label']} {p['conf']:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for x, y, c in p["kpts"]:
            if c > 0.3:
                cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), -1)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    main()
