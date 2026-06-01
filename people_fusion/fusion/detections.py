#!/usr/bin/env python3
"""Host-side receiver for the detector container's 2D pose stream.

The detector publishes one UDP JSON datagram per processed frame:
    {"ts": float, "frame_w": int, "frame_h": int,
     "persons": [{"box": [x1,y1,x2,y2], "conf": float, "kpts": [[x,y,c], ...]}]}

This binds the localhost port and keeps the latest datagram available to the fusion loop.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np

COCO_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class Person2D:
    box: np.ndarray          # [x1, y1, x2, y2]
    conf: float
    kpts: np.ndarray         # (J, 3): x, y, conf  (empty for non-pose detections)
    label: str = "person"
    source: str = "pose"

    @property
    def has_skeleton(self) -> bool:
        return len(self.kpts) > 0


@dataclass
class DetectionSet:
    ts: float
    frame_w: int
    frame_h: int
    persons: list[Person2D] = field(default_factory=list)


class DetectionReceiver:
    def __init__(self, host: str = "127.0.0.1", port: int = 7700) -> None:
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: DetectionSet | None = None
        self.received = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.5)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="detection-rx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._sock:
            self._sock.close()

    def get_latest(self, max_age: float | None = None) -> DetectionSet | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        if max_age is not None and (time.time() - latest.ts) > max_age:
            return None
        return latest

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
                persons = [
                    Person2D(
                        box=np.asarray(p["box"], dtype=np.float64),
                        conf=float(p["conf"]),
                        kpts=np.asarray(p.get("kpts", []), dtype=np.float64).reshape(-1, 3),
                        label=str(p.get("label", "person")),
                        source=str(p.get("source", "pose")),
                    )
                    for p in msg.get("persons", [])
                ]
                det = DetectionSet(float(msg["ts"]), int(msg["frame_w"]), int(msg["frame_h"]), persons)
            except (ValueError, KeyError):
                continue
            with self._lock:
                self._latest = det
                self.received += 1


if __name__ == "__main__":
    rx = DetectionReceiver()
    rx.start()
    print("listening on udp 127.0.0.1:7700 ...", flush=True)
    try:
        while True:
            time.sleep(1.0)
            det = rx.get_latest(max_age=5.0)
            if det is None:
                print("(no recent detections)", flush=True)
            else:
                print(f"ts={det.ts:.2f} {det.frame_w}x{det.frame_h} persons={len(det.persons)} rx={rx.received}", flush=True)
    except KeyboardInterrupt:
        rx.stop()
