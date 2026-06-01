#!/usr/bin/env python3
"""Host-side Unitree L2 reader.

Spawns the prebuilt `cloud_csv_udp` SDK binary and parses its stdout into point-cloud
frames. The CSV protocol (see examples/cloud_csv_udp.cpp) is:

    FRAME,<count>,<device_stamp>,<num_points>,<ring_num>
    x,y,z,intensity,ring
    ... (num_points lines) ...
    END

Each parsed frame is stored as the latest `(host_ts, device_stamp, points Nx5)` where the
columns are [x, y, z, intensity, ring]. `host_ts` is wall-clock time at END, used to
time-match against camera detections (same clock domain on the Spark).

Parsing mirrors visualize_lidar.py:112-163 so behavior stays consistent with the viewer.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

DEFAULT_BIN = "/home/aimakeradmin/Documents/Github/unilidar_sdk2/unitree_lidar_sdk/bin/cloud_csv_udp"


@dataclass
class LidarFrame:
    host_ts: float
    device_stamp: float
    points: np.ndarray  # Nx5 float64: x, y, z, intensity, ring


class LidarReader:
    def __init__(self, binary_path: str = DEFAULT_BIN) -> None:
        self.binary_path = binary_path
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: LidarFrame | None = None
        self.frames_received = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="lidar-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)

    def get_latest(self) -> LidarFrame | None:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [self.binary_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            print(f"[lidar] failed to launch reader: {exc}", flush=True)
            return

        print("[lidar] reader started", flush=True)
        device_stamp = 0.0
        frame_points: list[list[float]] = []

        for line in iter(self._proc.stdout.readline, ""):
            if self._stop.is_set() or not line:
                break
            line = line.strip()
            if not line:
                continue

            if line.startswith("FRAME,"):
                frame_points = []
                parts = line.split(",")
                device_stamp = float(parts[2]) if len(parts) > 2 else 0.0
                continue
            if line == "END":
                if frame_points:
                    points = np.asarray(frame_points, dtype=np.float64)
                    with self._lock:
                        self._latest = LidarFrame(time.time(), device_stamp, points)
                        self.frames_received += 1
                    if self.frames_received % 20 == 0:
                        print(f"[lidar] frames received: {self.frames_received}", flush=True)
                continue
            if line.startswith(("Unitree", "[UDPHandler]", "IMU", "ERROR")):
                print(f"[lidar] {line}", flush=True)
                continue

            parts = line.split(",", 5)
            if len(parts) < 3:
                continue
            try:
                frame_points.append(
                    [
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]) if len(parts) > 3 else 0.0,
                        float(parts[4]) if len(parts) > 4 else 0.0,
                    ]
                )
            except ValueError:
                continue

        print("[lidar] reader stopped", flush=True)


if __name__ == "__main__":
    # Smoke test: print frame stats for a few seconds.
    reader = LidarReader()
    reader.start()
    try:
        for _ in range(20):
            time.sleep(1.0)
            frame = reader.get_latest()
            if frame is not None:
                pts = frame.points
                print(
                    f"latest: {len(pts)} pts | stamp={frame.device_stamp} | "
                    f"x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
                    f"z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]",
                    flush=True,
                )
    finally:
        reader.stop()
