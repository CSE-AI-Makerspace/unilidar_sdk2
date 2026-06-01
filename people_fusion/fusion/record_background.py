#!/usr/bin/env python3
"""Record the static-background model from the empty room (host, L2 only).

Run with the room EMPTY: warms up the L2, accumulates frames into a voxel background, and
saves it. The viewer then subtracts it so only people / new objects get tracked.

  python record_background.py --seconds 20 --out ../calib/background.npz
"""
from __future__ import annotations

import argparse
import time

from background import BackgroundModel
from lidar_reader import LidarReader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--dilate", type=int, default=1, help="expand background by N voxels (jitter tolerance)")
    ap.add_argument("--out", default="../calib/background.npz")
    ap.add_argument("--warmup-timeout", type=float, default=70.0)
    a = ap.parse_args()

    reader = LidarReader()
    reader.start()
    print("[bg] waiting for L2 frames (keep the room EMPTY)...", flush=True)

    frames: list = []
    last = None
    t_first = None
    t_start = time.time()
    while True:
        f = reader.get_latest()
        now = time.time()
        if f is not None and f.host_ts != last:
            last = f.host_ts
            if t_first is None:
                t_first = now
                print("[bg] frames flowing; recording...", flush=True)
            frames.append(f.points.copy())
        if t_first is None and now - t_start > a.warmup_timeout:
            reader.stop()
            raise SystemExit("[bg] no L2 frames within warmup timeout")
        if t_first is not None and now - t_first >= a.seconds:
            break
        time.sleep(0.05)

    reader.stop()
    model = BackgroundModel(voxel=a.voxel, dilate=a.dilate)
    n, vox = model.accumulate(frames)
    model.save(a.out)
    print(f"[bg] built from {n} frames -> {vox} static voxels; saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
