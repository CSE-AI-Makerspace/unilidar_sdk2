#!/usr/bin/env python3
"""Record raw L2 point-cloud frames to .npy files (host).

Provides real point-cloud data for the Direction-A notebook (cluster extraction / classifier
training / inline viewing). Each file is an Nx5 array: x, y, z, intensity, ring.

  python record_clouds.py --n 40 --out ../notebooks/datasets/lidar_clouds
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from lidar_reader import LidarReader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="../notebooks/datasets/lidar_clouds")
    ap.add_argument("--every", type=float, default=0.25, help="seconds between saved frames")
    ap.add_argument("--warmup-timeout", type=float, default=70.0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    reader = LidarReader()
    reader.start()
    print("[rec] waiting for L2 frames...", flush=True)

    saved, last, first, tlast = 0, None, None, 0.0
    t0 = time.time()
    while saved < a.n:
        f = reader.get_latest()
        now = time.time()
        if f is not None and f.host_ts != last:
            last = f.host_ts
            if first is None:
                first = now
                print("[rec] frames flowing", flush=True)
            if now - tlast >= a.every:
                np.save(os.path.join(a.out, f"cloud_{saved:04d}.npy"), f.points.astype("float32"))
                saved += 1
                tlast = now
                if saved % 10 == 0:
                    print(f"[rec] saved {saved}/{a.n}", flush=True)
        if first is None and now - t0 > a.warmup_timeout:
            reader.stop()
            raise SystemExit("[rec] no L2 frames within warmup timeout")
        time.sleep(0.03)

    reader.stop()
    print(f"[rec] saved {saved} clouds to {a.out}", flush=True)


if __name__ == "__main__":
    main()
