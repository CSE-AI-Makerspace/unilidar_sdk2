#!/usr/bin/env python3
"""LiDAR→world rigid transform from reference-point correspondences (host, numpy only).

Printed ArUco tags are camera-only (no LiDAR depth signature), so we tie the L2 into the
marker world frame with a few LiDAR-visible reference objects placed on known marker
centres. Provide their LiDAR-frame and world-frame coordinates and this solves the rigid
transform (Kabsch / Umeyama without scale) -> calib/lidar_to_world.json, which
calibrate_extrinsic.py compose then combines with world_to_cam.

Input calib/lidar_world_correspondences.json:
  [{"lidar": [x,y,z], "world": [x,y,z]}, ...]   # >= 3 non-collinear pairs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform mapping src -> dst (both Nx3). Returns 4x4."""
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = dc - R @ sc
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib")
    args = ap.parse_args()
    out = Path(args.out)

    pairs = json.loads((out / "lidar_world_correspondences.json").read_text())
    if len(pairs) < 3:
        raise SystemExit(f"need >=3 correspondences, got {len(pairs)}")
    src = np.asarray([p["lidar"] for p in pairs], np.float64)
    dst = np.asarray([p["world"] for p in pairs], np.float64)

    T = kabsch(src, dst)
    resid = np.linalg.norm((src @ T[:3, :3].T + T[:3, 3]) - dst, axis=1)
    print(f"[l2w] {len(pairs)} pairs  rms={np.sqrt((resid**2).mean()):.3f}m  max={resid.max():.3f}m", flush=True)
    (out / "lidar_to_world.json").write_text(json.dumps({"T_lidar_to_world": T.tolist()}, indent=2))
    print(f"[l2w] wrote {out/'lidar_to_world.json'}\n{T}", flush=True)


if __name__ == "__main__":
    main()
