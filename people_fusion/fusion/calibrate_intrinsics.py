#!/usr/bin/env python3
"""Fisheye intrinsic calibration for the Amcrest camera (runs in the detector container).

Two stages:
  capture   - pull RTSP frames, detect a checkerboard, auto-save well-spread views.
  calibrate - run cv2.fisheye.calibrate over the saved views -> calib/intrinsics.json.

The Amcrest is a strong fisheye, so we use the Kannala-Brandt model (cv2.fisheye), which is
what camera_model.FisheyeCamera expects (K + 4 distortion coeffs).

Usage (inside container, repo mounted):
  python calibrate_intrinsics.py capture   --cols 9 --rows 6 --square 0.025 --n 20 --out calib
  python calibrate_intrinsics.py calibrate --cols 9 --rows 6 --square 0.025        --out calib

`--cols/--rows` are *inner corner* counts; `--square` is the square size in metres.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


def rtsp_url() -> str:
    explicit = os.environ.get("RTSP_URL", "").strip()
    if explicit:
        return explicit
    user = os.environ.get("RTSP_USERNAME", "admin")
    pw = os.environ.get("RTSP_PASSWORD", "")
    host = os.environ.get("RTSP_HOST", "10.0.0.24")
    sub = os.environ.get("RTSP_SUBTYPE", "0")
    auth = f"{user}:{pw}@" if pw else f"{user}@"
    return f"rtsp://{auth}{host}:554/cam/realmonitor?channel=1&subtype={sub}"


def object_points(cols: int, rows: int, square: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square


def find_corners(gray: np.ndarray, cols: int, rows: int):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)


def capture(args) -> None:
    out = Path(args.out) / "intrinsics_views"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(rtsp_url(), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    saved = 0
    last_center = None
    print(f"[intr] capturing {args.n} checkerboard views to {out} ...", flush=True)
    while saved < args.n:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, args.cols, args.rows)
        if corners is None:
            continue
        center = corners.mean(axis=0).ravel()
        # auto-accept only views whose board moved enough from the last accepted one
        if last_center is not None and np.linalg.norm(center - last_center) < args.min_move:
            continue
        last_center = center
        np.save(out / f"view_{saved:02d}.npy", corners)
        cv2.imwrite(str(out / f"view_{saved:02d}.png"), frame)
        saved += 1
        print(f"[intr] saved view {saved}/{args.n}", flush=True)
    cap.release()
    print("[intr] capture done", flush=True)


def calibrate(args) -> None:
    out = Path(args.out)
    views = sorted((out / "intrinsics_views").glob("view_*.npy"))
    if len(views) < 5:
        raise SystemExit(f"need >=5 views, found {len(views)}")
    sample = cv2.imread(str(views[0].with_suffix(".png")))
    h, w = sample.shape[:2]
    objp = object_points(args.cols, args.rows, args.square)

    objpoints = [objp.reshape(-1, 1, 3) for _ in views]
    imgpoints = [np.load(v).reshape(-1, 1, 2).astype(np.float32) for v in views]

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    rms, K, D, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints, (w, h), K, D, flags=flags, criteria=term)

    result = {"model": "fisheye_kb", "width": w, "height": h, "K": K.tolist(),
              "D": D.ravel().tolist(), "rms_px": float(rms), "n_views": len(views)}
    (out / "intrinsics.json").write_text(json.dumps(result, indent=2))
    print(f"[intr] rms={rms:.3f}px  K=\n{K}\n D={D.ravel()}", flush=True)
    print(f"[intr] wrote {out/'intrinsics.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--cols", type=int, default=9)
        p.add_argument("--rows", type=int, default=6)
        p.add_argument("--square", type=float, default=0.025)
        p.add_argument("--out", default="calib")
        if name == "capture":
            p.add_argument("--n", type=int, default=20)
            p.add_argument("--min-move", type=float, default=40.0)
    args = ap.parse_args()
    (capture if args.cmd == "capture" else calibrate)(args)


if __name__ == "__main__":
    main()
