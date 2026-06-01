#!/usr/bin/env python3
"""Camera↔LiDAR extrinsic via the floor ArUco markers (camera side runs in the container).

Pipeline (see plan): the floor markers define a world frame; the camera locates itself in
that frame with PnP, the LiDAR is tied to the same frame separately (calibrate_lidar_world.py
on the host), and the two compose into T_lidar->cam.

Modes:
  camera  - detect ArUco (DICT_ARUCO_ORIGINAL) in a live/saved frame, solve PnP against the
            measured marker world corners -> calib/world_to_cam.json.
  compose - T_lidar->cam = T_world->cam @ inv(T_lidar->world) -> calib/extrinsic.json,
            reading world_to_cam.json + lidar_to_world.json.

Needs calib/intrinsics.json (fisheye K,D) and calib/markers_world.json:
  {"dictionary":"DICT_ARUCO_ORIGINAL",
   "markers": {"2": [[x,y,z],...4 corners TL,TR,BR,BL in world m], "4": [...]}}
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
    user, pw = os.environ.get("RTSP_USERNAME", "admin"), os.environ.get("RTSP_PASSWORD", "")
    host, sub = os.environ.get("RTSP_HOST", "10.0.0.24"), os.environ.get("RTSP_SUBTYPE", "0")
    auth = f"{user}:{pw}@" if pw else f"{user}@"
    return f"rtsp://{auth}{host}:554/cam/realmonitor?channel=1&subtype={sub}"


def grab_frame() -> np.ndarray:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(rtsp_url(), cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame = None
    for _ in range(30):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        time.sleep(0.05)
    cap.release()
    if frame is None:
        raise SystemExit("could not grab a frame")
    return frame


def camera(args) -> None:
    out = Path(args.out)
    intr = json.loads((out / "intrinsics.json").read_text())
    K = np.asarray(intr["K"], np.float64).reshape(3, 3)
    D = np.asarray(intr["D"], np.float64).reshape(-1, 1)[:4]
    layout = json.loads((out / "markers_world.json").read_text())
    dict_id = getattr(cv2.aruco, layout.get("dictionary", "DICT_ARUCO_ORIGINAL"))

    frame = cv2.imread(args.image) if args.image else grab_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        raise SystemExit("no markers detected")
    ids = ids.flatten().tolist()
    print(f"[extr] detected markers: {ids}", flush=True)

    obj_pts, img_pts = [], []
    for mc, mid in zip(corners, ids):
        world = layout["markers"].get(str(mid))
        if world is None:
            print(f"[extr] marker {mid} not in layout, skipping", flush=True)
            continue
        obj_pts.append(np.asarray(world, np.float64))         # (4,3)
        img_pts.append(mc.reshape(4, 2).astype(np.float64))   # (4,2)
    if len(obj_pts) < 1:
        raise SystemExit("no detected markers found in layout")
    obj_pts = np.concatenate(obj_pts, axis=0)
    img_pts = np.concatenate(img_pts, axis=0)

    # Undistort the fisheye image points to the pinhole model, then PnP with K (no dist).
    undist = cv2.fisheye.undistortPoints(img_pts.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, undist, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()

    # reprojection error
    proj, _ = cv2.fisheye.projectPoints(obj_pts.reshape(-1, 1, 3), rvec, tvec, K, D)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())
    res = {"T_world_to_cam": T.tolist(), "markers_used": ids, "reproj_px": err}
    (out / "world_to_cam.json").write_text(json.dumps(res, indent=2))
    print(f"[extr] reproj err = {err:.2f}px -> {out/'world_to_cam.json'}", flush=True)


def compose(args) -> None:
    out = Path(args.out)
    w2c = np.asarray(json.loads((out / "world_to_cam.json").read_text())["T_world_to_cam"]).reshape(4, 4)
    l2w = np.asarray(json.loads((out / "lidar_to_world.json").read_text())["T_lidar_to_world"]).reshape(4, 4)
    T_lidar_to_cam = w2c @ l2w
    (out / "extrinsic.json").write_text(json.dumps({"T_lidar_to_cam": T_lidar_to_cam.tolist()}, indent=2))
    print(f"[extr] composed T_lidar_to_cam -> {out/'extrinsic.json'}\n{T_lidar_to_cam}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("camera")
    pc.add_argument("--out", default="calib")
    pc.add_argument("--image", default="", help="use a saved frame instead of live RTSP")
    px = sub.add_parser("compose")
    px.add_argument("--out", default="calib")
    args = ap.parse_args()
    (camera if args.cmd == "camera" else compose)(args)


if __name__ == "__main__":
    main()
