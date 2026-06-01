#!/usr/bin/env python3
"""Back-project a monocular depth map into a colored 3D point cloud and render it (host).

Visualizes the Direction-B output. Uses a nominal pinhole (no calibration yet) purely for
display; the real pipeline would use the calibrated fisheye intrinsics. Renders via the
legacy Open3D Visualizer on DISPLAY=:2 (proven to work headless here) and captures a PNG.

open3d + numpy only (no cv2) so it runs in lidar_venv.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import open3d as o3d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fov", type=float, default=90.0, help="nominal horizontal FOV (deg)")
    ap.add_argument("--stride", type=int, default=4, help="pixel subsample")
    a = ap.parse_args()

    img = np.asarray(o3d.io.read_image(a.image))  # HxWx3 RGB uint8
    depth = np.load(a.depth).astype("float32")
    h, w = depth.shape
    if img.shape[:2] != (h, w):
        # nearest-neighbour resize via index mapping (no cv2)
        ys = (np.linspace(0, img.shape[0] - 1, h)).astype(int)
        xs = (np.linspace(0, img.shape[1] - 1, w)).astype(int)
        img = img[ys][:, xs]

    s = a.stride
    fx = fy = (w / 2.0) / math.tan(math.radians(a.fov) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    us, vs = np.meshgrid(np.arange(0, w, s), np.arange(0, h, s))
    z = depth[::s, ::s].ravel()
    u, v = us.ravel().astype("float32"), vs.ravel().astype("float32")
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    cols = img[::s, ::s].reshape(-1, 3) / 255.0
    keep = (z > 0) & (z < np.percentile(z, 99))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep]))
    pcd.colors = o3d.utility.Vector3dVector(cols[keep])
    print(f"[cloud] {int(keep.sum())} points; z range {z.min():.2f}-{z.max():.2f} m", flush=True)

    c = np.asarray(pcd.get_axis_aligned_bounding_box().get_center())
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=1280, height=960)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.02, 0.02, 0.025])
    ctr = vis.get_view_control()
    ctr.set_lookat(c)
    ctr.set_front([-0.45, -0.35, -0.82])   # oblique view (camera looks along -front)
    ctr.set_up([0.0, -1.0, 0.0])           # image +y is down
    ctr.set_zoom(0.5)
    for _ in range(8):
        vis.poll_events()
        vis.update_renderer()
    vis.capture_screen_image(a.out, do_render=True)
    vis.destroy_window()
    print(f"[cloud] render -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
