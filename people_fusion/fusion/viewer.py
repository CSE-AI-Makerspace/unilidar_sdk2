#!/usr/bin/env python3
"""Live 3D viewer for L2 people/object tracking, with in-scene text labels.

Uses Open3D's O3DVisualizer (gui API) because the legacy Visualizer used by
visualize_lidar.py cannot render 3D text. Shows the live point cloud (height-colored), a 3D
box per tracked object, a floating "<label> #<id>" label, and — in fusion mode — a lifted
3D skeleton for each person.

Modes:
  - classical (default): geometry only, no calibration needed. VIEW_MODE=people (human-sized
    clusters) or VIEW_MODE=objects (all clusters). Labels are size-based ("person?"/"object").
  - fusion: set CALIB_DIR to a folder with intrinsics.json + extrinsic.json and run the
    detector container. Boxes/labels carry the real YOLO class and people get 3D skeletons.

Runs on the headless display (DISPLAY=:2) so it shows through the existing VNC path.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import open3d as o3d
import open3d.visualization as vis
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from classical import DetectConfig, PeopleTracker, detect_clusters, detect_people

LABEL_COLORS = {
    "person": (0.15, 0.95, 0.35),
    "person?": (0.95, 0.85, 0.15),
    "object": (0.35, 0.65, 1.0),
}
# COCO-17 skeleton edges
COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
]


def height_colors(points: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    lo = float(z.min())
    rng = max(float(z.max() - lo), 1e-6)
    t = np.clip((z - lo) / rng, 0.0, 1.0)
    return np.column_stack([0.15 + 0.70 * t, 0.85 - 0.45 * t, 1.00 - 0.65 * t])


def color_for(label: str) -> tuple:
    key = "person" if label in ("person", "person?") else "object"
    return LABEL_COLORS.get(label, LABEL_COLORS[key])


class LiveViewer:
    def __init__(self, title: str = "Unitree L2 — People & Objects", width: int = 1920, height: int = 1080):
        self.roi_radius = float(os.environ.get("ROI_RADIUS", "10.0"))
        self.roi_z = float(os.environ.get("ROI_Z", "3.0"))
        self.view_mode = os.environ.get("VIEW_MODE", "people")

        self.app = gui.Application.instance
        self.app.initialize()
        self.win = vis.O3DVisualizer(title, width, height)
        self.win.show_settings = False
        self.win.set_background([0.02, 0.02, 0.025, 1.0], None)
        self.win.show_skybox(False)
        self.win.set_on_close(self._on_close)
        self.app.add_window(self.win)

        self.cloud_mat = rendering.MaterialRecord()
        self.cloud_mat.shader = "defaultUnlit"
        self.cloud_mat.point_size = 2.5
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 3.0
        self.skel_mat = rendering.MaterialRecord()
        self.skel_mat.shader = "unlitLine"
        self.skel_mat.line_width = 5.0

        from lidar_reader import LidarReader

        self.reader = LidarReader()
        self.tracker = PeopleTracker(gate=0.8, max_age=1.5, min_hits=3)
        self.cfg = DetectConfig()

        # Static-background subtraction (fixed scene) to suppress furniture false-positives.
        self.bg = None
        bg_path = os.environ.get("BACKGROUND", "") or os.path.join(os.path.dirname(__file__), "..", "calib", "background.npz")
        if os.path.exists(bg_path):
            from background import BackgroundModel
            self.bg = BackgroundModel.load(bg_path)
            print(f"[viewer] background subtraction ON ({len(self.bg.occupied)} static voxels)", flush=True)
        else:
            print("[viewer] no background model; detecting on full cloud", flush=True)

        self._stop = threading.Event()
        self._dyn_names: set[str] = set()
        self._cloud_added = False
        self._camera_set = False
        self._frames = 0

        # Optional fusion mode.
        self.rx = None
        self.fusion = None
        calib_dir = os.environ.get("CALIB_DIR", "")
        if calib_dir and os.path.exists(os.path.join(calib_dir, "intrinsics.json")) \
                and os.path.exists(os.path.join(calib_dir, "extrinsic.json")):
            from camera_model import FisheyeCamera
            from detections import DetectionReceiver
            from fuse import FusionPipeline
            cam = FisheyeCamera.from_files(
                os.path.join(calib_dir, "intrinsics.json"),
                os.path.join(calib_dir, "extrinsic.json"),
            )
            self.fusion = FusionPipeline(camera=cam)
            self.rx = DetectionReceiver()
            print("[viewer] FUSION mode (calibration loaded)", flush=True)
        else:
            print(f"[viewer] classical mode (VIEW_MODE={self.view_mode})", flush=True)

    def start(self) -> None:
        self.reader.start()
        if self.rx is not None:
            self.rx.start()
        threading.Thread(target=self._worker, name="viewer-worker", daemon=True).start()
        self.app.run()  # blocks on the gui (main) thread until the window closes

    def _on_close(self) -> bool:
        self._stop.set()
        self.reader.stop()
        if self.rx is not None:
            self.rx.stop()
        return True

    def _worker(self) -> None:
        last_ts = None
        while not self._stop.is_set():
            frame = self.reader.get_latest()
            if frame is None or frame.host_ts == last_ts:
                time.sleep(0.05)
                continue
            last_ts = frame.host_ts

            pts = frame.points
            xy2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
            roi = (xy2 < self.roi_radius ** 2) & (np.abs(pts[:, 2]) < self.roi_z)
            cropped = pts[roi]
            if len(cropped) < 50:
                time.sleep(0.05)
                continue
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cropped[:, :3]))
            pcd.colors = o3d.utility.Vector3dVector(height_colors(cropped[:, :3]))

            # items: (id, label, bbox_min, bbox_max, color, skeleton|None)
            items = []
            if self.fusion is not None:
                det = self.rx.get_latest(max_age=0.5)
                for o in self.fusion.fuse_frame(frame, det):
                    items.append((o.track_id, o.label, o.bbox_min, o.bbox_max, color_for(o.label), o.skeleton))
            else:
                fg = cropped[self.bg.foreground_mask(cropped)] if self.bg is not None else cropped
                sg = self.bg is not None  # background already removed the floor
                if self.view_mode == "objects":
                    dets = detect_clusters(fg, self.cfg, min_points=15, max_dim=2.0, skip_ground=sg)
                else:
                    dets = detect_people(fg, self.cfg, skip_ground=sg)
                tracks = self.tracker.update(dets, frame.host_ts)
                for t in tracks:
                    if not dets:
                        continue
                    i = int(np.argmin([np.linalg.norm(d.centroid[:2] - t.centroid[:2]) for d in dets]))
                    d = dets[i]
                    items.append((t.track_id, t.label, d.bbox_min.copy(), d.bbox_max.copy(), color_for(t.label), None))

            self.app.post_to_main_thread(self.win, lambda p=pcd, it=items: self._apply(p, it))
            time.sleep(0.08)

    def _apply(self, pcd: o3d.geometry.PointCloud, items: list) -> None:
        if self._cloud_added:
            self.win.remove_geometry("cloud")
        self.win.add_geometry("cloud", pcd, self.cloud_mat)
        self._cloud_added = True

        for name in self._dyn_names:
            self.win.remove_geometry(name)
        self._dyn_names.clear()
        self.win.clear_3d_labels()

        for tid, label, bmin, bmax, color, skeleton in items:
            aabb = o3d.geometry.AxisAlignedBoundingBox(bmin, bmax)
            aabb.color = color
            bname = f"box_{tid}"
            self.win.add_geometry(bname, aabb, self.line_mat)
            self._dyn_names.add(bname)
            top = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, bmax[2] + 0.15]
            self.win.add_3d_label(top, f"{label} #{tid}")

            if skeleton is not None:
                ls = self._skeleton_lineset(skeleton, color)
                if ls is not None:
                    sname = f"skel_{tid}"
                    self.win.add_geometry(sname, ls, self.skel_mat)
                    self._dyn_names.add(sname)

        if not self._camera_set and len(pcd.points) > 100:
            pts_np = np.asarray(pcd.points)
            center = np.median(pts_np, axis=0)            # robust to stray far points
            eye = center + np.array([0.0, -4.0, 12.0])     # mostly overhead, slight tilt
            self.win.setup_camera(55.0, center.tolist(), eye.tolist(), [0.0, 1.0, 0.0])
            self._camera_set = True
        self._frames += 1
        if self._frames % 25 == 0:
            print(f"[viewer] rendered {self._frames} frames, {len(items)} tracks", flush=True)
        self.win.post_redraw()

    @staticmethod
    def _skeleton_lineset(skeleton: np.ndarray, color) -> o3d.geometry.LineSet | None:
        # skeleton: (J,4) x,y,z,valid
        pts, idx, remap = [], [], {}
        for j in range(len(skeleton)):
            if skeleton[j, 3] > 0:
                remap[j] = len(pts)
                pts.append(skeleton[j, :3])
        lines = [[remap[a], remap[b]] for a, b in COCO_EDGES if a in remap and b in remap]
        if not lines:
            return None
        ls = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.asarray(pts)),
            o3d.utility.Vector2iVector(np.asarray(lines)),
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
        return ls


if __name__ == "__main__":
    LiveViewer().start()
