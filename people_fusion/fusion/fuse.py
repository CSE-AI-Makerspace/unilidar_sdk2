#!/usr/bin/env python3
"""Camera↔LiDAR fusion pipeline.

Per frame:
  - Project the L2 cloud into the (calibrated) fisheye image.
  - For each 2D detection (person/object) from the detector container, gather the LiDAR
    points whose projection lands in its box -> metric 3D centroid + extent. For people,
    lift the 2D skeleton to 3D by sampling depth at each keypoint.
  - Run classical cluster detection over the full cloud for the wide FOV outside the camera
    cone; drop clusters already explained by a fused detection.
  - Track everything (class-agnostic NN centroid tracker) for stable IDs across frames.

Falls back to classical-only when no camera calibration is supplied, so it is runnable
before calibration and gains semantic labels + skeletons once calibrated.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from camera_model import FisheyeCamera, lift_keypoints, points_in_box
from classical import DetectConfig, detect_clusters
from detections import DetectionReceiver, DetectionSet
from lidar_reader import LidarFrame, LidarReader


@dataclass
class FusedObject:
    label: str
    source: str                 # "fusion" (camera+lidar) or "classical" (lidar only)
    centroid: np.ndarray        # xyz
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    conf: float
    skeleton: np.ndarray | None = None  # (J,4): x,y,z,valid  (people only)
    track_id: int = -1


@dataclass
class _Track:
    track_id: int
    centroid: np.ndarray
    label: str
    last_ts: float
    hits: int = 1


class _ObjectTracker:
    """Greedy nearest-neighbour centroid tracker; stamps track_id onto FusedObjects."""

    def __init__(self, gate: float = 0.8, max_age: float = 1.5, min_hits: int = 2) -> None:
        self.gate = gate
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: dict[int, _Track] = {}
        self._next = 1

    def update(self, objs: list[FusedObject], ts: float) -> list[FusedObject]:
        unmatched = set(range(len(objs)))
        for tr in sorted(self._tracks.values(), key=lambda t: t.last_ts):
            best_i, best_d = -1, self.gate
            for i in unmatched:
                d = float(np.linalg.norm(objs[i].centroid[:2] - tr.centroid[:2]))
                if d < best_d:
                    best_i, best_d = i, d
            if best_i >= 0:
                o = objs[best_i]
                tr.centroid, tr.label, tr.last_ts = o.centroid, o.label, ts
                tr.hits += 1
                o.track_id = tr.track_id
                unmatched.discard(best_i)
        for i in unmatched:
            tid = self._next
            self._next += 1
            self._tracks[tid] = _Track(tid, objs[i].centroid, objs[i].label, ts)
            objs[i].track_id = tid
        self._tracks = {k: v for k, v in self._tracks.items() if ts - v.last_ts <= self.max_age}
        confirmed = {t.track_id for t in self._tracks.values() if t.hits >= self.min_hits}
        return [o for o in objs if o.track_id in confirmed]


class FusionPipeline:
    def __init__(
        self,
        camera: FisheyeCamera | None = None,
        time_tol: float = 0.2,
        min_box_points: int = 12,
        dedup_xy: float = 0.6,
        cfg: DetectConfig | None = None,
    ) -> None:
        self.camera = camera
        self.time_tol = time_tol
        self.min_box_points = min_box_points
        self.dedup_xy = dedup_xy
        self.cfg = cfg or DetectConfig()
        self.tracker = _ObjectTracker()

    def fuse_frame(self, frame: LidarFrame, det: DetectionSet | None) -> list[FusedObject]:
        objs: list[FusedObject] = []

        # Camera fusion (only if calibrated and a time-matched detection set is available).
        if self.camera is not None and det is not None and abs(det.ts - frame.host_ts) <= self.time_tol:
            pixels, valid = self.camera.project(frame.points)
            for person in det.persons:
                inside = points_in_box(pixels, valid, person.box)
                if int(inside.sum()) < self.min_box_points:
                    continue
                pts = frame.points[inside, :3]
                skeleton = (
                    lift_keypoints(person.kpts, pixels, frame.points, inside)
                    if len(person.kpts)
                    else None
                )
                objs.append(
                    FusedObject(
                        label=person.label,
                        source="fusion",
                        centroid=pts.mean(axis=0),
                        bbox_min=pts.min(axis=0),
                        bbox_max=pts.max(axis=0),
                        conf=person.conf,
                        skeleton=skeleton,
                    )
                )

        # Classical clusters for the wide FOV / fallback; skip ones already fused.
        for d in detect_clusters(frame.points, self.cfg):
            if any(np.linalg.norm(o.centroid[:2] - d.centroid[:2]) < self.dedup_xy for o in objs):
                continue
            objs.append(
                FusedObject(
                    label=d.label,
                    source="classical",
                    centroid=d.centroid,
                    bbox_min=d.bbox_min,
                    bbox_max=d.bbox_max,
                    conf=0.5,
                )
            )

        return self.tracker.update(objs, frame.host_ts)


def main() -> None:
    import os

    calib_dir = os.environ.get("CALIB_DIR", "")
    camera = None
    if calib_dir:
        intr = os.path.join(calib_dir, "intrinsics.json")
        extr = os.path.join(calib_dir, "extrinsic.json")
        if os.path.exists(intr) and os.path.exists(extr):
            camera = FisheyeCamera.from_files(intr, extr)
            print(f"[fuse] loaded calibration from {calib_dir}", flush=True)
    if camera is None:
        print("[fuse] no calibration -> classical-only mode", flush=True)

    reader = LidarReader()
    reader.start()
    rx = DetectionReceiver()
    rx.start()
    pipe = FusionPipeline(camera=camera)

    # Data-harvesting hook (Direction-A): once calibrated, log camera-labeled LiDAR cluster
    # features to train a LiDAR-only person classifier later. Dormant unless HARVEST_FILE set.
    harvest_path = os.environ.get("HARVEST_FILE", "")
    harvest = open(harvest_path, "a") if harvest_path else None
    if harvest:
        print(f"[fuse] harvesting labeled cluster features -> {harvest_path}", flush=True)

    try:
        last = None
        while True:
            frame = reader.get_latest()
            if frame is None or frame.host_ts == last:
                time.sleep(0.05)
                continue
            last = frame.host_ts
            det = rx.get_latest(max_age=0.5)
            objs = pipe.fuse_frame(frame, det)
            if objs:
                summary = ", ".join(
                    f"{o.label}#{o.track_id}({o.source[0]})" for o in objs
                )
                skels = sum(1 for o in objs if o.skeleton is not None)
                print(f"[fuse] {len(objs)} objs | {summary} | skeletons={skels}", flush=True)
            if harvest and objs:
                for o in objs:
                    dims = (o.bbox_max - o.bbox_min).tolist()
                    harvest.write(json.dumps({
                        "ts": frame.host_ts, "label": o.label, "source": o.source,
                        "centroid": [round(float(v), 3) for v in o.centroid],
                        "dims": [round(float(v), 3) for v in dims], "conf": round(float(o.conf), 3),
                    }) + "\n")
                harvest.flush()
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        rx.stop()
        if harvest:
            harvest.close()


if __name__ == "__main__":
    main()
