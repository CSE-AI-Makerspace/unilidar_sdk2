#!/usr/bin/env python3
"""Classical LiDAR people detection + tracking (no ML).

Pipeline per frame:
  1. Structural-plane cascade (remove_structure): RANSAC out the floor, then the ceiling and
     walls (large horizontal/vertical planes), so they don't survive into clustering as junk.
  2. HDBSCAN clustering of the remaining points (handles range-varying density without a
     fixed eps, unlike plain DBSCAN).
  3. Human-size filter on each cluster (height + footprint + min points).
  4. Nearest-neighbour centroid tracker assigns stable IDs across frames.

This is the wide-FOV fallback that covers everything outside the camera cone, and the
baseline that fusion augments with skeletons inside it. Equivalent in spirit to
visualize_lidar.py's plane segmentation + people-height filter, kept standalone here so it
has no viewer side effects.

Convention: L2 frame has +z up (matches the viewer's height color scheme).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from sklearn.cluster import HDBSCAN


@dataclass
class DetectConfig:
    ground_distance_threshold: float = 0.05
    ground_ransac_n: int = 3
    ground_num_iterations: int = 120
    ground_normal_z_min: float = 0.8  # |nz| above this => treat plane as floor
    # structural-plane removal (walls + ceiling), shared RANSAC peeler
    remove_walls: bool = True
    remove_ceiling: bool = True
    plane_distance_threshold: float = 0.05
    plane_min_points: int = 150      # only strip planes at least this big (don't eat objects)
    plane_max_removals: int = 4      # how many structural planes to peel off per frame
    wall_normal_z_max: float = 0.3   # |nz| below this => plane is vertical (a wall)
    ceiling_normal_z_min: float = 0.8  # |nz| above this => horizontal plane
    ceiling_min_height: float = 2.0  # a horizontal plane this high (m) counts as ceiling
    hdbscan_min_cluster_size: int = 25  # smallest blob HDBSCAN will call a cluster
    hdbscan_min_samples: int = 12       # how conservative the density estimate is
    min_cluster_points: int = 25
    height_min: float = 0.8
    height_max: float = 2.2
    footprint_max: float = 0.9


def cluster_labels(non_ground: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """HDBSCAN cluster labels for non-ground points; -1 marks noise."""
    return HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        copy=True,  # don't mutate the caller's array; also future-proofs the sklearn default
    ).fit_predict(non_ground)


@dataclass
class Detection:
    centroid: np.ndarray  # xyz
    bbox_min: np.ndarray  # xyz
    bbox_max: np.ndarray  # xyz
    n_points: int
    label: str = "person"


def remove_ground(points_xyz: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    if len(points_xyz) < cfg.ground_ransac_n + 1:
        return points_xyz
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_xyz))
    plane, inliers = pcd.segment_plane(
        distance_threshold=cfg.ground_distance_threshold,
        ransac_n=cfg.ground_ransac_n,
        num_iterations=cfg.ground_num_iterations,
    )
    normal = np.asarray(plane[:3], dtype=np.float64)
    norm = np.linalg.norm(normal) or 1.0
    if abs(normal[2] / norm) < cfg.ground_normal_z_min:
        # Dominant plane is not the floor (e.g. a wall) -> keep all points.
        return points_xyz
    mask = np.ones(len(points_xyz), dtype=bool)
    mask[inliers] = False
    return points_xyz[mask]


def _peel_planes(points_xyz: np.ndarray, cfg: DetectConfig, is_target) -> np.ndarray:
    """Iterative RANSAC peeler shared by the plane removers.

    Drops large planes for which `is_target(nz, plane_z)` is True (nz = |normal_z|, plane_z =
    mean height). Other large planes are looked past but *kept*, and all non-planar points
    survive -- people/objects don't form large planes. Bounded by plane_max_removals.
    """
    remaining = np.ascontiguousarray(points_xyz)
    kept: list[np.ndarray] = []
    removed = 0
    checks = 0
    while removed < cfg.plane_max_removals and checks < cfg.plane_max_removals + 4:
        checks += 1
        if len(remaining) < cfg.ground_ransac_n + 1:
            break
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining))
        plane, inliers = pcd.segment_plane(
            distance_threshold=cfg.plane_distance_threshold,
            ransac_n=cfg.ground_ransac_n,
            num_iterations=cfg.ground_num_iterations,
        )
        inliers = np.asarray(inliers)
        if len(inliers) < cfg.plane_min_points:
            break  # no large planes left to peel
        normal = np.asarray(plane[:3], dtype=np.float64)
        normal /= np.linalg.norm(normal) or 1.0
        nz = abs(normal[2])
        plane_z = float(remaining[inliers, 2].mean())
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        if is_target(nz, plane_z):
            remaining = np.ascontiguousarray(remaining[mask])  # drop it
            removed += 1
        else:
            kept.append(remaining[inliers])  # keep it, but look underneath
            remaining = np.ascontiguousarray(remaining[mask])
    if kept:
        remaining = np.vstack([remaining, *kept])
    return remaining


def remove_ceiling(points_xyz: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Strip large high horizontal planes (the ceiling)."""
    return _peel_planes(
        points_xyz, cfg,
        lambda nz, z: nz >= cfg.ceiling_normal_z_min and z >= cfg.ceiling_min_height,
    )


def remove_walls(points_xyz: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Strip large near-vertical planes (walls)."""
    return _peel_planes(points_xyz, cfg, lambda nz, z: nz <= cfg.wall_normal_z_max)


def remove_structure(points_xyz: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Cascade the discrete removers: floor -> ceiling -> walls."""
    pts = remove_ground(points_xyz, cfg)
    if cfg.remove_ceiling:
        pts = remove_ceiling(pts, cfg)
    if cfg.remove_walls:
        pts = remove_walls(pts, cfg)
    return pts


def detect_people(points: np.ndarray, cfg: DetectConfig | None = None, skip_ground: bool = False) -> list[Detection]:
    cfg = cfg or DetectConfig()
    if points is None or len(points) == 0:
        return []
    xyz = points[:, :3]
    non_ground = xyz if skip_ground else remove_structure(xyz, cfg)
    if len(non_ground) < cfg.hdbscan_min_cluster_size:
        return []

    labels = cluster_labels(non_ground, cfg)

    detections: list[Detection] = []
    for label in range(labels.max() + 1) if labels.size and labels.max() >= 0 else []:
        cluster = non_ground[labels == label]
        if len(cluster) < cfg.min_cluster_points:
            continue
        bmin = cluster.min(axis=0)
        bmax = cluster.max(axis=0)
        height = bmax[2] - bmin[2]
        footprint = max(bmax[0] - bmin[0], bmax[1] - bmin[1])
        if not (cfg.height_min <= height <= cfg.height_max):
            continue
        if footprint > cfg.footprint_max:
            continue
        detections.append(Detection(cluster.mean(axis=0), bmin, bmax, len(cluster), "person"))
    return detections


def detect_clusters(points: np.ndarray, cfg: DetectConfig | None = None,
                    min_points: int = 20, max_dim: float = 3.0, skip_ground: bool = False) -> list[Detection]:
    """Detect *all* non-ground clusters, labeled tentatively by size.

    Human-sized clusters -> "person?"; everything else -> "object". This is the
    class-agnostic view used by the live viewer before camera fusion supplies real labels.
    """
    cfg = cfg or DetectConfig()
    if points is None or len(points) == 0:
        return []
    xyz = points[:, :3]
    non_ground = xyz if skip_ground else remove_structure(xyz, cfg)
    if len(non_ground) < cfg.hdbscan_min_cluster_size:
        return []
    labels = cluster_labels(non_ground, cfg)

    out: list[Detection] = []
    for label in range(labels.max() + 1) if labels.size and labels.max() >= 0 else []:
        cluster = non_ground[labels == label]
        if len(cluster) < min_points:
            continue
        bmin = cluster.min(axis=0)
        bmax = cluster.max(axis=0)
        height = bmax[2] - bmin[2]
        footprint = max(bmax[0] - bmin[0], bmax[1] - bmin[1])
        if max(height, footprint) > max_dim:
            continue
        human = (cfg.height_min <= height <= cfg.height_max) and (footprint <= cfg.footprint_max)
        tag = "person?" if human else "object"
        out.append(Detection(cluster.mean(axis=0), bmin, bmax, len(cluster), tag))
    return out


@dataclass
class Track:
    track_id: int
    centroid: np.ndarray
    last_ts: float
    hits: int = 1
    misses: int = 0
    label: str = "person"
    history: list[np.ndarray] = field(default_factory=list)


class PeopleTracker:
    """Greedy nearest-neighbour tracker over cluster centroids (xy gate)."""

    def __init__(self, gate: float = 0.8, max_age: float = 1.5, min_hits: int = 2) -> None:
        self.gate = gate
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection], ts: float) -> list[Track]:
        unmatched = set(range(len(detections)))
        # Greedy match each existing track to the nearest detection within the xy gate.
        for track in sorted(self._tracks.values(), key=lambda t: t.last_ts):
            best_i, best_d = -1, self.gate
            for i in unmatched:
                d = float(np.linalg.norm(detections[i].centroid[:2] - track.centroid[:2]))
                if d < best_d:
                    best_i, best_d = i, d
            if best_i >= 0:
                det = detections[best_i]
                track.centroid = det.centroid
                track.label = det.label
                track.last_ts = ts
                track.hits += 1
                track.misses = 0
                track.history.append(det.centroid)
                unmatched.discard(best_i)
            else:
                track.misses += 1

        for i in unmatched:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(tid, detections[i].centroid, ts,
                                      label=detections[i].label, history=[detections[i].centroid])

        # Age out stale tracks.
        self._tracks = {tid: t for tid, t in self._tracks.items() if ts - t.last_ts <= self.max_age}
        return [t for t in self._tracks.values() if t.hits >= self.min_hits]


if __name__ == "__main__":
    import time

    from lidar_reader import LidarReader

    reader = LidarReader()
    reader.start()
    tracker = PeopleTracker()
    try:
        while True:
            time.sleep(0.1)
            frame = reader.get_latest()
            if frame is None:
                continue
            dets = detect_people(frame.points)
            tracks = tracker.update(dets, frame.host_ts)
            if dets:
                ids = ", ".join(
                    f"#{t.track_id}@({t.centroid[0]:.1f},{t.centroid[1]:.1f},{t.centroid[2]:.1f})" for t in tracks
                )
                print(f"{len(dets)} clusters | tracks: {ids}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
