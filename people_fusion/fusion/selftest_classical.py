#!/usr/bin/env python3
"""Synthetic sanity test for classical.py (no live LiDAR needed).

Builds a floor plane plus two human-sized vertical blobs and asserts that
detect_people finds exactly two clusters and the tracker keeps stable IDs across frames.
"""
from __future__ import annotations

import numpy as np

from classical import DetectConfig, PeopleTracker, detect_people

rng = np.random.default_rng(0)


def make_floor(extent: float = 3.0, n: int = 6000) -> np.ndarray:
    xy = rng.uniform(-extent, extent, size=(n, 2))
    z = rng.normal(0.0, 0.01, size=(n, 1))
    return np.hstack([xy, z])


def make_person(cx: float, cy: float, n: int = 300) -> np.ndarray:
    xy = rng.normal([cx, cy], 0.12, size=(n, 2))
    z = rng.uniform(0.1, 1.75, size=(n, 1))
    return np.hstack([xy, z])


def make_frame(people_xy: list[tuple[float, float]]) -> np.ndarray:
    parts = [make_floor()] + [make_person(cx, cy) for cx, cy in people_xy]
    xyz = np.vstack(parts)
    extra = np.zeros((len(xyz), 2))  # intensity, ring columns
    return np.hstack([xyz, extra])


def main() -> int:
    cfg = DetectConfig()
    tracker = PeopleTracker(min_hits=1)

    # Frame 1: two people.
    dets1 = detect_people(make_frame([(1.0, 0.5), (-1.5, -1.0)]), cfg)
    print(f"frame1: {len(dets1)} detections")
    for d in dets1:
        print(f"  centroid=({d.centroid[0]:.2f},{d.centroid[1]:.2f},{d.centroid[2]:.2f}) "
              f"h={d.bbox_max[2]-d.bbox_min[2]:.2f} n={d.n_points}")
    t1 = tracker.update(dets1, ts=0.0)
    ids1 = sorted(t.track_id for t in t1)

    # Frame 2: both people shifted slightly -> IDs must persist.
    dets2 = detect_people(make_frame([(1.15, 0.55), (-1.6, -0.9)]), cfg)
    t2 = tracker.update(dets2, ts=0.1)
    ids2 = sorted(t.track_id for t in t2)

    print(f"frame1 detections={len(dets1)} tracks={ids1}")
    print(f"frame2 detections={len(dets2)} tracks={ids2}")

    ok = len(dets1) == 2 and len(dets2) == 2 and ids1 == ids2 and len(ids1) == 2
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
