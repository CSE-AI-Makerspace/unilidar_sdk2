#!/usr/bin/env python3
"""Voxel-based static-background model for the fixed L2 (no camera/calibration needed).

Built from empty-room frames: any voxel consistently occupied is "static structure" (floor,
furniture, walls). At runtime we drop points in those voxels so only the *dynamic* foreground
(people / new objects) survives → kills the "chair = person?" false positives.

The background is **dilated** by `dilate` voxels so it tolerates LiDAR jitter (points wobble
across voxel boundaries; without dilation a halo of edge points around furniture survives and
clusters into false people).

Fixed-scene groundwork for Direction-A (LiDAR learns to count). numpy only.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

_OFF = np.int64(1024)      # index shift so all packed coords are positive (±205 m at 0.2 m)
_BASE = np.int64(4096)


class BackgroundModel:
    def __init__(self, voxel: float = 0.2, dilate: int = 1) -> None:
        self.voxel = voxel
        self.dilate = dilate
        self.occupied: np.ndarray | None = None  # packed int64 voxel keys (dilated)

    def _idx(self, pts: np.ndarray) -> np.ndarray:
        return np.floor(pts[:, :3] / self.voxel).astype(np.int64)

    def _pack(self, idx: np.ndarray) -> np.ndarray:
        x, y, z = idx[:, 0] + _OFF, idx[:, 1] + _OFF, idx[:, 2] + _OFF
        return (x * _BASE + y) * _BASE + z

    def _unpack(self, keys: np.ndarray) -> np.ndarray:
        z = keys % _BASE - _OFF
        y = (keys // _BASE) % _BASE - _OFF
        x = (keys // (_BASE * _BASE)) % _BASE - _OFF
        return np.stack([x, y, z], axis=1)

    def _dilate_keys(self, keys: np.ndarray) -> np.ndarray:
        if self.dilate <= 0 or len(keys) == 0:
            return np.unique(keys)
        idx = self._unpack(np.asarray(keys, dtype=np.int64))
        d = self.dilate
        offs = np.array([[i, j, k] for i in range(-d, d + 1) for j in range(-d, d + 1) for k in range(-d, d + 1)])
        big = (idx[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        return np.unique(self._pack(big))

    def accumulate(self, frames: list[np.ndarray], stable_frac: float = 0.25) -> tuple[int, int]:
        cnt: Counter[int] = Counter()
        for pts in frames:
            cnt.update(np.unique(self._pack(self._idx(pts))).tolist())
        thresh = max(1, int(stable_frac * len(frames)))
        keys = np.array([k for k, c in cnt.items() if c >= thresh], dtype=np.int64)
        self.occupied = self._dilate_keys(keys)
        return len(frames), int(len(self.occupied))

    def foreground_mask(self, pts: np.ndarray) -> np.ndarray:
        if self.occupied is None or len(self.occupied) == 0:
            return np.ones(len(pts), dtype=bool)
        return ~np.isin(self._pack(self._idx(pts)), self.occupied)

    def save(self, path: str) -> None:
        np.savez(path, voxel=self.voxel, dilate=self.dilate, occupied=self.occupied)

    @classmethod
    def load(cls, path: str) -> "BackgroundModel":
        d = np.load(path)
        m = cls(float(d["voxel"]), int(d["dilate"]))
        m.occupied = d["occupied"]
        return m
