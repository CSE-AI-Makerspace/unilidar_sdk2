#!/usr/bin/env python3
"""Fisheye camera model + LiDAR→image projection (host side, numpy only).

Implements the OpenCV fisheye (Kannala-Brandt) projection in pure numpy so the host fusion
process needs no opencv — calibration (which uses cv2.fisheye / cv2.aruco) runs in the
detector container and only hands over JSON (K, D, extrinsic). The projection here matches
cv2.fisheye.projectPoints exactly so it stays consistent with the calibration.

Frames:
  - LiDAR frame: raw L2 points (x,y,z), +z up.
  - Camera frame: P_cam = R @ P_lidar + t, from the 4x4 extrinsic T_lidar->cam.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FisheyeCamera:
    K: np.ndarray            # 3x3 intrinsics
    D: np.ndarray            # (4,) Kannala-Brandt distortion [k1,k2,k3,k4]
    width: int
    height: int
    T_lidar_to_cam: np.ndarray  # 4x4

    @classmethod
    def from_files(cls, intrinsics_json: str | Path, extrinsic_json: str | Path) -> "FisheyeCamera":
        intr = json.loads(Path(intrinsics_json).read_text())
        extr = json.loads(Path(extrinsic_json).read_text())
        return cls(
            K=np.asarray(intr["K"], dtype=np.float64).reshape(3, 3),
            D=np.asarray(intr["D"], dtype=np.float64).reshape(-1)[:4],
            width=int(intr["width"]),
            height=int(intr["height"]),
            T_lidar_to_cam=np.asarray(extr["T_lidar_to_cam"], dtype=np.float64).reshape(4, 4),
        )

    def to_cam(self, points_lidar: np.ndarray) -> np.ndarray:
        """Nx3 LiDAR points -> Nx3 camera-frame points."""
        R = self.T_lidar_to_cam[:3, :3]
        t = self.T_lidar_to_cam[:3, 3]
        return points_lidar @ R.T + t

    def project(self, points_lidar: np.ndarray, margin: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Project Nx3 LiDAR points to pixels.

        Returns (pixels Nx2 float, valid Nx bool). `valid` is True for points in front of
        the camera whose pixel lands within the image (expanded by `margin`).
        """
        if points_lidar is None or len(points_lidar) == 0:
            return np.empty((0, 2)), np.empty((0,), dtype=bool)

        cam = self.to_cam(points_lidar[:, :3])
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        in_front = z > 1e-6
        z_safe = np.where(in_front, z, 1.0)

        a = x / z_safe
        b = y / z_safe
        r = np.sqrt(a * a + b * b)
        theta = np.arctan(r)

        k1, k2, k3, k4 = self.D
        theta2 = theta * theta
        theta_d = theta * (1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)

        scale = np.where(r > 1e-9, theta_d / np.where(r > 1e-9, r, 1.0), 1.0)
        xp = a * scale
        yp = b * scale

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        u = fx * xp + cx
        v = fy * yp + cy
        pixels = np.column_stack([u, v])

        valid = (
            in_front
            & (u >= -margin) & (u < self.width + margin)
            & (v >= -margin) & (v < self.height + margin)
        )
        return pixels, valid

    def cam_depth(self, points_lidar: np.ndarray) -> np.ndarray:
        """Camera-frame Z (forward depth) per point."""
        return self.to_cam(points_lidar[:, :3])[:, 2]


def points_in_box(pixels: np.ndarray, valid: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Boolean mask of points whose projected pixel falls inside a [x1,y1,x2,y2] box."""
    x1, y1, x2, y2 = box
    inside = (
        valid
        & (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2)
        & (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
    )
    return inside


def lift_keypoints(
    kpts_2d: np.ndarray,
    pixels: np.ndarray,
    points_lidar: np.ndarray,
    inside_mask: np.ndarray,
    radius: float = 25.0,
    kpt_conf_min: float = 0.3,
) -> np.ndarray:
    """Lift 2D keypoints to 3D by nearest projected LiDAR point within `radius` px.

    kpts_2d: (J,3) [x,y,conf]; pixels/points_lidar: projected cloud; inside_mask: which
    cloud points to consider (e.g. the person's box points). Returns (J,4) [x,y,z,valid].
    """
    cand_px = pixels[inside_mask]
    cand_pts = points_lidar[inside_mask, :3]
    out = np.zeros((len(kpts_2d), 4))
    if len(cand_px) == 0:
        return out
    for j, (kx, ky, kc) in enumerate(kpts_2d):
        if kc < kpt_conf_min:
            continue
        d = np.hypot(cand_px[:, 0] - kx, cand_px[:, 1] - ky)
        i = int(np.argmin(d))
        if d[i] <= radius:
            out[j, :3] = cand_pts[i]
            out[j, 3] = 1.0
    return out
