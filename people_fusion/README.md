# people_fusion — Camera↔LiDAR people & object tracking (Spark04)

Tracks people (and optionally objects) in 3D by fusing the **Unitree L2 LiDAR** (metric 3D,
wide FOV) with the **Amcrest fisheye IP camera** (YOLO detection/pose, skeletons). Built to
run fully Spark-local on the GB10 (Blackwell, aarch64) under the repo's hard rules: no host
apt / base-image mutation, Docker only for helper services.

## Architecture

```
Amcrest 10.0.0.24 ── RTSP ─► detector container (NGC PyTorch, GPU)
                              multi-model YOLO (pose+det+world) ── UDP JSON ─┐
                                                                            ▼
Unitree L2 192.168.1.62 ─► cloud_csv_udp ─► fusion (host, open3d) ─► O3DVisualizer (:2 → VNC)
                                            project+associate, 3D skeletons, classical wide-FOV
```

- **AI in the container** (Blackwell-ready torch verified: sm_121, torch 2.x / CUDA 13).
- **LiDAR + geometry + fusion + viewer on the host** (`lidar_venv`: open3d + numpy only).
- They talk over a localhost UDP detection stream.

## Components (`fusion/`)
| File | Role |
|------|------|
| `lidar_reader.py` | spawn `cloud_csv_udp`, parse `FRAME/END` CSV → `(host_ts, stamp, Nx5)` |
| `classical.py` | RANSAC ground removal + DBSCAN + size filter + NN tracker (class-agnostic) |
| `camera_model.py` | fisheye (Kannala-Brandt) projection + keypoint lifting (numpy only) |
| `detections.py` | host receiver for the detector's UDP stream |
| `fuse.py` | project cloud → associate to 2D detections → 3D pos + extent + lifted skeleton; classical for the wide FOV |
| `viewer.py` | O3DVisualizer live view: cloud + 3D boxes + text labels + skeletons |
| `calibrate_intrinsics.py` | fisheye intrinsics from a checkerboard (container) |
| `calibrate_extrinsic.py` | ArUco floor markers → camera→world; compose → `extrinsic.json` (container) |
| `calibrate_lidar_world.py` | LiDAR→world from reference-point correspondences (host) |
| `background.py` / `record_background.py` | fixed-scene voxel background; suppress furniture (host) |
| `depth_demo.py` / `depth_to_cloud.py` | Direction-B preview: monocular metric depth → 3D cloud |

Detector lives in `detector/` (`detect.py`, `Dockerfile`). Calibration outputs go in `calib/`.

## Run

```bash
# build the detector image (once)
make fusion-build

# start the multi-model detector (export the camera password first)
export RTSP_PASSWORD='...'
make fusion-detector MODELS=pose,det      # or MODELS=pose,det,world (+ WORLD_CLASSES in detect.py)

# record the static background once, with the room EMPTY (suppresses furniture)
make fusion-record-bg

# start the labeled 3D viewer on DISPLAY=:2 (view via the existing VNC tunnel)
make fusion-viewer VIEW_MODE=people        # or VIEW_MODE=objects
#   fusion mode (real labels + skeletons): make fusion-viewer CALIB_DIR=people_fusion/calib

make fusion-status     # processes + recent log
make fusion-stop       # stop viewer + detector + cloud reader
```

The L2 takes ~30–50 s to produce point frames after a cold start.

## Calibration (one-time, fixed install)
1. **Intrinsics** (container): `calibrate_intrinsics.py capture` then `calibrate` with a checkerboard.
2. **Marker layout**: measure the floor ArUco markers (confirmed `DICT_ARUCO_ORIGINAL`, IDs 2 & 4)
   into `calib/markers_world.json` (4 world corners per marker, metres).
3. **Camera→world** (container): `calibrate_extrinsic.py camera`.
4. **LiDAR→world** (host): place reference objects on known marker centres, fill
   `calib/lidar_world_correspondences.json`, run `calibrate_lidar_world.py`.
5. **Compose** (container): `calibrate_extrinsic.py compose` → `calib/extrinsic.json`.

Then restart the viewer with `CALIB_DIR=people_fusion/calib` for semantic labels + 3D skeletons.

## Status
- ✅ Detector: multi-model YOLO (pose+det+world) live at ~15 fps on the GB10 GPU.
- ✅ Classical LiDAR tracking: live, labeled 3D boxes in the viewer.
- ✅ Background subtraction (`make fusion-record-bg` with the room empty) cuts furniture false-positives.
- ✅ Data-harvesting hook + Direction A/B previews in place; full cross-modal learning is post-calibration.
- ✅ Fusion + calibration code complete; execution pending the one-time physical calibration.
