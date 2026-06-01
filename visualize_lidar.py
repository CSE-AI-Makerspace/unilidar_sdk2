#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import json
import subprocess
import time
import threading
import signal
from collections import deque
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("lidar_view_config.json")
DEFAULT_CONFIG = {
    "active_mode": "live",
    "modes": {
        "live": {
            "description": "Show only the newest LiDAR point-cloud frame.",
            "accumulate": False,
            "max_frames": 1,
            "max_points": 0,
            "voxel_size": 0.0,
        },
        "accumulate": {
            "description": "Accumulate recent LiDAR frames into a rolling composite point cloud.",
            "accumulate": True,
            "max_frames": 40,
            "max_points": 250000,
            "voxel_size": 0.0,
        },
    },
    "render": {
        "point_size": 2.0,
        "background": [0.02, 0.02, 0.025],
        "color_scheme": "height",
        "color_schemes": ["height", "distance", "intensity", "ring"],
    },
    "filters": {
        "statistical_outlier": {
            "enabled": False,
            "nb_neighbors": 20,
            "std_ratio": 2.0,
        },
        "plane_segmentation": {
            "enabled": False,
            "distance_threshold": 0.03,
            "ransac_n": 3,
            "num_iterations": 80,
            "remove_plane": True,
            "min_inliers": 100,
        },
    },
    "custom_calibration": {
        "enabled": False,
        "ground_plane": None,
        "base_transform_matrix": None,
        "transform_matrix": None,
        "yaw_degrees": 0.0,
        "ground_distance_threshold": 0.03,
        "ground_ransac_n": 3,
        "ground_num_iterations": 120,
        "ground_min_inliers": 500,
    },
}


def load_config():
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        loaded = json.load(config_file)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in loaded.items():
        if key not in {"modes", "render", "filters", "custom_calibration"}:
            merged[key] = value
    merged["render"].update(loaded.get("render", {}))
    for mode_name, mode_config in loaded.get("modes", {}).items():
        if isinstance(mode_config, dict) and isinstance(merged["modes"].get(mode_name), dict):
            merged["modes"][mode_name].update(mode_config)
        else:
            merged["modes"][mode_name] = mode_config
    for filter_name, filter_config in loaded.get("filters", {}).items():
        if isinstance(filter_config, dict) and isinstance(merged["filters"].get(filter_name), dict):
            merged["filters"][filter_name].update(filter_config)
        else:
            merged["filters"][filter_name] = filter_config
    merged["custom_calibration"].update(loaded.get("custom_calibration", {}))
    return merged


def save_config(config):
    with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")

# Storage for point cloud data
latest_points = None
buffer_lock = threading.Lock()
stop_event = threading.Event()
lidar_process = None
frames_received = 0
config = load_config()
active_mode = config.get("active_mode", "live")
accumulated_frames = deque()
force_rebuild = False
reset_view_on_next_cloud = True
running = True
active_view_preset = "default"
people_height_filter = False
last_calibration_points = None

def read_lidar_stream():
    """Run the Lidar UDP example and parse point cloud output"""
    global lidar_process, latest_points, frames_received
    try:
        lidar_process = subprocess.Popen(
            ["/home/aimakeradmin/Documents/Github/unilidar_sdk2/unitree_lidar_sdk/bin/cloud_csv_udp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        print("Lidar stream started. Reading point cloud data...")
        frame_points = []

        for line in iter(lidar_process.stdout.readline, ''):
            if stop_event.is_set() or not line:
                break

            line = line.strip()
            if not line:
                continue
            if line.startswith("FRAME,"):
                frame_points = []
                continue
            if line == "END":
                if frame_points:
                    with buffer_lock:
                        latest_points = np.asarray(frame_points, dtype=np.float64)
                        frames_received += 1
                    if frames_received % 20 == 0:
                        print(f"Rendered source frames: {frames_received}")
                continue
            if line.startswith(("Unitree", "[UDPHandler]", "IMU", "ERROR")):
                print(line)
                continue

            parts = line.split(",", 5)
            if len(parts) < 3:
                continue
            try:
                intensity = float(parts[3]) if len(parts) > 3 else 0.0
                ring = float(parts[4]) if len(parts) > 4 else 0.0
                frame_points.append([
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    intensity,
                    ring,
                ])
            except ValueError:
                continue

    except Exception as e:
        print(f"Lidar error: {e}")

def take_latest_points():
    global latest_points
    with buffer_lock:
        points = latest_points
        latest_points = None
    return points

def colorize_points(points):
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    scheme = config["render"].get("color_scheme", "height")
    if scheme == "distance":
        values = np.linalg.norm(points[:, :3], axis=1)
    elif scheme == "intensity" and points.shape[1] > 3:
        values = points[:, 3]
    elif scheme == "ring" and points.shape[1] > 4:
        values = points[:, 4]
    else:
        values = points[:, 2]

    value_min = float(np.min(values))
    value_range = max(float(np.max(values) - value_min), 1e-6)
    normalized = np.clip((values - value_min) / value_range, 0.0, 1.0)
    colors = np.column_stack((
        0.15 + 0.70 * normalized,
        0.85 - 0.45 * normalized,
        1.00 - 0.65 * normalized,
    ))
    return colors


def apply_people_height_filter(points):
    if not people_height_filter or points.size == 0:
        return points
    z = points[:, 2]
    return points[(z >= 0.6) & (z <= 2.2)]


def apply_statistical_outlier_filter(points):
    filter_config = config["filters"]["statistical_outlier"]
    if not filter_config.get("enabled", False) or points.size == 0:
        return points

    nb_neighbors = int(filter_config.get("nb_neighbors", 20))
    if len(points) <= nb_neighbors:
        return points

    temp_cloud = o3d.geometry.PointCloud()
    temp_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    _, indices = temp_cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=float(filter_config.get("std_ratio", 2.0)),
    )
    if len(indices) == 0:
        return points
    return points[np.asarray(indices, dtype=np.int64)]


def apply_plane_segmentation(points):
    filter_config = config["filters"]["plane_segmentation"]
    if not filter_config.get("enabled", False) or points.size == 0:
        return points

    ransac_n = int(filter_config.get("ransac_n", 3))
    min_inliers = int(filter_config.get("min_inliers", 100))
    if len(points) < max(ransac_n, min_inliers):
        return points

    temp_cloud = o3d.geometry.PointCloud()
    temp_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    try:
        _, inliers = temp_cloud.segment_plane(
            distance_threshold=float(filter_config.get("distance_threshold", 0.03)),
            ransac_n=ransac_n,
            num_iterations=int(filter_config.get("num_iterations", 80)),
        )
    except RuntimeError as exc:
        print(f"Plane segmentation skipped: {exc}")
        return points

    if len(inliers) < min_inliers:
        return points

    if filter_config.get("remove_plane", True):
        mask = np.ones(len(points), dtype=bool)
        mask[np.asarray(inliers, dtype=np.int64)] = False
        return points[mask]
    return points


def apply_native_filters(points):
    points = apply_people_height_filter(points)
    points = apply_statistical_outlier_filter(points)
    points = apply_plane_segmentation(points)
    return points


def normalize_vector(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return vector
    return vector / norm


def rotation_from_vectors(source, target):
    source = normalize_vector(np.asarray(source, dtype=np.float64))
    target = normalize_vector(np.asarray(target, dtype=np.float64))
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot > 0.999999:
        return np.eye(3)
    if dot < -0.999999:
        axis = np.cross(source, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, [0.0, 1.0, 0.0])
        axis = normalize_vector(axis)
        return rotation_matrix_from_axis_angle(axis, np.pi)

    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / max(np.dot(cross, cross), 1e-9))


def rotation_matrix_from_axis_angle(axis, angle_radians):
    axis = normalize_vector(np.asarray(axis, dtype=np.float64))
    x, y, z = axis
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ])


def transform_points(points, matrix):
    if points.size == 0:
        return points
    transformed = points.copy()
    xyz = np.column_stack((points[:, :3], np.ones(len(points), dtype=np.float64)))
    transformed[:, :3] = (np.asarray(matrix, dtype=np.float64) @ xyz.T).T[:, :3]
    return transformed


def update_calibration_transform():
    calibration = config["custom_calibration"]
    base_matrix = calibration.get("base_transform_matrix")
    if not base_matrix:
        calibration["transform_matrix"] = None
        return

    yaw = np.deg2rad(float(calibration.get("yaw_degrees", 0.0)))
    yaw_matrix = np.eye(4)
    yaw_matrix[:3, :3] = rotation_matrix_from_axis_angle([0.0, 0.0, 1.0], yaw)
    calibration["transform_matrix"] = (yaw_matrix @ np.asarray(base_matrix, dtype=np.float64)).tolist()


def apply_custom_calibration(points):
    calibration = config["custom_calibration"]
    if not calibration.get("enabled", False):
        return points
    matrix = calibration.get("transform_matrix")
    if not matrix:
        return points
    return transform_points(points, matrix)


def active_mode_config():
    return config["modes"].get(active_mode, config["modes"]["live"])


def set_mode(mode):
    global active_mode, force_rebuild, reset_view_on_next_cloud
    if mode not in config["modes"]:
        print(f"Unknown mode: {mode}")
        return
    active_mode = mode
    config["active_mode"] = mode
    accumulated_frames.clear()
    force_rebuild = True
    reset_view_on_next_cloud = True
    save_config(config)
    print_status()


def update_point_size(delta):
    current = float(config["render"].get("point_size", 2.0))
    config["render"]["point_size"] = round(max(1.0, min(10.0, current + delta)), 1)
    render_options.point_size = config["render"]["point_size"]
    save_config(config)
    print_status()


def update_accumulation_frames(delta):
    global force_rebuild
    mode_config = config["modes"]["accumulate"]
    mode_config["max_frames"] = max(1, min(300, int(mode_config.get("max_frames", 40)) + delta))
    while len(accumulated_frames) > mode_config["max_frames"]:
        accumulated_frames.popleft()
    force_rebuild = True
    save_config(config)
    print_status()


def update_voxel_size(delta):
    global force_rebuild
    mode_config = config["modes"]["accumulate"]
    current = float(mode_config.get("voxel_size", 0.02))
    mode_config["voxel_size"] = round(max(0.0, min(0.20, current + delta)), 3)
    force_rebuild = True
    save_config(config)
    print_status()


def toggle_statistical_outlier():
    global force_rebuild
    filter_config = config["filters"]["statistical_outlier"]
    filter_config["enabled"] = not filter_config.get("enabled", False)
    force_rebuild = True
    save_config(config)
    print_status()


def toggle_plane_segmentation():
    global force_rebuild
    filter_config = config["filters"]["plane_segmentation"]
    filter_config["enabled"] = not filter_config.get("enabled", False)
    force_rebuild = True
    save_config(config)
    print_status()


def cycle_color_scheme():
    color_schemes = config["render"].get("color_schemes", DEFAULT_CONFIG["render"]["color_schemes"])
    current = config["render"].get("color_scheme", "height")
    try:
        next_index = (color_schemes.index(current) + 1) % len(color_schemes)
    except ValueError:
        next_index = 0
    config["render"]["color_scheme"] = color_schemes[next_index]
    save_config(config)
    print_status()


def calibrate_ground_from_current_cloud():
    global force_rebuild, reset_view_on_next_cloud
    if last_calibration_points is None or len(last_calibration_points) == 0:
        print("Custom calibration: no current/accumulated cloud available yet.")
        return

    calibration = config["custom_calibration"]
    points = last_calibration_points
    ransac_n = int(calibration.get("ground_ransac_n", 3))
    min_inliers = int(calibration.get("ground_min_inliers", 500))
    if len(points) < max(ransac_n, min_inliers):
        print(f"Custom calibration: not enough points for ground plane ({len(points)} available).")
        return

    temp_cloud = o3d.geometry.PointCloud()
    temp_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    try:
        plane_model, inliers = temp_cloud.segment_plane(
            distance_threshold=float(calibration.get("ground_distance_threshold", 0.03)),
            ransac_n=ransac_n,
            num_iterations=int(calibration.get("ground_num_iterations", 120)),
        )
    except RuntimeError as exc:
        print(f"Custom calibration: ground plane failed: {exc}")
        return

    if len(inliers) < min_inliers:
        print(f"Custom calibration: ground plane rejected, only {len(inliers)} inliers.")
        return

    plane_model = np.asarray(plane_model, dtype=np.float64)
    plane_norm = max(float(np.linalg.norm(plane_model[:3])), 1e-9)
    normal = plane_model[:3] / plane_norm
    d = float(plane_model[3]) / plane_norm
    if normal[2] < 0:
        normal = -normal
        d = -d

    rotation = rotation_from_vectors(normal, [0.0, 0.0, 1.0])
    base_matrix = np.eye(4)
    base_matrix[:3, :3] = rotation
    base_matrix[:3, 3] = [0.0, 0.0, d]

    calibration["enabled"] = True
    calibration["ground_plane"] = [float(normal[0]), float(normal[1]), float(normal[2]), d]
    calibration["base_transform_matrix"] = base_matrix.tolist()
    update_calibration_transform()
    force_rebuild = True
    reset_view_on_next_cloud = True
    save_config(config)
    print(
        "Custom calibration: saved ground plane ax+by+cz+d=0 as "
        f"{normal[0]:.5f}x + {normal[1]:.5f}y + {normal[2]:.5f}z + {d:.5f} = 0 "
        f"using {len(inliers)} inliers."
    )
    print_status()


def nudge_yaw(delta_degrees):
    global force_rebuild, reset_view_on_next_cloud
    calibration = config["custom_calibration"]
    if not calibration.get("base_transform_matrix"):
        print("Custom calibration: calibrate ground with G before yaw nudging.")
        return
    calibration["yaw_degrees"] = round(float(calibration.get("yaw_degrees", 0.0)) + delta_degrees, 2)
    calibration["enabled"] = True
    update_calibration_transform()
    force_rebuild = True
    reset_view_on_next_cloud = True
    save_config(config)
    print_status()


def toggle_world_transform():
    global force_rebuild, reset_view_on_next_cloud
    calibration = config["custom_calibration"]
    if not calibration.get("transform_matrix"):
        print("Custom calibration: no saved transform yet. Press G to calibrate ground first.")
        return
    calibration["enabled"] = not calibration.get("enabled", False)
    force_rebuild = True
    reset_view_on_next_cloud = True
    save_config(config)
    print_status()


def clear_accumulation():
    global force_rebuild, reset_view_on_next_cloud
    accumulated_frames.clear()
    force_rebuild = True
    reset_view_on_next_cloud = True
    print("Accumulated cloud cleared.")


def reset_reframe():
    global reset_view_on_next_cloud
    reset_view_on_next_cloud = True
    print("Reset/reframe requested.")


def save_current_cloud():
    if len(pcd.points) == 0:
        print("No point cloud available to save yet.")
        return
    output_path = Path("/tmp") / f"unitree_l2_cloud_{time.strftime('%Y%m%d_%H%M%S')}.ply"
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Saved current cloud: {output_path}")


def build_display_points(points):
    global last_calibration_points
    mode_config = active_mode_config()
    if mode_config.get("accumulate", False):
        accumulated_frames.append(points)
        max_frames = int(mode_config.get("max_frames", 40))
        while len(accumulated_frames) > max_frames:
            accumulated_frames.popleft()
        display_points = np.concatenate(list(accumulated_frames), axis=0)
        max_points = int(mode_config.get("max_points", 0))
        if max_points > 0 and len(display_points) > max_points:
            display_points = display_points[-max_points:]
        voxel_size = float(mode_config.get("voxel_size", 0.0))
        if voxel_size > 0 and len(display_points) > 0:
            temp_cloud = o3d.geometry.PointCloud()
            temp_cloud.points = o3d.utility.Vector3dVector(display_points[:, :3])
            display_points = np.asarray(temp_cloud.voxel_down_sample(voxel_size).points)
        last_calibration_points = display_points.copy()
        display_points = apply_custom_calibration(display_points)
        return apply_native_filters(display_points)
    accumulated_frames.clear()
    last_calibration_points = points.copy()
    points = apply_custom_calibration(points)
    return apply_native_filters(points)


def apply_view_preset(name):
    global active_view_preset, people_height_filter, reset_view_on_next_cloud, force_rebuild
    active_view_preset = name
    people_height_filter = name == "people-height"
    if people_height_filter:
        force_rebuild = True

    view_control = vis.get_view_control()
    if name == "top-down":
        view_control.set_front([0.0, 0.0, -1.0])
        view_control.set_up([0.0, 1.0, 0.0])
        view_control.set_zoom(0.62)
    elif name == "front":
        view_control.set_front([1.0, 0.0, 0.0])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_zoom(0.72)
    elif name == "side":
        view_control.set_front([0.0, -1.0, 0.0])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_zoom(0.72)
    elif name == "rear":
        view_control.set_front([-1.0, 0.0, 0.0])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_zoom(0.72)
    elif name == "isometric":
        view_control.set_front([-0.7, -0.7, -0.45])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_zoom(0.68)
    elif name == "lidar-origin":
        view_control.set_front([1.0, 0.0, 0.0])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_zoom(0.36)
    elif name == "people-height":
        view_control.set_front([0.0, -1.0, -0.15])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_zoom(0.62)
    elif name == "reset":
        people_height_filter = False
        reset_view_on_next_cloud = True

    print(f"View preset: {name}")


def print_status():
    mode_config = active_mode_config()
    outlier_config = config["filters"]["statistical_outlier"]
    plane_config = config["filters"]["plane_segmentation"]
    calibration = config["custom_calibration"]
    print(
        "Mode={mode} accumulate={accumulate} max_frames={max_frames} "
        "voxel_size={voxel_size} point_size={point_size} color={color} "
        "outlier={outlier} plane_segmentation={plane} world_transform={world_transform} "
        "yaw={yaw} view={view} people_height_filter={people_filter}".format(
            mode=active_mode,
            accumulate=mode_config.get("accumulate", False),
            max_frames=mode_config.get("max_frames", 1),
            voxel_size=mode_config.get("voxel_size", 0.0),
            point_size=config["render"].get("point_size", 2.0),
            color=config["render"].get("color_scheme", "height"),
            outlier=outlier_config.get("enabled", False),
            plane=plane_config.get("enabled", False),
            world_transform=calibration.get("enabled", False),
            yaw=calibration.get("yaw_degrees", 0.0),
            view=active_view_preset,
            people_filter=people_height_filter,
        )
    )


def print_help():
    print(
        "\nKeyboard controls:\n"
        "  L        live/current-frame mode\n"
        "  A        accumulated rolling-cloud mode\n"
        "  C        clear accumulated cloud\n"
        "  [ / ]    decrease/increase accumulated frame count\n"
        "  , / .    decrease/increase accumulation voxel size\n"
        "  - / =    decrease/increase point size\n"
        "  K        cycle point color scheme\n"
        "  O        toggle statistical outlier removal\n"
        "  P        toggle dominant plane segmentation/removal\n"
        "  G        custom/not-native: calibrate ground from current/accumulated cloud\n"
        "  y / Y    custom/not-native: nudge world yaw -/+ 2 degrees\n"
        "  T        custom/not-native: toggle raw vs world-aligned transform\n"
        "  1        top-down floor plan view\n"
        "  2        front view\n"
        "  3        side view\n"
        "  4        rear/opposite side view\n"
        "  5        isometric overview\n"
        "  6        LiDAR-origin view\n"
        "  7        people-height band view/filter\n"
        "  0        reset/reframe\n"
        "  s / S    save current displayed cloud to /tmp\n"
        "  H        print this help\n"
        "  Q        quit\n"
    )
    print_status()

# Create visualization window
vis = o3d.visualization.VisualizerWithKeyCallback()
if not vis.create_window("Unitree Lidar L2 Point Cloud", width=1280, height=960):
    raise RuntimeError("Open3D could not create a visualization window. Check DISPLAY/XAUTHORITY.")

# Create empty point cloud
pcd = o3d.geometry.PointCloud()
geometry_added = False
render_options = vis.get_render_option()
render_options.background_color = np.asarray(config["render"].get("background", [0.02, 0.02, 0.025]))
render_options.point_size = float(config["render"].get("point_size", 2.0))


def quit_viewer(_vis):
    global running
    running = False
    return False


def register_callbacks():
    vis.register_key_callback(ord("L"), lambda _vis: (set_mode("live"), False)[1])
    vis.register_key_callback(ord("A"), lambda _vis: (set_mode("accumulate"), False)[1])
    vis.register_key_callback(ord("C"), lambda _vis: (clear_accumulation(), False)[1])
    vis.register_key_callback(ord("H"), lambda _vis: (print_help(), False)[1])
    vis.register_key_callback(ord("S"), lambda _vis: (save_current_cloud(), False)[1])
    vis.register_key_callback(ord("s"), lambda _vis: (save_current_cloud(), False)[1])
    vis.register_key_callback(ord("K"), lambda _vis: (cycle_color_scheme(), False)[1])
    vis.register_key_callback(ord("O"), lambda _vis: (toggle_statistical_outlier(), False)[1])
    vis.register_key_callback(ord("P"), lambda _vis: (toggle_plane_segmentation(), False)[1])
    vis.register_key_callback(ord("G"), lambda _vis: (calibrate_ground_from_current_cloud(), False)[1])
    vis.register_key_callback(ord("y"), lambda _vis: (nudge_yaw(-2.0), False)[1])
    vis.register_key_callback(ord("Y"), lambda _vis: (nudge_yaw(2.0), False)[1])
    vis.register_key_callback(ord("T"), lambda _vis: (toggle_world_transform(), False)[1])
    vis.register_key_callback(ord("Q"), quit_viewer)
    vis.register_key_callback(ord("["), lambda _vis: (update_accumulation_frames(-5), False)[1])
    vis.register_key_callback(ord("]"), lambda _vis: (update_accumulation_frames(5), False)[1])
    vis.register_key_callback(ord(","), lambda _vis: (update_voxel_size(-0.005), False)[1])
    vis.register_key_callback(ord("."), lambda _vis: (update_voxel_size(0.005), False)[1])
    vis.register_key_callback(ord("-"), lambda _vis: (update_point_size(-0.5), False)[1])
    vis.register_key_callback(ord("="), lambda _vis: (update_point_size(0.5), False)[1])
    vis.register_key_callback(ord("1"), lambda _vis: (apply_view_preset("top-down"), False)[1])
    vis.register_key_callback(ord("2"), lambda _vis: (apply_view_preset("front"), False)[1])
    vis.register_key_callback(ord("3"), lambda _vis: (apply_view_preset("side"), False)[1])
    vis.register_key_callback(ord("4"), lambda _vis: (apply_view_preset("rear"), False)[1])
    vis.register_key_callback(ord("5"), lambda _vis: (apply_view_preset("isometric"), False)[1])
    vis.register_key_callback(ord("6"), lambda _vis: (apply_view_preset("lidar-origin"), False)[1])
    vis.register_key_callback(ord("7"), lambda _vis: (apply_view_preset("people-height"), False)[1])
    vis.register_key_callback(ord("0"), lambda _vis: (reset_reframe(), False)[1])


register_callbacks()

# Start Lidar reader thread
lidar_thread = threading.Thread(target=read_lidar_stream, daemon=True)
lidar_thread.start()

print("Visualization started. Close window to exit.")
print_help()
frame_count = 0

try:
    while running:
        points = take_latest_points()
        if points is not None and len(points) > 0:
            display_points = build_display_points(points)
            if display_points.size == 0:
                continue
            pcd.points = o3d.utility.Vector3dVector(display_points[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(colorize_points(display_points))

            if not geometry_added:
                vis.add_geometry(pcd, reset_bounding_box=True)
                geometry_added = True
            elif reset_view_on_next_cloud:
                vis.update_geometry(pcd)
                vis.reset_view_point(True)
                reset_view_on_next_cloud = False
            elif force_rebuild:
                vis.update_geometry(pcd)
                force_rebuild = False
            else:
                vis.update_geometry(pcd)
            frame_count += 1

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.033)  # ~30 FPS

except KeyboardInterrupt:
    print(f"\nVisualization stopped. Rendered {frame_count} frames.")
finally:
    stop_event.set()
    if lidar_process and lidar_process.poll() is None:
        lidar_process.send_signal(signal.SIGINT)
        try:
            lidar_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            lidar_process.kill()
    vis.destroy_window()
