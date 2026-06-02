#!/usr/bin/env bash
# Launch the lidar_venv ipykernel with the ROS 2 (Jazzy) environment sourced, so the
# notebook can `import rclpy` and subscribe to /unilidar/cloud while using lidar_venv's
# open3d/numpy. Used by the "Python (lidar_venv + ROS2)" Jupyter kernelspec.
set -e

# Source ROS 2 so rclpy + message packages land on PYTHONPATH.
source /opt/ros/jazzy/setup.bash

# Hand off to the venv interpreter; "$@" carries the connection-file args Jupyter passes.
exec /home/aimakeradmin/Documents/Github/unilidar_sdk2/lidar_venv/bin/python \
  -m ipykernel_launcher "$@"
