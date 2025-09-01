#!/bin/bash

# === [1] Source ROS ========================
source /opt/ros/noetic/setup.bash

# === [2] Workspace path (relative to this script)
WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# === [3] Source workspace environment
source "$WORKSPACE/devel/setup.bash"

# === [4] Start roscore in background if not already running
if ! pgrep -f "roscore" > /dev/null; then
    echo "[INFO] Starting roscore..."
    gnome-terminal -- bash -c "roscore; exec bash" &
    sleep 3  # give it time to start
else
    echo "[INFO] roscore already running."
fi

# # === [5] Optional: Rosbag path (if using fallback simulation)
# BAG_FILE="$WORKSPACE/src/imu_listener_pkg/bags/MH_05_difficult.bag"

# === [6] Launch inference node
roslaunch imu_listener_pkg inference.launch
