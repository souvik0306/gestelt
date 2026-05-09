#!/bin/bash
set -e

WS_DIR="$HOME/px4_sanity/gestelt_ws4"

# Create workspace folder
echo "[1/4] Create fresh workspace and clone repositories"
rm -rf "$WS_DIR" && mkdir -p "$WS_DIR/src"

git clone -b souvik_ai_uncertainty https://github.com/souvik0306/gestelt.git "$WS_DIR/src/gestelt"
git clone -b master https://github.com/souvik0306/mavros.git "$WS_DIR/src/mavros"
git clone -b px4_base https://github.com/souvik0306/mavlink.git "$WS_DIR/src/mavlink"

# Install Python and MAVROS runtime dependencies
echo "[2/4] Install dependencies and GeographicLib datasets"
pip3 install --user onnxruntime pymavlink numpy scipy
cd "$HOME"
wget -O install_geographiclib_datasets.sh https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash ./install_geographiclib_datasets.sh

# Import third party repositories required by gestelt
echo "[3/4] Import third party repositories and verify custom MAVLink XML"
cd "$WS_DIR/src/gestelt"
vcs import < thirdparty.repos --recursive
grep -R "AI_IMU_NOISE" "$WS_DIR/src/mavlink/message_definitions/v1.0/common.xml"

# Catkin build only the minimal AI IMU and MAVROS packages
echo "[4/4] Build MAVROS and AI IMU packages from source workspace"
cd "$WS_DIR"
source /opt/ros/noetic/setup.bash
catkin init || true
catkin clean --workspace "$WS_DIR" -y
catkin build --workspace "$WS_DIR" ai_msgs mavros_msgs libmavconn mavros mavros_extras imu_listener_pkg
source "$WS_DIR/devel/setup.bash"
export PYTHONPATH="$WS_DIR/src/gestelt/imu_listener_pkg/src:$PYTHONPATH"

# Final verification checks
echo "[CHECK] Verify workspace packages and generated MAVLink headers"
rospack find mavros
rospack find ai_msgs
rospack find imu_listener_pkg

find "$WS_DIR/build" "$WS_DIR/devel" -name "mavlink_msg_ai_imu_noise.h"
grep -R "AI_IMU_NOISE" "$WS_DIR/build" "$WS_DIR/devel" 2>/dev/null | head
echo "[DONE] Workspace setup completed successfully"