#!/bin/bash
set -e

WS_DIR="$HOME/px4_sanity/gestelt_ws3"

# Create workspace folder
echo "[1/6] Create a fresh workspace:"
rm -rf "$WS_DIR"
mkdir -p "$WS_DIR/src"

# Clone custom gestelt, MAVROS, and MAVLink repositories
echo "[2/6] Clone custom gestelt, MAVROS, and MAVLink repositories:"
git clone -b souvik_ai_uncertainty https://github.com/souvik0306/gestelt.git "$WS_DIR/src/gestelt"
git clone -b master https://github.com/souvik0306/mavros.git "$WS_DIR/src/mavros"
git clone -b px4_base https://github.com/souvik0306/mavlink.git "$WS_DIR/src/mavlink"

# Install Python and MAVROS runtime dependencies
echo "[3/6] Install Python and MAVROS runtime dependencies"
pip3 install --user onnxruntime pymavlink numpy scipy
cd "$HOME"
wget -O install_geographiclib_datasets.sh https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash ./install_geographiclib_datasets.sh
rm -f install_geographiclib_datasets.sh

# Import third party repositories required by gestelt
echo "[4/6] Import third party repositories required by gestelt"
cd "$WS_DIR/src/gestelt"
vcs import < thirdparty.repos --recursive

# Generate the custom MAVLink headers for the AI IMU noise message and copy them to the ROS include directory
echo "[5/6] Generate and install custom MAVLink headers"
rm -rf /tmp/custom_mavlink_headers
cd "$WS_DIR/src/mavlink"
mavgen.py \
  --lang=C \
  --wire-protocol=2.0 \
  --output=/tmp/custom_mavlink_headers \
  message_definitions/v1.0/common.xml
sudo cp -r /tmp/custom_mavlink_headers/* /opt/ros/noetic/include/mavlink/v2.0/

# Catkin build only the minimal AI IMU and MAVROS packages
echo "[6/6] Catkin build only the minimal AI IMU and MAVROS packages"
cd "$WS_DIR"
source /opt/ros/noetic/setup.bash
catkin init
catkin clean --workspace "$WS_DIR" -y
catkin build --workspace "$WS_DIR" ai_msgs mavros_msgs libmavconn mavros mavros_extras imu_listener_pkg
source "$WS_DIR/devel/setup.bash"
export PYTHONPATH="$WS_DIR/src/gestelt/imu_listener_pkg/src:$PYTHONPATH"

rospack find mavros
rospack find ai_msgs
rospack find imu_listener_pkg

AI_HEADER="/opt/ros/noetic/include/mavlink/v2.0/common/mavlink_msg_ai_imu_noise.h"

if [ -f "$AI_HEADER" ]; then
    echo "AI_IMU_NOISE header exists"
else
    echo "AI_IMU_NOISE header missing"
    exit 1
fi