#!/bin/bash
set -e

WS_DIR="$HOME/gestelt_ws_ai"

source /opt/ros/noetic/setup.bash

cd "$WS_DIR"
source devel/setup.bash

export PYTHONPATH="$WS_DIR/src/gestelt/imu_listener_pkg/src:$PYTHONPATH"

echo "[1/4] Checking ROS packages"
rospack find mavros
rospack find ai_msgs
rospack find imu_listener_pkg

echo "[2/4] Detecting FCU serial device"
FCU_DEVICE=""

if [ -e "/dev/ttyAML0" ]; then
    FCU_DEVICE="/dev/ttyAML0"
fi

if [ -z "$FCU_DEVICE" ]; then
    echo "No FCU serial device found"
    echo "Available serial devices:"
    ls /dev/ttyAML* 2>/dev/null || true
    exit 1
fi
echo "Found FCU device: $FCU_DEVICE"

echo "[3/4] Checking permissions"
if [ ! -r "$FCU_DEVICE" ] || [ ! -w "$FCU_DEVICE" ]; then
    echo "No read/write permission for $FCU_DEVICE"
    echo "Try:"
    echo "sudo usermod -aG dialout \$USER"
    echo "Then log out and log back in."
    exit 1
fi

echo "[4/4] Starting MAVROS and AI IMU client"
# FCU URL is configured inside ai_imu_inference.launch 
# [IMPORTANT: UPDATE THE LAUNCH FILE IF YOU WANT TO USE A DIFFERENT CONNECTION METHOD]
roslaunch imu_listener_pkg ai_imu_inference.launch