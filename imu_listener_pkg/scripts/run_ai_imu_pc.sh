#!/bin/bash
set -e

SESSION="gz_sim_single_uav"
WS_DIR="$HOME/px4_sanity/gestelt_ws5"
PX4_DIR="$HOME/Ai_imu_radxa_ws/PX4-Autopilot"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n main

tmux send-keys -t "$SESSION:main.0" "
source /opt/ros/noetic/setup.bash
cd $PX4_DIR
make px4_sitl gazebo
" C-m

tmux split-window -h -t "$SESSION:main.0"

tmux send-keys -t "$SESSION:main.1" "
sleep 5
source /opt/ros/noetic/setup.bash
cd $WS_DIR
source devel/setup.bash
export PYTHONPATH=$WS_DIR/src/gestelt/imu_listener_pkg/src:\$PYTHONPATH
roslaunch imu_listener_pkg ai_imu_inference.launch
" C-m
# FCU URL is configured inside ai_imu_inference.launch 
# [IMPORTANT: UPDATE THE LAUNCH FILE IF YOU WANT TO USE A DIFFERENT CONNECTION METHOD]

tmux attach -t "$SESSION"