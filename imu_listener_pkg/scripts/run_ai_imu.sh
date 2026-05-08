#!/bin/bash
set -e

SESSION="ai_imu_test"
WS_DIR="$HOME/px4_sanity/gestelt_ws2"
PX4_DIR="$HOME/Ai_imu_ws_noise/PX4-Autopilot"
FCU_URL="udp://:14540@localhost:14580"

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
roslaunch imu_listener_pkg ai_imu_inference.launch fcu_url:=$FCU_URL
" C-m

tmux attach -t "$SESSION"
