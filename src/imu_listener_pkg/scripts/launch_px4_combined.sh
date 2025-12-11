#!/bin/bash

SESSION="px4_sim"
PX4_ROOT="${PX4_ROOT:-$HOME/Ai_imu_ws/PX4-Autopilot/}"
PX4_LAUNCH_CMD="${PX4_LAUNCH_CMD:-make px4_sitl gazebo}"
AI_CLIENT_PATH="$HOME/Ai_imu_ws/gestelt/src/imu_listener_pkg/src/ai_imu_client.py"

# Verify paths
[ ! -f "$AI_CLIENT_PATH" ] && echo "[ERROR] AI client not found: $AI_CLIENT_PATH" && exit 1
[ ! -d "$PX4_ROOT" ] && echo "[ERROR] PX4 directory not found: $PX4_ROOT" && exit 1

# Kill existing session
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

# Create session
tmux new-session -d -s "$SESSION"
tmux split-window -h -p 80 -t "$SESSION":0

# Pane 0: QGroundControl
tmux select-pane -t "$SESSION":0.0 -T "QGroundControl"
tmux send-keys -t "$SESSION":0.0 "cd ~/Downloads && ./QGroundControl.AppImage" C-m

# Pane 1: AI IMU Client (loads model first)
tmux select-pane -t "$SESSION":0.1 -T "AI IMU Client"
tmux send-keys -t "$SESSION":0.1 "python3 $AI_CLIENT_PATH" C-m

# Pane 2: PX4 SITL + Gazebo (waits for model)
tmux split-window -v -t "$SESSION":0.1
tmux select-pane -t "$SESSION":0.2 -T "PX4 SITL + Gazebo"
tmux send-keys -t "$SESSION":0.2 "sleep 5 && cd $PX4_ROOT && $PX4_LAUNCH_CMD" C-m

# Pane 3: MAVROS (waits for PX4)
tmux split-window -v -t "$SESSION":0.2
tmux select-pane -t "$SESSION":0.3 -T "MAVROS"
tmux send-keys -t "$SESSION":0.3 "sleep 8 && source /opt/ros/noetic/setup.bash && roslaunch mavros px4.launch fcu_url:=udp://:14540@localhost:14580 fcu_protocol:=v2.0" C-m

# Note: Rosbag recording is now automatic - starts/stops with AI client for perfect time alignment

# Set pane borders
tmux set-option -g pane-border-status top
tmux set-option -g pane-border-format "#{pane_index}: #{pane_title}"

# Attach
tmux attach -t "$SESSION"
