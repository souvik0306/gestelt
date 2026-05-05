#!/bin/bash

SESSION="px4_sim"
WS_DIR="$HOME/Ai_imu_ws_noise_unc"
AI_CLIENT_PATH="$HOME/Ai_imu_ws_noise_unc/src/gestelt/imu_listener_pkg/src/ai_imu_client_swap.py"

SCRIPT_DIR="$HOME/Ai_imu_ws_noise_unc/src/gestelt/gestelt_bringup"
PX4_DIR="$HOME/Ai_imu_ws_noise_unc/PX4-Autopilot"
# Kill existing sessions
tmux kill-session -t "$SESSION" 2>/dev/null
tmux kill-session -t "gz_sim_single_uav" 2>/dev/null

# Create session with 5 equal panes
tmux new-session -d -s "gz_sim_single_uav"
tmux split-window -t "gz_sim_single_uav":0.0 -h
tmux split-window -t "gz_sim_single_uav":0.0 -h
tmux split-window -t "gz_sim_single_uav":0.0 -h
tmux split-window -t "gz_sim_single_uav":0.0 -h
tmux select-layout -t "gz_sim_single_uav" tiled

tmux select-pane -t "gz_sim_single_uav":0.0 -T "AI Client"
tmux select-pane -t "gz_sim_single_uav":0.1 -T "Gazebo+PX4"
tmux select-pane -t "gz_sim_single_uav":0.2 -T "Trajectory Server"
tmux select-pane -t "gz_sim_single_uav":0.3 -T "Planner"
tmux select-pane -t "gz_sim_single_uav":0.4 -T "Mission"

tmux set-option -g pane-border-status top
tmux set-option -g pane-border-format "#{pane_index}: #{pane_title}"

# Attach and launch all panes
tmux attach -t "gz_sim_single_uav" \; \
  send-keys -t 0 "source /opt/ros/noetic/setup.bash && source $WS_DIR/devel/setup.bash && until rostopic list 2>/dev/null | grep -q '^/mavros/state$'; do echo '[AI Client] waiting for /mavros/state ...'; sleep 1; done && python3 $AI_CLIENT_PATH" C-m \; \
  send-keys -t 1 "sleep 4 && source /opt/ros/noetic/setup.bash && source $WS_DIR/devel/setup.bash && source $PX4_DIR/Tools/setup_gazebo.bash $PX4_DIR $PX4_DIR/build/px4_sitl_default && export ROS_PACKAGE_PATH=\$ROS_PACKAGE_PATH:$SCRIPT_DIR:$PX4_DIR:$PX4_DIR/Tools/sitl_gazebo && export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$SCRIPT_DIR/simulation/models && export GAZEBO_RESOURCE_PATH=\$GAZEBO_RESOURCE_PATH:$SCRIPT_DIR/simulation && roslaunch gestelt_bringup sitl_drone.launch gui:=true" C-m \; \
  send-keys -t 2 "sleep 2 && source $HOME/Ai_imu_ws_noise_unc/devel/setup.bash && export ROS_PACKAGE_PATH=$HOME/Ai_imu_ws_noise_unc/src:\$ROS_PACKAGE_PATH && roslaunch trajectory_server trajectory_server_node.launch rviz_config:=gz_sim" #C-m \; \
  send-keys -t 3 "sleep 15 && source $HOME/Ai_imu_ws_noise_unc/devel/setup.bash && export ROS_PACKAGE_PATH=$HOME/Ai_imu_ws_noise_unc/src:\$ROS_PACKAGE_PATH && roslaunch trajectory_planner trajectory_planner_node.launch" #C-m \; \
  send-keys -t 4 "sleep 50 & source $HOME/Ai_imu_ws_noise_unc/devel/setup.bash && export ROS_PACKAGE_PATH=$HOME/Ai_imu_ws_noise_unc/src:\$ROS_PACKAGE_PATH && roslaunch gestelt_bringup demo_trajectory_mission.launch" #C-m