#!/bin/bash
#
# Start IMU Inference Pipeline in TMUX
#
# Layout:
#   ┌─────────────────────────────┬──────────────┐
#   │                             │   roscore    │
#   │      Inference Node         │──────────────│
#   │        (Large)              │  rosbag/src  │
#   │                             │              │
#   └─────────────────────────────┴──────────────┘
#
# Usage:
#   ./start_inference_tmux.sh                    # Use rosbag (default)
#   ./start_inference_tmux.sh --mavros           # Use MAVROS/PX4
#   ./start_inference_tmux.sh --topic /custom    # Use custom topic
#   ./start_inference_tmux.sh --bag /path/to/bag # Custom rosbag file
#

set -e

# Check tmux installation
if ! command -v tmux &> /dev/null; then
    echo "[ERROR] tmux not installed. Run: sudo apt-get install tmux"
    exit 1
fi

# Parse arguments
USE_ROSBAG="true"
USE_MAVROS="false"
IMU_TOPIC="/imu0"
BAG_FILE=""
SESSION_NAME="imu_inference"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mavros) USE_ROSBAG="false"; USE_MAVROS="true"; IMU_TOPIC="/mavros/imu/data"; shift ;;
        --topic) USE_ROSBAG="false"; USE_MAVROS="false"; IMU_TOPIC="$2"; shift 2 ;;
        --bag) BAG_FILE="$2"; shift 2 ;;
        --session) SESSION_NAME="$2"; shift 2 ;;
        *) echo "Usage: $0 [--mavros] [--topic /topic] [--bag /path/to/bag] [--session name]"; exit 1 ;;
    esac
done

# Setup paths
WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# Verify workspace
if [ ! -f "$WORKSPACE/devel/setup.bash" ]; then
    echo "[ERROR] Workspace not built! Run: cd $WORKSPACE && catkin build"
    exit 1
fi

# Kill existing session
tmux has-session -t $SESSION_NAME 2>/dev/null && tmux kill-session -t $SESSION_NAME

# Display config
echo "=================================================="
echo "  IMU Inference Pipeline"
echo "=================================================="
echo "Workspace: $WORKSPACE"
echo "Session:   $SESSION_NAME"
echo "Mode:      $([ "$USE_ROSBAG" = "true" ] && echo "Rosbag" || [ "$USE_MAVROS" = "true" ] && echo "MAVROS" || echo "Custom")"
echo "IMU Topic: $IMU_TOPIC"
[ -n "$BAG_FILE" ] && echo "Bag File:  $BAG_FILE"
echo "=================================================="

# Source command for all panes
SOURCE_CMD="source /opt/ros/noetic/setup.bash && source $WORKSPACE/devel/setup.bash"

# Create tmux session and layout
tmux new-session -d -s $SESSION_NAME -n "IMU_Pipeline"
tmux set-option -g pane-border-status top
tmux set-option -g pane-border-format "#{pane_index}: #{pane_title}"
tmux split-window -h -p 30
tmux select-pane -t 1
tmux split-window -v -p 50

# Pane 1: ROS Core
tmux select-pane -t 1
tmux send-keys "printf '\033]2;ROS Core\033\\'" C-m
tmux send-keys "$SOURCE_CMD" C-m
tmux send-keys "echo '═══ ROS CORE ═══' && sleep 1 && roscore" C-m

# Pane 2: Rosbag/Source
tmux select-pane -t 2
tmux send-keys "printf '\033]2;IMU Source\033\\'" C-m
tmux send-keys "$SOURCE_CMD" C-m
tmux send-keys "sleep 3" C-m

if [ "$USE_ROSBAG" = "true" ]; then
    tmux send-keys "echo '═══ ROSBAG PLAYBACK ═══'" C-m
    tmux send-keys "echo 'Waiting for inference node...'" C-m
    tmux send-keys "while ! rosnode list 2>/dev/null | grep -q 'imu_inference_node'; do sleep 1; echo -n '.'; done && echo" C-m
    tmux send-keys "sleep 15 && echo 'Starting playback...'" C-m
    if [ -n "$BAG_FILE" ]; then
        tmux send-keys "rosbag play --clock $BAG_FILE" C-m
    else
        tmux send-keys "rosbag play --clock \$(rospack find imu_listener_pkg)/bags/MH_02_easy.bag" C-m
    fi
elif [ "$USE_MAVROS" = "true" ]; then
    tmux send-keys "echo '═══ MAVROS (PX4 SITL) ═══'" C-m
    tmux send-keys "roslaunch mavros px4.launch fcu_url:=udp://:14540@localhost:14557" C-m
else
    tmux send-keys "echo '═══ CUSTOM IMU SOURCE ═══'" C-m
    tmux send-keys "echo 'Listening on: $IMU_TOPIC'" C-m
    tmux send-keys "echo 'Start your IMU publisher...'" C-m
fi

# Pane 0: Inference Node (Main)
tmux select-pane -t 0
tmux send-keys "printf '\033]2;Inference Node\033\\'" C-m
tmux send-keys "$SOURCE_CMD" C-m
tmux send-keys "echo '═══════════════════════════════════════════════════════════════'" C-m
tmux send-keys "echo '  IMU INFERENCE NODE (AirIMU ONNX)'" C-m
tmux send-keys "echo '═══════════════════════════════════════════════════════════════'" C-m
tmux send-keys "echo 'IMU Topic: $IMU_TOPIC | Model: airimu_cpu_fp32.onnx'" C-m
tmux send-keys "echo '═══════════════════════════════════════════════════════════════'" C-m
tmux send-keys "sleep 3 && echo 'Loading model...'" C-m
tmux send-keys "roslaunch imu_listener_pkg inference.launch use_rosbag:=false use_mavros:=false imu_topic:=$IMU_TOPIC" C-m

# Attach to session
echo
echo "TMUX session '$SESSION_NAME' created!"
echo
echo "Controls:"
echo "  Switch panes: Ctrl+b + arrows"
echo "  Detach:       Ctrl+b + d"
echo "  Kill:         tmux kill-session -t $SESSION_NAME"
echo
sleep 2
tmux attach-session -t $SESSION_NAME
