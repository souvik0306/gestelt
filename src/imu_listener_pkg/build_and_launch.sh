#!/bin/bash

# Detect workspace root (two levels up from this script)
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PKG_NAME="imu_listener_pkg"
PKG_PATH="$WS_DIR/src/$PKG_NAME"

# Ensure src exists at workspace root
if [ ! -d "$WS_DIR/src" ]; then
    mkdir "$WS_DIR/src"
fi

# Move package into src if not already there
if [ ! -d "$PKG_PATH" ]; then
    # If running from inside the package, move it up
    THIS_PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    mv "$THIS_PKG" "$WS_DIR/src/"
fi

# Build the package
cd "$WS_DIR"
catkin build "$PKG_NAME"

# Source the workspace
source "$WS_DIR/devel/setup.bash"

# Launch the node
roslaunch "$PKG_NAME" inference.launch