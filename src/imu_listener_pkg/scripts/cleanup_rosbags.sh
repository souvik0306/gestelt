#!/bin/bash

echo "Cleaning up rosbag processes and active files..."

# Kill all rosbag processes
pkill -9 -f "rosbag record"

# Wait a moment
sleep 1

# Find and rename .active files
BAGS_DIR="$HOME/Ai_imu_ws1/src/gestelt/imu_listener_pkg/bags"

if [ -d "$BAGS_DIR" ]; then
    cd "$BAGS_DIR"
    
    for active_file in *.bag.active; do
        if [ -f "$active_file" ]; then
            # Remove .active extension
            final_name="${active_file%.active}"
            
            echo "Finalizing: $active_file -> $final_name"
            
            # Use rosbag reindex to properly close the file
            rosbag reindex "$active_file"
            
            # Rename to remove .active
            mv "$active_file" "$final_name" 2>/dev/null || echo "File already renamed"
            
            echo "✓ Finalized: $final_name"
        fi
    done
    
    echo "Done!"
else
    echo "Bags directory not found: $BAGS_DIR"
fi
