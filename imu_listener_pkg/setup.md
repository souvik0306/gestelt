# imu_listener_pkg

This package provides ROS nodes and scripts for IMU data inference using neural network models.

## Setup Instructions

### 1. Install Dependencies

Install required Python packages:
```sh
pip3 install -r /home/souvik03/Downloads/gestelt/imu_listener_pkg/requirements.txt
```

### 2. Build the Package

Navigate to your ROS workspace and build:
```sh
cd ~/gestelt_ws
catkin build imu_listener_pkg
```

### 3. Source the Workspace

Before running any scripts or launch files, source your workspace:
```sh
source ~/gestelt_ws/devel/setup.bash
```

### 4. Run Inference

You can start inference using the provided shell script:
```sh
/home/souvik03/Downloads/gestelt/imu_listener_pkg/scripts/start_inference.sh
```

Alternatively, you can use the ROS launch file:
```sh
roslaunch imu_listener_pkg inference.launch
```

Or run the main node directly:
```sh
rosrun imu_listener_pkg imu_listener.py
```

## File Structure

- `src/imu_listener.py` — Main IMU listener node
- `models/airimu_euroc.onnx` — Pretrained neural network model
- `bags/` — Example ROS bag files
- `results/` — Output and timing results
- `launch/inference.launch` — ROS launch file
- `scripts/start_inference.sh` — Script to start inference

## Notes

- Make sure you have ROS and Catkin properly set up.
- Edit `inference.launch` or `start_inference.sh` if you need to change model or bag file paths.
- For troubleshooting, check the output logs and ensure all dependencies are installed.
