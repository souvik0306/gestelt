# imu_listener_pkg

> **Quick Start:**
>
> - **First time or after code changes:**  
>   Run `./build_and_launch.sh` from the package directory to build and launch.
>
> - **Subsequent runs (workspace already built):**  
>   Run `./scripts/start_inference.sh` to launch inference only.

---

This package provides ROS nodes and scripts for IMU data inference using neural network models.

## Setup Instructions

### 1. Install Dependencies

Install required Python packages:
```sh
cd ~/Downloads/gestelt/src/imu_listener_pkg
pip3 install -r requirements.txt
```

### 2. Prepare Workspace Structure

If your workspace does not have a `src` directory, create it and move your package inside:
```sh
cd ~/Downloads/gestelt
mkdir -p src
mv imu_listener_pkg src/
```
*(Skip this step if your package is already inside `src/`)*

### 3. Build the Package

Navigate to your workspace root and build:
```sh
cd ~/Downloads/gestelt
catkin build imu_listener_pkg        # or catkin_make if that’s what the repo uses
```

### 4. Source the Workspace

Before running any scripts or launch files, source your workspace:
```sh
source ~/Downloads/gestelt/devel/setup.bash
```

### 5. Run Inference

You can start inference using the provided shell script:
```sh
~/Downloads/gestelt/src/imu_listener_pkg/scripts/start_inference.sh
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
