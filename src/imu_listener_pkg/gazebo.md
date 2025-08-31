# PX4 Gazebo MAVROS Simulation with IMU Integration

This guide describes how to simulate a drone using PX4 SITL in Gazebo with MAVROS, run inference on IMU data, and send commands such as takeoff and land.

---

## Prerequisites

- ROS Noetic installed
- PX4 Firmware cloned at `~/src/PX4-Autopilot`
- MAVROS installed
- All required `source` paths set correctly
- Gazebo world launches and displays the drone

---

## Run Sequence

### 1. Source ROS and PX4 setup

```bash
source /opt/ros/noetic/setup.bash
source ~/src/PX4-Autopilot/Tools/setup_gazebo.bash ~/src/PX4-Autopilot ~/src/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/src/PX4-Autopilot
```

### 2. Launch PX4 SITL and Gazebo

```bash
cd ~/src/PX4-Autopilot
make px4_sitl_default gazebo
```

### 3. Launch MAVROS in a new terminal

```bash
source /opt/ros/noetic/setup.bash
roslaunch mavros px4.launch fcu_url:="udp://:14540@localhost:14557"
```

Wait for output similar to:

```
[ INFO] Ready for takeoff!
```

### 4. Check MAVROS connection

```bash
rostopic echo /mavros/state
```

Ensure the output shows:

- `connected: True`
- `armed: False` (initially)
- `mode: MANUAL` or `AUTO.TAKEOFF`

### 5. Arm the drone

```bash
rosservice call /mavros/cmd/arming "value: true"
```

### 6. Send takeoff command (altitude: 2.0m)

```bash
rosservice call /mavros/cmd/takeoff "min_pitch: 0.0
yaw: 0.0
latitude: 0.0
longitude: 0.0
altitude: 2.0"
```

### 7. (Optional) Land the drone

```bash
rosservice call /mavros/cmd/land "{}"
```

### 8. (Optional) Run IMU Inference Node

```bash
rosrun imu_listener_pkg imu_listener.py
```