# PX4 Gazebo MAVROS Simulation with IMU Integration

This guide describes how to simulate a drone using PX4 SITL in Gazebo with MAVROS, run inference on IMU data, and send commands such as takeoff and land.

---

## Prerequisites

- ROS Noetic installed
- PX4 Firmware cloned at `~/Downloads/gestelt_ws/PX4-Autopilot`
- MAVROS installed
- All required `source` paths set correctly
- Gazebo world launches and displays the drone

---

## Run Sequence

### 1. Source ROS and PX4 setup

```bash
source /opt/ros/noetic/setup.bash
source ~/Downloads/gestelt_ws/PX4-Autopilot/Tools/setup_gazebo.bash ~/Downloads/gestelt_ws/PX4-Autopilot ~/Downloads/gestelt_ws/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/Downloads/gestelt_ws/PX4-Autopilot
```

### 2. Launch PX4 SITL and Gazebo

```bash
cd ~/Downloads/gestelt_ws/PX4-Autopilot
make px4_sitl gazebo
```
or
```bash
make px4_sitl_default gazebo
```

### 3. Switching Branch and Running SITL Drone Bringup

To switch to the `19_April_agile_demo` git branch and run the SITL drone bringup script:

```sh
cd ~/Downloads/gestelt_ws/src/gestelt
git checkout 19_April_agile_demo
cd gestelt_bringup/scripts
./sitl_drone_bringup.sh
```

### 4. Launch MAVROS in a new terminal

```bash
source /opt/ros/noetic/setup.bash
roslaunch mavros px4.launch fcu_url:="udp://:14540@localhost:14557"
```

Wait for output similar to:

```
[ INFO] Ready for takeoff!
```

### 5. Check MAVROS connection

```bash
rostopic echo /mavros/state
```

Ensure the output shows:

- `connected: True`
- `armed: False` (initially)
- `mode: MANUAL` or `AUTO.TAKEOFF`

### 6. Arm the drone

```bash
rosservice call /mavros/cmd/arming "value: true"
```
Example output:
```
success: True
result: 0
```

### 7. Set mode to AUTO.TAKEOFF

```bash
rosservice call /mavros/set_mode "custom_mode: 'AUTO.TAKEOFF'"
```
Example output:
```
mode_sent: True
```

### 8. Send takeoff command (altitude: 2.0m)

```bash
rosservice call /mavros/cmd/takeoff "min_pitch: 0.0
yaw: 0.0
latitude: 0.0
longitude: 0.0
altitude: 2.0"
```

### 9. (Optional) Land the drone

```bash
rosservice call /mavros/cmd/land "{}"
```

### 10. (Optional) Run IMU Inference Node

```bash
rosrun imu_listener_pkg imu_listener.py
```