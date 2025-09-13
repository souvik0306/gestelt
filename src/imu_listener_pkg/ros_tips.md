## Terminal 1: Start `roscore`

This starts the ROS Master which all nodes communicate through.

```bash
# Terminal 1
roscore
```

Leave this running. Do NOT close this terminal.

---

## Terminal 2: Source workspace and play `.bag`

Open a new terminal and run:

```bash
# Terminal 2

# 1. Source your catkin workspace (so your Python node is recognized)
source ~/Downloads/ros_projects/devel/setup.bash

# 2. Optionally: source ROS if not in ~/.bashrc
source /opt/ros/noetic/setup.bash

# 3. Play your bag file (this will publish topics like /imu0 or /fcu/imu)
rosbag play ~/Downloads/V1_02_medium.bag
```

---

## Terminal 3: Echo IMU data

If you want to inspect the raw messages being published:

```bash
# Terminal 3
source /opt/ros/noetic/setup.bash
rostopic list                  # see available topics
rostopic echo /imu0            # or /fcu/imu or another IMU topic
```

---

## Terminal 4: Run your custom IMU listener node

```bash
# Terminal 4 
source ~/Downloads/ros_projects/devel/setup.bash
rosrun imu_listener_pkg imu_listener.py
```

---

## Terminal 5: Echo the corrected IMU topic

After running your node, you can view the corrected IMU messages:

```bash
# Terminal 5
source ~/Downloads/ros_projects/devel/setup.bash
rostopic echo /corrected_imu
```

This will display the messages published by your `CorrectedIMUPublisher` class.

---

## Terminal 6: Launch everything with ROS launch

You can use a launch file to start both the bag playback and your IMU inference node automatically:

```bash
# Terminal 6
cd ~/ros_projects
catkin_make
source devel/setup.bash
roslaunch imu_listener_pkg inference.launch
```

This will play your bag file and start your custom IMU node as defined

---

## Terminal 7: Launch everything using the shell script

You can use the provided shell script to automate starting ROS, sourcing your workspace, and launching the inference node:

```bash
# Terminal 7
cd ~/ros_projects/src/imu_listener_pkg/scripts
bash start_inference.sh
```

This script will:
- Ensure `roscore` is running
- Source your ROS and workspace environments
- Launch the bag playback and your IMU inference node as defined in the launch file

---

## Popular ROS Topic Commands

Here are some useful ROS commands for working with topics:

```bash
rostopic list                      # List all available topics
rostopic echo /topic_name          # Print messages from a topic
rostopic hz /topic_name            # Show the publishing rate of a topic
rostopic info /topic_name          # Show type and publishers/subscribers of a topic
rosnode info /node_name            # Show details about a running node
```

---

## Quick Reference Table

| Item             | What it does                                | Real-world analogy                     |
| ---------------- | ------------------------------------------- | -------------------------------------- |
| `roscore`        | Starts the master and enables communication | Like a server (lets devices chat)    |
| `rostopic echo`  | See what's being published to a topic       | Listening to active ros topics      |
| Folder structure | Organizes code for ROS to compile and run   | Clean layout                    |

## Sample IMU message format from V1_02_medium.bag 
```
header:
  stamp:
    secs: 1403715533
    nsecs: 142143000
  frame_id: "imu4"

orientation:
  x: 0.0
  y: 0.0
  z: 0.0
  w: 1.0

orientation_covariance: [99999.9, 0, 0, 0, 99999.9, 0, 0, 0, 99999.9]

angular_velocity:
  x: -0.49
  y: 0.26
  z: 0.12

angular_velocity_covariance: [0, 0, 0, 0, 0, 0, 0, 0, 0]

linear_acceleration:
  x: 9.9
  y: 0.98
  z: -3.3

linear_acceleration_covariance: [0, 0, 0, 0, 0, 0, 0, 0, 0]
```