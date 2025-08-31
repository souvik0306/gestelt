# ROS Catkin Workspace Setup Guide (Beginner Friendly)

This guide will help you set up a ROS catkin workspace, create a package, add your Python script, and run your node. All commands are for Ubuntu Linux.

---

## 1. Install Required Tools

First, install the GNU C++ compiler (`g++`), which is required by CMake:

```bash
sudo apt update
sudo apt install g++
```

---

## 2. Create a Catkin Workspace

A catkin workspace is where you build and manage your ROS packages.

1. Create the workspace and the `src` directory:

   ```bash
   mkdir -p ~/Downloads/ros_projects/src
   ```
2. Navigate to the workspace root:

   ```bash
   cd ~/Downloads/ros_projects
   ```
3. Initialize the workspace:

   ```bash
   catkin_make
   ```

This will create the `build/`, `devel/`, and `src/` directories.

---

## 3. Create a ROS Package

1. Move into the `src` directory:

   ```bash
   cd ~/Downloads/ros_projects/src
   ```
2. Create a new package (replace `imu_listener_pkg` with your preferred name):

   ```bash
   catkin_create_pkg imu_listener_pkg std_msgs rospy sensor_msgs
   ```

This creates a new folder `imu_listener_pkg` with the necessary files.

---

## 4. Add Your Python Script

1. Move into your new package:

   ```bash
   cd imu_listener_pkg
   ```
2. Create a folder for Python scripts:

   ```bash
   mkdir scripts
   ```
3. Move your Python script into the `scripts` folder (adjust the path if needed):

   ```bash
   mv ~/Downloads/ros_projects_old/imu_listener.py scripts/
   ```
4. Make the script executable:

   ```bash
   chmod +x scripts/imu_listener.py
   ```

---

## 5. Configure CMakeLists.txt for Python Scripts

1. Open the `CMakeLists.txt` file in your package:

   ```bash
   gedit CMakeLists.txt
   ```
2. Find or add the following lines to enable installation of Python scripts:

   ```cmake
   catkin_install_python(PROGRAMS
     scripts/imu_listener.py
     DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
   )
   ```
3. Save and close the file.

---

## 6. Build the Workspace

1. Go back to the root of your workspace:

   ```bash
   cd ~/Downloads/ros_projects
   ```
2. Build the workspace:

   ```bash
   catkin_make
   ```

---

## 7. Source the Workspace

Before running any ROS nodes, you need to source the workspace setup file so ROS can find your packages.

1. Source the setup file:

   ```bash
   source devel/setup.bash
   ```
2. (Optional) To automatically source this file in every new terminal, add the following line to your `~/.bashrc`:

   ```bash
   echo "source ~/Downloads/ros_projects/devel/setup.bash" >> ~/.bashrc
   ```

---

## 8. Run Your Node

Now you can run your Python node using `rosrun`:

```bash
rosrun imu_listener_pkg imu_listener.py
```

This will start your script and it will listen for `/imu/data` or whichever topic you have set in your script.

---

## 9. Folder Structure Overview

Your workspace should look like this:

```
~/Downloads/ros_projects/
├── build/
├── devel/
└── src/
    └── imu_listener_pkg/
        ├── CMakeLists.txt
        ├── package.xml
        └── src/
            └── imu_listener.py
```

---

You have now set up a ROS catkin workspace, created a package, added your Python script, and run
