#!/bin/bash

# Please DO NOT run with sudo.
# This bash will launch all the components of the system, in the order of:
# 0. px4 (mavros & gcs_bridge)

init_path=~/gestelt_ws_ai

# delay between launching various modules
module_delay=2.0

# check whether is running as sudo
if [ "$EUID" -eq 0 ]
    then echo "Please DO NOT run as root."
    exit
fi

if [ "${STY}" != "" ]
    then echo "You are running the script in a screen environment. Please quit the screen."
    exit
fi

output="$(screen -ls)"
if [[ $output != *"No Sockets found"* ]]; then
    echo "There are some screen sessions alive. Please run 'pkill screen' before launching uavos."
    exit
fi

echo "The system is booting..."

cd ${init_path}

# roscore
screen -d -m -S roscore bash -c "source /opt/ros/noetic/setup.bash; source devel/setup.bash; roscore; exec bash -i"
sleep ${module_delay}
sleep ${module_delay}
sleep ${module_delay}
echo "roscore ready."

# preparation
cd ${init_path}
source /opt/ros/noetic/setup.bash
source devel/setup.bash

#################################################################################################################################
# vrpn_client_ros
screen -d -m -S vrpn_client_ros bash -c "source /opt/ros/noetic/setup.bash; source devel/setup.bash; roslaunch vrpn_client_ros sample.launch; exec bash -i"
sleep ${module_delay}
sleep ${module_delay}
sleep ${module_delay}
echo "vrpn_client_ros ready."
#################################################################################################################################

#################################################################################################################################
# mavros
screen -d -m -S mavros bash -c "source /opt/ros/noetic/setup.bash; source devel/setup.bash; roslaunch mavros px4.launch; exec bash -i"
sleep ${module_delay}
sleep ${module_delay}
sleep ${module_delay}
echo "mavros ready."
#################################################################################################################################

# MAVROS verification
rostopic list | grep mavros >/dev/null

if [ $? -ne 0 ]; then
    echo "MAVROS failed to start. Check using:"
    echo "screen -r mavros"
    exit
fi

echo "ALL GREEN ! System is started."