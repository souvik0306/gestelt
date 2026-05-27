FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV WS_DIR=/root/px4_sanity/gestelt_ws5
ENV PYTHONPATH=/root/px4_sanity/gestelt_ws5/src/gestelt/imu_listener_pkg/src

SHELL ["/bin/bash", "-c"]

# Base tools
RUN apt-get update && apt-get install -y \
    lsb-release gnupg2 curl wget git sudo \
    python3-pip \
    build-essential cmake pkg-config \
    geographiclib-tools \
    && rm -rf /var/lib/apt/lists/*

# ROS repository
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Full ROS + MAVROS dependencies
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    ros-noetic-mavros \
    ros-noetic-mavros-extras \
    ros-noetic-mavlink \
    python3-vcstool \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install --no-cache-dir onnxruntime numpy scipy

# Create workspace
RUN mkdir -p $WS_DIR/src

# Clone repositories
RUN git clone -b souvik_radxa_v116 https://github.com/souvik0306/gestelt.git $WS_DIR/src/gestelt \
    && git clone -b master https://github.com/souvik0306/mavros.git $WS_DIR/src/mavros \
    && git clone -b souvik_radxa_v1_16 https://github.com/souvik0306/mavlink.git $WS_DIR/src/mavlink

# GeographicLib datasets
RUN cd /root \
    && wget -O install_geographiclib_datasets.sh https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh \
    && bash ./install_geographiclib_datasets.sh

# Import third party dependencies
RUN cd $WS_DIR/src/gestelt \
    && vcs import < thirdparty.repos --recursive

# Build workspace
RUN cd $WS_DIR \
    && source /opt/ros/noetic/setup.bash \
    && catkin init \
    && catkin build ai_msgs mavros_msgs libmavconn mavros mavros_extras imu_listener_pkg

# Rosbag storage directory
RUN mkdir -p $WS_DIR/src/gestelt/imu_listener_pkg/data

WORKDIR $WS_DIR

CMD ["/bin/bash"]