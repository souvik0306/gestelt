cmake_minimum_required(VERSION 3.0.2)
project(imu_listener_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  mavros_msgs
  ai_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs geometry_msgs mavros_msgs ai_msgs
)

catkin_install_python(PROGRAMS
  src/ai_imu_client.py
  src/ai_imu_client_buffer.py
  src/realtime_imu_inference.py
  src/realtime_imu_inference_buffer.py
  src/test_inference_speed.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY scripts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY bags
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)