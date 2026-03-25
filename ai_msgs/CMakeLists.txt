cmake_minimum_required(VERSION 2.8.3)
project(ai_msgs)

## Find catkin and message packages
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  ImuNoise.msg
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)
