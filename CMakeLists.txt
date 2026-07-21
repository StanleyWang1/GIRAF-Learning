cmake_minimum_required(VERSION 3.0.2)
project(giraf_optitrak_teleop)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rosgraph
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS geometry_msgs rosgraph rospy sensor_msgs std_msgs
)

catkin_install_python(
  PROGRAMS scripts/giraf_optitrak_teleop_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
