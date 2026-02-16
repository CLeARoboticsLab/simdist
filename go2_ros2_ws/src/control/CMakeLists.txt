cmake_minimum_required(VERSION 3.8)
project(control)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(unitree_go REQUIRED)

# Include directories
include_directories(include)


# Add executables
add_executable(cmd_vel_pub src/cmd_vel_pub.cpp)
ament_target_dependencies(cmd_vel_pub rclcpp geometry_msgs tf2 tf2_geometry_msgs nav_msgs unitree_go)

add_executable(state_machine src/state_machine.cpp src/motor_crc.cpp)
ament_target_dependencies(state_machine rclcpp std_msgs unitree_go)


# Install targets
install(TARGETS cmd_vel_pub state_machine
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts
ament_python_install_package(${PROJECT_NAME})

# Install Python entry points
# install(PROGRAMS
#   control/go2_controller_node.py
#   DESTINATION lib/${PROJECT_NAME}
# )

ament_package()
