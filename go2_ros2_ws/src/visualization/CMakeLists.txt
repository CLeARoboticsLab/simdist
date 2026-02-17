cmake_minimum_required(VERSION 3.8)
project(visualization)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python scripts
ament_python_install_package(${PROJECT_NAME})

# Install Python entry points
install(PROGRAMS
  visualization/joint_state_pub.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
