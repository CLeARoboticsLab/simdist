cmake_minimum_required(VERSION 3.8)
project(simdist_controller)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python scripts
ament_python_install_package(${PROJECT_NAME})

# Install Python entry points
install(PROGRAMS
  simdist_controller/simdist_controller_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
