cmake_minimum_required(VERSION 3.8)
project(measurement)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(unitree_go REQUIRED)


add_executable(repub_body_imu src/repub_body_imu.cpp)
ament_target_dependencies(repub_body_imu
        rclcpp
        sensor_msgs
        unitree_go
        tf2_ros
        tf2
        tf2_geometry_msgs
)

add_executable(imu_to_clock src/imu_to_clock.cpp)
ament_target_dependencies(imu_to_clock rclcpp sensor_msgs rosgraph_msgs)

add_executable(transform_cloud src/transform_cloud.cpp)
ament_target_dependencies(transform_cloud rclcpp sensor_msgs tf2_ros tf2 tf2_geometry_msgs Eigen3)

add_executable(state_transform_and_filter src/state_transform_and_filter.cpp)
ament_target_dependencies(state_transform_and_filter rclcpp nav_msgs tf2_ros tf2_geometry_msgs)

add_executable(laser_mapping_simple src/laser_mapping_simple.cpp)
ament_target_dependencies(laser_mapping_simple
  rclcpp
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
)

add_executable(path_viz src/path_viz.cpp)
ament_target_dependencies(path_viz
  rclcpp
  geometry_msgs
  nav_msgs
)

install(TARGETS
    repub_body_imu
    DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS imu_to_clock DESTINATION lib/${PROJECT_NAME})

install(TARGETS transform_cloud DESTINATION lib/${PROJECT_NAME})

install(TARGETS state_transform_and_filter DESTINATION lib/${PROJECT_NAME})

install(TARGETS laser_mapping_simple DESTINATION lib/${PROJECT_NAME})

install(TARGETS path_viz DESTINATION lib/${PROJECT_NAME})

target_include_directories(state_transform_and_filter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(transform_cloud PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Install Python scripts
ament_python_install_package(${PROJECT_NAME})

# Install Python entry points
install(PROGRAMS
  measurement/observer.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  measurement/mocap.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  measurement/pose_kf.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
