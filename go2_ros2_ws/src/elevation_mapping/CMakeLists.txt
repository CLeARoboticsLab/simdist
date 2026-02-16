cmake_minimum_required(VERSION 3.8)
project(elevation_mapping)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)


add_executable(pointcloud_to_gridmap src/pointcloud_to_gridmap.cpp)
ament_target_dependencies(pointcloud_to_gridmap rclcpp sensor_msgs grid_map_ros grid_map_msgs pcl_ros pcl_conversions)

add_executable(flatten_gridmap src/flatten_gridmap.cpp)
ament_target_dependencies(flatten_gridmap rclcpp grid_map_ros grid_map_msgs std_msgs)

add_executable(body_aligned_z_tf_pub src/body_aligned_z_tf_pub.cpp)
ament_target_dependencies(body_aligned_z_tf_pub rclcpp tf2 tf2_geometry_msgs)

add_executable(world_tf_pub src/world_tf_pub.cpp)
ament_target_dependencies(world_tf_pub rclcpp tf2 tf2_ros sensor_msgs geometry_msgs)

add_executable(terrain_analysis_simple src/terrain_analysis_simple.cpp)
ament_target_dependencies(terrain_analysis_simple
  rclcpp
  grid_map_ros
  grid_map_msgs
  nav_msgs
  pcl_ros
  pcl_conversions
  sensor_msgs
)

install(TARGETS
  pointcloud_to_gridmap
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
flatten_gridmap
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
body_aligned_z_tf_pub
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
world_tf_pub
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  terrain_analysis_simple
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
