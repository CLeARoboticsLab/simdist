#include "grid_map_msgs/msg/grid_map.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <vector>

struct TimedPoint {
  pcl::PointXYZI pt;
  double timestamp;
};
std::deque<TimedPoint> point_buffer_;

class TerrainMapper : public rclcpp::Node {
public:
  TerrainMapper() : Node("terrain_analysis_simple") {
    this->declare_parameter<double>("resolution", 0.1);
    this->declare_parameter<double>("range", 3.0);
    this->declare_parameter<double>("point_lifetime", 5.0);
    this->declare_parameter<double>("pub_rate", 20.0);

    this->get_parameter("resolution", grid_resolution_);
    this->get_parameter("range", range_limit_);
    this->get_parameter("point_lifetime", point_lifetime_);
    double pub_rate;
    this->get_parameter("pub_rate", pub_rate);

    // Wait until clock is non-zero (valid)
    while (rclcpp::ok() && this->get_clock()->now().seconds() == 0.0) {
      RCLCPP_INFO(this->get_logger(), "Waiting for valid ROS time...");
      rclcpp::sleep_for(std::chrono::milliseconds(250));
    }

    start_t_ = this->get_clock()->now().seconds();
    scan_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/registered_scan", 10,
        std::bind(&TerrainMapper::pointCloudCallback, this,
                  std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/state_estimation", 10,
        std::bind(&TerrainMapper::odomCallback, this, std::placeholders::_1));

    terrain_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/terrain_map", 10);
    grid_map_pub_ = this->create_publisher<grid_map_msgs::msg::GridMap>(
        "/terrain_grid_map", 10);

    map_.setFrameId("map");
    map_.setGeometry(grid_map::Length(2 * range_limit_, 2 * range_limit_),
                     grid_resolution_, grid_map::Position(0.0, 0.0));
    map_.add("elevation");
    map_.add("sum");
    map_.add("count");

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / pub_rate)),
        std::bind(&TerrainMapper::publishTerrainMap, this));
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    current_position_ = msg->pose.pose.position;
    map_.move(grid_map::Position(current_position_.x, current_position_.y));
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::fromROSMsg(*msg, cloud);
    double now = this->get_clock()->now().seconds();

    for (const auto &pt : cloud.points) {
      point_buffer_.push_back(TimedPoint{pt, now});
    }

    // Trim old points from the front of the buffer
    while (!point_buffer_.empty() &&
           (now - point_buffer_.front().timestamp > point_lifetime_)) {
      point_buffer_.pop_front();
    }
  }

  void publishTerrainMap() {
    std::lock_guard<std::mutex> lock(map_mutex_);
    pcl::PointCloud<pcl::PointXYZI> map_cloud;
    rclcpp::Time now = this->get_clock()->now();
    map_["sum"].setZero();
    map_["count"].setZero();
    map_["elevation"].setConstant(NAN);

    for (const auto &tp : point_buffer_) {
      grid_map::Index index;
      if (!map_.getIndex(grid_map::Position(tp.pt.x, tp.pt.y), index))
        continue;

      map_.at("sum", index) += tp.pt.z;
      map_.at("count", index) += 1.0f;
    }

    for (grid_map::GridMapIterator it(map_); !it.isPastEnd(); ++it) {
      const auto &index = *it;
      float count = map_.at("count", index);
      if (count > 0.0f) {
        map_.at("elevation", index) = map_.at("sum", index) / count;
        grid_map::Position pos;
        map_.getPosition(index, pos);
        pcl::PointXYZI p;
        p.x = pos.x();
        p.y = pos.y();
        p.z = map_.at("elevation", index);
        p.intensity = p.z;
        map_cloud.points.push_back(p);
      }
    }

    sensor_msgs::msg::PointCloud2 out_pc;
    pcl::toROSMsg(map_cloud, out_pc);
    out_pc.header.stamp = now;
    out_pc.header.frame_id = "map";
    terrain_pub_->publish(out_pc);

    std::unique_ptr<grid_map_msgs::msg::GridMap> msg =
        grid_map::GridMapRosConverter::toMessage(map_);
    grid_map_pub_->publish(std::move(*msg));
  }

  grid_map::GridMap map_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr terrain_pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::mutex map_mutex_;
  double start_t_;

  geometry_msgs::msg::Point current_position_;
  double grid_resolution_;
  double range_limit_;
  double point_lifetime_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TerrainMapper>());
  rclcpp::shutdown();
  return 0;
}
