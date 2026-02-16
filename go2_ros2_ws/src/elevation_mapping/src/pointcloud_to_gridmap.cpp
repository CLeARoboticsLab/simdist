#include "grid_map_msgs/msg/grid_map.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include <random>

class PointCloudToGridMap : public rclcpp::Node {
public:
  PointCloudToGridMap()
      : Node("pointcloud_to_gridmap"),
        tf_buffer_(this->get_clock(), tf2::durationFromSec(60.0)),
        tf_listener_(tf_buffer_) {
    // Declare parameters
    this->declare_parameter<std::string>("topics.terrain_cloud",
                                         "/terrain_map");
    this->declare_parameter<std::string>("topics.elevation_grid",
                                         "/elevation_grid");
    this->declare_parameter<std::string>("topics.elevation_grid_filled",
                                         "/elevation_grid_filled");
    this->declare_parameter<double>("grid_resolution", 0.05);
    this->declare_parameter<double>("grid_length_x", 3.0);
    this->declare_parameter<double>("grid_length_y", 3.0);
    this->declare_parameter<double>("grid_offset_x", 0.0);
    this->declare_parameter<double>("grid_offset_y", 0.0);
    this->declare_parameter<std::string>("grid_layer", "elevation");
    this->declare_parameter<std::string>("aggregation", "min");
    this->declare_parameter<int>("num_neighbors", 5);
    this->declare_parameter<bool>("height_in_world_z_axis", true);
    this->declare_parameter<std::string>("frames.body", "body");
    this->declare_parameter<std::string>("frames.body_z_up", "body_z_up");

    // Get parameters
    std::string body_frame, body_z_up_frame;
    bool height_in_world_z_axis;
    this->get_parameter("topics.terrain_cloud", input_topic_);
    this->get_parameter("topics.elevation_grid", output_topic_);
    this->get_parameter("topics.elevation_grid_filled", output_filled_topic_);
    this->get_parameter("grid_resolution", grid_resolution_);
    this->get_parameter("grid_length_x", grid_length_x_);
    this->get_parameter("grid_length_y", grid_length_y_);
    this->get_parameter("grid_offset_x", grid_offset_x_);
    this->get_parameter("grid_offset_y", grid_offset_y_);
    this->get_parameter("grid_layer", grid_layer_);
    this->get_parameter("aggregation", aggregation_);
    this->get_parameter("num_neighbors", num_neighbors_);
    this->get_parameter("height_in_world_z_axis", height_in_world_z_axis);
    this->get_parameter("frames.body", body_frame);
    this->get_parameter("frames.body_z_up", body_z_up_frame);

    if (height_in_world_z_axis) {
      target_frame_ = body_z_up_frame;
    } else {
      target_frame_ = body_frame;
    }

    // Subscriber for PointCloud2
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, 10,
        std::bind(&PointCloudToGridMap::pointCloudCallback, this,
                  std::placeholders::_1));

    // Publisher for original GridMap
    grid_map_pub_ =
        this->create_publisher<grid_map_msgs::msg::GridMap>(output_topic_, 10);

    // Publisher for filled GridMap
    grid_map_filled_pub_ = this->create_publisher<grid_map_msgs::msg::GridMap>(
        output_filled_topic_, 10);

    RCLCPP_INFO(this->get_logger(), "PointCloud to GridMap node started.");
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr
      grid_map_filled_pub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string input_topic_, output_topic_, output_filled_topic_, grid_layer_,
      target_frame_, aggregation_;
  double grid_resolution_, grid_length_x_, grid_length_y_, grid_offset_x_,
      grid_offset_y_;
  int num_neighbors_;

  void
  pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    // Look up transform from PointCloud2 frame to body frame
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
      transform_stamped = tf_buffer_.lookupTransform(
          target_frame_, cloud_msg->header.frame_id, cloud_msg->header.stamp,
          rclcpp::Duration::from_seconds(0.1));
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Could not transform %s to %s: %s",
                  cloud_msg->header.frame_id.c_str(), target_frame_.c_str(),
                  ex.what());
      return;
    }

    // Convert PointCloud2 to PCL format
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Create a GridMap object centered in the body frame
    grid_map::GridMap map({grid_layer_, "sum", "count"});
    map.setFrameId(target_frame_);
    // Add grid_resolution_ to make grid centered on a edge
    map.setGeometry(grid_map::Length(grid_length_x_ + grid_resolution_,
                                     grid_length_y_ + grid_resolution_),
                    grid_resolution_,
                    grid_map::Position(grid_offset_x_, grid_offset_y_));

    // Transform and insert points into GridMap
    for (const auto &point : cloud->points) {
      geometry_msgs::msg::PointStamped point_in, point_out;
      point_in.header.frame_id = cloud_msg->header.frame_id;
      point_in.point.x = point.x;
      point_in.point.y = point.y;
      point_in.point.z = point.z;

      try {
        tf2::doTransform(point_in, point_out, transform_stamped);
      } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "Point transformation failed: %s",
                    ex.what());
        continue;
      }

      grid_map::Position position(point_out.point.x, point_out.point.y);
      if (map.isInside(position)) {
        grid_map::Index index;
        map.getIndex(position, index);

        if (aggregation_ == "min") {
          if (std::isnan(map.at(grid_layer_, index))) {
            map.at(grid_layer_, index) = static_cast<float>(point_out.point.z);
          } else {
            map.at(grid_layer_, index) =
                std::min(map.at(grid_layer_, index),
                         static_cast<float>(point_out.point.z));
          }
        } else if (aggregation_ == "mean") {
          if (std::isnan(map.at("sum", index))) {
            map.at("sum", index) = static_cast<float>(point_out.point.z);
            map.at("count", index) = 1.0;
          } else {
            map.at("sum", index) += static_cast<float>(point_out.point.z);
            map.at("count", index) += 1.0;
          }
        }
      }
    }

    // If using mean aggregation, compute the mean
    if (aggregation_ == "mean") {
      for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        if (map.at("count", index) > 0) {
          map.at(grid_layer_, index) =
              map.at("sum", index) / map.at("count", index);
        } else {
          map.at(grid_layer_, index) = std::numeric_limits<float>::quiet_NaN();
        }
      }
    }

    // Fill NaN values
    grid_map::GridMap filled_map = map; // Copy of the original map

    for (grid_map::GridMapIterator it(filled_map); !it.isPastEnd(); ++it) {
      const grid_map::Index index(*it);

      if (std::isnan(filled_map.at(grid_layer_, index))) {
        std::vector<float> neighbors;
        grid_map::Position center_position;
        filled_map.getPosition(index, center_position); // Correct usage

        for (grid_map::CircleIterator circle_it(filled_map, center_position,
                                                grid_resolution_ * 3);
             !circle_it.isPastEnd(); ++circle_it) {
          if (!std::isnan(filled_map.at(grid_layer_, *circle_it))) {
            neighbors.push_back(filled_map.at(grid_layer_, *circle_it));
          }
          if (neighbors.size() >= num_neighbors_)
            break; // Stop if we found enough
        }

        if (!neighbors.empty()) {
          // Choose method: mean of nearest N
          float sum = std::accumulate(neighbors.begin(), neighbors.end(), 0.0);
          filled_map.at(grid_layer_, index) = sum / neighbors.size();
        }
      }
    }

    // Publish original GridMap
    std::unique_ptr<grid_map_msgs::msg::GridMap> message =
        grid_map::GridMapRosConverter::toMessage(map);
    if (message) {
      grid_map_pub_->publish(std::move(*message));
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to convert GridMap to message.");
    }

    // Publish filled GridMap
    std::unique_ptr<grid_map_msgs::msg::GridMap> filled_message =
        grid_map::GridMapRosConverter::toMessage(filled_map);
    if (filled_message) {
      grid_map_filled_pub_->publish(std::move(*filled_message));
    } else {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to convert Filled GridMap to message.");
    }
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudToGridMap>());
  rclcpp::shutdown();
  return 0;
}
