#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using std::placeholders::_1;

class TransformCloud : public rclcpp::Node {
public:
  TransformCloud()
      : Node("sensor_transformer"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {

    // Declare and get parameters
    std::string input_topic, output_topic;
    this->declare_parameter<std::string>("topics.lidar_cloud",
                                         "/utlidar/cloud");
    this->declare_parameter<std::string>("topics.lidar_cloud_transformed",
                                         "/utlidar/transformed_cloud");
    this->declare_parameter<double>("cam_offset", 0.046825);
    this->declare_parameter<double>("x_filter_min", -0.7);
    this->declare_parameter<double>("x_filter_max", -0.1);
    this->declare_parameter<double>("y_filter_min", -0.3);
    this->declare_parameter<double>("y_filter_max", 0.3);
    this->declare_parameter<double>("z_filter_min", -0.6);
    this->declare_parameter<double>("z_filter_max", 0.0);

    this->get_parameter("topics.lidar_cloud", input_topic);
    this->get_parameter("topics.lidar_cloud_transformed", output_topic);
    this->get_parameter("cam_offset", cam_offset_);
    this->get_parameter("x_filter_min", x_filter_min_);
    this->get_parameter("x_filter_max", x_filter_max_);
    this->get_parameter("y_filter_min", y_filter_min_);
    this->get_parameter("y_filter_max", y_filter_max_);
    this->get_parameter("z_filter_min", z_filter_min_);
    this->get_parameter("z_filter_max", z_filter_max_);

    // Adjust z-filter limits based on cam offset
    z_filter_min_ -= cam_offset_;
    z_filter_max_ -= cam_offset_;

    // Subscribe to input point cloud topic
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic, 50, std::bind(&TransformCloud::cloud_callback, this, _1));

    // Publisher for transformed point cloud
    cloud_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, 50);
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  double cam_offset_;
  double x_filter_min_, x_filter_max_;
  double y_filter_min_, y_filter_max_;
  double z_filter_min_, z_filter_max_;

  bool is_in_filter_box(const Eigen::Vector3f &point) {
    return (point.x() > x_filter_min_ && point.x() < x_filter_max_ &&
            point.y() > y_filter_min_ && point.y() < y_filter_max_ &&
            point.z() > z_filter_min_ && point.z() < z_filter_max_);
  }

  void
  cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    try {
      // Lookup transform from body to lidar
      geometry_msgs::msg::TransformStamped transform_stamped =
          tf_buffer_.lookupTransform("body", "lidar", rclcpp::Time(0));

      // Convert transform to Eigen
      Eigen::Matrix3f rotation_matrix =
          Eigen::Quaternionf(transform_stamped.transform.rotation.w,
                             transform_stamped.transform.rotation.x,
                             transform_stamped.transform.rotation.y,
                             transform_stamped.transform.rotation.z)
              .toRotationMatrix();

      Eigen::Vector3f translation(transform_stamped.transform.translation.x,
                                  transform_stamped.transform.translation.y,
                                  transform_stamped.transform.translation.z);

      // Create output point cloud
      sensor_msgs::msg::PointCloud2 transformed_cloud = *cloud_msg;
      transformed_cloud.header.frame_id = "body";

      sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");

      std::vector<std::array<float, 3>> transformed_points;

      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        Eigen::Vector3f point(*iter_x, *iter_y, *iter_z);

        // Apply transformation
        Eigen::Vector3f transformed_point =
            (rotation_matrix * point) + translation;
        transformed_point.z() -= cam_offset_;

        // Filter points
        if (!is_in_filter_box(transformed_point)) {
          transformed_points.push_back({transformed_point.x(),
                                        transformed_point.y(),
                                        transformed_point.z()});
        }
      }

      // Resize cloud and copy transformed points
      transformed_cloud.data.resize(transformed_points.size() *
                                    cloud_msg->point_step);
      transformed_cloud.width = transformed_points.size();
      transformed_cloud.row_step =
          transformed_cloud.width * cloud_msg->point_step;

      auto out_iter_x =
          sensor_msgs::PointCloud2Iterator<float>(transformed_cloud, "x");
      auto out_iter_y =
          sensor_msgs::PointCloud2Iterator<float>(transformed_cloud, "y");
      auto out_iter_z =
          sensor_msgs::PointCloud2Iterator<float>(transformed_cloud, "z");

      for (size_t i = 0; i < transformed_points.size();
           ++i, ++out_iter_x, ++out_iter_y, ++out_iter_z) {
        *out_iter_x = transformed_points[i][0];
        *out_iter_y = transformed_points[i][1];
        *out_iter_z = transformed_points[i][2];
      }

      // Publish transformed point cloud
      cloud_pub_->publish(transformed_cloud);

    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                           "Could not transform from 'body' to 'lidar': %s",
                           ex.what());
    }
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TransformCloud>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
