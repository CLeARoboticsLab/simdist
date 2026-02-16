#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp> // for doTransform

class LaserMappingSimple : public rclcpp::Node {
public:
  LaserMappingSimple()
      : Node("laser_mapping_simple"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/utlidar/transformed_cloud", 10,
        std::bind(&LaserMappingSimple::pointcloudCallback, this,
                  std::placeholders::_1));

    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/registered_scan", 10);
  }

private:
  void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    try {
      geometry_msgs::msg::TransformStamped transform_stamped =
          tf_buffer_.lookupTransform(target_frame_, msg->header.frame_id,
                                     msg->header.stamp,
                                     rclcpp::Duration::from_seconds(0.1));

      sensor_msgs::msg::PointCloud2 transformed_cloud;
      tf2::doTransform(*msg, transformed_cloud, transform_stamped);
      transformed_cloud.header.frame_id = target_frame_;
      pub_->publish(transformed_cloud);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "TF transform failed: %s", ex.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string target_frame_ = "camera_init";
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaserMappingSimple>());
  rclcpp::shutdown();
  return 0;
}
