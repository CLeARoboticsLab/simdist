#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/imu.hpp>

class IMUToClockNode : public rclcpp::Node {
public:
  IMUToClockNode() : Node("imu_to_clock") {
    // Declare parameter
    this->declare_parameter<std::string>("topics.imu_lidar", "/imu/data");

    // Get the IMU topic name from parameter
    this->get_parameter("topics.imu_lidar", imu_topic_);
    RCLCPP_INFO(this->get_logger(), "Subscribing to IMU topic: %s",
                imu_topic_.c_str());

    // Create subscriber
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, 10,
        std::bind(&IMUToClockNode::imu_callback, this, std::placeholders::_1));

    // Create clock publisher
    clock_pub_ =
        this->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
  }

private:
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    rosgraph_msgs::msg::Clock clock_msg;
    clock_msg.clock = msg->header.stamp; // Extract timestamp from IMU message
    clock_pub_->publish(clock_msg);
  }

  std::string imu_topic_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IMUToClockNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
