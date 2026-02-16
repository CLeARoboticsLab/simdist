#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

class PathPublisher : public rclcpp::Node {
public:
  PathPublisher() : Node("path_publisher_node") {
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/state_estimation", 1,
        std::bind(&PathPublisher::odom_callback, this, std::placeholders::_1));

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path", 10);

    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(100), // 10 Hz
                                std::bind(&PathPublisher::publish_path, this));
  }

private:
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    latest_odom_ = msg;
    if (need_to_init_odom_) {
      path_buffer_.clear();
      path_.header.frame_id = msg->header.frame_id;
      need_to_init_odom_ = false;
    }
  }

  void publish_path() {
    if (need_to_init_odom_) {
      return;
    }

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = latest_odom_->header;
    pose_stamped.pose = latest_odom_->pose.pose;
    path_buffer_.push_back(pose_stamped);

    if (path_buffer_.size() > max_path_length_) {
      path_buffer_.erase(path_buffer_.begin());
    }

    path_.header.stamp = this->get_clock()->now();
    path_.poses = path_buffer_;
    path_pub_->publish(path_);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::Odometry::SharedPtr latest_odom_;
  std::vector<geometry_msgs::msg::PoseStamped> path_buffer_;
  nav_msgs::msg::Path path_;
  bool need_to_init_odom_ = true;
  size_t max_path_length_ = 1000;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathPublisher>());
  rclcpp::shutdown();
  return 0;
}