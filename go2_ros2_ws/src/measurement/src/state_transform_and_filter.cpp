#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class StateTransformAndFilter : public rclcpp::Node {
public:
  StateTransformAndFilter()
      : Node("state_transform_and_filter"), callback_counter_(0),
        tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
    // Load parameters
    this->declare_parameter<std::string>("topics.state", "/odom");
    this->declare_parameter<std::string>("topics.state_transform_and_filtered",
                                         "/odom_filtered");
    this->declare_parameter<std::string>("frames.world", "world");
    this->declare_parameter<std::string>("frames.init", "init");
    this->declare_parameter<double>("state_filter.alpha", 0.0);
    this->declare_parameter<int>("state_filter.downsample_ratio", 1);

    odom_topic_ = this->get_parameter("topics.state").as_string();
    filtered_odom_topic_ =
        this->get_parameter("topics.state_transform_and_filtered").as_string();
    world_frame_ = this->get_parameter("frames.world").as_string();
    init_frame_ = this->get_parameter("frames.init").as_string();
    alpha_ = this->get_parameter("state_filter.alpha").as_double();
    downsample_ratio_ =
        this->get_parameter("state_filter.downsample_ratio").as_int();

    // Subscribe to odometry topic
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 1000,
        std::bind(&StateTransformAndFilter::odomCallback, this,
                  std::placeholders::_1));

    // Publisher for filtered odometry
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
        filtered_odom_topic_, 1000);
  }

private:
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::string odom_topic_, filtered_odom_topic_;
  std::string world_frame_, init_frame_;
  nav_msgs::msg::Odometry filtered_odom_;
  bool has_data_ = false;
  double alpha_;
  int downsample_ratio_;
  int callback_counter_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  double applyLowPassFilter(double new_value, double old_value) {
    return alpha_ * new_value + (1 - alpha_) * old_value;
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    geometry_msgs::msg::TransformStamped transform_stamped;
    std::string init_frame = msg->header.frame_id;
    try {
      transform_stamped = tf_buffer_.lookupTransform(world_frame_, init_frame,
                                                     tf2::TimePointZero);
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                           "Could not transform odometry: %s", ex.what());
      return;
    }

    geometry_msgs::msg::PoseStamped pose_init, pose_world;
    pose_init.header = msg->header;
    pose_init.pose = msg->pose.pose;
    tf2::doTransform(pose_init, pose_world, transform_stamped);

    if (!has_data_) {
      filtered_odom_ = *msg;
      has_data_ = true;
    } else {
      // Apply low-pass filter
      filtered_odom_.pose.pose.position.x = applyLowPassFilter(
          pose_world.pose.position.x, filtered_odom_.pose.pose.position.x);
      filtered_odom_.pose.pose.position.y = applyLowPassFilter(
          pose_world.pose.position.y, filtered_odom_.pose.pose.position.y);
      filtered_odom_.pose.pose.position.z = applyLowPassFilter(
          pose_world.pose.position.z, filtered_odom_.pose.pose.position.z);

      filtered_odom_.pose.pose.orientation.x =
          applyLowPassFilter(pose_world.pose.orientation.x,
                             filtered_odom_.pose.pose.orientation.x);
      filtered_odom_.pose.pose.orientation.y =
          applyLowPassFilter(pose_world.pose.orientation.y,
                             filtered_odom_.pose.pose.orientation.y);
      filtered_odom_.pose.pose.orientation.z =
          applyLowPassFilter(pose_world.pose.orientation.z,
                             filtered_odom_.pose.pose.orientation.z);
      filtered_odom_.pose.pose.orientation.w =
          applyLowPassFilter(pose_world.pose.orientation.w,
                             filtered_odom_.pose.pose.orientation.w);

      // twist remains unchanged in transformation
      filtered_odom_.twist.twist.linear.x = applyLowPassFilter(
          msg->twist.twist.linear.x, filtered_odom_.twist.twist.linear.x);
      filtered_odom_.twist.twist.linear.y = applyLowPassFilter(
          msg->twist.twist.linear.y, filtered_odom_.twist.twist.linear.y);
      filtered_odom_.twist.twist.linear.z = applyLowPassFilter(
          msg->twist.twist.linear.z, filtered_odom_.twist.twist.linear.z);

      filtered_odom_.twist.twist.angular.x = applyLowPassFilter(
          msg->twist.twist.angular.x, filtered_odom_.twist.twist.angular.x);
      filtered_odom_.twist.twist.angular.y = applyLowPassFilter(
          msg->twist.twist.angular.y, filtered_odom_.twist.twist.angular.y);
      filtered_odom_.twist.twist.angular.z = applyLowPassFilter(
          msg->twist.twist.angular.z, filtered_odom_.twist.twist.angular.z);
    }

    // Only publish every `downsample_ratio_` callbacks
    if (++callback_counter_ >= downsample_ratio_) {
      filtered_odom_.header.stamp = msg->header.stamp;
      filtered_odom_.header.frame_id = world_frame_;
      odom_pub_->publish(filtered_odom_);
      callback_counter_ = 0;
    }
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StateTransformAndFilter>());
  rclcpp::shutdown();
  return 0;
}
