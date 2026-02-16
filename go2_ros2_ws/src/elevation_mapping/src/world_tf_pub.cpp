#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/static_transform_broadcaster.h>

class GravityAlignmentNode : public rclcpp::Node {
public:
  GravityAlignmentNode() : Node("gravity_alignment_node"), imu_count_(0) {
    this->declare_parameter<std::string>("topics.imu_body", "/imu_body");
    this->declare_parameter<std::string>("frames.world", "world");
    this->declare_parameter<std::string>("frames.init", "init");
    this->declare_parameter<int>("req_imu_count", 100);

    imu_body_topic_ = this->get_parameter("topics.imu_body").as_string();
    world_frame_ = this->get_parameter("frames.world").as_string();
    init_frame_ = this->get_parameter("frames.init").as_string();
    req_imu_count_ = this->get_parameter("req_imu_count").as_int();

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_body_topic_, 100,
        std::bind(&GravityAlignmentNode::imuCallback, this,
                  std::placeholders::_1));
  }

private:
  std::string imu_body_topic_, world_frame_, init_frame_;

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    tf2::Vector3 acc(msg->linear_acceleration.x, msg->linear_acceleration.y,
                     msg->linear_acceleration.z);
    if (acc.length() < 1e-3)
      return; // Ignore near-zero acceleration values

    gravity_accum_ += acc;
    imu_count_++;

    if (imu_count_ >= req_imu_count_) {
      tf2::Vector3 gravity = gravity_accum_ / imu_count_;
      gravity.normalize();

      // z-axis of world frame, in the init frame, is aligned with gravity
      tf2::Vector3 z_world = gravity;
      z_world.normalize();

      // Align y-axes of world and init frames
      tf2::Vector3 y_world(0, 1, 0);

      // Compute x-axis to maintain a right-handed coordinate system
      tf2::Vector3 x_world = y_world.cross(z_world);
      x_world.normalize();

      // Ensure orthogonality by recomputing y_world
      y_world = z_world.cross(x_world);
      y_world.normalize();

      // Create rotation
      tf2::Matrix3x3 R_init_to_world(x_world.x(), y_world.x(), z_world.x(),
                                     x_world.y(), y_world.y(), z_world.y(),
                                     x_world.z(), y_world.z(), z_world.z());
      tf2::Matrix3x3 R_world_to_init = R_init_to_world.transpose();
      tf2::Quaternion q;
      R_world_to_init.getRotation(q);

      publishStaticTF(q);
      imu_sub_.reset(); // Stop listening to IMU after publishing
    }
  }

  void publishStaticTF(const tf2::Quaternion &q) {
    static tf2_ros::StaticTransformBroadcaster static_broadcaster(this);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = rclcpp::Time(0);
    transform.header.frame_id = world_frame_;
    transform.child_frame_id = init_frame_;
    transform.transform.translation.x = 0.0;
    transform.transform.translation.y = 0.0;
    transform.transform.translation.z = 0.0;
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    transform.transform.rotation.w = q.w();

    static_broadcaster.sendTransform(transform);
    RCLCPP_INFO(this->get_logger(), "Published static transform from %s to %s.",
                world_frame_.c_str(), init_frame_.c_str());
  }

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  tf2::Vector3 gravity_accum_ = tf2::Vector3(0, 0, 0);
  int imu_count_, req_imu_count_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GravityAlignmentNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
