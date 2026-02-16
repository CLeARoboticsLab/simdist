#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

class AlignZFramePublisher : public rclcpp::Node {
public:
  AlignZFramePublisher() : Node("body_aligned_z_tf_pub") {

    this->declare_parameter<std::string>("frames.world", "world");
    this->declare_parameter<std::string>("frames.body", "body");
    this->declare_parameter<std::string>("frames.body_z_up", "body_z_up");

    this->get_parameter("frames.world", world_frame_);
    this->get_parameter("frames.body", body_frame_);
    this->get_parameter("frames.body_z_up", body_z_up_frame_);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1),
        std::bind(&AlignZFramePublisher::publish_aligned_frame, this));
  }

private:
  void publish_aligned_frame() {
    try {
      geometry_msgs::msg::TransformStamped transform_stamped;
      transform_stamped = tf_buffer_->lookupTransform(world_frame_, body_frame_,
                                                      tf2::TimePointZero);

      // Extract translation
      double x = transform_stamped.transform.translation.x;
      double y = transform_stamped.transform.translation.y;
      double z = transform_stamped.transform.translation.z;

      // Extract quaternion
      tf2::Quaternion q_body(transform_stamped.transform.rotation.x,
                             transform_stamped.transform.rotation.y,
                             transform_stamped.transform.rotation.z,
                             transform_stamped.transform.rotation.w);

      // Convert to roll-pitch-yaw
      double roll, pitch, yaw;
      tf2::Matrix3x3(q_body).getRPY(roll, pitch, yaw);

      // Create a new quaternion that only retains yaw (aligning Z-axis with the
      // world)
      tf2::Quaternion q_new;
      q_new.setRPY(0, 0, yaw); // Zero out roll and pitch

      // Publish the new transform
      geometry_msgs::msg::TransformStamped new_transform;
      new_transform.header.stamp = this->now();
      new_transform.header.frame_id = world_frame_;
      new_transform.child_frame_id = body_z_up_frame_;

      new_transform.transform.translation.x = x;
      new_transform.transform.translation.y = y;
      new_transform.transform.translation.z = z;

      new_transform.transform.rotation.x = q_new.x();
      new_transform.transform.rotation.y = q_new.y();
      new_transform.transform.rotation.z = q_new.z();
      new_transform.transform.rotation.w = q_new.w();

      tf_broadcaster_->sendTransform(new_transform);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(),
                           3000, // Throttle period in milliseconds
                           "Transform lookup failed: %s", ex.what());
    }
  }

  std::string world_frame_, body_frame_, body_z_up_frame_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AlignZFramePublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
