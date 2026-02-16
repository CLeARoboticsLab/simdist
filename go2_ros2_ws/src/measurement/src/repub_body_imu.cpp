#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "unitree_go/msg/low_state.hpp"
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

class LowStateIMUPublisher : public rclcpp::Node {
public:
  LowStateIMUPublisher()
      : Node("go2_imu"), initialized_(false), gravity_initialized_(false) {
    std::string lowstate_topic, imu_body_topic, imu_body_nograv_topic;
    this->declare_parameter<std::string>("topics.lowstate", "/lowstate");
    this->declare_parameter<std::string>("topics.imu_body", "/imu_body");
    this->declare_parameter<std::string>("topics.imu_body_nograv",
                                         "/imu_body_nograv");
    this->declare_parameter<double>("imu_repub.rate", 200.0);
    this->declare_parameter<double>("imu_repub.alpha", 0.3);
    this->declare_parameter<double>("imu_repub.gravity_init_time", 5.0);
    this->get_parameter("topics.lowstate", lowstate_topic);
    this->get_parameter("topics.imu_body", imu_body_topic);
    this->get_parameter("topics.imu_body_nograv", imu_body_nograv_topic);
    this->get_parameter("imu_repub.rate", rate_);
    this->get_parameter("imu_repub.alpha", alpha_);
    this->get_parameter("imu_repub.gravity_init_time", gravity_init_time_);

    // Wait until clock is non-zero (valid)
    while (rclcpp::ok() && this->get_clock()->now().seconds() == 0.0) {
      RCLCPP_INFO(this->get_logger(), "Waiting for valid ROS time...");
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
    start_time_ = this->get_clock()->now();
    RCLCPP_INFO(this->get_logger(), "Start time initialized: %.3f",
                start_time_.seconds());

    imu_publisher_ =
        this->create_publisher<sensor_msgs::msg::Imu>(imu_body_topic, 100);
    imu_nograv_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>(
        imu_body_nograv_topic, 100);

    subscription_ = this->create_subscription<unitree_go::msg::LowState>(
        lowstate_topic, 1000,
        std::bind(&LowStateIMUPublisher::lowStateCallback, this,
                  std::placeholders::_1));

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000 / rate_)),
        std::bind(&LowStateIMUPublisher::publishFilteredIMU, this));
  }

private:
  void lowStateCallback(const unitree_go::msg::LowState::SharedPtr msg) {
    const auto &imu_state = msg->imu_state;

    if (!initialized_) {
      for (int i = 0; i < 3; ++i) {
        acc_avg_[i] = imu_state.accelerometer[i];
        gyro_avg_[i] = imu_state.gyroscope[i];
      }
      for (int i = 0; i < 4; ++i) {
        orientation_[i] = imu_state.quaternion[i];
      }
      initialized_ = true;
      return;
    }

    for (int i = 0; i < 3; ++i) {
      acc_avg_[i] =
          (1 - alpha_) * acc_avg_[i] + alpha_ * imu_state.accelerometer[i];
      gyro_avg_[i] =
          (1 - alpha_) * gyro_avg_[i] + alpha_ * imu_state.gyroscope[i];
    }

    for (int i = 0; i < 4; ++i) {
      orientation_[i] = imu_state.quaternion[i]; // no filtering for orientation
    }

    // Estimate gravity while initializing
    if (!gravity_initialized_) {
      auto now = this->get_clock()->now();
      double elapsed = (now - start_time_).seconds();
      for (int i = 0; i < 3; ++i) {
        gravity_estimate_[i] += imu_state.accelerometer[i];
      }
      gravity_sample_count_++;

      if (elapsed > gravity_init_time_) {
        for (int i = 0; i < 3; ++i) {
          gravity_estimate_[i] /= gravity_sample_count_;
        }
        for (int i = 0; i < 4; ++i) {
          initial_orientation_[i] = orientation_[i];
        }
        gravity_initialized_ = true;
        RCLCPP_INFO(this->get_logger(),
                    "Gravity direction initialized: [%.3f, %.3f, %.3f]",
                    gravity_estimate_[0], gravity_estimate_[1],
                    gravity_estimate_[2]);
      }
    }
  }

  void publishFilteredIMU() {
    if (!initialized_)
      return;

    auto imu_msg = sensor_msgs::msg::Imu();
    imu_msg.header.stamp = this->get_clock()->now();
    imu_msg.header.frame_id = "body";

    // unitree's order is wxyz, but ROS's order is xyzw
    imu_msg.orientation.x = orientation_[1];
    imu_msg.orientation.y = orientation_[2];
    imu_msg.orientation.z = orientation_[3];
    imu_msg.orientation.w = orientation_[0];

    imu_msg.angular_velocity.x = gyro_avg_[0];
    imu_msg.angular_velocity.y = gyro_avg_[1];
    imu_msg.angular_velocity.z = gyro_avg_[2];

    imu_msg.linear_acceleration.x = acc_avg_[0];
    imu_msg.linear_acceleration.y = acc_avg_[1];
    imu_msg.linear_acceleration.z = acc_avg_[2];

    imu_publisher_->publish(imu_msg);

    if (gravity_initialized_) {
      // Build tf2 quaternions
      tf2::Quaternion q_current(orientation_[1], orientation_[2],
                                orientation_[3], orientation_[0]);
      tf2::Quaternion q_initial(
          initial_orientation_[1], initial_orientation_[2],
          initial_orientation_[3], initial_orientation_[0]);

      tf2::Matrix3x3 R_current(q_current); // body_current → world
      tf2::Matrix3x3 R_initial(q_initial); // body_initial → world

      // Rotation from initial body frame → current body frame
      // R_rel = R_current^-1 * R_initial = R_current.transpose() * R_initial
      tf2::Matrix3x3 R_rel = R_current.transpose() * R_initial;

      // Rotate gravity vector into current body frame
      tf2::Vector3 gravity_initial(gravity_estimate_[0], gravity_estimate_[1],
                                   gravity_estimate_[2]);
      tf2::Vector3 gravity_current = R_rel * gravity_initial;

      // Subtract rotated gravity vector from accelerometer
      tf2::Vector3 acc(acc_avg_[0], acc_avg_[1], acc_avg_[2]);
      tf2::Vector3 acc_nograv = acc - gravity_current;

      sensor_msgs::msg::Imu imu_nograv = imu_msg;
      imu_nograv.linear_acceleration.x = acc_nograv.x();
      imu_nograv.linear_acceleration.y = acc_nograv.y();
      imu_nograv.linear_acceleration.z = acc_nograv.z();

      imu_nograv_publisher_->publish(imu_nograv);
    }
  }

  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_nograv_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  double rate_, alpha_, gravity_init_time_;
  bool initialized_;
  bool gravity_initialized_;
  rclcpp::Time start_time_;

  std::array<double, 3> acc_avg_{};
  std::array<double, 3> gyro_avg_{};
  std::array<double, 4> orientation_{}; // just copied, not filtered
  std::array<double, 3> gravity_estimate_{};
  std::array<double, 4> initial_orientation_{};
  size_t gravity_sample_count_ = 0;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LowStateIMUPublisher>());
  rclcpp::shutdown();
  return 0;
}
