#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <unitree_go/msg/wireless_controller.hpp>

class CmdVelPublisher : public rclcpp::Node {
public:
  CmdVelPublisher() : Node("cmd_vel_publisher") {
    // Declare parameters with default values
    this->declare_parameter<float>("cmd.rate", 50);
    this->declare_parameter<std::string>("cmd.type", "constant");

    this->declare_parameter<std::vector<double>>("cmd.constant",
                                                 {0.0, 0.0, 0.0});
    this->declare_parameter<std::vector<double>>(
        "cmd.wirelesscontroller_maxvels", {1.5, 0.5, 1.0});

    this->declare_parameter<float>("cmd.straight_line.vx", 0.0);
    this->declare_parameter<float>("cmd.straight_line.Kp_y", 2.0);
    this->declare_parameter<float>("cmd.straight_line.Kd_y", 0.5);
    this->declare_parameter<float>("cmd.straight_line.Kp_yaw", 2.0);
    this->declare_parameter<float>("cmd.straight_line.Kd_yaw", 0.5);

    this->declare_parameter<float>("cmd.straight_line_integral.vx", 0.0);
    this->declare_parameter<float>("cmd.straight_line_integral.Kp_x", 2.0);
    this->declare_parameter<float>("cmd.straight_line_integral.Kd_x", 0.5);
    this->declare_parameter<float>("cmd.straight_line_integral.Kp_y", 2.0);
    this->declare_parameter<float>("cmd.straight_line_integral.Kd_y", 0.5);
    this->declare_parameter<float>("cmd.straight_line_integral.Kp_yaw", 2.0);
    this->declare_parameter<float>("cmd.straight_line_integral.Kd_yaw", 0.5);

    this->declare_parameter<float>("cmd.wirelesscontrol_position.Kp_x", 2.0);
    this->declare_parameter<float>("cmd.wirelesscontrol_position.Kd_x", 0.0);
    this->declare_parameter<float>("cmd.wirelesscontrol_position.Kp_y", 2.0);
    this->declare_parameter<float>("cmd.wirelesscontrol_position.Kd_y", 0.0);
    this->declare_parameter<float>("cmd.wirelesscontrol_position.Kp_yaw", 2.0);
    this->declare_parameter<float>("cmd.wirelesscontrol_position.Kd_yaw", 0.0);

    this->declare_parameter<std::vector<double>>("cmd.max_vels",
                                                 {1.5, 0.5, 1.0});
    this->declare_parameter<std::vector<double>>("cmd.min_vels",
                                                 {-0.5, -0.5, -1.0});

    this->declare_parameter<std::string>("topics.cmd_vel", "/cmd_vel");
    this->declare_parameter<std::string>("topics.robot_state", "/robot_state");
    this->declare_parameter<std::string>("topics.wirelesscontroller",
                                         "/wirelesscontroller");
    this->declare_parameter<std::string>("topics.state_transform_and_filtered",
                                         "/state_transform_and_filtered");
    this->declare_parameter<std::string>("topics.mocap_odom", "/mocap_odom");

    // Retrieve parameters
    this->get_parameter("cmd.rate", rate_);
    this->get_parameter("cmd.type", cmd_type_);
    this->get_parameter("cmd.max_vels", max_vels_);
    this->get_parameter("cmd.min_vels", min_vels_);

    this->get_parameter("cmd.wirelesscontroller_maxvels",
                        wirelesscontroller_maxvels_);

    if (cmd_type_ == "constant") {
      this->get_parameter("cmd.constant", cmd_constant_);
      if (cmd_constant_.size() != 3) {
        RCLCPP_ERROR(this->get_logger(),
                     "cmd.constant must have exactly 3 elements [linear.x, "
                     "linear.y, angular.z]");
        rclcpp::shutdown();
        return;
      }
    } else if (cmd_type_ == "straight_line") {
      this->get_parameter("cmd.straight_line.vx", cmd_straight_line_vx_);
      this->get_parameter("cmd.straight_line.Kp_y", Kp_y_);
      this->get_parameter("cmd.straight_line.Kd_y", Kd_y_);
      this->get_parameter("cmd.straight_line.Kp_yaw", Kp_yaw_);
      this->get_parameter("cmd.straight_line.Kd_yaw", Kd_yaw_);
    } else if (cmd_type_ == "straight_line_integral") {
      this->get_parameter("cmd.straight_line_integral.vx",
                          cmd_straight_line_vx_);
      this->get_parameter("cmd.straight_line_integral.Kp_x", Kp_x_);
      this->get_parameter("cmd.straight_line_integral.Kd_x", Kd_x_);
      this->get_parameter("cmd.straight_line_integral.Kp_y", Kp_y_);
      this->get_parameter("cmd.straight_line_integral.Kd_y", Kd_y_);
      this->get_parameter("cmd.straight_line_integral.Kp_yaw", Kp_yaw_);
      this->get_parameter("cmd.straight_line_integral.Kd_yaw", Kd_yaw_);
    } else if (cmd_type_ == "wirelesscontroller") {
    } else if (cmd_type_ == "wirelesscontrol_position") {
      this->get_parameter("cmd.wirelesscontrol_position.Kp_x", Kp_x_);
      this->get_parameter("cmd.wirelesscontrol_position.Kd_x", Kd_x_);
      this->get_parameter("cmd.wirelesscontrol_position.Kp_y", Kp_y_);
      this->get_parameter("cmd.wirelesscontrol_position.Kd_y", Kd_y_);
      this->get_parameter("cmd.wirelesscontrol_position.Kp_yaw", Kp_yaw_);
      this->get_parameter("cmd.wirelesscontrol_position.Kd_yaw", Kd_yaw_);
    } else {
      RCLCPP_ERROR(this->get_logger(),
                   "Invalid cmd.type: %s. Supported types: constant, "
                   "wirelesscontroller, straight_line, straight_line_integral, "
                   "wirelesscontrol_position",
                   cmd_type_.c_str());
      rclcpp::shutdown();
      return;
    }

    this->get_parameter("topics.cmd_vel", cmd_vel_topic_);
    this->get_parameter("topics.robot_state", topics_robot_state_);
    this->get_parameter("topics.wirelesscontroller",
                        topics_wirelesscontroller_);
    std::string topics_state_transform_and_filtered, topics_mocap_odom;
    this->get_parameter("topics.state_transform_and_filtered",
                        topics_state_transform_and_filtered);
    this->get_parameter("topics.mocap_odom", topics_mocap_odom);

    cmd_wirelesscontroller_ = {0.0f, 0.0f, 0.0f};
    need_to_init_odom_ = true;

    // Create subscriptions
    robot_state_sub_ = this->create_subscription<std_msgs::msg::String>(
        topics_robot_state_, 10,
        std::bind(&CmdVelPublisher::robot_state_callback, this,
                  std::placeholders::_1));
    wirelesscontroller_sub_ =
        this->create_subscription<unitree_go::msg::WirelessController>(
            topics_wirelesscontroller_, 10,
            std::bind(&CmdVelPublisher::wireless_controller_callback, this,
                      std::placeholders::_1));
    odom_topic_ = topics_state_transform_and_filtered;
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 1,
        std::bind(&CmdVelPublisher::odom_callback, this,
                  std::placeholders::_1));

    // Create publisher
    cmd_vel_pub_ =
        this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    // Timer to publish at specified rate
    dt_ = 1.0f / rate_;
    auto period = std::chrono::milliseconds(static_cast<int>(1000.0f * dt_));
    timer_ = this->create_wall_timer(
        period, std::bind(&CmdVelPublisher::publish_cmd_vel, this));
  }

private:
  void publish_cmd_vel() {
    float vx, vy, yaw_rate;

    if (robot_state_ != "WALKING" || need_to_init_odom_) {
      vx = 0.0f;
      vy = 0.0f;
      yaw_rate = 0.0f;
    } else if (cmd_type_ == "constant") {
      vx = cmd_constant_[0];
      vy = cmd_constant_[1];
      yaw_rate = cmd_constant_[2];
    }

    else if (cmd_type_ == "wirelesscontroller") {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      vx = cmd_wirelesscontroller_[0];
      vy = cmd_wirelesscontroller_[1];
      yaw_rate = cmd_wirelesscontroller_[2];
    }

    else if (cmd_type_ == "straight_line") {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      vx = cmd_straight_line_vx_;
      vy = -Kp_y_ * odom_.pose.pose.position.y -
           Kd_y_ * odom_.twist.twist.linear.y;
      float yaw = tf2::getYaw(odom_.pose.pose.orientation);
      yaw_rate = -Kp_yaw_ * yaw - Kd_yaw_ * odom_.twist.twist.angular.z;
    }

    else if (cmd_type_ == "straight_line_integral") {
      std::lock_guard<std::mutex> lock(cmd_mutex_);

      x_des_ += cmd_straight_line_vx_ * dt_;
      float dx_global = x_des_ - odom_.pose.pose.position.x;
      vx = Kp_x_ * dx_global - Kd_x_ * odom_.twist.twist.linear.x;
      vy = -Kp_y_ * odom_.pose.pose.position.y -
           Kd_y_ * odom_.twist.twist.linear.y;
      float yaw = tf2::getYaw(odom_.pose.pose.orientation);
      yaw_rate = -Kp_yaw_ * yaw - Kd_yaw_ * odom_.twist.twist.angular.z;
    }

    else if (cmd_type_ == "wirelesscontrol_position") {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      float yaw = tf2::getYaw(odom_.pose.pose.orientation);

      float vx_in = cmd_wirelesscontroller_[0];
      float vy_in = cmd_wirelesscontroller_[1];
      float yaw_rate_in = cmd_wirelesscontroller_[2];
      x_des_ += (std::cos(yaw) * vx_in - std::sin(yaw) * vy_in) * dt_;
      y_des_ += (std::sin(yaw) * vx_in + std::cos(yaw) * vy_in) * dt_;
      yaw_des_ += yaw_rate_in * dt_;
      yaw_des_ = std::atan2(std::sin(yaw_des_), std::cos(yaw_des_));

      float dx_global = x_des_ - odom_.pose.pose.position.x;
      float dy_global = y_des_ - odom_.pose.pose.position.y;
      float dx_body = std::cos(yaw) * dx_global + std::sin(yaw) * dy_global;
      float dy_body = -std::sin(yaw) * dx_global + std::cos(yaw) * dy_global;
      vx = Kp_x_ * dx_body - Kd_x_ * odom_.twist.twist.linear.x;
      vy = Kp_y_ * dy_body - Kd_y_ * odom_.twist.twist.linear.y;

      float dyaw = yaw_des_ - yaw;
      dyaw = std::atan2(std::sin(dyaw), std::cos(dyaw));
      yaw_rate = Kp_yaw_ * dyaw - Kd_yaw_ * odom_.twist.twist.angular.z;
    }

    else {
      RCLCPP_ERROR(this->get_logger(), "Invalid cmd.type: %s",
                   cmd_type_.c_str());
      rclcpp::shutdown();
      return;
    }

    vx = std::clamp(vx, static_cast<float>(min_vels_[0]),
                    static_cast<float>(max_vels_[0]));
    vy = std::clamp(vy, static_cast<float>(min_vels_[1]),
                    static_cast<float>(max_vels_[1]));
    yaw_rate = std::clamp(yaw_rate, static_cast<float>(min_vels_[2]),
                          static_cast<float>(max_vels_[2]));

    geometry_msgs::msg::Twist twist_msg;
    twist_msg.linear.x = vx;
    twist_msg.linear.y = vy;
    twist_msg.angular.z = yaw_rate;

    cmd_vel_pub_->publish(twist_msg);
  }

  void robot_state_callback(const std_msgs::msg::String::SharedPtr msg) {
    robot_state_ = msg->data;
    if (robot_state_ == "WALKING" || robot_state_ == "STAND") {
      need_to_init_odom_ = true;
      x_des_ = 0.0f;
      y_des_ = 0.0f;
      yaw_des_ = 0.0f;
    }
  }

  void wireless_controller_callback(
      const unitree_go::msg::WirelessController::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    cmd_wirelesscontroller_ = {
        msg->ly *
            static_cast<float>(wirelesscontroller_maxvels_[0]), // linear-x
        msg->lx *
            -static_cast<float>(wirelesscontroller_maxvels_[1]), // linear-y
        msg->rx *
            -static_cast<float>(wirelesscontroller_maxvels_[2]) // angular-z
    };
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(cmd_mutex_);

    if (need_to_init_odom_) {
      init_odom_ = *msg;
    }

    // Compute relative position
    geometry_msgs::msg::Point init_pos = init_odom_.pose.pose.position;
    geometry_msgs::msg::Point cur_pos = msg->pose.pose.position;

    tf2::Vector3 delta_pos(cur_pos.x - init_pos.x, cur_pos.y - init_pos.y,
                           cur_pos.z - init_pos.z);

    tf2::Quaternion init_q, cur_q;
    tf2::fromMsg(init_odom_.pose.pose.orientation, init_q);
    tf2::fromMsg(msg->pose.pose.orientation, cur_q);

    tf2::Quaternion init_q_inv = init_q.inverse();
    tf2::Vector3 rel_pos = tf2::quatRotate(init_q_inv, delta_pos);
    tf2::Quaternion rel_q = init_q_inv * cur_q;

    // Save relative odometry
    odom_.header = msg->header;
    odom_.pose.pose.position.x = rel_pos.x();
    odom_.pose.pose.position.y = rel_pos.y();
    odom_.pose.pose.position.z = rel_pos.z();
    odom_.pose.pose.orientation = tf2::toMsg(rel_q);
    odom_.twist = msg->twist; // velocity is already in body frame

    if (need_to_init_odom_) {
      need_to_init_odom_ = false;
    }
  }

  float rate_, dt_;
  float x_des_ = 0.0f;
  float y_des_ = 0.0f;
  float yaw_des_ = 0.0f;
  std::string cmd_type_;
  std::vector<double> cmd_constant_;
  std::vector<double> wirelesscontroller_maxvels_;
  std::vector<float> cmd_wirelesscontroller_;
  float cmd_straight_line_vx_, Kp_x_, Kd_x_, Kp_y_, Kd_y_, Kp_yaw_, Kd_yaw_;
  std::vector<double> max_vels_, min_vels_;
  std::mutex cmd_mutex_;
  std::string robot_state_;
  std::string cmd_vel_topic_, topics_robot_state_, topics_wirelesscontroller_;
  std::string odom_topic_;
  nav_msgs::msg::Odometry odom_, init_odom_;
  bool need_to_init_odom_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_state_sub_;
  rclcpp::Subscription<unitree_go::msg::WirelessController>::SharedPtr
      wirelesscontroller_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelPublisher>());
  rclcpp::shutdown();
  return 0;
}
