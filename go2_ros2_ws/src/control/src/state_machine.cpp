#include "control/motor_crc.h"
#include <chrono>
#include <fcntl.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <termios.h>
#include <unistd.h>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/wireless_controller.hpp>

enum class RobotState {
  OFF,
  PRONE,
  STANDING,
  STAND,
  WALKING,
  PRONING,
  RECOVERY
};

enum class InputSource { Keyboard, Controller };

class StateMachine : public rclcpp::Node {
public:
  StateMachine()
      : Node("state_machine",
             rclcpp::NodeOptions().use_intra_process_comms(true)) {

    RCLCPP_INFO(this->get_logger(), "Starting state machine");
    get_parameters();
    current_joint_pos_ = PRONE_JOINT_ANGLES;
    transition_joint_pos_ = PRONE_JOINT_ANGLES;
    transition_start_joint_pos_ = PRONE_JOINT_ANGLES;
    desired_joint_pos_ = STAND_JOINT_ANGLES;

    lowstate_sub_ = create_subscription<unitree_go::msg::LowState>(
        topics_lowstate_, 1,
        std::bind(&StateMachine::lowstate_callback, this,
                  std::placeholders::_1));

    joint_cmd_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        topics_joint_pos_cmd_, 1,
        std::bind(&StateMachine::joint_cmd_callback, this,
                  std::placeholders::_1));

    wirelesscontroller_sub_ =
        create_subscription<unitree_go::msg::WirelessController>(
            topics_wirelesscontroller_, 1,
            std::bind(&StateMachine::handle_wireless_controller_input, this,
                      std::placeholders::_1));

    lowcmd_pub_ = create_publisher<unitree_go::msg::LowCmd>(topics_lowcmd_, 10);

    robot_state_pub_ =
        create_publisher<std_msgs::msg::String>(topics_robot_state_, 10);

    transition_state(RobotState::OFF);
    loop_thread_ = std::thread(&StateMachine::run_loop, this);

    RCLCPP_INFO(this->get_logger(), "State machine started.");
    RCLCPP_INFO(this->get_logger(), "Press the Start button / space bar to go to prone");
    RCLCPP_INFO(this->get_logger(),
                "Press any other button / key during any state to return to prone");
  }

  ~StateMachine() {
    RCLCPP_INFO(this->get_logger(), "Shutting down state machine...");
    shutdown();
    running_ = false;
    if (loop_thread_.joinable()) {
      loop_thread_.join();
    }
  }

private:
  void get_parameters() {
    declare_parameter<std::string>("topics.joint_pos_cmd", "/joint_pos_cmd");
    declare_parameter<std::string>("topics.lowstate", "/lowstate");
    declare_parameter<std::string>("topics.lowcmd", "/lowcmd");
    declare_parameter<std::string>("topics.robot_state", "/robot_state");
    declare_parameter<std::string>("topics.wirelesscontroller",
                                   "/wirelesscontroller");
    declare_parameter<float>("state_machine.rate", 100.0);
    declare_parameter<float>("state_machine.standing_duration", 2.0);
    declare_parameter<float>("state_machine.proning_duration", 2.0);
    declare_parameter<float>("state_machine.stand_Kp", 100.0);
    declare_parameter<float>("state_machine.walking_Kp", 25.0);
    declare_parameter<float>("state_machine.proning_Kp", 25.0);
    declare_parameter<float>("state_machine.prone_Kp", 5.0);
    declare_parameter<float>("state_machine.Kd", 1.0);

    get_parameter("topics.joint_pos_cmd", topics_joint_pos_cmd_);
    get_parameter("topics.lowstate", topics_lowstate_);
    get_parameter("topics.lowcmd", topics_lowcmd_);
    get_parameter("topics.robot_state", topics_robot_state_);
    get_parameter("topics.wirelesscontroller", topics_wirelesscontroller_);
    get_parameter("state_machine.rate", state_machine_rate_);
    get_parameter("state_machine.standing_duration", standing_duration_);
    get_parameter("state_machine.proning_duration", proning_duration_);
    get_parameter("state_machine.stand_Kp", stand_Kp_);
    get_parameter("state_machine.walking_Kp", walking_Kp_);
    get_parameter("state_machine.proning_Kp", proning_Kp_);
    get_parameter("state_machine.prone_Kp", prone_Kp_);
    get_parameter("state_machine.Kd", Kd_);
  }

  void lowstate_callback(const unitree_go::msg::LowState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    for (int i = 0; i < 12; ++i) {
      current_joint_pos_[i] = msg->motor_state[i].q;
    }
  }

  void
  joint_cmd_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    for (int i = 0; i < 12; ++i) {
      desired_joint_pos_[i] = msg->data[i];
    }
  }

  void run_loop() {
    rclcpp::Rate rate(state_machine_rate_);
    while (running_ && rclcpp::ok()) {
      handle_keyboard_input();
      control_loop();
      rate.sleep(); // TODO: this does not actually respect sim time in Humble
    }
  }

  void control_loop() {
    std::lock_guard<std::mutex> lock(control_mutex_);
    auto lowcmd_msg = create_lowcmd_message();
    lowcmd_pub_->publish(lowcmd_msg);
  }

  unitree_go::msg::LowCmd create_lowcmd_message() {
    compute_transition();

    unitree_go::msg::LowCmd cmd;
    for (int i = 0; i < 12; ++i) {
      cmd.motor_cmd[i].mode = 0x01;
      cmd.motor_cmd[i].dq = 0.0;
      cmd.motor_cmd[i].kd = Kd_;
      cmd.motor_cmd[i].tau = 0.0;
      switch (state_) {
      case RobotState::OFF:
        cmd.motor_cmd[i].q = PosStopF;
        cmd.motor_cmd[i].dq = VelStopF;
        cmd.motor_cmd[i].kp = 0.0;
        cmd.motor_cmd[i].kd = 0.0;
        break;
      case RobotState::PRONE:
        cmd.motor_cmd[i].q = PRONE_JOINT_ANGLES[i];
        cmd.motor_cmd[i].kp = prone_Kp_;
        break;
      case RobotState::STANDING:
        cmd.motor_cmd[i].q = transition_joint_pos_[i];
        cmd.motor_cmd[i].kp = stand_Kp_;
        break;
      case RobotState::STAND:
        cmd.motor_cmd[i].q = STAND_JOINT_ANGLES[i];
        cmd.motor_cmd[i].kp = stand_Kp_;
        break;
      case RobotState::WALKING:
        cmd.motor_cmd[i].q = desired_joint_pos_[i];
        cmd.motor_cmd[i].kp = walking_Kp_;
        break;
      case RobotState::PRONING:
        cmd.motor_cmd[i].q = transition_joint_pos_[i];
        if (i < 6)
          cmd.motor_cmd[i].kp = proning_Kp_;
        else
          // Reduce Kp for back joints to prevent lidar from hitting the ground
          cmd.motor_cmd[i].kp = proning_Kp_ / 1.5;
        break;
      case RobotState::RECOVERY:
        cmd.motor_cmd[i].q = RECOVERY_JOINT_ANGLES[i];
        cmd.motor_cmd[i].kp = stand_Kp_;
        break;
      }

      if (state_ != RobotState::OFF) {
        // Enforce joint limits
        if (cmd.motor_cmd[i].q < MIN_JOINT_ANGLES[i])
          cmd.motor_cmd[i].q = MIN_JOINT_ANGLES[i];
        else if (cmd.motor_cmd[i].q > MAX_JOINT_ANGLES[i])
          cmd.motor_cmd[i].q = MAX_JOINT_ANGLES[i];
      }
    }
    get_crc(cmd);
    return cmd;
  }

  void compute_transition() {
    if (state_ != RobotState::STANDING && state_ != RobotState::PRONING)
      return;

    transition_progress_ +=
        1.0 / (state_ == RobotState::STANDING
                   ? standing_duration_ * state_machine_rate_
                   : proning_duration_ * state_machine_rate_);
    transition_progress_ = std::min(transition_progress_, 1.0f);
    for (int i = 0; i < 12; ++i) {
      transition_joint_pos_[i] =
          transition_start_joint_pos_[i] +
          transition_progress_ *
              (state_ == RobotState::STANDING
                   ? STAND_JOINT_ANGLES[i] - transition_start_joint_pos_[i]
                   : PRONE_JOINT_ANGLES[i] - transition_start_joint_pos_[i]);
    }
    if (transition_progress_ >= 1.0f) {
      if (state_ == RobotState::STANDING)
        transition_state(RobotState::STAND);
      else if (state_ == RobotState::PRONING)
        transition_state(RobotState::PRONE);
      transition_progress_ = 0.0f;
    }
  }

  void transition_state(RobotState new_state) {
    RCLCPP_INFO(this->get_logger(), "Transitioning to state: %s",
                state_to_string(new_state).c_str());

    switch (new_state) {
    case RobotState::OFF:
      break;
    case RobotState::PRONE:
      RCLCPP_INFO(this->get_logger(), "Press the start button / space bar to stand");
      break;
    case RobotState::STANDING:
      break;
    case RobotState::STAND:
      RCLCPP_INFO(this->get_logger(), "Press the start button / space bar to walk");
      RCLCPP_INFO(this->get_logger(), "Press any other button / key to return to prone");
      break;
    case RobotState::WALKING:
      RCLCPP_INFO(this->get_logger(), "Press the start button / space bar to stand");
      RCLCPP_INFO(this->get_logger(), "Press any other button / key to go to recovery");
      break;
    case RobotState::PRONING:
      break;
    case RobotState::RECOVERY:
      RCLCPP_INFO(this->get_logger(), "Press the start button / space bar to stand");
      RCLCPP_INFO(this->get_logger(), "Press any other button / key to return to prone");
      break;
    }

    if (new_state == RobotState::STANDING || new_state == RobotState::PRONING)
      transition_progress_ = 0.0;
    transition_start_joint_pos_ = current_joint_pos_;
    state_ = new_state;
    publish_state();
  }

  void publish_state() {
    std_msgs::msg::String msg;
    msg.data = state_to_string(state_);
    robot_state_pub_->publish(msg);
  }

  static int get_key_input() {
    struct termios oldt, newt;
    int ch;
    int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, old_flags | O_NONBLOCK);

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, old_flags);

    return (ch != EOF) ? ch : -1; // Return -1 if no input was detected
  }

  void handle_key_input(int key, InputSource source) {
    if (key == -1 || key == 0)
      return;

    bool is_space = false;
    bool is_non_space = false;

    if (source == InputSource::Keyboard) {
      is_space = (key == ' ');
      is_non_space = (key != ' ' && key != -1);
    } else if (source == InputSource::Controller) {
      is_space = (key == 4);
      is_non_space = (key != 4 && key != 0);
    }

    switch (state_) {
    case RobotState::OFF:
      if (is_space)
        transition_state(RobotState::PRONE);
      break;
    case RobotState::PRONE:
      if (is_space)
        transition_state(RobotState::STANDING);
      break;
    case RobotState::STANDING:
      if (is_non_space)
        transition_state(RobotState::PRONING);
      break;
    case RobotState::STAND:
      if (is_space)
        transition_state(RobotState::WALKING);
      else if (is_non_space)
        transition_state(RobotState::PRONING);
      break;
    case RobotState::WALKING:
      if (is_space)
        transition_state(RobotState::STAND);
      else if (is_non_space)
        transition_state(RobotState::RECOVERY);
      break;
    case RobotState::PRONING:
      break;
    case RobotState::RECOVERY:
      if (is_space)
        transition_state(RobotState::STANDING);
      else if (is_non_space)
        transition_state(RobotState::PRONING);
      break;
    }
  }

  void handle_keyboard_input() {
    int key = get_key_input();
    handle_key_input(key, InputSource::Keyboard);
  }

  void handle_wireless_controller_input(
      const unitree_go::msg::WirelessController::SharedPtr msg) {
    uint16_t key = msg->keys;

    if (key == 0)
      return;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_controller_input_time_);

    if (elapsed.count() < controller_input_cooldown_ms_)
      return;

    last_controller_input_time_ = now;

    handle_key_input(static_cast<int>(key), InputSource::Controller);
  }

  void shutdown() {
    transition_state(RobotState::OFF);
    control_loop();
    RCLCPP_INFO(this->get_logger(), "State machine safely shut down");
    rclcpp::shutdown();
  }

  std::string state_to_string(RobotState state) {
    switch (state) {
    case RobotState::OFF:
      return "OFF";
    case RobotState::PRONE:
      return "PRONE";
    case RobotState::STANDING:
      return "STANDING";
    case RobotState::STAND:
      return "STAND";
    case RobotState::WALKING:
      return "WALKING";
    case RobotState::PRONING:
      return "PRONING";
    case RobotState::RECOVERY:
      return "RECOVERY";
    default:
      return "UNKNOWN";
    }
  }

  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
      joint_cmd_sub_;
  rclcpp::Subscription<unitree_go::msg::WirelessController>::SharedPtr
      wirelesscontroller_sub_;
  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robot_state_pub_;

  std::string topics_joint_pos_cmd_, topics_lowstate_, topics_lowcmd_,
      topics_robot_state_, topics_wirelesscontroller_;
  double state_machine_rate_, standing_duration_, proning_duration_;
  double stand_Kp_, walking_Kp_, proning_Kp_, prone_Kp_, Kd_;
  RobotState state_;
  float transition_progress_ = 0.0f;
  bool running_ = true;
  std::thread loop_thread_;
  std::mutex control_mutex_;
  std::chrono::steady_clock::time_point last_controller_input_time_;
  const int controller_input_cooldown_ms_ = 250;

  std::vector<float> desired_joint_pos_;
  std::vector<float> current_joint_pos_;
  std::vector<float> transition_joint_pos_, transition_start_joint_pos_;

  const std::vector<float> PRONE_JOINT_ANGLES = {-0.35, 1.36,  -2.65, 0.35,
                                                 1.36,  -2.65, -0.35, 1.36,
                                                 -2.65, 0.35,  1.36,  -2.65};
  const std::vector<float> STAND_JOINT_ANGLES = {
      -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5};
  const std::vector<float> RECOVERY_JOINT_ANGLES = {
      -0.3, 1.0, -1.85, 0.3, 1.0, -1.85, -0.3, 1.23, -2.15, 0.3, 1.23, -2.15};
  const std::vector<float> MIN_JOINT_ANGLES = {-1.04, -0.52, -2.72, -1.04,
                                               -0.52, -2.72, -1.04, -0.52,
                                               -2.72, -1.04, -0.52, -2.72};
  const std::vector<float> MAX_JOINT_ANGLES = {1.04,  4.53,  -0.83, 1.04,
                                               4.53,  -0.83, 1.04,  4.53,
                                               -0.83, 1.04,  4.53,  -0.83};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StateMachine>();
  rclcpp::spin(node);
  return 0;
}