#include "grid_map_msgs/msg/grid_map.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <algorithm>
#include <vector>

class GridMapToVec : public rclcpp::Node {
public:
  GridMapToVec() : Node("gridmap_to_vec") {
    // Declare parameters
    this->declare_parameter<std::string>("topics.elevation_grid_filled",
                                         "/elevation_grid_filled");
    this->declare_parameter<std::string>("topics.elevation_vec",
                                         "/elevation_vec");
    this->declare_parameter<std::string>("grid_layer", "elevation");
    this->declare_parameter<bool>("normalize_for_nn", false);
    this->declare_parameter<float>("nan_replacement",
                                   0.0); // Default replacement for NaNs

    // Get parameters
    this->get_parameter("topics.elevation_grid_filled", input_topic_);
    this->get_parameter("topics.elevation_vec", output_topic_);
    this->get_parameter("grid_layer", grid_layer_);
    this->get_parameter("normalize_for_nn", normalize_);
    this->get_parameter("nan_replacement", nan_replacement_);

    // Subscriber to filled GridMap
    grid_map_sub_ = this->create_subscription<grid_map_msgs::msg::GridMap>(
        input_topic_, 10,
        std::bind(&GridMapToVec::gridMapCallback, this, std::placeholders::_1));

    // Publisher for flattened neural network input
    vec_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        output_topic_, 10);
  }

private:
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr vec_pub_;

  std::string input_topic_, output_topic_, grid_layer_;
  bool normalize_;
  float nan_replacement_;

  void gridMapCallback(const grid_map_msgs::msg::GridMap::SharedPtr msg) {
    grid_map::GridMap map;
    grid_map::GridMapRosConverter::fromMessage(*msg, map);

    if (!map.exists(grid_layer_)) {
      RCLCPP_WARN(this->get_logger(), "Grid layer '%s' does not exist.",
                  grid_layer_.c_str());
      return;
    }

    // Get grid dimensions
    size_t num_cells_x = map.getSize()(0);
    size_t num_cells_y = map.getSize()(1);

    // Flatten the grid map into a 1D vector
    std::vector<float> vec(num_cells_x * num_cells_y);
    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::lowest();

    // Starting at the bottom right corner, iterate over the grid map
    // x is iterated first
    int idx = 0;
    for (int y = num_cells_y - 1; y >= 0; y--) {
      for (int x = num_cells_x - 1; x >= 0; x--) {
        float value = map.at(grid_layer_, grid_map::Index(x, y));

        if (std::isnan(value)) {
          value = nan_replacement_; // Replace NaNs
        }

        // Track min/max for normalization
        if (normalize_) {
          min_value = std::min(min_value, value);
          max_value = std::max(max_value, value);
        }

        vec[idx] = value;
        idx++;
      }
    }

    // Apply normalization if enabled
    if (normalize_ && max_value > min_value) {
      for (auto &val : vec) {
        val = (val - min_value) / (max_value - min_value); // Normalize to [0,1]
      }
    }

    // Publish as Float32MultiArray
    std_msgs::msg::Float32MultiArray output_msg;
    output_msg.data = vec;
    vec_pub_->publish(output_msg);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GridMapToVec>());
  rclcpp::shutdown();
  return 0;
}
