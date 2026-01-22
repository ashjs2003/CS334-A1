/**
 * \file gita_test_node.cpp
 * \brief Provides a test utility for controlling a ros2-foxy-enabled robot over the network
 *
 * \copyright
 * Copyright (c) 2025 Piaggio Fast Forward (PFF), Inc.
 * All Rights Reserved. Reproduction, publication,
 * or re-transmittal not allowed without written permission of PFF, Inc.
 */

#include <cstdio>
#include "gita_test/gita_test_node.hpp"

gitaTest::gitaTest()
  : Node("gita_test_node") {

    // Add ROS parameter for robot ID and read it in
    int robot_id;
    this->declare_parameter<int>("robot_id", 0); // Default to ID 0
    this->get_parameter("robot_id", robot_id);

    RCLCPP_INFO(this->get_logger(), "Controlling robot '%d'", robot_id);
    std::string robot_namespace = "/gita_" + std::to_string(robot_id);

    // setup network communication (second value is QOS history depth)
    // publishers
    m_twist_command_pub    = this->create_publisher<geometry_msgs::msg::Twist>((robot_namespace + "/twist_cmd"), 10);
    m_standing_command_pub = this->create_publisher<example_interfaces::msg::Bool>((robot_namespace + "/standing_cmd"), 10);
    m_pairing_command_pub  = this->create_publisher<example_interfaces::msg::Bool>((robot_namespace + "/pairing_cmd"), 10);
    m_source_command_pub   = this->create_publisher<example_interfaces::msg::Bool>((robot_namespace + "/source_cmd"), 10);
    // subscribers
    m_twist_feedback_sub          = this->create_subscription<geometry_msgs::msg::Twist>((robot_namespace + "/robot_twist"), 10, std::bind(&gitaTest::robotTwistCallback, this, _1));
    m_robot_pose_feedback_sub     = this->create_subscription<geometry_msgs::msg::Pose>((robot_namespace + "/robot_pose"), 10, std::bind(&gitaTest::robotPoseCallback, this, _1));
    m_track_position_feedback_sub = this->create_subscription<geometry_msgs::msg::Point>((robot_namespace + "/track_position"), 10, std::bind(&gitaTest::trackPositionCallback, this, _1));
    m_standing_feedback_sub       = this->create_subscription<example_interfaces::msg::Bool>((robot_namespace + "/robot_standing"), 10, std::bind(&gitaTest::robotStandingCallback, this, _1));
    m_pairing_feedback_sub        = this->create_subscription<example_interfaces::msg::Bool>((robot_namespdddace + "/robot_paired"), 10, std::bind(&gitaTest::robotPairingCallback, this, _1));

    // setup components to make the gita node pleasent to use
    m_key_getter = std::make_unique<CurrentKeyGetter>();

    constexpr float max_lin_accel = 1.5; // m/s^2
    constexpr float max_ang_accel = 2; // rads^2
    constexpr float max_lin_speed = 2.0; // m/s
    constexpr float max_ang_speed = 2.0; // rad/s

    // setup speed and acceleration limits
    m_max_lin_speed = std::abs(max_lin_speed);
    m_max_ang_speed = std::abs(max_ang_speed);
    m_lin_accel_limiter = std::make_unique<AccelLimiter>(max_lin_accel);
    m_ang_accel_limiter = std::make_unique<AccelLimiter>(max_ang_accel);

    // set the loop period
    float key_polling_frequency  = 10;
    m_command_loop_period        = std::chrono::milliseconds(int(1e3f / key_polling_frequency));
    m_command_acquisition_thread = std::thread(&gitaTest::getKeyboardCommands, this);

    // setup ROS loop timing
    m_main_loop_timer = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&gitaTest::run, this));
}

void gitaTest::getKeyboardCommands() {

    std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds> next_loop_time = std::chrono::steady_clock::now();

    static bool standing        = false;
    static bool pairing         = false;
    static bool external_source = false;
    static char last_key = 0;

    // if we are not shutdown
    while (!m_shutdown.load()) {

        // set the time for the next loop
        next_loop_time += m_command_loop_period;

        // get the latest key press infor from the key getter
        std::pair<char, bool> keyboard_state = m_key_getter->getLatestKey();

        float lin_out = 0.0; // m/s
        float ang_out = 0.0; // rad/s

        // if a key is pressed
        if (keyboard_state.second) {

            uint64_t current_timestamp = (std::chrono::steady_clock::now()).time_since_epoch().count();

            switch (keyboard_state.first) {
                // Forwards
                case 'w':
                    lin_out = m_max_lin_speed * 0.5;
                    if (last_key != 'w') {
                        std::cout << current_timestamp << " - w pressed" << std::endl;
                    }
                    break;
                // Forwards (fast)
                case 'W':
                    lin_out = m_max_lin_speed;
                    if (last_key != 'W') {
                        std::cout << current_timestamp <<  " - shift+w pressed" << std::endl;
                    }
                    break;
                // Right turn
                case 'd':
                    ang_out = -m_max_lin_speed * 0.5;
                    if (last_key != 'd') {
                        std::cout << current_timestamp << " - d pressed" << std::endl;
                    }
                    break;
                // Right turn (fast)
                case 'D':
                    ang_out = -m_max_ang_speed;
                    if (last_key != 'D') {
                        std::cout << current_timestamp << " - shift+d pressed" << std::endl;
                    }
                    break;
                // Left turn
                case 'a':
                    ang_out = m_max_ang_speed * 0.5;
                    if (last_key != 'a') {
                        std::cout << current_timestamp << " - a pressed" << std::endl;
                    }
                    break;
                // Left turn (fast)
                case 'A':
                    ang_out = m_max_ang_speed;
                    if (last_key != 'A') {
                        std::cout << current_timestamp << " - shift+a pressed" << std::endl;
                    }
                    break;
                // Backwards
                case 's':
                    lin_out = -m_max_lin_speed * 0.5;
                    if (last_key != 's') {
                        std::cout << current_timestamp << " - s pressed" << std::endl;
                    }
                    break;
                // Backwards (fast)
                case 'S':
                    lin_out = -m_max_lin_speed;
                    if (last_key != 'S') {
                        std::cout << current_timestamp << " - shift+s pressed" << std::endl;
                    }
                    break;
                // Pair/Unpair
                case 'p':
                    {
                        example_interfaces::msg::Bool pairing_msg;
                        pairing_msg.data = !pairing;
                        m_pairing_command_pub->publish(pairing_msg);

                        std::cout << current_timestamp << " - p pressed" << std::endl;

                        pairing = !pairing;
                        break;
                    }
                // External/internal twist source
                case 'x':
                    {
                        example_interfaces::msg::Bool source_msg;
                        source_msg.data = !external_source;
                        m_source_command_pub->publish(source_msg);

                        std::cout << current_timestamp << " - x pressed" << std::endl;

                        external_source = !external_source;
                        break;
                    }
                // Sit/Stand
                case 'c':
                    {
                        example_interfaces::msg::Bool standing_msg;
                        standing_msg.data = !standing;
                        m_standing_command_pub->publish(standing_msg);

                        std::cout << current_timestamp << " - c pressed" << std::endl;

                        standing = !standing;
                        break;
                    }
                // Brake/stop as fast as possible
                case 'b':
                {
                    // reset acceleration limits
                    m_lin_accel_limiter->reset();
                    m_ang_accel_limiter->reset();

                    std::cout << current_timestamp << " - b pressed" << std::endl;

                    break;
                }
            }

            // record the last key that was pressed
            last_key = keyboard_state.first;
        } else {
            // record that a key was not pressed
            last_key = 0;
        }

        {            
            // update the velocity output
            std::unique_lock twist_lock(m_twist_mutex);
            m_linear_velocity_cmd = lin_out;
            m_angular_velocity_cmd = ang_out;
        }

        // sleep until the next loop
        std::this_thread::sleep_until(next_loop_time);
    }
}

gitaTest::~gitaTest() {
    // stop the key acquisition thread
    m_shutdown.store(true);
    m_command_acquisition_thread.join();
}

void gitaTest::run() {

    // if we are not shutdown
    if (!m_shutdown.load()) {

        geometry_msgs::msg::Twist twist;

        float lin_out = 0;
        float ang_out = 0;
        {
            std::shared_lock twist_lock(m_twist_mutex);
            lin_out = m_linear_velocity_cmd;
            ang_out = m_angular_velocity_cmd;
        }

        // limit the acceleration of the output
        m_lin_accel_limiter->limitOutput(lin_out);
        m_ang_accel_limiter->limitOutput(ang_out);

        // set twist command output
        twist.linear.x  = std::clamp(m_lin_accel_limiter->getOutput(), -m_max_lin_speed, m_max_lin_speed);
        twist.angular.z = std::clamp(m_ang_accel_limiter->getOutput(), -m_max_ang_speed, m_max_ang_speed);

        // always publish, otherwise we time out
        m_twist_command_pub->publish(twist);

        std::this_thread::yield();

        // if we need to shutdown
  } else {
    // shutdown the node
    rclcpp::shutdown();
  }
}

int main(int argc, char ** argv)
{
  // proceed through the ROS2 node lifecycle
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<gitaTest>());
  rclcpp::shutdown();

  return 0;
}
