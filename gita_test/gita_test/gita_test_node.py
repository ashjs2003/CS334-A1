#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Pose, Point
from example_interfaces.msg import Bool

import threading
import time
import sys
import termios
import tty
from math import copysign


# ------------------------------------------------------------
# Acceleration limiter (faithful to C++ behavior)
# ------------------------------------------------------------
class AccelLimiter:
    def __init__(self, max_accel):
        self.max_accel = abs(max_accel)
        self.output = 0.0
        self.last_time = time.time()

    def reset(self):
        self.output = 0.0
        self.last_time = time.time()

    def limit(self, target):
        now = time.time()
        dt = max(now - self.last_time, 1e-6)

        delta = target - self.output
        max_delta = self.max_accel * dt

        if abs(delta) > max_delta:
            delta = copysign(max_delta, delta)

        self.output += delta
        self.last_time = now
        return self.output


# ------------------------------------------------------------
# Main Node
# ------------------------------------------------------------
class GitaTestNode(Node):

    def __init__(self):
        super().__init__('gita_test_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('robot_id', 0)
        self.robot_id = self.get_parameter('robot_id').value
        self.ns = f'/gita_{self.robot_id}'

        self.get_logger().info(f"Controlling robot '{self.robot_id}'")

        # ---------------- Publishers ----------------
        self.twist_pub = self.create_publisher(
            Twist, f'{self.ns}/twist_cmd', 10
        )
        self.standing_pub = self.create_publisher(
            Bool, f'{self.ns}/standing_cmd', 10
        )
        self.pairing_pub = self.create_publisher(
            Bool, f'{self.ns}/pairing_cmd', 10
        )
        self.source_pub = self.create_publisher(
            Bool, f'{self.ns}/source_cmd', 10
        )

        # ---------------- Subscribers ----------------
        self.create_subscription(
            Twist, f'{self.ns}/robot_twist', self.robot_twist_cb, 10
        )
        self.create_subscription(
            Pose, f'{self.ns}/robot_pose', self.robot_pose_cb, 10
        )
        self.create_subscription(
            Point, f'{self.ns}/track_position', self.track_position_cb, 10
        )
        self.create_subscription(
            Bool, f'{self.ns}/robot_standing', self.robot_standing_cb, 10
        )
        self.create_subscription(
            Bool, f'{self.ns}/robot_paired', self.robot_paired_cb, 10
        )

        # ---------------- Motion limits ----------------
        self.max_lin_speed = 2.0
        self.max_ang_speed = 2.0

        self.lin_limiter = AccelLimiter(max_accel=1.5)
        self.ang_limiter = AccelLimiter(max_accel=2.0)

        # ---------------- Shared state ----------------
        self.cmd_lock = threading.Lock()
        self.target_linear = 0.0
        self.target_angular = 0.0

        self.standing = False
        self.paired = False
        self.external_source = False

        self.shutdown_flag = False

        # ---------------- Keyboard thread ----------------
        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop, daemon=True
        )
        self.keyboard_thread.start()

        # ---------------- Publish loop ----------------
        self.timer = self.create_timer(0.05, self.publish_loop)

    # ------------------------------------------------------------
    # Keyboard handling (faithful mapping)
    # ------------------------------------------------------------
    def keyboard_loop(self):
        settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while not self.shutdown_flag:
                key = sys.stdin.read(1)
                with self.cmd_lock:

                    if key == 'w':
                        self.target_linear = 0.5 * self.max_lin_speed
                        self.get_logger().info(f"Moved half step forward")

                    elif key == 'W':
                        self.target_linear = self.max_lin_speed
                        self.get_logger().info(f"Moved Forward")

                    elif key == 's':
                        self.target_linear = -0.5 * self.max_lin_speed
                        self.get_logger().info(f"Moved half step backward")

                    elif key == 'S':
                        self.target_linear = -self.max_lin_speed
                        self.get_logger().info(f"Moved Backward")

                    elif key == 'a':
                        self.target_angular = 0.5 * self.max_ang_speed
                        self.get_logger().info(f"Turned half step left")

                    elif key == 'A':
                        self.target_angular = self.max_ang_speed
                        self.get_logger().info(f"Turned left")

                    elif key == 'd':
                        self.target_angular = -0.5 * self.max_ang_speed
                        self.get_logger().info(f"Turned half step right")

                    elif key == 'D':
                        self.target_angular = -self.max_ang_speed
                        self.get_logger().info(f"Turned right")

                    elif key == 'b':  # brake
                        self.lin_limiter.reset()
                        self.ang_limiter.reset()
                        self.target_linear = 0.0
                        self.target_angular = 0.0
                        self.get_logger().info(f"Brake")

                    elif key == 'c':  # sit / stand
                        self.standing = not self.standing
                        self.standing_pub.publish(Bool(data=self.standing))
                        self.get_logger().info(f"Standing: {self.standing}")

                    elif key == 'p':  # pair
                        self.paired = not self.paired
                        self.pairing_pub.publish(Bool(data=self.paired))
                        self.get_logger().info(f"Paired: {self.paired}")

                    elif key == 'x':  # source
                        self.external_source = not self.external_source
                        self.source_pub.publish(Bool(data=self.external_source))
                        self.get_logger().info(
                            f"External source: {self.external_source}"
                        )

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    # ------------------------------------------------------------
    # Publish loop (always publishing, timeout-safe)
    # ------------------------------------------------------------
    def publish_loop(self):
        with self.cmd_lock:
            lin = self.lin_limiter.limit(self.target_linear)
            ang = self.ang_limiter.limit(self.target_angular)

        lin = max(min(lin, self.max_lin_speed), -self.max_lin_speed)
        ang = max(min(ang, self.max_ang_speed), -self.max_ang_speed)

        msg = Twist()
        msg.linear.x = lin
        msg.angular.z = ang

        self.twist_pub.publish(msg)

    # ------------------------------------------------------------
    # Feedback callbacks (logging only, like C++)
    # ------------------------------------------------------------
    def robot_twist_cb(self, msg: Twist):
        pass

    def robot_pose_cb(self, msg: Pose):
        pass

    def track_position_cb(self, msg: Point):
        pass

    def robot_standing_cb(self, msg: Bool):
        self.standing = msg.data

    def robot_paired_cb(self, msg: Bool):
        self.paired = msg.data

    # ------------------------------------------------------------
    def destroy_node(self):
        self.shutdown_flag = True
        super().destroy_node()


# ------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = GitaTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
