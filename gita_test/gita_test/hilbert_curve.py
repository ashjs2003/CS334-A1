#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
import math
import time

# ===============================
# Helpers
# ===============================

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
    return math.atan2(siny, cosy)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ===============================
# Hilbert geometry
# ===============================

def hilbert_index_to_xy(index, order):
    n = 2 ** order
    x = y = 0
    t = index
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def generate_hilbert_points(order, L):
    N = 2 ** order
    cell = L / (N - 1)
    pts = []
    for i in range(N * N):
        gx, gy = hilbert_index_to_xy(i, order)
        pts.append((gx * cell, gy * cell))
    return pts

# ===============================
# Pure Pursuit utilities
# ===============================

def advance_index_to_closest(path, x, y, start_idx):
    """Find closest point index, searching forward from start_idx."""
    best_i = start_idx
    best_d = float("inf")
    # small forward window is enough for dense paths
    end = min(len(path), start_idx + 200)
    for i in range(start_idx, end):
        d = math.hypot(path[i][0] - x, path[i][1] - y)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d

def find_lookahead_point(path, x, y, closest_idx, Ld):
    """Return index and point that is >= Ld away along the polyline from closest."""
    # walk forward accumulating arc length
    acc = 0.0
    i = closest_idx
    while i + 1 < len(path) and acc < Ld:
        acc += dist(path[i], path[i+1])
        i += 1
    return i, path[i]

def estimate_corner_severity(path, idx, window=6):
    """
    Rough corner metric: angle change between segments around idx.
    Returns |dtheta| in radians (0 = straight, pi = U-turn).
    """
    i0 = max(1, idx - window)
    i1 = min(len(path) - 2, idx + window)

    # segment headings
    a0 = math.atan2(path[i0][1]-path[i0-1][1], path[i0][0]-path[i0-1][0])
    a1 = math.atan2(path[i1+1][1]-path[i1][1], path[i1+1][0]-path[i1][0])
    return abs(wrap(a1 - a0))

# ===============================
# Node
# ===============================

class GitaHilbertPurePursuit(Node):
    def __init__(self):
        super().__init__('gita_hilbert_pure_pursuit')

        self.declare_parameter('robot_id', 1)
        rid = self.get_parameter('robot_id').value
        self.ns = f'/gita_{rid}'

        self.twist_pub = self.create_publisher(Twist, f'{self.ns}/twist_cmd', 10)
        self.create_subscription(Pose, f'{self.ns}/robot_pose', self.pose_cb, 10)

        # ---- Path params ----
        self.order = 3
        self.L = 1.5
        self.base_path = generate_hilbert_points(self.order, self.L)
        self.path = None  # will be anchored at start pose

        # ---- Control params ----
        self.dt = 0.05

        # lookahead (bigger = smoother but less accurate)
        self.Ld_min = 0.12
        self.Ld_max = 0.35
        self.Ld = 0.20

        # speed/limits
        self.v_max = 0.35
        self.v_min = 0.08
        self.w_max = 1.6

        # smoothness constraints (rate limits)
        self.w_dot_max = 3.0      # rad/s^2 (angular accel limit)
        self.v_dot_max = 0.6      # m/s^2

        # stopping
        self.goal_tol = 0.06

        # internal state
        self.pose = None
        self.yaw = 0.0
        self.started = False
        self.closest_idx = 0

        self.v_cmd_prev = 0.0
        self.w_cmd_prev = 0.0
        self.t_last = time.time()

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("Pure Pursuit Hilbert ready. Waiting for pose...")

    def pose_cb(self, msg):
        self.pose = msg.position
        self.yaw = yaw_from_quat(msg.orientation)

        # Anchor path at first pose so path starts where robot is
        if not self.started:
            x0, y0 = self.pose.x, self.pose.y
            # shift so first point equals current pose
            bx0, by0 = self.base_path[0]
            dx, dy = x0 - bx0, y0 - by0
            self.path = [(px + dx, py + dy) for (px, py) in self.base_path]
            self.started = True
            self.closest_idx = 0
            self.get_logger().info(
                f"Path anchored at start. order={self.order}, points={len(self.path)}"
            )

    def control_loop(self):
        if not self.started or self.pose is None or self.path is None:
            return

        now = time.time()
        dt = max(1e-3, now - self.t_last)
        self.t_last = now

        x, y, th = self.pose.x, self.pose.y, self.yaw

        # --- goal check ---
        gx, gy = self.path[-1]
        if math.hypot(gx - x, gy - y) < self.goal_tol:
            self.twist_pub.publish(Twist())
            self.get_logger().info("Reached final goal. Stopping.")
            return

        # --- closest point (search forward) ---
        self.closest_idx, d_closest = advance_index_to_closest(self.path, x, y, self.closest_idx)

        # --- adaptive lookahead for smoother turns ---
        # More corner severity => shorter lookahead for accuracy.
        corner = estimate_corner_severity(self.path, self.closest_idx, window=6)
        # map corner in [0, ~pi/2] to lookahead in [Ld_max .. Ld_min]
        t = clamp(corner / (math.pi / 2), 0.0, 1.0)
        self.Ld = (1.0 - t) * self.Ld_max + t * self.Ld_min

        # --- lookahead target ---
        tgt_idx, (xt, yt) = find_lookahead_point(self.path, x, y, self.closest_idx, self.Ld)

        # transform target to robot frame
        dx = xt - x
        dy = yt - y
        # angle to target in world
        ang = math.atan2(dy, dx)
        alpha = wrap(ang - th)

        # Pure Pursuit curvature
        kappa = (2.0 * math.sin(alpha)) / max(self.Ld, 1e-3)

        # choose speed (slow down on sharper curvature)
        # v ~ v_max / (1 + gain*|kappa|)
        v_des = self.v_max / (1.0 + 1.2 * abs(kappa))
        v_des = clamp(v_des, self.v_min, self.v_max)

        w_des = v_des * kappa
        w_des = clamp(w_des, -self.w_max, self.w_max)

        # --- rate limit for smoothness ---
        dv = clamp(v_des - self.v_cmd_prev, -self.v_dot_max * dt, self.v_dot_max * dt)
        dw = clamp(w_des - self.w_cmd_prev, -self.w_dot_max * dt, self.w_dot_max * dt)

        v_cmd = self.v_cmd_prev + dv
        w_cmd = self.w_cmd_prev + dw

        self.v_cmd_prev = v_cmd
        self.w_cmd_prev = w_cmd

        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.twist_pub.publish(cmd)

# ===============================
def main():
    rclpy.init()
    node = GitaHilbertPurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
