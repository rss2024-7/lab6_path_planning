import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker
from wall_follower.visualization_tools import VisualizationTools
import numpy as np

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 1  # FILL IN #
        self.speed = 4.0  # FILL IN #
        self.wheelbase_length = 0.3  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        
        self.pose_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        
        self.point_pub = self.create_publisher(Marker, "/closest_point", 1)

    def pose_callback(self, odometry_msg):
        def rotation_matrix(angle):
            return np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
        
        # convert [x y theta] to 3x3 transform matrix
        def transform_matrix(pose):
            X = np.eye(3)
            X[:2, :2] = rotation_matrix(pose[2])
            X[:2, -1] = np.array(pose[:2])
            return X
        if not self.trajectory.points: return
        car_pos_x = odometry_msg.pose.pose.position.x
        car_pos_y = odometry_msg.pose.pose.position.y
        car_angle = 2 * np.arctan2(odometry_msg.pose.pose.orientation.z, odometry_msg.pose.pose.orientation.w)

        car_vec = np.array([car_pos_x, car_pos_y])

        points = np.array(self.trajectory.points)
        # self.get_logger().info("%s" % points)
        traj_x = points[:, 0]
        traj_y = points[:, 1]

        closest_segment_index = self.find_closest_segment(traj_x, traj_y, car_pos_x, car_pos_y)

        
        seg_end_x, seg_end_y = traj_x[closest_segment_index + 1], traj_y[closest_segment_index + 1]
        seg_angle_error = np.clip(np.abs(np.arctan2(seg_end_y - car_pos_y, seg_end_x - car_pos_x) - car_angle), 0, np.pi/2)
        self.lookahead = 2.0 - (seg_angle_error / (np.pi / 2)) * 1.0

        lookahead_points = self.get_lookahead_point(traj_x[closest_segment_index], traj_y[closest_segment_index],
                                                    traj_x[closest_segment_index + 1], traj_y[closest_segment_index + 1],
                                                    car_pos_x, car_pos_y)

        closest_point_x, closest_point_y = seg_end_x, seg_end_y
        if lookahead_points is not None:
            # # self.get_logger().info("no points found")
            # drive_msg = AckermannDriveStamped()
            # drive_msg.drive.speed = self.lookahead * 2.0
            # self.drive_pub.publish(drive_msg)
            # return
            closest_point_x, closest_point_y = lookahead_points
        # VisualizationTools.plot_line(np.array([car_pos_x, closest_point_x]), np.array([car_pos_y, closest_point_y]), self.point_pub, frame="map")
        VisualizationTools.plot_line(np.array([car_pos_x, traj_x[closest_segment_index + 1]]), np.array([car_pos_y, traj_y[closest_segment_index + 1]]), self.point_pub, frame="map")
        # closest_point_x = traj_x[0]
        # closest_point_y = traj_y[0]

        angle_error = np.arctan2(closest_point_y - car_pos_y, closest_point_x - car_pos_x) - car_angle
        angle_error /= 4
        angle_error = np.clip(angle_error, -0.34, 0.34)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.lookahead * 2.0
        # self.get_logger().info("%s" % (self.lookahead * 2.0))
        drive_msg.drive.steering_angle = angle_error
        self.drive_pub.publish(drive_msg)

    def find_closest_segment(self, x, y, car_x, car_y):
        def min_dist(x1, y1, x2, y2, px, py):
            v = np.array([x1, y1])
            w = np.array([x2, y2])
            p = np.array([px, py])
            l2 = np.dot(w - v, w - v)
            if (l2 == 0.0): return np.linalg.norm(p - v)

            t = max(0, min(1, np.dot(p - v, w  - v) / l2))
            projection = v + t * (w - v)
            return np.linalg.norm(p - projection)
        
        distances = np.zeros(len(x) - 1)
        for i in range(len(x) - 1):
            distances[i] = min_dist(x[i], y[i], x[i + 1], y[i + 1], car_x, car_y)
            if np.linalg.norm(np.array([x[i + 1] - car_x, y[i + 1] - car_y])) < self.lookahead:
                distances[i] += 100
        return np.where(distances == np.min(distances))[0][0]

    def get_lookahead_point(self, x1, y1, x2, y2, origin_x, origin_y):
        Q = np.array([origin_x, origin_y])                  # Centre of circle
        r = self.lookahead                  # Radius of circle

        P1 = np.array([x1, y1])      # Start of line segment
        P2 = np.array([x2, y2])
        V = P2 - P1  # Vector along line segment
        a = np.dot(V, V)
        b = 2 * np.dot(V, P1 - Q)
        c = np.dot(P1, P1) + np.dot(Q, Q) - 2 * np.dot(P1, Q) - r**2

        disc = (b**2 - 4 * a * c)
        if disc < 0:
            return None
        
        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)

        if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
            return None
        if not (0 <= t2 <= 1):
            return P1 + t1 * V
        if not (0 <= t1 <= 1):
            return P1 + t2 * V

        intersect1 = P1 + t1 * V
        intersect2 = P1 + t2 * V

        intersect = intersect1
        if np.linalg.norm(P2 - intersect2) < np.linalg.norm(P2 - intersect2):
            intersect = intersect2

        return intersect

        t = max(0, min(1, - b / (2 * a)))
        return P1 + t * V


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
