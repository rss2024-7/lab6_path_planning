"""
Add the following to dockerfile:

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install drake


To test:
ros2 launch path_planning sim_plan.launch.xml
ros2 launch path_planning build_trajectory.launch.xml
"""


import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

import numpy as np


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        # To assist in building convex sets
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, 
            "/initialpose",
            self.pose_callback,
            1
        )

        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        # Vpolytope constructor: Constructs the polytope from a d-by-n matrix, where d is the ambient dimension, and n is the number of vertices.
        self.convex_sets = [
            np.array([0, 282],
                     [0, 345],
                     [45, 282],
                     [95, 345]),
        ]

    def map_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        map_width = msg.info.width
        map_height = msg.info.height
        map_resolution = msg.info.resolution  # resolution in meters/cell
        map_data = msg.data


    def pose_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        position_x = msg.pose.pose.position.x
        position_y = msg.pose.pose.position.y
        position_z = msg.pose.pose.position.z
        orientation_x = msg.pose.pose.orientation.x
        orientation_y = msg.pose.pose.orientation.y
        orientation_z = msg.pose.pose.orientation.z
        orientation_w = msg.pose.pose.orientation.w
        theta = 2 * np.atan2(orientation_z, orientation_w)
        self.current_pose = np.array([position_x, position_y, theta])

    def goal_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        position_x = msg.pose.position.x
        position_y = msg.pose.position.y
        position_z = msg.pose.position.z
        orientation_x = msg.pose.orientation.x
        orientation_y = msg.pose.orientation.y
        orientation_z = msg.pose.orientation.z
        orientation_w = msg.pose.orientation.w
        theta = 2 * np.atan2(orientation_z, orientation_w)
        self.goal_pose = np.array([position_x, position_y, theta])

        assert self.current_pose is not None
        assert self.map is not None

        self.plan_path(self.current_pose, self.goal_pose, self.map)

    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


    def pose_callback(self, msg):
        """
        Helper for manually defining convex sets
        """
        self.get_logger().info("initial pose")
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        angle = 2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.get_logger().info(f"initial pose set: {x}, {y}, {angle}")


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
