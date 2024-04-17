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

        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map_size = None

    def map_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution  # resolution in meters/cell
        self.map_data = msg.data
        self.map_orientation = msg.info.origin.orientation
        self.map_position = msg.info.origin.position
    
        self.map_size = (self.map_width, self.map_height)

        self.get_logger().info("orientation: ", str(self.map_orientation))
        self.get_logger().info("position: ", str(self.map_orientation))
        


    def pose_cb(self, msg):
        # gets the initial pose 
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
        
        # reset the path
        self.trajectory.clear()
        self.trajectory.addPoint((position_x, position_y))

    def goal_cb(self, msg):
        # gets the goal pose
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
        self.goal_pose = np.array([position_x, position_y, theta]) # ***

        assert self.current_pose is not None
        assert self.map is not None

        # add the last point to the trajectory
        self.trajectory.addPoint((position_x, position_y))

        self.plan_path(self.current_pose, self.goal_pose, self.map)

    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


    # added
    def pixel_to_real(pixel):
        x = 25.900000
        y = 48.50000
        theta = 3.14
        scale = 0.0504
        transform = np.array([[np.cos(theta), -np.sin(theta), x],
                    [np.sin(theta), np.cos(theta), y],
                    [0,0,1]])

        pixel = np.array([pixel[0], pixel[1]]) * scale
        pixel = np.array([*pixel,1])
        pixel = np.linalg.inv(transform) @ pixel
        point = pixel
        return point

    def real_to_pixel(point):
        point = np.array([point[0], point[1], 1])
        x = 25.900000
        y = 48.50000
        theta = 3.14
        scale = 0.0504
        transform = np.array([[np.cos(theta), -np.sin(theta), x],
                    [np.sin(theta), np.cos(theta), y],
                    [0,0,1]])

        point = transform @ point
        point = point / scale
        pixel = point
        return pixel


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()