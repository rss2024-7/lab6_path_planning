import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
from .rrt_star import RRT_star
from .rrt_star import Joint

import numpy as np
import cv2


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

        self.tree_pub = self.create_publisher(
            PoseArray,
            "tree",
            10
        )


        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map_size = None

        x = 25.900000
        y = 48.50000
        theta = 3.14
        resolution = 0.0504
        self.transform = np.array([[np.cos(theta), -np.sin(theta), x],
                    [np.sin(theta), np.cos(theta), y],
                    [0,0,1]])



        self.get_logger().info("------READY-----")

    def map_cb(self, msg):
        self.map = msg
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution  # resolution in meters/cell
        self.map_data = msg.data
        self.map_orientation = msg.info.origin.orientation
        self.map_position = msg.info.origin.position
        self.map_info = (self.map_data, self.map_width, self.map_height, self.map_resolution)

        self.get_logger().info("map height: " + str(self.map_height))
        self.get_logger().info("map width: " + str(self.map_width))
        self.get_logger().info("map data length: " + str(len(self.map_data)))
        self.get_logger().info("map [0][0]: " + str(self.pixel_to_real([0,0], self.transform, self.map_resolution)))
        self.get_logger().info("map [0][map width]: " + str(self.pixel_to_real([0,self.map_width], self.transform, self.map_resolution)))
        self.get_logger().info("map [map height][0]: " + str(self.pixel_to_real([self.map_height,0], self.transform, self.map_resolution)))
        self.get_logger().info("map [map height][map width]: " + str(self.pixel_to_real([self.map_height,self.map_width], self.transform, self.map_resolution)))
        self.get_logger().info("origin (25.9, 48.5) to pixel: " + str(self.real_to_pixel([25.9, 48.5], self.transform, self.map_resolution)))


        # self.get_logger().info("orientation: ", str(self.map_orientation))
        # self.get_logger().info("position: ", str(self.map_orientation))
        # self.get_logger().info("map: " + np.array2string(self.map_data))
        


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
        theta = 2 * np.arctan2(orientation_z, orientation_w)
        self.current_pose = np.array([position_x, position_y, theta]) 


        self.get_logger().info("starting pose: " + np.array2string(self.current_pose))

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
        theta = 2 * np.arctan2(orientation_z, orientation_w)
        self.goal_pose = np.array([position_x, position_y, theta]) # ***

        
        self.get_logger().info("goal pose: " + np.array2string(self.goal_pose))

        assert self.current_pose is not None
        assert self.map is not None


        self.plan_path(self.current_pose, self.goal_pose, self.map)

    def plan_path(self, start_point, end_point, map):

        path = RRT_star(self.map_info, start_point, end_point).make_tree()
        
        self.trajectory.points = path

        # self.trajectory.clear()
        # self.trajectory.addPoint(start_point)

        # point = self.real_to_pixel(end_point, self.transform, self.map_resolution)

        # self.get_logger().info("point: " + np.array2string(end_point))
        # self.get_logger().info("pixel: " + np.array2string(point))


        # TO-DO add intermediate points

        # self.trajectory.addPoint(end_point)
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


    def pixel_to_real(self, pixel, transform, resolution, column_row=False):
        # if column_row = True, pixel format is [v (column index), u (row index)] 
        if column_row:
            pixel = np.array([pixel[0], pixel[1]]) 
        else:
            pixel = np.array([pixel[1], pixel[0]]) 

        pixel = pixel * resolution
        pixel = np.array([*pixel,1])
        pixel = np.linalg.inv(transform) @ pixel
        point = pixel

        # returns real x,y 
        return point

    def real_to_pixel(self, point, transform, resolution, column_row=False):
        #takes in [x,y] real coordinate
        point = np.array([point[0], point[1], 1])
        point = transform @ point
        point = point / resolution
        pixel = np.floor(point)

        if column_row: # returns [v (column index), u (row index)] 
            return np.array([pixel[0], pixel[1]])
        else:
            # returns [u (row index), v (column index)]
            return np.array([pixel[1], pixel[0]])
    
    def index_from_pixel(self, pixel, map_width, column_row=False):
        # calculate the index of the row-major map
        if column_row: # pixel = [v, u]
            return int(pixel[1] * map_width + pixel[0])
        else: # pixel = [u, v]
            return int(pixel[0] * map_width + pixel[1])

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()