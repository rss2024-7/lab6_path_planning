import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

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
    
        # self.map_size = (self.map_width, self.map_height)


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


        self.get_logger().info("current pose: " + np.array2string(self.current_pose))
        
        # reset the path
        # self.trajectory.clear()
        # self.trajectory.addPoint((position_x, position_y))

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

        
        self.get_logger().info("goal: " + np.array2string(self.goal_pose))

        assert self.current_pose is not None
        assert self.map is not None



        # add the last point to the trajectory
        # self.trajectory.addPoint((position_x, position_y))

        
        

        self.plan_path(self.current_pose, self.goal_pose, self.map)

    def plan_path(self, start_point, end_point, map):
        # self.get_logger().info("orientation: " + str(self.map_orientation))
        self.get_logger().info("goal: " + np.array2string(self.goal_pose))
        pixel  = self.real_to_pixel(self.goal_pose)
        self.get_logger().info("pixel: " + np.array2string(pixel))
        real  = self.pixel_to_real(pixel)
        self.get_logger().info("real: " + np.array2string(real))

        probability = self.map_data[self.index_from_pixel(pixel)]
        self.get_logger().info("probability: " + str(probability))

        # self.get_logger().info("saving image")
        # occupancy = np.zeros([self.map_height, self.map_width])
        # for i in range(self.map_height):
        #     for j in range(self.map_width):
        #         ix = self.index_from_pixel([i,j])
        #         if self.map_data[ix] == 100:
        #             occupancy[i][j] = 255

        # cv2.imwrite("./occupancy.png", occupancy)

        # self.get_logger().info("image saved")


        self.trajectory.clear()
        self.trajectory.addPoint(start_point)


        # add intermediate points

        self.trajectory.addPoint(end_point)
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


    def pixel_to_real(self, pixel, column_row=False):
        # if column_row = True, pixel format is [v (column index), u (row index)] 
        if column_row:
            pixel = np.array([pixel[0], pixel[1]]) 
        else:
            pixel = np.array([pixel[1], pixel[0]]) 

        pixel = pixel * self.map_resolution
        pixel = np.array([*pixel,1])
        pixel = np.linalg.inv(self.transform) @ pixel
        point = pixel

        # returns real x,y 
        return point

    def real_to_pixel(self, point,column_row=False):
        #takes in [x,y] real coordinate
        point = np.array([point[0], point[1], 1])
        point = self.transform @ point
        point = point / self.map_resolution
        pixel = np.floor(point)

        if column_row: # returns [v (column index), u (row index)] 
            return np.array([pixel[0], pixel[1]])
        else:
            # returns [u (row index), v (column index)]
            return np.array([pixel[1], pixel[0]])
    
    def index_from_pixel(self, pixel, column_row=False):
        # calculate the index of the row-major map
        if column_row: # pixel = [v, u]
            return int(pixel[1] * self.map_width + pixel[0])
        else: # pixel = [u, v]
            return int(pixel[0] * self.map_width + pixel[1])

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()