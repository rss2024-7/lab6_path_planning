import rclpy
from rclpy.node import Node
import numpy as np
import math
from .rrt_jm import Nd, RRT

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory


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
            self.initial_pose_topic, #self.initial_pose_topic
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.start = None
        self.goal = None
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.grid_probabilities = []
        self.obstacles = []

        self.get_logger().info("-----Initialized-----")

    def map_cb(self, msg):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        
        self.grid_probabilities = msg.data # get occupancy prob of each cell [0, 100, -1]
        self.get_logger().info("-----Received map_info-----")
        indices = np.where(msg.data == -1)
        # indices is empty, verify unknown value is -1
        self.get_logger().info(f"Indices: {indices}")
        x_inds = int(indices/self.map_width)
        y_inds = indices%self.map_width
        obst_inds = np.transpose(np.vstack((x_inds, y_inds)))
        
        # print("obst_inds:", obst_inds[10, :])
        self.obstacles = set(map(tuple, obst_inds)) # array of tuples (x,y)
        # self.get_logger().info("-----Received map_info-----")

    def pose_cb(self, pose):
        self.start = np.array([pose.pose.pose.position.x,
                                pose.pose.pose.position.y])  # only x,y coords
        self.get_logger().info("starting pose: " + np.array2string(self.start))

    def goal_cb(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])  # only x,y coords
        self.get_logger().info("goal pose: " + np.array2string(self.goal))
        path = PoseArray()
        
        assert self.current_pose is not None
        assert self.map is not None
        rrt = RRT(self.start, self.goal, step_size=5, max_iter=1000, threshold=1, \
                  min_steer=math.pi/2, obs=self.obstacles, map_info=[self.map_resolution, \
                                                                     [self.map_width, self.map_height],
                                                                     self.obstacles])
        final_path = rrt.rrt_search()
        # convert list of np.array([x,y]) to list of tuples
        path_arr = np.array(final_path)
        
        path_points = list(map(tuple, path_arr.reshape(len(final_path), 2)))
        print("path_points:", path_points[10, :])
        self.trajectory.points = path_points
        
        # publish trajectory
        self.plan_path()

    def plan_path(self):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
