"""
Add the following to dockerfile:

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install drake
RUN python3 -m pip install opencv-python


To test:
ros2 launch racecar_simulator simulate.launch.xml
ros2 launch path_planning sim_plan.launch.xml

or

ros2 launch path_planning build_trajectory.launch.xml
"""


import rclpy
from rclpy.node import Node

assert rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Point as ROSPoint
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from .utils import LineTrajectory

import time
import numpy as np
import pydot

from pydrake.all import (
    GraphOfConvexSetsOptions,
    GcsTrajectoryOptimization,
    Point,
    VPolytope,
    Solve,
    CompositeTrajectory,
    PiecewisePolynomial,
)

from.convex_set_build_tool import convex_sets


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "/odom")
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
        
        # FOR SIMULATION TESTING
        self.odom_sub = self.create_subscription(
            Odometry, 
            self.odom_topic,
            self.odom_cb,
            1
        )

        self.polygon_marker_pub = self.create_publisher(MarkerArray, 'polygons', 1)

        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.display_convex_sets()

        self.get_logger().info(f"{convex_sets}")
        self.get_logger().info("=============================READY=============================")


    def map_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        map_width = msg.info.width
        map_height = msg.info.height
        map_resolution = msg.info.resolution  # resolution in meters/cell
        map_data = msg.data


    def odom_cb(self, msg):
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

        # print(f"odom current_pose: {self.current_pose}")


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
        theta = 2 * np.arctan2(orientation_z, orientation_w)
        self.current_pose = np.array([position_x, position_y, theta])

        print(f"current_pose: {self.current_pose}")


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
        theta = 2 * np.arctan2(orientation_z, orientation_w)
        self.goal_pose = np.array([position_x, position_y, theta])

        print(f"goal_pose set: {self.goal_pose}")

        assert self.current_pose is not None

        self.plan_path(self.current_pose, self.goal_pose, self.map)


    def visualize_connectivity(self, regions):
        """
        Create and display SVG graph of region connectivity
        """
        numEdges = 0

        graph = pydot.Dot("GCS region connectivity")
        keys = list(regions.keys())
        for k in keys:
            graph.add_node(pydot.Node(k))
        for i in range(len(keys)):
            v1 = regions[keys[i]]
            for j in range(i + 1, len(keys)):
                v2 = regions[keys[j]]
                if v1.IntersectsWith(v2):
                    numEdges += 1
                    graph.add_edge(pydot.Edge(keys[i], keys[j], dir="both"))

        svg = graph.create_svg()

        with open('gcs_regions_connectivity.svg', 'wb') as svg_file:
            svg_file.write(svg)

        return numEdges
    

    def display_convex_sets(self):
        marker_array = MarkerArray()
        marker_id = 0

        for poly_index, polygon in enumerate(convex_sets):
            color = ColorRGBA()
            color.r = np.random.random()
            color.g = np.random.random()
            color.b = np.random.random()
            color.a = 1.0

            for point in polygon:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.25
                marker.scale.y = 0.25
                marker.scale.z = 0.25
                marker.color = color
                marker.pose.orientation.w = 1.0
                marker.pose.position = ROSPoint(x=float(point[0]), y=float(point[1]), z=0.0)
                marker.id = marker_id
                marker_id += 1

                marker_array.markers.append(marker)

        self.polygon_marker_pub.publish(marker_array)
        self.get_logger().info('Publishing Polygons MarkerArray.')
    

    def plan_path(self, start_point, end_point, map):
        self.get_logger().info(f"PLANNING PATH FROM {start_point} TO {end_point}")

        self.display_convex_sets()

        start_pos = start_point[:2]
        goal_pos = end_point[:2]
        
        # Convert convex_sets list to dictionary of Vpolytope objects, with numbers as keys
        gcs_regions = {}
        for i, convex_set in enumerate(convex_sets):
            # Transpose convex_set to be d x n where d is ambient dimension and n is number of points
            gcs_regions[i+1] = VPolytope(convex_set.T)

        gcs_regions["start"] = Point(start_pos)
        gcs_regions["goal"] = Point(goal_pos)

        self.visualize_connectivity(gcs_regions)
        self.get_logger().info("Connectivity graph saved to gcs_regions_connectivity.svg.")

        edges = []

        gcs = GcsTrajectoryOptimization(len(start_pos))
        gcs_regions = gcs.AddRegions(list(gcs_regions.values()), order=1)
        source = gcs.AddRegions([Point(start_pos)], order=0)
        target = gcs.AddRegions([Point(goal_pos)], order=0)
        edges.append(gcs.AddEdges(source, gcs_regions))
        edges.append(gcs.AddEdges(gcs_regions, target))
        
        gcs.AddTimeCost()
        gcs.AddVelocityBounds(np.array([1.0, 1.0]), np.array([1.0, 1.0]))

        options = GraphOfConvexSetsOptions()
        options.preprocessing = True
        options.max_rounded_paths = 5  # Max number of distinct paths to compare during random rounding; only the lowest cost path is returned.
        start_time = time.time()
        self.get_logger().info("Starting gcs.SolvePath.")
        traj, result = gcs.SolvePath(source, target, options)
        self.get_logger().info(f"GCS SolvePath Runtime: {time.time() - start_time}")

        if not result.is_success():
            self.get_logger().info("GCS Solve Fail.")
            return
        
        self.get_logger().info(traj)

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
