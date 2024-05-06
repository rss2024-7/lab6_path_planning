"""
Add the following to dockerfile:

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install opencv-python

To test:
ros2 launch racecar_simulator simulate.launch.xml

To test only path planner:
    ros2 launch path_planning sim_plan.launch.xml

To test path planner + path follower:
    ros2 launch path_planning sim_plan_follow.launch.xml
"""


"""
Add the following to dockerfile:

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install drake
RUN python3 -m pip install opencv-python
"""


import rclpy
from rclpy.node import Node

assert rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, PointStamped 
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Point as ROSPoint
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from .utils import LineTrajectory
from .visualization_tools import VisualizationTools

import time
import numpy as np
import pydot

from .nominal_path import nominal_path


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.TURN_RADIUS = 0.9
        self.ARM_LENGTH = 0.75

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

        self.shell_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.shell_cb,
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

        self.trajectory_pub = self.create_publisher(Marker, "viz/trajectory", 10)
        self.closest_pt_pub = self.create_publisher(Marker, "viz/closest_pt", 1)
        self.shell_pub = self.create_publisher(Marker, "viz/shell_pub", 1)
        self.circle_pub = self.create_publisher(Marker, "viz/circle_pub", 1)
        self.line_pub = self.create_publisher(Marker, "viz/line_pub", 1)
        self.line_id = 0  # Initialize a counter for line marker IDs

        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.trajectory = nominal_path  # initialize trajectory to nominal

        self.get_logger().info("=============================READY=============================")


    def map_cb(self, msg):
        self.map = msg
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution  # resolution in meters/cell
        self.map_data = msg.data

        points = self.lane_traj.points
        for i in range(len(points) - 1):
            self.edit_map(np.array(points[i]), np.array(points[i + 1]))

        # self.map_data = self.padded_map_grid
        self.map_orientation = msg.info.origin.orientation
        self.map_position = msg.info.origin.position
        self.map_info = (self.map_data, self.map_width, self.map_height, self.map_resolution)


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

        print(f"goal pose set: {self.goal_pose}")

        
    def shell_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        position_x = msg.point.x
        position_y = msg.point.y
        position_z = msg.point.z
        self.shell_pose = np.array([position_x, position_y])

        self.plan_path(self.shell_pose, self.map)

        self.get_logger().info(f"shell_pose set: {self.goal_pose}")


    def publish_point(self, point, publisher, r, g, b):
        if publisher.get_subscription_count() > 0:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = 0
            marker.type = 2  # sphere
            marker.action = 0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0
            publisher.publish(marker)
        elif publisher.get_subscription_count() == 0:
            self.get_logger().info("Not publishing point, no subscribers")


    def publish_circle(self, center, radius):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "circle"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        num_points = 50
        for i in range(num_points + 1):  # +1 to close the circle
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = 0.0
            marker.points.append(ROSPoint(x=x, y=y, z=z))

        marker.scale.x = 0.05  # Line width

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker.lifetime = Duration(sec=0, nanosec=0)  # 0 means the marker never auto-deletes

        self.circle_pub.publish(marker)


    def publish_line(self, p1, p2):
        line_strip = Marker()
        line_strip.type = Marker.LINE_STRIP
        line_strip.header.frame_id = "map"
        line_strip.id = self.line_id  # Set a unique ID for each line

        line_strip.scale.x = 0.01
        line_strip.scale.y = 0.01
        line_strip.color.a = 1.
        line_strip.color.r = 0.0
        line_strip.color.g = 0.0
        line_strip.color.b = 1.0

        p = ROSPoint()
        p.x = p1[0]
        p.y = p1[1]
        line_strip.points.append(p)
        p = ROSPoint()
        p.x = p2[0]
        p.y = p2[1]
        line_strip.points.append(p)

        self.line_pub.publish(line_strip)
        self.line_id += 1  # Increment the ID for the next line


    def find_tangent_point(self, p, c, r, shell_pos):
        """
        Calculate the point on circle centered at c with radius r such that the 
        line drawn from this point to p is tangent ot the circle, such that this
        point is closer to shell_pos than the other tangent point.
        """
        d = np.linalg.norm(p - c)  # distance from p to center of circle

        if d <= r:
            self.get_logger().info("The point p is inside the circle. No tangent line exists.")
            return np.array([None, None])

        theta = np.arccos(r/d)

        radius_vec = (p-c)/d * r

        rotation1 = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
        rotation2 = np.array([[np.cos(-theta), -np.sin(-theta)],
                              [np.sin(-theta), np.cos(-theta)]])

        center_to_tangent1 = rotation1 @ radius_vec
        center_to_tangent2 = rotation2 @ radius_vec

        t1 = c + center_to_tangent1
        t2 = c + center_to_tangent2

        if np.linalg.norm(shell_pos - t1) < np.linalg.norm(shell_pos - t2):
            return t1
        else:
            return t2


    def real_to_pixel(self, pt, column_row=False):
        #takes in [x,y] real coordinate
        point = np.array([pt[0], pt[1], 1.0])
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
    

    def check_segment_collision(self, p1, p2):
        vec = p2-p1
        dist = np.linalg.norm(vec)
        num_intervals = int(dist/0.1)

        for interval in range(1,num_intervals):
            coord = p1 + interval/num_intervals * vec
            pixel = self.real_to_pixel(coord)
            if self.map_data[self.index_from_pixel(pixel)] != 0:
                return True
        return False
    

    def plan_path(self, shell_point, map):
        self.get_logger().info(f"PLANNING PATH DEVIATION TO {shell_point}")

        VisualizationTools.plot_line([point['x'] for point in self.trajectory['points']], [point['y'] for point in self.trajectory['points']], self.trajectory_pub)

        shell_pos = shell_point[:2]

        # Visualize shell markers
        self.publish_point(shell_pos, self.shell_pub, 0.0, 1.0, 0.0)
        
        # Find closest point on path to the designated point
        # First, find closest endpoint to the designated point
        closest_dist_sq = np.inf
        closest_idx = 0
        for i in range(len(self.trajectory["points"])):
            new_dist_sq = (self.trajectory["points"][i]["x"] - shell_pos[0])**2 + (self.trajectory["points"][i]["y"] - shell_pos[1])**2
            if new_dist_sq < closest_dist_sq:
                closest_dist_sq = new_dist_sq
                closest_idx = i
            
        # Now, search the two segments adjacent to the closest endpoint to find
        # the true closest point to the designated point
        closest_dist_sq = np.inf
        closest_pt = None
        closest_segment = []  # will contain two indices of the trajectory that contain the closest point
        for i in range(closest_idx-1, closest_idx+1):
            if i < 0 or i >= len(self.trajectory["points"])-1:
                continue

            start = np.array([self.trajectory["points"][i]["x"], self.trajectory["points"][i]["y"]])
            end = np.array([self.trajectory["points"][i+1]["x"], self.trajectory["points"][i+1]["y"]])

            start_to_point = shell_point - start
            start_to_end = end - start

            segment_length_squared = np.dot(start_to_end, start_to_end)
            
            projection = np.dot(start_to_point, start_to_end) / segment_length_squared

            # Clamp the projection parameter to the range [0, 1]
            projection = max(0, min(1, projection))
            closest_pt_estimate = start + projection * start_to_end
            closest_pt_estimate_dist = np.linalg.norm(shell_point - closest_pt_estimate)

            if (closest_pt_estimate_dist < closest_dist_sq):
                closest_dist_sq = closest_pt_estimate_dist
                closest_pt = closest_pt_estimate
                closest_segment = [i, i+1]
        
        self.get_logger().info(f"closest_pt: {closest_pt}")
        self.publish_point(closest_pt, self.closest_pt_pub, 1.0, 0.5, 0.0)

        circle_center = shell_pos + (closest_pt - shell_pos) / np.linalg.norm(closest_pt - shell_pos) * (self.TURN_RADIUS + self.ARM_LENGTH)
        self.get_logger().info(f"circle_center: {circle_center}")
        self.publish_circle(circle_center, self.TURN_RADIUS)

        if np.sqrt(closest_dist_sq) < self.ARM_LENGTH:
            stop_point = closest_pt
        else:
            # shell point is at a "corner", between two line segments of the nominal trajectory
            if self.trajectory["points"][closest_idx]["x"] == closest_pt[0] and self.trajectory["points"][closest_idx]["y"] == closest_pt[1]:
                pass
            else:
                for pt in np.linspace(np.array([self.trajectory["points"][closest_segment[0]]["x"], self.trajectory["points"][closest_segment[0]]["y"]]),
                                      np.array([self.trajectory["points"][closest_segment[1]]["x"], self.trajectory["points"][closest_segment[1]]["y"]]),
                                      50):
                    tangent_pt = self.find_tangent_point(pt, circle_center, self.TURN_RADIUS, shell_pos)
                    if tangent_pt.any() == None:
                        continue
                    in_collision = self.check_segment_collision(pt, tangent_pt)
                    if not in_collision:
                        self.get_logger().info("hi")
                        self.publish_line(pt, tangent_pt)
            stop_point = shell_pos + (closest_pt - shell_pos) / np.linalg.norm(closest_pt - shell_pos) * (self.ARM_LENGTH)
            

        

        # traj_pose_array = PoseArray()
        # length_sum = 0.0
        # previous_point = None
        # for t in np.linspace(traj.start_time(), traj.end_time(), 100):
        #     self.get_logger().info(f"{traj.value(t)}")

        #     pose = Pose()
        #     pose.position.x = float(traj.value(t)[0,0])
        #     pose.position.y = float(traj.value(t)[1,0])
        #     pose.position.z = 0.0  # Assuming z is 0 for 2D coordinates
        #     pose.orientation.w = 1.0  # Neutral orientation
        #     traj_pose_array.poses.append(pose)

        #     current_point = np.array([pose.position.x, pose.position.y])

        #     # Calculate distance from the previous point if it exists
        #     if previous_point is not None:
        #         distance = np.linalg.norm(current_point - previous_point)
        #         length_sum += distance

        #     # Update previous_point to the current point for the next iteration
        #     previous_point = current_point

        # # set frame so visualization works
        # traj_pose_array.header.frame_id = "/map"  # replace with your frame id

        # self.traj_pub.publish(traj_pose_array)

        # self.get_logger().info(f"Total length of the trajectory: {length_sum}")


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
