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
        self.ARM_LENGTH = 0.5

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
        self.deviation_point_pub = self.create_publisher(Marker, "viz/deviation_point", 1)
        self.circle_pub = self.create_publisher(Marker, "viz/circle_pub", 1)
        self.line_pub = self.create_publisher(Marker, "viz/line_pub", 1)
        self.id = 0  # Initialize a counter for line marker IDs

        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.trajectory = [nominal_path.copy()]  # initialize trajectory to nominal
        self.nominal_trajectory = nominal_path.copy()
        self.traj_idx = 0  # between 0 and 3, depending on whether we're going for shell 1, 2, 3, or returning to start.
        self.traj_colors = [(1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0), (1.0, 1.0, 0.5)]

        x = 25.900000
        y = 48.50000
        theta = 3.14
        resolution = 0.0504
        self.transform = np.array([[np.cos(theta), -np.sin(theta), x],
                                    [np.sin(theta), np.cos(theta), y],
                                    [0,0,1]])
        
        self.viz_tools = VisualizationTools()

        self.get_logger().info("=============================PLANNER READY=============================")


    def map_cb(self, msg):
        self.map = msg
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution  # resolution in meters/cell
        self.map_data = msg.data

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

        # print(f"goal pose set: {self.goal_pose}")

        self.viz_tools.plot_line([point['x'] for point in self.trajectory[self.traj_idx]["points"]], [point['y'] for point in self.trajectory[self.traj_idx]["points"]], self.trajectory_pub, color=(1.0, 1.0, 1.0), increment_id=True, scale=0.15)

        
    def shell_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        position_x = msg.point.x
        position_y = msg.point.y
        position_z = msg.point.z
        self.shell_pose = np.array([position_x, position_y])

        self.plan_path(self.shell_pose, self.map)

        self.get_logger().info(f"shell_pose set: {self.shell_pose}")


    def publish_point(self, point, publisher, r, g, b, size=0.3):
        if publisher.get_subscription_count() > 0:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = self.id
            marker.type = 2  # sphere
            marker.action = 0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.orientation.w = 1.0
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = size
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0
            publisher.publish(marker)
            self.id += 1
        elif publisher.get_subscription_count() == 0:
            self.get_logger().info("Not publishing point, no subscribers")


    def publish_circle(self, center, radius, id=0):
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
        line_strip.id = self.id  # Set a unique ID for each line

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
        self.id += 1  # Increment the ID for the next line


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
    

    def circle_to_trajectory(self, c, r, pt1, pt2, shell_pos, num_points=40):
        num_points = int(num_points*np.linalg.norm(pt2-pt1))
        
        # Calculate angles for pt1 and pt2
        theta1 = np.arctan2(pt1[1] - c[1], pt1[0] - c[0])
        theta2 = np.arctan2(pt2[1] - c[1], pt2[0] - c[0])
        
        # Normalize angles between 0 and 2*pi
        theta1 = theta1 if theta1 >= 0 else theta1 + 2 * np.pi
        theta2 = theta2 if theta2 >= 0 else theta2 + 2 * np.pi

        # Determine direction of the shortest path
        if (theta2 - theta1) % (2 * np.pi) > np.pi:
            reverse_path = True
            theta1, theta2 = theta2, theta1
        else:
            reverse_path = False

        # Create an array of angles from theta1 to theta2
        if theta1 > theta2:
            theta2 += 2 * np.pi
        angles = np.linspace(theta1, theta2, num_points)
        
        x_coords = c[0] + r * np.cos(angles)
        y_coords = c[1] + r * np.sin(angles)

        if reverse_path:
            x_coords = np.flip(x_coords)
            y_coords = np.flip(y_coords)

        closest_pt_to_shell_idx = None
        closest_pt_to_shell_dist = np.inf
        for i in range(np.shape(x_coords)[0]):
            if np.linalg.norm(np.array([x_coords[i], y_coords[i]]) - shell_pos) < closest_pt_to_shell_dist:
                closest_pt_to_shell_dist = np.linalg.norm(np.array([x_coords[i], y_coords[i]]) - shell_pos)
                closest_pt_to_shell_idx = i

            # in_collision = self.check_segment_collision(np.array([x_coords[i], y_coords[i]]), np.array([x_coords[i+1], y_coords[i+1]]))
            # if in_collision:
            #     self.get_logger().info("circle in collision")
            #     return np.array([None]), np.array([None])
        
        for i in range(len(x_coords)):  # Visualize if no collision
            time.sleep(0.02)
            self.publish_point(np.array([x_coords[i], y_coords[i]]), self.deviation_point_pub, 0.0, 0.0, 1.0, size=0.05)

        return np.vstack((x_coords, y_coords)).T, closest_pt_to_shell_idx
    

    def plan_deviation(self, path_pt_idx, closest_pt, circle_center, shell_pos, s_e):
        for pt in np.linspace(np.array([self.trajectory[self.traj_idx]["points"][path_pt_idx]["x"], self.trajectory[self.traj_idx]["points"][path_pt_idx]["y"]]), closest_pt, 50):
            tangent_pt = self.find_tangent_point(pt, circle_center, self.TURN_RADIUS, shell_pos)
            if tangent_pt.any() == None:
                continue
            in_collision = self.check_segment_collision(pt, tangent_pt)
            if not in_collision:
                self.publish_line(pt, tangent_pt)
                if s_e == "s":
                    self.publish_point(tangent_pt, self.deviation_point_pub, 0.0, 1.0, 1.0, size=0.1)
                if s_e == "e":
                    self.publish_point(tangent_pt, self.deviation_point_pub, 1.0, 0.0, 1.0, size=0.1)
                return tangent_pt  # once we find a non-colliding tangent point, return it


    def plan_path(self, shell_point, map):
        self.get_logger().info(f"PLANNING PATH TO {shell_point}")

        self.id += 1

        shell_pos = shell_point[:2]

        # Visualize shell markers
        self.publish_point(shell_pos, self.shell_pub, 0.0, 1.0, 0.0)
        
        # Find closest point on path to the designated point
        # First, find closest endpoint to the designated point
        closest_dist_sq = np.inf
        closest_idx = 0
        for i in range(len(self.trajectory[self.traj_idx]["points"])):
            new_dist_sq = (self.trajectory[self.traj_idx]["points"][i]["x"] - shell_pos[0])**2 + (self.trajectory[self.traj_idx]["points"][i]["y"] - shell_pos[1])**2
            if new_dist_sq < closest_dist_sq:
                closest_dist_sq = new_dist_sq
                closest_idx = i
            
        # Now, search the two segments adjacent to the closest endpoint to find
        # the true closest point to the designated point
        closest_dist_sq = np.inf
        closest_pt = None
        closest_segment = []  # will contain two indices of the trajectory that contain the closest point
        for i in range(closest_idx-1, closest_idx+1):
            if i < 0 or i >= len(self.trajectory[self.traj_idx]["points"])-1:
                continue

            start = np.array([self.trajectory[self.traj_idx]["points"][i]["x"], self.trajectory[self.traj_idx]["points"][i]["y"]])
            end = np.array([self.trajectory[self.traj_idx]["points"][i+1]["x"], self.trajectory[self.traj_idx]["points"][i+1]["y"]])

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
            circle_pts = [closest_pt]
            closest_circle_pt_to_shell_idx = 0
        else:
            # shell point is at a "corner", between two line segments of the nominal trajectory
            if np.linalg.norm(circle_center - np.array([self.trajectory[self.traj_idx]["points"][closest_idx]["x"], self.trajectory[self.traj_idx]["points"][closest_idx]["y"]])) <= self.TURN_RADIUS:
                if closest_idx > 0:
                    deviation_pt_1 = self.plan_deviation(closest_idx-1, closest_pt, circle_center, shell_pos, "s")
                if closest_idx < len(self.trajectory[self.traj_idx]["points"])-1:
                    deviation_pt_2 = self.plan_deviation(closest_idx+1, closest_pt, circle_center, shell_pos, "e")
                last_traj_idx_to_follow = closest_idx-1
                next_traj_idx_to_follow = last_traj_idx_to_follow + 2
            else:
                deviation_pt_1 = self.plan_deviation(closest_segment[0], closest_pt, circle_center, shell_pos, "s")
                deviation_pt_2 = self.plan_deviation(closest_segment[1], closest_pt, circle_center, shell_pos, "e")
                last_traj_idx_to_follow = closest_segment[0]
                next_traj_idx_to_follow = last_traj_idx_to_follow + 1

            circle_pts, closest_circle_pt_to_shell_idx = self.circle_to_trajectory(circle_center, self.TURN_RADIUS, deviation_pt_1, deviation_pt_2, shell_pos)

        # Update the "global" trajectory
        self.nominal_trajectory = {"points": self.trajectory[self.traj_idx]["points"][:last_traj_idx_to_follow+1] + [{"x": p[0], "y": p[1]} for p in circle_pts] + self.trajectory[self.traj_idx]["points"][next_traj_idx_to_follow:]}

        self.get_logger().info(f"closest_circle_pt_to_shell_idx: {closest_circle_pt_to_shell_idx}")
        self.trajectory[self.traj_idx]["points"] = self.trajectory[self.traj_idx]["points"][:last_traj_idx_to_follow+1]  # remove points beyond
        self.trajectory[self.traj_idx]["points"] += [{"x": p[0], "y": p[1]} for p in circle_pts[:closest_circle_pt_to_shell_idx+1]]  # add points in circle deviation

        self.get_logger().info(f" self.trajectory[{self.traj_idx}]: { self.trajectory[self.traj_idx]}")

        self.viz_tools.plot_line([point['x'] for point in self.trajectory[self.traj_idx]["points"]], [point['y'] for point in self.trajectory[self.traj_idx]["points"]], self.trajectory_pub, color=self.traj_colors[self.traj_idx], increment_id=True)
        self.id += 1

        # Publish PoseArray
        traj_pose_array = PoseArray()
        for point in self.trajectory[self.traj_idx]["points"]:
            pose = Pose()
            pose.position.x = point['x']
            pose.position.y = point['y']
            pose.position.z = 0.0  # Assuming z is 0 as it's not provided in the path
            traj_pose_array.poses.append(pose)

        # set frame so visualization works
        traj_pose_array.header.frame_id = "/map"  # replace with your frame id

        self.traj_pub.publish(traj_pose_array)

        self.traj_idx += 1

        # Modify the next trajectory to start at the end of the current trajectory
        self.trajectory.append({"points": self.nominal_trajectory["points"][last_traj_idx_to_follow + closest_circle_pt_to_shell_idx+1:]})

        self.get_logger().info(f"self.trajectory[{self.traj_idx}]: { self.trajectory[self.traj_idx]}")

        self.viz_tools.plot_line([point['x'] for point in self.trajectory[self.traj_idx]["points"]], [point['y'] for point in self.trajectory[self.traj_idx]["points"]], self.trajectory_pub, color=self.traj_colors[self.traj_idx])
        self.id += 1


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
