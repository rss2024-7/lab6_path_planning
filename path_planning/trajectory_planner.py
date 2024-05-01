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

import time
import numpy as np
import pydot

from .nominal_path import nominal_path

# nominal_path = {"points": [{"x": -10.65359878540039, "y": 25.74323272705078}, {"x": -8.669516800306914, "y": 25.491402076756714}, {"x": -6.771459653802023, "y": 24.861021978243354}, {"x": -6.801376866974625, "y": 22.861245750672882}, {"x": -6.340533575487584, "y": 21.929129508267145}, {"x": -7.242911516921321, "y": 20.14427263702682}, {"x": -8.497119525884397, "y": 18.586401567184108}, {"x": -9.761628735583386, "y": 18.09327602119255}, {"x": -10.913687700463186, "y": 16.458418185954244}, {"x": -12.037173237741136, "y": 14.80379428683781}, {"x": -13.18392314864232, "y": 13.165208115210536}, {"x": -13.026948713940847, "y": 11.384866982995156}, {"x": -15.000921247168504, "y": 11.063257844345795}, {"x": -16.080865556393285, "y": 9.379892082256935}, {"x": -17.24757483647936, "y": 7.755456963281409}, {"x": -18.724532325765274, "y": 6.406903451238878}, {"x": -18.68659939325515, "y": 4.407263210437753}, {"x": -18.300866764530642, "y": 2.4448131246681095}, {"x": -20.2798935019388, "y": 2.1559304969044746}, {"x": -22.10273069638282, "y": 1.3329696457387261}, {"x": -22.49271821865058, "y": -0.6286393194924829}, {"x": -24.492435313612525, "y": -0.6622776921784046}, {"x": -26.49215240857447, "y": -0.6959160648643263}, {"x": -28.491869503536414, "y": -0.7295544375502481}, {"x": -30.49158659849836, "y": -0.7631928102361698}, {"x": -32.4913036934603, "y": -0.7968311829220915}, {"x": -34.49102078842225, "y": -0.8304695556080133}, {"x": -36.490636214022715, "y": -0.8696888072667361}, {"x": -38.49036523326575, "y": -0.9026106783752893}, {"x": -40.49009425250878, "y": -0.9355325494838425}, {"x": -42.48982327175182, "y": -0.9684544205923956}, {"x": -44.48955229099485, "y": -1.0013762917009488}, {"x": -46.489281310237885, "y": -1.034298162809502}, {"x": -48.48901032948092, "y": -1.0672200339180553}, {"x": -50.48873934872395, "y": -1.1001419050266086}, {"x": -52.11481857299805, "y": -1.166604995727539}]}


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

        self.closest_pt_pub = self.create_publisher(Marker, "viz/closest_pt", 1)
        self.shell_pub = self.create_publisher(Marker, "viz/shell_pub", 1)

        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

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
        self.get_logger().info("Before Publishing point")
        if publisher.get_subscription_count() > 0:
            self.get_logger().info("Publishing point")
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = 0
            marker.type = 2  # sphere
            marker.action = 0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0
            publisher.publish(marker)
        elif publisher.get_subscription_count() == 0:
            self.get_logger().info("Not publishing point, no subscribers")
    

    def plan_path(self, shell_point, map):
        self.get_logger().info(f"PLANNING PATH DEVIATION TO {shell_point}")

        goal_pos = shell_point[:2]

        # Visualize shell markers
        self.publish_point(goal_pos, self.shell_pub, 0.0, 1.0, 0.0)
        
        # Find closes point on path to the designated point
        # First, find closest endpoint to the designated point
        closest_dist_sq = np.inf
        closest_idx = 0
        for i in range(len(nominal_path["points"])):
            new_dist_sq = (nominal_path["points"][i]["x"] - goal_pos[0])**2 + (nominal_path["points"][i]["y"] - goal_pos[1])**2
            if new_dist_sq < closest_dist_sq:
                closest_dist_sq = new_dist_sq
                closest_idx = i
            
        # Now, search the two segments adjacent to the closest endpoint to find
        # the true closest point to the designated point
        closest_dist_sq = np.inf
        closest_pt = None
        for i in range(closest_idx-1, closest_idx+1):
            if i < 0 or i >= len(nominal_path["points"]):
                continue

            start = np.array([nominal_path["points"][i]["x"], nominal_path["points"][i]["y"]])
            end = np.array([nominal_path["points"][i+1]["x"], nominal_path["points"][i+1]["y"]])

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
        
        self.publish_point(closest_pt, self.closest_pt_pub, 1.0, 0.5, 0.0)


            

        ANGLE_INCREMENT = 0.1  # radians
        for angle in np.arange(0, 2*np.pi, ANGLE_INCREMENT):
            pass
            

        

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
