import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PointStamped, PoseWithCovarianceStamped
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

        # Adjust lookahead based on angle to lookahead point
        # higher angle error ~ lower lookahead distance
        self.min_lookahead = 1.0 
        self.max_lookahead = 2.0 

        self.speed_to_lookahead = 2.0
        
        # the angle to target s.t. the lookahead will be at its minimum
        self.min_lookahead_angle = np.deg2rad(90) 

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
        
        self.point_sub = self.create_subscription(PointStamped,
                                                 "/clicked_point",
                                                 self.point_callback,
                                                 1)
        self.init_point_sub = self.create_subscription(PoseWithCovarianceStamped,
                                            "/initialpose",
                                            self.init_callback,
                                            1)
        
        self.target_pub = self.create_publisher(Marker, "/target_point", 1)
        self.radius_pub = self.create_publisher(Marker, "/radius", 1)

        self.max_steer = 0.34

        self.initialized_traj = False

    def point_callback(self, point_msg):
        self.get_logger().info("adding point")
        self.trajectory.addPoint((point_msg.point.x, point_msg.point.y))
        self.trajectory.publish_viz(duration=0.0)
        self.trajectory.save("trajectory_custom.traj")

    def pose_callback(self, odometry_msg):

        # no trajectory to follow
        if not self.initialized_traj: return

        # retrieve odometry data
        car_pos_x = odometry_msg.pose.pose.position.x
        car_pos_y = odometry_msg.pose.pose.position.y
        car_angle = 2 * np.arctan2(odometry_msg.pose.pose.orientation.z, odometry_msg.pose.pose.orientation.w)

        # process trajectory points into np arrays
        points = np.array(self.trajectory.points)
        traj_x = points[:, 0]
        traj_y = points[:, 1]

        # get info about closest segment
        closest_segment_index = self.find_closest_segment(traj_x, traj_y, car_pos_x, car_pos_y)
        seg_end_x, seg_end_y = traj_x[closest_segment_index + 1], traj_y[closest_segment_index + 1]
        car_to_seg_end_x, car_to_seg_end_y = self.to_car_frame(seg_end_x, seg_end_y, car_pos_x, car_pos_y, car_angle)

        # on last segment and past end
        if (closest_segment_index + 1 == len(traj_x) - 1 and \
            car_to_seg_end_x < 0): 
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)
            return


        lookahead_point = self.get_lookahead_point(traj_x[closest_segment_index], traj_y[closest_segment_index],
                                                    traj_x[closest_segment_index + 1], traj_y[closest_segment_index + 1],
                                                    car_pos_x, car_pos_y)

        # default target point is end of closest segment (in case no lookahead point found)
        target_point_x, target_point_y = seg_end_x, seg_end_y

        # if found lookahead point, set target to it
        if lookahead_point is not None:
            target_point_x, target_point_y = lookahead_point

        # convert target point to the car's frame
        car_to_target_x, car_to_target_y = self.to_car_frame(target_point_x, target_point_y, car_pos_x, car_pos_y, car_angle)


        # Visualize Stuff
        VisualizationTools.plot_line(np.array([0, car_to_target_x]), np.array([0, car_to_target_y]), self.target_pub, frame="base_link")
        angles = np.linspace(-np.pi, np.pi, 100)
        circle_x = self.lookahead * np.cos(angles)
        circle_y = self.lookahead * np.sin(angles)
        circle_x = 0.9 * np.cos(angles)
        circle_y = 0.9 * np.sin(angles)
        VisualizationTools.plot_line(circle_x, circle_y, self.radius_pub, frame="base_link")


        # angle to target point
        angle_error = np.arctan2(car_to_target_y, car_to_target_x)

        self.lookahead = self.max_lookahead \
                            - np.clip(np.abs(angle_error), 0, 
                                        self.min_lookahead_angle) / self.min_lookahead_angle \
                                        * (self.max_lookahead - self.min_lookahead)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.lookahead * self.speed_to_lookahead

        steer_angle = np.arctan2((self.wheelbase_length*np.sin(angle_error)), 
                                 0.5*self.lookahead + self.wheelbase_length*np.cos(angle_error))
        
        steer_angle = np.clip(steer_angle, -self.max_steer, self.max_steer)
        drive_msg.drive.steering_angle = steer_angle
        self.drive_pub.publish(drive_msg)

    def find_closest_segment(self, x, y, car_x, car_y):
        """Finds closest line segment in trajectory and returns its index.
            Code based on https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/1501725#1501725
            and modified to use numpy arrays for better speed

        Args:
            x (1D np array): x values of trajectory
            y (1D np array): y values of trajectorffy
            car_x (float): x position of car
            car_y (float): y position of car
        Returns:
            int: index of start of closest line segment in the trajectory arrays
        """
        points = np.vstack((x, y)).T
        v = points[:-1, :] # segment start points
        w = points[1:, :] # segment end points
        p = np.array([[car_x, car_y]])
        
        l2 = np.sum((w - v)**2, axis=1)

        t = np.maximum(0, np.minimum(1, np.sum((p - v) * (w - v), axis=1) / l2))

        projections = v + t[:, np.newaxis] * (w - v)
        min_distances = np.linalg.norm(p - projections, axis=1)

        # if too close to end point of segment, take it out of consideration for closest line segment
        end_point_distances = np.linalg.norm(w-p, axis=1)
        min_distances[np.where(end_point_distances[:-1] < self.lookahead)] += np.inf

        closest_segment_index = np.where(min_distances == np.min(min_distances))[0][0]

        return closest_segment_index
    
        # # Non-Numpy Vectorized Version (keeping in case it's faster)
        # def min_dist(x1, y1, x2, y2, px, py):
        #     v = np.array([x1, y1])
        #     w = np.array([x2, y2])
        #     p = np.array([px, py])
        #     l2 = np.dot(w - v, w - v)
        #     if (l2 == 0.0): return np.linalg.norm(p - v)

        #     t = max(0, min(1, np.dot(p - v, w  - v) / l2))
        #     projection = v + t * (w - v)
        #     return np.linalg.norm(p - projection)
        
        # distances = np.zeros(len(x) - 1)
        # for i in range(len(x) - 1):
        #     distances[i] = min_dist(x[i], y[i], x[i + 1], y[i + 1], car_x, car_y)
        #     if np.linalg.norm(np.array([x[i + 1] - car_x, y[i + 1] - car_y])) < self.lookahead:
        #         distances[i] += 100
        # return np.where(distances == np.min(distances))[0][0]


    def get_lookahead_point(self, x1, y1, x2, y2, origin_x, origin_y):
        """Based on https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/1501725#1501725.
            Finds lookahead point (intersection between circle and line segment).

        Args:
            x1 (float): line segment start x
            y1 (float): line segment start y
            x2 (float): line segment end x
            y2 (float): line segment end y
            origin_x (float): center of circle x
            origin_y (float): center of circle y

        Returns:
            1D np array of size 2: point of intersection (not necessarily on the line segment). 
            None if no intersection even if line segment is extended
        """
        Q = np.array([origin_x, origin_y])                  # Centre of circle
        r = self.lookahead  # Radius of circle

        P1 = np.array([x1, y1])  # Start of line segment
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

        t = max(t1, t2)

        if (t < 0): return None

        return P1 + t * V
    
    # convert a point from the map frame to the car frame
    def to_car_frame(self, x, y, car_x, car_y, car_angle):
        def rotation_matrix(angle):
            return np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
    
        # convert [x y theta] to 3x3 transform matrix
        def transform_matrix(pose):
            X = np.eye(3)
            X[:2, :2] = rotation_matrix(pose[2])
            X[:2, -1] = np.array(pose[:2])
            return X
        world_to_car = transform_matrix([car_x, car_y, car_angle])
        world_to_target = transform_matrix([x, y, 0.0])
        car_to_target = np.linalg.inv(world_to_car) @ world_to_target

        return car_to_target[:2, 2]


    def trajectory_callback(self, msg):
        traj_len = len(msg.poses)

        self.get_logger().info(f"Receiving new trajectory {traj_len} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.trajectory.update_distances()
        self.get_logger().info(f"Trajectory length: {self.trajectory.distance_along_trajectory(traj_len-1)}")

        # self.trajectory.save("current_trajectory.traj")

        self.initialized_traj = True

    def init_callback(self, msg):
        return


def main(args=None):

    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()