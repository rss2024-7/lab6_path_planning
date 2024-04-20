import rclpy

import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseArray, Point
from std_msgs.msg import Header
import os
from typing import List, Tuple
import json

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory

import random

# to-do: download occupany grid from ROS
 
EPSILON = 0.00000000001


def check_collision(p1, p2, grid):
    # Check if there is a collision between p1 and p2
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1) 
    for x in range(x1, x2 + 1): 
        y = round(m*x + y1)
        if grid[x][y] == 0:
            return True
    return False

def make_vertices(n, grid, start_point, end_point):
    vertices = [start_point, end_point]
    n = 1000
    # for _ in range(n):
    #     random_point = (x,y)
    #     random_pixel = point_to_pixel(random_point)
    #     while get_grid_value(random_pixel, grid) == 0:
    #         random_point = (x,y)
    #         random_pixel = point_to_pixel(random_point)
    #     vertices.append(random_point)

    return vertices

def add_edges(vertices):
    # for vertex in vertices, check collision and add edge
    graph = {vertex: [] for vertex in vertices} # graph is a dictionary. keys = vertices, vertex: [vertices that share an edge with vertex]
    # for i in range(len(vertices)):
    #     for j in range(i, len(j)):
    #         if not check_collision(vertex[i], vertex[j]):
    #             vertex[i].append(j)
    #             vertex[j].append(i)
    return graph

def shortest_path(graph):
    # do Dikstra's, expand on the point that is closest to the goal point (the heuristic)
    # start_point = vertex[0]
    # p2 = ()
    # p3 = ()
    # end_point = vertex[1]
    # nodes = [start_point, p2, p3, end_point]
    return None

