import numpy as np

def pixel_to_real(pixel):
    # need to check if this returns [u, v] or [v, u]
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
    # need to check if this returns [x, y] or [y, x]
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


class Point:
    def __init__(self, coords, path_len_from_start, dist_to_end):
        self.coords = coords # (x,y) in real coordinates NOT pixel
        self.path_len = path_len_from_start # path length from starting point
        self.dist_to_end = dist_to_end # NOT path distance
        self.parent = None

# class RRT_star:
