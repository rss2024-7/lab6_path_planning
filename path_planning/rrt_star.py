import numpy as np
import random
import math

class Joint:
    def __init__(self, coords, pixel, path_len_from_start):
        self.coords = coords # (x,y) in real coordinates NOT pixel
        self.pixel = pixel
        self.path_len_from_start = path_len_from_start # path length from starting point
        # self.dist_to_end = dist_to_end # NOT path distance
        self.parent = None

class RRT_star:
    def __init__(self, map_info, start, end):
        self.max_distance = 1 # in meters

        self.map_data = map_info[0] # AKA occupancy grid
        self.map_width = map_info[1] #1730
        self.map_height = map_info[2] #1300
        self.map_resolution = map_info[3]

        self.start = np.array([start[0], start[1]]) #[x,y]
        self.end = np.array([end[0],end[1]]) #[x,y]

        self.hall_pixels = np.where(self.map_data == 0) # the places that are beige
        self.wall_pixels = np.where(self.map_data == 100)
        self.nodes = []

        # tabulate the distance from each other
        # self.pixel_distance_table = np.array()


        # TO-DO: make less jank
        x = 25.900000
        y = 48.50000
        theta = 3.14
        resolution = 0.0504
        self.transform = np.array([[np.cos(theta), -np.sin(theta), x],
                    [np.sin(theta), np.cos(theta), y],
                    [0,0,1]])

        self.x_bounds = (-61.3, 25.9)
        self.y_bounds = (-16.2, 48.5)

        
        self.start_pixel = self.real_to_pixel(self.start)
        self.end_pixel = self.real_to_pixel(self.end)
        self.end_index = self.index_from_pixel(self.end_pixel)




    def sample_node(self):
        # returns true if start node has been connected with end node, false otherwise
        end_node_sample_frequency = 1

        valid_pixel = False
        pixel = None
        coords = None
        path_len_from_start = None
        closest_node = None # aka the parent node

        used_end_node = False
        while not valid_pixel:
            if np.random.rand() < end_node_sample_frequency:
                used_end_node = True
                pixel = self.end_pixel
            else:
                pixel = self.pixel_from_index(np.random.choice(self.hall_pixels)) ## warning???

            if self.no_collisions(pixel):
                coords = self.pixel_to_real(pixel)
                closest_node, dist_to_closest_node = self.find_closest_node(coords)
                if dist_to_closest_node <= self.max_distance and used_end_node:
                    path_len_from_start = closest_node.path_len_from_start + dist_to_closest_node
                    end_node = Joint(self.end, self.end_pixel, path_len_from_start)
                    end_node.parent = closest_node
                    self.nodes.append(end_node)
                    return True

                if dist_to_closest_node > self.max_distance:
                    x2, y2 = coords[0], coords[1]
                    x1, y1 = closest_node.coords[0], closest_node.coords[1]
                    # theta = np.arctan2((x1,y1), (x2,y2))
                    a=np.array([coords[0], coords[1]])
                    b=closest_node.coords

                    a_n = np.linalg.norm(a)
                    b_n = np.linalg.norm(b)

                    theta = np.arccos(np.dot(a, b)/(a_n*b_n))

                    coords = np.array([x1 + self.max_distance * math.cos(theta), y1 + self.max_distance * math.sin(theta)])
                    dist_to_closest_node = self.max_distance
                    pixel = self.real_to_pixel(coords)

                path_len_from_start = closest_node.path_len_from_start + dist_to_closest_node

                valid_pixel = True

        new_node = Joint(coords, pixel, path_len_from_start)  # def __init__(self, coords, pixel, path_len_from_start, dist_to_end)
        new_node.parent = closest_node

        self.nodes.append(new_node)
        return False
    
    def find_closest_node(self, new_node_coords):
        closest_node = self.nodes[0]
        min_dist = 100000
        for node in self.nodes:
            dist = np.linalg.norm(node.coords-new_node_coords)
            if dist < min_dist:
                min_dist = dist
                closest_node = node

        return closest_node, min_dist



    def no_collisions(self, pixel): #also need to check if theta is a valid steering angle
        # TO-DO implement this
        return True



    def make_tree(self):
        found_end = False
        start_node = Joint(self.start, self.start_pixel, 0) # coords, pixel, path_len_from_start, dist_to_end
        self.nodes.append(start_node)
        while not found_end:
            found_end = self.sample_node()

        end_node = self.nodes[-1]
        real_coords = [self.end]

        parent = end_node.parent
        while parent:
            real_coords.append((parent.coords[0], parent.coords[1]))
            parent = parent.parent
        

        return real_coords[::-1]
        


    # utils 
    def pixel_to_real(self, pix, column_row=False):
        # if column_row = True, pixel format is [v (column index), u (row index)] 
        if column_row:
            pixel = np.array([pix[0], pix[1]]) 
        else:
            pixel = np.array([pix[1], pix[0]]) 

        pixel = pixel * self.map_resolution
        pixel = np.array([*pixel,1.0])
        pixel = np.linalg.inv(self.transform) @ pixel
        point = pixel

        # returns real x,y 
        return np.array([point[0], point[1]])

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
    
    def pixel_from_index(self, index, column_row=False):
        # calculate the index of the row-major map
        u = int(index/self.map_width)
        v = index - u * self.map_width
        if column_row: # pixel = [v, u]
            return np.array([v,u])
        else: # pixel = [u, v]
            return np.array([u,v])
