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

class RRT:
    def __init__(self, map_info, start, end):
        self.max_distance = 2 # in meters

        self.map_data = map_info[0] # AKA occupancy grid
        self.map_width = map_info[1] #1730
        self.map_height = map_info[2] #1300
        self.map_resolution = map_info[3]

        self.start = np.array([start[0], start[1]]) #[x,y]
        self.end = np.array([end[0],end[1]]) #[x,y]

        # self.hall_pixels = np.where(self.map_data == 0)[0] # the places that are beige
        self.hall_pixels = []
        for i in range(len(self.map_data)):
            if self.map_data[i] == 0:
                self.hall_pixels.append(i)

        # self.wall_pixels = np.where(self.map_data == 100)[0]
        self.nodes = []


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
        end_node_sample_frequency = 0.2

        valid_pixel = False
        pixel = None
        coords = None
        path_len_from_start = None
        closest_node = None # aka the parent node
        ix = None

        used_end_node = False
        while not valid_pixel:
            used_end_node = False
            if np.random.rand() < end_node_sample_frequency: # use end node
                used_end_node = True
                pixel = self.end_pixel
            else:
                # ix = random.randint(0,len(self.map_data)-1)
                # while self.map_data[ix] != 0:
                #     ix = random.randint(0,len(self.map_data)-1)
                ix = np.random.choice(self.hall_pixels)
                pixel = self.pixel_from_index(ix)

            coords = self.pixel_to_real(pixel)
            closest_node, dist_to_closest_node = self.find_closest_node(coords)

            if self.no_collisions_pixel(pixel, closest_node.pixel):
                if used_end_node: # end node has been found
                    if dist_to_closest_node < self.max_distance: 
                        path_len_from_start = closest_node.path_len_from_start + dist_to_closest_node
                        end_node = Joint(self.end, self.end_pixel, path_len_from_start)
                        end_node.parent = closest_node
                        self.nodes.append(end_node)
                        return True, (self.end[0], self.end[1])

                if dist_to_closest_node > self.max_distance:
                    vec = coords-closest_node.coords
                    coords = closest_node.coords + self.max_distance/dist_to_closest_node * vec
                    dist_to_closest_node = self.max_distance
                    pixel = self.real_to_pixel(coords) 
    
                # valid_pixel = True

                path_len_from_start = closest_node.path_len_from_start + dist_to_closest_node
                new_node = Joint(coords, pixel, path_len_from_start)  # def __init__(self, coords, pixel, path_len_from_start, dist_to_end)
                new_node.parent = closest_node
                self.nodes.append(new_node)
                return False, (coords[0], coords[1])
    
    def find_closest_node(self, new_node_coords): # in real coordinates
        closest_node = self.nodes[0]
        min_dist = 1000000
        for node in self.nodes:
            dist = np.linalg.norm(node.coords-new_node_coords)
            if dist < min_dist:
                min_dist = dist
                closest_node = node

        return closest_node, min_dist

    def no_collisions_pixel(self, pixel1, pixel2): #also need to check if theta is a valid steering angle
        if self.map_data[self.index_from_pixel(pixel2)] != 0:
            return False
        
        vec = pixel2-pixel1
        dist = np.linalg.norm(vec)

        num_intervals = int(dist/0.2)
        for interval in range(1,num_intervals):
            pixel = np.round(pixel1 + interval/num_intervals * vec)
            pixel = pixel.astype(int)
            if self.map_data[self.index_from_pixel(pixel)] != 0:
                return False
            
        return True

    def no_collisions_dist(self, p1, p2): #also need to check if theta is a valid steering angle
        vec = p2-p1
        dist = np.linalg.norm(vec)
        num_intervals = int(dist/0.1)

        for interval in range(1,num_intervals):
            coord = p1 + interval/num_intervals * vec
            pixel = self.real_to_pixel(coord)
            if self.map_data[self.index_from_pixel(pixel)] != 0:
                return False
        return True



    # def make_tree(self):
    #     # found_end = False
    #     # start_node = Joint(self.start, self.start_pixel, 0) # coords, pixel, path_len_from_start, dist_to_end
    #     # self.nodes.append(start_node)
    #     # while not found_end:
    #     #     found_end, coords = self.sample_node()

    #     # end_node = self.nodes[-1]
    #     # real_coords = [self.end]

    #     # parent = end_node.parent
    #     # while parent:
    #     #     real_coords.append((parent.coords[0], parent.coords[1]))
    #     #     parent = parent.parent
        

    #     # return real_coords[::-1]
    #     return True
        


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
