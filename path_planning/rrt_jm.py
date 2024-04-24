import numpy as np
import math

# def euclid_dist(start_pt, end_pt):
#     ## Args: start_pt, end_pt as 2-elem lists [x-coord, y-coord]
#     diff = np.array(end_pt) - np.array(start_pt)
#     return np.sqrt(diff[0]**2 + diff[1]**2)

class Nd:
    def __init__(self, state):
        self.state = state
        self.parent = None
        self.cost = 0

class RRT:
    def __init__(self, start, goal, step_size, max_iter, threshold, min_steer, map_info):
        self.start = Nd(start)
        self.goal = goal
        self.step_size = step_size
        self.min_steer = min_steer
        self.max_iter = max_iter
        self.threshold = threshold
        self.counter = 0
        self.tree = [self.start]
        self.map_resolution = map_info[0]
        self.map_dims = map_info[1]
        self.obstacles = map_info[2]
        theta1 = np.pi
        x = 25.900000
        y = 48.50000
        self.x_bounds = (-61.3, 25.9)
        self.y_bounds = (-16.2, 48.5)
        self.transform = np.array([[np.cos(theta1), -np.sin(theta1), x],
                    [np.sin(theta1), np.cos(theta1), y],
                    [0,0,1]])
        
    # def goal_reached(new_node):
    #     # Implement Broad + Narrow phase check
    #     raise NotImplementedError
    
    def rrt_star(self):
        for _ in range(self.max_iter):
            rand_point = self.random_point()
            # print("sample:", rand_point)
            self.counter += 1
            nearest_node = self.find_nearest_node(rand_point)
            # print("nearest_node:", nearest_node.state)
            # Collision check in steer function
            new_state = self.steer(nearest_node, rand_point)
            from_node_pix = self.real_to_pixel(nearest_node.state)
            # print("from_node pix:", from_node_pix)
            new_state_pix = self.real_to_pixel(new_state)
            # print("new_state_pix:", new_state_pix)
            if self.collision_free(from_node_pix, new_state_pix):
                nearest_nodes = self.find_nearest_nodes(new_state, radius=1)
                # min_dist = np.inf
                parent = nearest_node #default parent
                parent_dist = self.step_size #default dist
                min_cost = np.inf
                for node in nearest_nodes:
                    node_cost = self.construct_path(node)[1]
                    dist = np.linalg.norm(new_state - node.state)
                    cost = node_cost + dist
                    if cost < min_cost:
                        min_cost = cost
                        parent = node
                        parent_dist = dist
                new_node = Nd(new_state)
                new_node.parent = parent
                new_node.cost = parent_dist
                self.tree.append(new_node)
                # Check if goal is reached
                if np.linalg.norm(new_node.state - self.goal) < self.threshold:
                # if self.goal_reached(new_node.state):
                    return self.construct_path(new_node)
        # print("partial_path:", self.construct_path(self.tree[-1]))     
        return None
    
    def rrt_search(self):
        for _ in range(self.max_iter):
            rand_point = self.random_point()
            # print("sample:", rand_point)
            self.counter += 1
            nearest_node = self.find_nearest_node(rand_point)
            # print("nearest_node:", nearest_node.state)
            # Collision check in steer function
            # new_node = self.steer(nearest_node, rand_point)
            new_state = self.steer(nearest_node, rand_point)
            from_node_pix = self.real_to_pixel(nearest_node.state)
            # print("from_node pix:", from_node_pix)
            new_state_pix = self.real_to_pixel(new_state)
            # print("new_state_pix:", new_state_pix)
            if self.collision_free(from_node_pix, new_state_pix):
                new_node = Nd(new_state)
                new_node.parent = nearest_node
                self.tree.append(new_node)
                # Check if goal is reached
                if np.linalg.norm(new_node.state - self.goal) < self.threshold:
                # if self.goal_reached(new_node.state):
                    return self.construct_path(new_node)[0]
        # print("partial_path:", self.construct_path(self.tree[-1]))     
        return None

    def find_nearest_node(self, point):
        nearest_node = None
        min_distance = float('inf')
        # use np array + broadcasting here instead
        for node in self.tree:
            distance_to_point = np.linalg.norm(node.state - point)
            if distance_to_point < min_distance:
                nearest_node = node
                min_distance = distance_to_point
        return nearest_node
    
    def find_nearest_nodes(self, point, radius):
        nearest_nodes = []
        # use np array + broadcasting here instead
        for node in self.tree:
            distance_to_point = np.linalg.norm(node.state - point)
            if distance_to_point <= radius:
                nearest_nodes.append(node)
        return nearest_nodes
    
    def steer(self, from_node, to_point):
        # Get direction (unit vector)
        direction = (to_point - from_node.state)/np.linalg.norm(to_point - from_node.state)
        # print("direction:", direction)
        # Compute new state/point
        new_state = from_node.state + self.step_size * direction
        return new_state
        # print("new_state1:", new_state)
        #--------------------------------------------
        # if from_node.parent != None :
        #     # prev direction (unit vector)
        #     prev_direction = (-from_node.state + from_node.parent.state)/np.linalg.norm(-from_node.state + from_node.parent.state)
        #     # calculate angle difference to drive to to_point, theta in [0, pi]
        #     dir_diff = direction - prev_direction
            
        #     # constrain to min_steering angle 
        #     if abs(theta) < self.min_steer: #resample
        #             # side = 1
        #             # if theta < 0:
        #             #     side = -1
        #             # Need to correct new_direction calculation, use rotation matrix
        #             rotate = np.array([[np.cos(self.min_steer), -np.sin(self.min_steer)], 
        #                              [np.sin(self.min_steer), np.cos(self.min_steer)]] )
        #             rotate2 = np.array([[np.cos(-self.min_steer), -np.sin(-self.min_steer)], 
        #                              [np.sin(-self.min_steer), np.cos(-self.min_steer)]] )
        #             new_direction1 = rotate@np.transpose(prev_direction)
        #             new_direction2 = rotate2@np.transpose(prev_direction)
        #             dirs = np.array([new_direction1, new_direction2])
        #             new_states = from_node.state + self.step_size * dirs
        #             # print("new_states:", new_states)
        #             new_state = new_states[np.where(min(np.linalg.norm(new_states-to_point, axis=1)))][0]
                    # print("new_state2:", new_state)
                    # return None
        

    def construct_path(self, end_node):
        path = []
        cost = 0
        current_node = end_node
        while current_node is not None:
            path.append(current_node.state)
            cost += current_node.cost
            current_node = current_node.parent
        # cost = len(path) * self.step_size
        return path[::-1], cost

    def collision_free(self, point1, point2):
        # Check if the line segment between point1 and point2 is free of obstacles
        # Assume modified map info accounts for car dimensions, so may 
        # not need Broad+Narrow phase check

        # create vector array of cells touched by line
        diff = point2 - point1
        # print(diff)
        slope = intercept = 0
        if diff[0] != 0:
            slope = diff[1]/diff[0]
            intercept = point1[1] - slope*point1[0]
            x_arr = np.linspace(point1[0], point2[0], num=abs(int(diff[0]/self.map_resolution))+1)
            y_arr = slope*x_arr + intercept
        else: 
            if diff[1] == 0: # pt1 directly on top of pt2
                return True
            # slope = np.inf
            x_arr = np.full((1, abs(int(diff[1]/self.map_resolution))+1), point1[0] ) 
            y_arr = np.linspace(point1[1], point2[1], num=abs(int(diff[1]/self.map_resolution))+1)

        # check cells' occupancy probability
        inds = np.vstack((x_arr, y_arr))
        inds = np.transpose(inds)
        line_inds = set(map(tuple, inds)) # array of tuples (x,y)
        # print("pixel coords in line", line_inds)


        # set intersect indices and obstacle indices
        collisions = line_inds.intersection(self.obstacles)

        if collisions:
            return False
        return True

    def random_point(self):
        # Generate a random point within the search space
        # Bias sampling towards goal: every _ samples, use goal as sample point
        if self.counter != 0 and self.counter % 2 == 0:
            return self.goal
        
        return np.array([np.random.uniform(self.x_bounds[0], self.x_bounds[1]), \
                np.random.uniform(self.y_bounds[0], self.y_bounds[1])])  

    # def normalize(self, vector):
    #     magnitude = np.linalg.norm(vector)
    #     if magnitude == 0:
    #         return vector
    #     else:
    #         return vector / magnitude

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

# if __name__ == "__main__":
#     # start, goal, step_size, max_iter, threshold, min_steer, obs, map_dims)
#     start = np.array([0, 0])
#     goal = np.array([3, 3])
#     step_size = 1
#     max_iter = 5
#     threshold = 0.5
#     min_steer = np.pi/2
#     # obs = {(3, 6), (14,5), (14,6), (14,7), (2,6)}
#     obs = {(95,51), (92,50), (91, 46), (91,50)}
#     map_dims = [0.5, (5,5)]

#     rrt = RRT(start, goal, step_size, max_iter, threshold, min_steer, obs, map_dims)
#     new = Nd(np.array([1,1]))
#     new.parent = Nd(np.array([0,1]))
    # print(rrt.steer(new, np.array([0,0])))
    # final_path = rrt.rrt_search()
    # print("final_path:", final_path)
