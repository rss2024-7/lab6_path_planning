import numpy as np

# def euclid_dist(start_pt, end_pt):
#     ## Args: start_pt, end_pt as 2-elem lists [x-coord, y-coord]
#     diff = np.array(end_pt) - np.array(start_pt)
#     return np.sqrt(diff[0]**2 + diff[1]**2)

class Node:
    def __init__(self, state):
        self.state = state
        self.parent = None

class RRT:
    def __init__(self, start, goal, step_size, max_iter, threshold, max_steer):
        self.start = Node(start)
        self.goal = goal
        self.step_size = step_size
        self.max_steer = max_steer
        self.max_iter = max_iter
        self.threshold = threshold
        self.tree = [self.start]
        
    # def goal_reached(new_node):
    #     # Implement Broad + Narrow phase check
    #     raise NotImplementedError
    
    def rrt_search(self):
        for _ in range(self.max_iter):
            rand_point = self.random_point()
            nearest_node = self.find_nearest_node(rand_point)
            # Collision check in steer function
            new_node = self.steer(nearest_node, rand_point)
            if new_node:
                self.tree.append(new_node)
                # Check if goal is reached
                if np.linalg.norm(new_node.state - self.goal) < self.threshold:
                # if self.goal_reached(new_node.state):
                    return self.construct_path(new_node)
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
    
    def angle_diff(self, from_node, to_point): 
        # Find angle between current direction vector and prev direction vector
        
        # prev_mag = np.linalg.norm(from_node.state - from_node.parent.state)
        prev_direction = (from_node.state - from_node.parent.state)/np.linalg.norm(from_node.state - from_node.parent.state)

        # curr_mag =  np.linalg.norm(to_point - from_node.state)
        curr_direction = (to_point - from_node.state)/np.linalg.norm(to_point - from_node.state)

        # Use dot product to find angle_diff, mag of both vectors should be 1
        theta = np.arccos(np.dot(curr_direction, prev_direction))
    
        return abs(theta)
    
    def steer(self, from_node, to_point):
        # Get direction
        direction = (to_point - from_node.state)/np.linalg.norm(to_point - from_node.state)
        # Compute new state/point
        new_state = from_node.state + self.step_size * direction
        # constrain to max_steering angle 
        if self.angle_diff(from_node, new_state) > self.max_steer:
                # Need to correct new_direction calculation, use rotation matrix
                new_direction = np.array([np.cos(self.max_steer), np.sin(self.max_steer)])
                new_state = from_node.state + self.step_size * new_direction
        if self.collision_free(from_node.state, new_state):
            new_node = Node(new_state)
            new_node.parent = from_node
            return new_node
        else:
            return None

    def construct_path(self, end_node):
        path = []
        current_node = end_node
        while current_node is not None:
            path.append(current_node.state)
            current_node = current_node.parent
        return path[::-1]

    def collision_free(self, point1, point2):
        # Check if the line segment between point1 and point2 is free of obstacles
        # Assume modified map info accounts for car dimensions, so may 
        # not need Broad+Narrow phase check
        

        # Need to finish implementing this

        # create vector array of cells touched by line
        diff = point2 - point1

        # check cells' occupancy probability

        return True  

    def random_point(self):
        # Generate a random point within the search space
        # Bias sampling towards goal: every _ samples, use goal as sample point
        if self.counter % 200 == 0:
            return self.goal
        return np.random.uniform(0, 10, size=(2,)) # change 0, 10 to map width, height constraints

    def normalize(self, vector):
        magnitude = np.linalg.norm(vector)
        if magnitude == 0:
            return vector
        else:
            return vector / magnitude

# Example:
start = np.array([0, 0])
goal = np.array([10, 10])
step_size = 0.5
max_iter = 1000
threshold = 0.1

rrt = RRT(start, goal, step_size, max_iter, threshold)
final_path = rrt.rrt_search()
print(final_path)
