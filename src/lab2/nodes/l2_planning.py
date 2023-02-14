#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag


def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        rand = np.random.rand(3, 1)
        rand[0] = rand[0] * (self.bounds[0, 1] - self.bounds[0, 0])  + self.bounds[0, 0]
        rand[1] = rand[1] * (self.bounds[1, 1] - self.bounds[1, 0])  + self.bounds[1, 0] 
        rand[2] = 0 

        return rand
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        for i in self.nodes:
            if i.point == point:
                return True
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        closest = 1000000
        ind = False
        for i in range(len(self.nodes)):
            dist = np.sqrt((self.nodes[i].point[0]- point[0])**2 + (self.nodes[i].point[1]- point[1])**2)
            if dist < closest:
                closest = dist
                ind = i

        return ind
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        #print("TO DO: Implment a method to simulate a trajectory given a sampled point")
        final_pos = self.robot_controller(node_i, point_s)

        return final_pos
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced

        vel_cmds = [0, 0.025, 0.1, 0.26]
        rot_cmds = np.linspace(-1.82, 1.82, 11)

        self.all_opts = np.array(np.meshgrid(vel_cmds, rot_cmds)).T.reshape(-1, 2)
        self.all_opts_scaled = self.all_opts * self.num_substeps

        self.num_opts = self.all_opts_scaled.shape[0]

        local_paths = np.zeros([self.num_substeps + 1, self.num_opts, 3])
        for i in range(0, self.num_opts):
            local_paths[0, i, :] = node_i.reshape(3)

        #Create array to track if collision is present:
        collisions = np.zeros(self.num_opts)

        for t in range(1, self.num_substeps + 1):
            path_idx = 0
            for t_vel_idx in range(0, len(vel_cmds)):
                t_vel = vel_cmds[t_vel_idx]
                for r_vel_idx in range(0, len(rot_cmds)):
                    r_vel = rot_cmds[r_vel_idx]
                    # Stop propagating paths that have already collided:
                    if collisions[path_idx] == 1:
                        path_idx += 1
                        continue
                    
                    theta = local_paths[t-1,path_idx,2]
                    rot_mat = np.array([[np.cos(theta), 0],
                                        [np.sin(theta), 0],
                                        [0, 1]])
                    vel_vec = np.array([t_vel, r_vel])
                    q_dot = np.matmul(rot_mat, vel_vec) # shape (3, 1)
                    

                    local_paths[t,path_idx,:] = local_paths[t-1,path_idx,:] + q_dot.T * self.timestep
                    # TODO - implement collision checking here:
                    #if collision:
                        #collisions[path_idx] = 1
                    path_idx += 1


            valid_opts = np.where(collisions == 0)[0]
            final_cost = np.zeros_like(valid_opts, dtype=float)
            for n in range(0, len(valid_opts)):
                final_pos = local_paths[self.num_substeps, valid_opts[n], :]
                # check euclidian distance to goal:
                final_cost[n] = np.sqrt(np.sum(np.square(final_pos[:2] - point_s)))

            if final_cost.size == 0:  # hardcoded recovery if all options have collision
                final_pos = []

        return final_pos
    
    def trajectory_rollout(self, vel, rot_vel, node_i):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # NOT USED

        local_paths = np.zeros((3, self.num_substeps))
        local_paths[:,0] = node_i
        for cmd_idx in range(0, self.num_substeps):
            theta = local_paths[2,cmd_idx]
            rot_mat = np.array([[np.cos(theta), 0],
                                [np.sin(theta), 0],
                                [0, 1]])
            vel_vec = np.array([vel, rot_vel])
            q_dot = np.matmul(rot_mat, vel_vec) # shape (3, 1)

        local_paths[:,cmd_idx+1] = local_paths[:,cmd_idx] + q_dot.T * self.timestep
          
        return local_paths
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        #print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        origins_x = np.ones_like(point[0])*self.map_settings_dict["origin"][0]
        origins_y = np.ones_like(point[1])*self.map_settings_dict["origin"][1]
        resolutions = np.ones_like(point[0])*self.map_settings_dict["resolution"]
        origins = np.hstack((origins_x,origins_y))
        resolutions = np.hstack((resolutions,resolutions))
        return (point + origins) / resolutions

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        pixel_points = self.point_to_cell(points)
        xx, yy = disk(pixel_points, 2, shape=self.map_shape)

        return xx, yy
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        #for i in range(1): #Most likely need more iterations than this to complete the map!
        path_found = False
        min_attempts = 50 # number of attempts before trying to connect final node:
        attempts = 0
        while path_found == False:
            
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            closest_output = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            if closest_output == []: #no path found
              attempts += 1 
              continue

            #Add closest point to path:
            new_node = Node(closest_output, closest_node_id, 0) #todo - need cost?
            self.window.add_point(self.point_to_cell(new_node.point[:2]))
            self.window.add_line(self.point_to_cell(new_node.point[:2]), self.point_to_cell(self.nodes[closest_node_id].point[:2]))
            self.nodes += [new_node]
            self.nodes[closest_node_id].children_ids += [len(self.nodes)]

            if attempts > min_attempts:
                # start trying to connect the goal:
                point = self.goal_point 
                closest_node_id = self.closest_node(point)
                closest_output = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
                if np.sqrt(np.sum(np.square(closest_output[:2] - point[:2]))) < self.stopping_dist:
                    # if new point is within target radius:
                    path_found = True
                    
                new_node = Node(closest_output, closest_node_id, 0) #todo - need cost?
                self.window.add_point(point_to_cell(new_node.point[:2]))
                self.window.add_line(self.point_to_cell(new_node.point[:2]), self.point_to_cell(self.nodes[closest_node_id].point[:2]))
                self.nodes += [new_node]
                self.nodes[closest_node_id].children_ids += [len(self.nodes)]

            attempts += 1 

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
