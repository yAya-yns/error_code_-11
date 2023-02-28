#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.draw import disk
from scipy.linalg import block_diag
import scipy
import random 


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
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist, display_window=True):
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
        print(f'self.bounds: \n{self.bounds}')

        #Robot information
        self.robot_radius = 0.5 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m
        self.sim_stopping_dist = stopping_dist #m
        print(f'goal_point: {goal_point}')
        print(f'stopping_dist: {stopping_dist}')

        #Trajectory Simulation Parameters
        self.timestep = 2.0 #s used to be 1.0
        # self.num_substeps = 10
        self.num_substeps = int(1000 * self.timestep)

        #Planning storage
        node = np.zeros((3,1)) #WORLD
        node[2][0] = 1
        # node[0:2] = self.cell_to_point(node[0:2]) 
        self.nodes = [Node(node, -1, 0)]
        self.final_nodes = [Node(self.goal_point, -1, 0)]
        
        self.node_pts = np.zeros((3,1))[:2][None]
        # print(self.node_pts)

        self.trajectories = [] #list of all added trajectories we want to plot

        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot()
        # self.ax.axis('equal')
        # print(self.occupancy_map.shape)
        # self.ax.spy(1 - self.occupancy_map)
        # plt.show()
        # exit()

        # #RRT* Specific Parameters
        # self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        # self.zeta_d = np.pi
        # self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        # self.gamma_RRT = self.gamma_RRT_star + .1
        # self.epsilon = 2.5
        
        #Pygame window for visualization
        # if display_window:
        #     self.window = pygame_utils.PygameWindow(
        #         "Path Planner", (500, 500), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self, goal=False):
        #Return an [x,y] coordinate to drive the robot towards
        # print("TO DO: Sample point to drive towards")
        #maybe sample occ map first?
        #TODO this idk what it wants. not sure if this is enough. Hint is throwing me off
        # IN MAP POINTS

        # find random node in existing nodes:
        if goal:
            cent = self.point_to_cell(self.goal_point).squeeze()
        else:
            node_ind = random.randrange(len(self.nodes))
            # node_ind = len(self.nodes)-1
            cent = self.point_to_cell(self.nodes[node_ind].point[:2]).squeeze()
        map_circles_r, map_circles_c = disk((cent[0], cent[1]), 100) # radius of 5m 100 cells
        
        # # rand = np.random.rand(3,)
        # x = np.random.rand() * (self.bounds[0, 1] - self.bounds[0, 0])  + self.bounds[0, 0]
        # y = np.random.rand() * (self.bounds[1, 1] - self.bounds[1, 0])  + self.bounds[1, 0] 
        
        theta = np.random.rand() * 2 * np.pi  - np.pi

        random.randrange(len(map_circles_r))
        y = map_circles_c[random.randrange(len(map_circles_c))]
        x = map_circles_r[random.randrange(len(map_circles_r))]

        point = self.cell_to_point(np.array([x, y])[None, :].T)
        x = point[0][0]
        y = point[1][0]

        if x < self.bounds[0, 0]:
            x = self.bounds[0, 0]
        if x > self.bounds[0, 1]:
            x = self.bounds[0, 1]

        if y < self.bounds[1, 0]:
            y = self.bounds[1, 0]
        if y > self.bounds[1, 1]:
            y = self.bounds[1, 1]

        rand = np.array([x, y, theta])
        return rand
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        #print("TO DO: Check that nodes are not duplicates")
        
        # self.nodes is list of 
        # def __init__(self, point, parent_id, cost):
            # self.point = point # A 3 by 1 vector [x, y, theta]
            # self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
            # self.cost = cost # The cost to come to this node
            # self.children_ids = [] # The children node ids of this node
        
        for i in self.nodes:
            if (i.point == point).all():
                return True
        return False
    
    def closest_node(self, point : np.ndarray) -> int:
        '''
        Returns the index of the closest node
        '''
        assert point.shape == (2,), point.shape
        point = point.reshape(2, 1)

        print([n.point for n in self.nodes])
        nodes = np.hstack([n.point[0:2] for n in self.nodes])
        diff = nodes - point
        dist = np.sum(np.square(diff), axis=0)
        return np.argmin(dist)
    
    def simulate_trajectory(self, node_i : np.ndarray, point_s : np.ndarray) -> tuple:
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        # recieve map points
        assert node_i.shape == (3,), node_i.shape
        assert point_s.shape == (2,), point_s.shape
        
        # calculate vel and rot_vel to get from node_i to point_s
        vel, rot_vel = self.robot_controller(node_i, point_s)
        print(vel)
        robot_traj = self.trajectory_rollout(vel, rot_vel, node_i, point_s)
        return robot_traj, vel, rot_vel
    
    def robot_controller(self, node_i : np.ndarray, point_s : np.ndarray) -> tuple:
        '''
        This controller determines the velocities that will nominally move the robot from node i to node s
        Max velocities should be enforced
        '''
        assert node_i.shape == (3,)
        assert point_s.shape == (2,), point_s.shape

        # determine if point_s is on the right side of the robot
        robot_xy, theta = node_i[0:2], node_i[2]
        robot_right_direction = np.array([np.sin(theta), -np.cos(theta)])   # vector representing right direction of robot
        on_right = np.sign(np.dot(robot_right_direction, point_s - robot_xy))

        # compute the radius of the circle
        start_to_end_vec = point_s - robot_xy
        radius = 0.5 * np.linalg.norm(robot_xy - point_s) / np.dot(start_to_end_vec / np.linalg.norm(start_to_end_vec), robot_right_direction * on_right)

        # calculate v and w based on circle radius
        ret_v = radius * self.rot_vel_max
        ret_w = -1 * self.rot_vel_max * on_right

        # normalize v and w to ensure it doesn't exceed constraint
        if ret_v > self.vel_max:
            factor = self.vel_max / ret_v
            ret_v = factor * ret_v
            ret_w = factor * ret_w
        assert ret_v > 0, (ret_v, factor, ret_w)

        #  TODO # return a straight line if w=0
        # if ret_w == 0:
        #     temp = np.linspace([1, 1], [self.num_substeps, self.num_substeps], self.num_substeps).T - 1
        #     ret = np.zeros((3, self.num_substeps))
        #     ret[0:2, :] = temp * vel * robot_direction.reshape(2, 1) * self.timestep + robot_xy.reshape(2, 1) # broadcast
        #     ret[2, :] = theta
        #     return ret
        
        return ret_v, ret_w
        

    def is_collide(self, points : np.ndarray) -> bool:
        # check for multiple points if collides with circle 
        # RECIEVE MAP POINTS
        # robot radius: self.robot_radius
        # maps: self.occupancy_map (2d array, 0 represents obstacles)
        points = points.T
        obstacle_value = 0
        robot_radius_in_cell = np.ceil(self.robot_radius / self.map_settings_dict['resolution'])
        x_range = np.arange(0,len(self.occupancy_map[0]))  # col
        y_range = np.arange(0,len(self.occupancy_map))  # row
        print(points)
        print(self.map_shape)
        for point in points:
            cx = point[0]
            cy = point[1]
            robot_occupancy = (x_range[np.newaxis,:]-cx)**2 + (y_range[:,np.newaxis]-cy)**2 < robot_radius_in_cell**2  # (x,y) fashion
            if obstacle_value in self.occupancy_map[robot_occupancy]:  # colliding
                return True
            if abs(cy) >= self.map_shape[0] or abs(cx) >= self.map_shape[1]:
                return True
            if cy<0 or cx<0:
                return True
        return False

    
    def trajectory_rollout(self, vel : float, rot_vel : float, curr_pose : np.ndarray, goal_pose : np.ndarray) -> np.ndarray:
        # Given your chosen velocities determine the trajectory of the robot for your gtimestep
        # The returned trajectory should be a series of points to check for collisions
        #RECIEVE WORLD POINTS
        assert curr_pose.shape == (3,), curr_pose.shape
        assert type(vel) == type(rot_vel) and type(vel) in [np.float64, float], type(vel)
        assert vel > 0

        robot_xy, theta = curr_pose[0:2], curr_pose[2]
        robot_direction = np.array([np.cos(theta), np.sin(theta)])  # vector representing forward direction of robot

        # return a straight line if w=0
        if rot_vel == 0:
            temp = np.linspace([1, 1], [self.num_substeps, self.num_substeps], self.num_substeps).T - 1
            ret = np.zeros((3, self.num_substeps))
            ret[0:2, :] = temp * vel * robot_direction.reshape(2, 1) * self.timestep + robot_xy.reshape(2, 1) # broadcast
            ret[2, :] = theta
            return ret
        
        # extract sign from w
        on_right = np.sign(rot_vel) * -1
        rot_vel = np.abs(rot_vel)
        robot_right_direction = np.array([np.sin(theta), -np.cos(theta)])   # vector representing right direction of robot

        # get radius and circle centre
        radius = vel / rot_vel
        circle_centre = radius * on_right * robot_right_direction + robot_xy

        # generate trajectory along circle
        start_angle = theta + np.pi/2 * on_right
        delta_angle = rot_vel * self.timestep

        traj = np.zeros((3, self.num_substeps))
        for i in range(self.num_substeps):
            ang = start_angle - i * delta_angle * on_right
            traj[2, i] = theta - i * delta_angle * on_right
            pos = (circle_centre + radius * np.array([np.cos(ang), np.sin(ang)]))
            traj[0:2, i] = pos
            dist_to_goal_pose = np.sqrt(np.sum(np.square(pos - goal_pose[:2])))
            
            if dist_to_goal_pose < self.sim_stopping_dist and i < (self.num_substeps - 1):
                traj[2, i+1] = theta - (i+1) * delta_angle * on_right
                traj[0:2, i+1] = goal_pose[:2]
                traj = traj[:, :i+2]
                break
        return traj

    
    def point_to_cell(self, point):
        #Convert a series of [x,y] indicies to metric points in the map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        # map origin = [-21.0, -49.25, 0.000000]
        # point = [[],[],[]]
        #TODO smth wrong
        map_origin = self.map_settings_dict['origin'][0:2]
        res = self.map_settings_dict['resolution']
        occ_points = point - np.tile(map_origin, (point.shape[1], 1)).T
        occ_points = occ_points/res
        occ_points[1, :] = self.map_shape[0] - occ_points[1, :]
        # print(point)
        # print(np.tile(map_origin, (point.shape[1], 1)).T)
        # print("res: ", occ_points) 
        # input()
        return np.round(occ_points).astype(int)

    def cell_to_point(self, point):
        #Convert a series of [x,y] metric points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        # map origin = [-21.0, -49.25, 0.000000]

        map_origin = self.map_settings_dict['origin'][0:2]
        res = self.map_settings_dict['resolution']
        h = self.map_shape[1]
        point[1, :] = h - point[1, :]
        world_points = point*res + np.tile(map_origin, (point.shape[1], 1)).T

        # world_points = point*res
        # world_points[1, :] = h - world_points[1, :]
        # world_points =  world_points + np.tile(map_origin, (point.shape[1], 1)).T

        # print(point)
        # print(point*res)
        # print(np.tile(map_origin, (point.shape[1], 1)).T)
        # print("res: ", world_points), 
        # input()
        return world_points

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        # print("TO DO: Implement a method to get the pixel locations of the robot path")
        #RECIEVE MAP POINTS
        map_circles = []
        points = points.squeeze()
        for i in range(len(points)):
            world_circles_r, world_circles_c = disk((points[i][0], points[i][1]), self.robot_radius/self.map_settings_dict['resolution'])
            world_circles = np.array([world_circles_r, world_circles_c])
            valid_max = np.where(world_circles)
            map_circles.append(world_circles)
            # map_circles.append(self.point_to_cell(world_circles))
        return map_circles
        # return [], []
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i : Node, point_f : np.ndarray):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point

        # the following code assumes that point_f is outside of the robot's maximum turn circle
        # point_f.reshape(2)
        assert point_f.shape == (2,)
        assert self.num_substeps >= 3

        path = np.zeros((3, self.num_substeps))
        min_radius = 2 #TODO: get minimum radius from velocity and maximum angular velocity

        # first check if the point is at the left or right side of the robot
        robot_xy, theta = node_i.point[0:2], node_i.point[2]
        robot_direction = np.array([np.cos(theta), np.sin(theta)])  # vector representing forward direction of robot
        robot_right_direction = np.array([np.sin(theta), -np.cos(theta)])   # vector representing right direction of robot

        # use dot product to find whether point_f is on the right side of robot, and whether it is in front of the robot
        on_right = np.sign(np.dot(robot_right_direction, point_f - robot_xy))
        in_front = np.sign(np.dot(robot_direction, point_f - robot_xy))

        # shit tone of math based on geometry
        circle_centre = on_right * min_radius * robot_right_direction + robot_xy
        circle_centre_to_point_f = point_f - circle_centre
        circle_centre_to_point_f_length = np.linalg.norm(circle_centre - point_f)
        alpha = np.arccos(min_radius / circle_centre_to_point_f_length) * on_right
        beta = np.arctan2(circle_centre_to_point_f[1], circle_centre_to_point_f[0])
        turning_point = (circle_centre + min_radius * np.array([np.cos(alpha + beta), np.sin(alpha + beta)]))
        theta_at_turning_point = np.arctan2(point_f[1] - turning_point[1], point_f[0] - turning_point[0])

        # filling in the start, finish, and turning point
        path[:, -1] = np.concatenate([point_f, [theta_at_turning_point]], axis=0)
        path[:, -2] = np.concatenate([turning_point, [theta_at_turning_point]], axis=0)

        # filling in the remaining points in the arc
        start_angle = theta + on_right * np.pi/2
        circle_centre_to_turning_point = turning_point - circle_centre
        turning_point_angle = np.arctan2(circle_centre_to_turning_point[1], circle_centre_to_turning_point[0])
        delta_angle = (turning_point_angle - start_angle) / (self.num_substeps - 2)

        for i in range(self.num_substeps - 1):
            ang = start_angle + i * delta_angle
            pos = (circle_centre + min_radius * np.array([np.cos(ang), np.sin(ang)]))
            path[:, i] = np.concatenate([pos, [theta - i * delta_angle]])

        return path
    
    def cost_to_come(self, trajectory_o, rot_cost_coefficient=0):
        #The cost to get to a node from lavalle 
        # print("TO DO: Implement a cost to come metric")
        
        # assume trajectory_o is 3xn numpy array generated by 'connect_node_to_point()'
        # [x, y, theta]
        # From now, the cost is calculated the Euclidean distance between points, ignoring rotation(i.e rot_cost_coefficient=0). 
        # Potentially we can tune the cost function for better performance
        
        points = np.transpose(trajectory_o)  # changing 3xn to nx3
        total_cost = 0
        for i in range(len(points) - 1):
            point1 = points[i, 0 : 2]
            point2 = points[i + 1, 0 : 2]
            angle_difference = points[i, 2] - points[i + 1, 2]
            total_cost += np.linalg.norm(point1 - point2) + rot_cost_coefficient * angle_difference
        return total_cost
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        # print("TO DO: Update the costs of connected nodes after rewiring.")

        # based on the obeservation to the codebase: I assume self.nodes[node_id] refers to a specific node
        # which means, their node_id is just the index (order added to the self.nodes)

        if len(self.nodes[node_id].children_ids) == 0:  # reached to a leaf of a tree 
            return
        
        for children_id in self.nodes[node_id].children_ids:
            trajectory_from_cur_to_child = np.hstack([self.nodes[node_id].point, self.nodes[children_id].point])
            assert trajectory_from_cur_to_child.shape == (3,2)
            cost_between_cur_to_child = self.cost_to_come(trajectory_from_cur_to_child)
            self.nodes[children_id].cost = self.nodes[node_id].cost + cost_between_cur_to_child
            self.update_children(children_id)
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        path_finding = True
        plot = 0
        # self.window.add_point(np.array([-10, -10]))
        # input()
        while path_finding: #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space() #world
            while self.check_if_duplicate(point):
                point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point[:2])
            print("point: ", point)
            print("closest node: ", self.nodes[closest_node_id].point)

            #Simulate driving the robot towards the closest point
            curr_node = self.nodes[closest_node_id].point
            traj, vel, rot_vel = self.simulate_trajectory(curr_node.reshape(3), point[0:2])
            thresh = 2
            traj = self.point_to_cell(traj[:2])

            # 4. check if traj collides with map or obstacle
            if not self.is_collide(traj) :   # TODO: remove 'True or' once collision detection done 
                new_node = Node(np.vstack((point[:2].reshape(2, 1), 0)), closest_node_id, 1)  # constant cost for now
                self.nodes.append(new_node)
                self.trajectories.append(traj)
                plot += 1
                if plot % 10 == 0:
                    plot = 0
                    self.plot_nodes()
            else:
                print(f'new point trajectory has collision')
                continue

            # 6. check if close enough to goal
            if np.linalg.norm(new_node.point[:2] - self.goal_point) < self.stopping_dist:
                traj, vel, rot_vel = self.simulate_trajectory(point, self.goal_point[0:2].squeeze())
                traj = self.point_to_cell(traj[:2])
                if not self.is_collide(traj):   # TODO: remove 'True or' once collision detection done 
                    new_node = Node(np.vstack((self.goal_point[:2].reshape(2, 1), 0)), len(self.nodes) - 1, 1)
                    self.nodes.append(new_node)
                    self.trajectories.append(traj)
                    print('goal point found in rrt')
                    break

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
            # traj = np.array((3, self.num_substeps)) of points

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
    
    def plot_nodes(self):
        # Set the figure size
        start_node = np.zeros((1, 2))
        # plt.rcParams["figure.figsize"] = [self.map_shape[0], self.map_shape[1]]
        # plt.rcParams["figure.autolayout"] = True  
        plt.imshow(self.occupancy_map)

        # change each node from world to map points
        for node in self.nodes:
            # point = [[],[],[]]
            map_point = self.point_to_cell(node.point[:2])
            plt.plot([map_point[0]], [map_point[1]], 'ro')

        # plot goal in green
        map_goal = self.point_to_cell(self.goal_point)
        plt.plot([map_goal[0]], [map_goal[1]], 'go')

        # plot start in blue
        map_start = self.point_to_cell(start_node.T)
        plt.plot([map_start[0]], [map_start[1]], 'bo')

        # plot each trajectory
        for traj in self.trajectories:
            plt.plot(traj[0, :], traj[1, :], color='red', alpha=0.3)
        plt.show()
        return 0

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    # map_filename = "myhal.png"
    # map_setings_filename = "myhal.yaml"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[20], [-20]])# np.array([[7], [2]]) #m WORLD POINTS
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_planning()
    #print(nodes)
    #exit()
    node_path_metric = np.hstack(path_planner.recover_path())
    #print(nodes)
    #print(node_path_metric)
    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)

if __name__ == '__main__':
    main()