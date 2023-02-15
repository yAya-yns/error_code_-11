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
import scipy


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
        node = np.zeros((3,1))
        # node[0:2] = self.cell_to_point(node[0:2]) 
        self.nodes = [Node(node, -1, 0)]
        self.node_pts = np.zeros((3,1))[:2][None]

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
        # print("TO DO: Sample point to drive towards")
        #maybe sample occ map first?
        #TODO this idk what it wants. not sure if this is enough. Hint is throwing me off
        # IN WORLD POINTS
        rand = np.random.rand(3, 1)
        rand[0] = rand[0] * (self.bounds[0, 1] - self.bounds[0, 0])  + self.bounds[0, 0]
        rand[1] = rand[1] * (self.bounds[1, 1] - self.bounds[1, 0])  + self.bounds[1, 0] 
        rand[2] = 0 

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
            if i.point == point:
                return True
        return False
    
    def closest_node(self, point):
        # Returns the index of the closest node
        # print("TO DO: Implement a method to get the closest node to a sampled point")
        closest = 1000000
        ind = False
        print("point in closest: ", point)
        for i in range(len(self.nodes)):
            print("self.nodes[i]: ", self.nodes[i].point)
            dist = np.sqrt(np.square(np.sum(self.nodes[i].point.squeeze() - point.squeeze())))
            if dist < closest:
                closest = dist
                ind = i

        return ind
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        # recieve map points

        # print("TO DO: Implment a method to simulate a trajectory given a sampled point") uhhh 
        print("node_i:", node_i)
        print("point_s:", point_s)
        input()
        vel, rot_vel = self.robot_controller(node_i, point_s)
        vel = vel[None, :]
        rot_vel = rot_vel[None, :]
        print("vel: ", vel)
        print("rot_vel: ", rot_vel)

        robot_traj = self.trajectory_rollout(vel, rot_vel, node_i, point_s)
        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")
        
        # recieve WORLD points


        # print("node_i: ", node_i)
        # print("to point_s: ", point_s)

        interval = 0.05

        d_x = node_i[0] - point_s[0]
        d_y = node_i[1] - point_s[1]
        # print("dx dy ", d_x, d_y)
        d_theta = np.arctan2(point_s[1], point_s[0]) 
        if d_theta < 0:
            d_theta = 2*np.pi + d_theta
        # print("dtheta ", d_theta)

        # dt = np.sqrt(d_x**2 + d_y**2)/(self.vel_max/2) 

        vel = np.sqrt(d_x**2 + d_y**2)/self.num_substeps/7
        rot_vel = (d_theta - node_i[2])/self.num_substeps/7
        if abs(vel) > self.vel_max:
            if vel>0:
                vel = np.array([self.vel_max])
            else:
                vel = -np.array([self.vel_max])
        if abs(rot_vel) > self.rot_vel_max:
            if rot_vel>0:
                rot_vel = np.array([self.rot_vel_max])
            else:
                rot_vel = -np.array([self.rot_vel_max])

        vel_best = vel
        rot_vel_best = rot_vel

        if vel - interval > 0: vel_start = vel - interval
        else: vel_start = 0

        if vel + interval < self.vel_max: vel_stop = vel + interval
        else: vel_stop = self.vel_max

        if rot_vel - interval > 0: rot_vel_start = rot_vel - interval
        else: rot_vel_start = 0

        if rot_vel + interval < self.rot_vel_max: rot_vel_stop = rot_vel + interval
        else: rot_vel_stop = self.rot_vel_max

        rot_list = np.linspace(rot_vel_start, rot_vel_stop, num=5)
        vel_list = np.linspace(vel_start, vel_stop, num=5)

        traj_opts = np.zeros((rot_list.shape[0]**2, self.num_substeps, 3))
        i=0

        ind_list = np.zeros((rot_list.shape[0]**2, 2))
        for rot in range(rot_list.shape[0]):
            for vel in range(vel_list.shape[0]):
                curr_traj = self.trajectory_rollout(vel_list[vel][None, :], rot_list[rot][None, :], node_i, point_s)
                ind_list[i] = np.array([vel, rot])
                if self.is_collide(curr_traj[:, 0:2].T):
                    print("collision true")
                    traj_opts[i, :, :] = np.inf * np.ones((self.num_substeps, 3))
                    i += 1
                    continue
                traj_opts[i, :, :] = self.trajectory_rollout(vel_list[vel][None, :], rot_list[rot][None, :], node_i, point_s)
                i += 1

        best_dist = np.inf
        for i in range(traj_opts.shape[0]):
            dist = np.linalg.norm(traj_opts[i,-1, :] - point_s)
            if dist < best_dist:
                ind = ind_list[i]
                vel_best = vel_list[int(ind[0])]
                rot_vel_best = rot_list[int(ind[1])]

        return vel_best, rot_vel_best
        # if abs(vel) > self.vel_max:
        #     if vel>0:
        #         vel = np.array([self.vel_max])
        #     else:
        #         vel = -np.array([self.vel_max])
        # if abs(rot_vel) > self.rot_vel_max:
        #     if rot_vel>0:
        #         rot_vel = np.array([self.rot_vel_max])
        #     else:
        #         rot_vel = -np.array([self.rot_vel_max])
        # print("vel after ", vel)
        # print("rot_vel after ", rot_vel)
        # input()
        

    def is_collide(self, points):
        # check for multiple points if collides with circle 
        # RECIEVE MAP POINTS
        for j in range(points.shape[0]):
            if (points[j][0]<0 or points[j][0] > self.map_shape[0] or points[j][1]<0 or points[j][1]> self.map_shape[1]):
                print(points[j])
                print(self.map_shape)
                print("COLLISION OUT OF BOUNDS")
                return 1
            
        disks = self.points_to_robot_circle(points.T)
        # print("white: ",np.where(self.occupancy_map == 1))
        # print(self.bounds[0, 0])
        # print(self.bounds[1, 0])
        # print(self.bounds[0, 1])
        # print(self.bounds[1, 1])
        # print(self.occupancy_map[1][1])
        # RECIEVE MAP POINTS
        for disk in disks:
            if np.sum(self.occupancy_map[disk[0].astype(int), disk[1].astype(int)])<self.map_settings_dict['occupied_thresh']:
                print(disk, points, self.occupancy_map[disk[0].astype(int), disk[1].astype(int)])
                return 1
            else:
                return 0
    
    def trajectory_rollout(self, vel, rot_vel, curr_pose, goal):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")
        #RECIEVE WORLD POINTS
        traj = np.zeros((3, self.num_substeps))
        paths = []
        n = rot_vel.shape[0]

        time = self.timestep/self.num_substeps * np.ones((3,1))
        theta = curr_pose[2][0]
        # curr_pose[0:2] = self.cell_to_point(curr_pose[0:2]) #change to world for correct units
        print("curr_pose: ", curr_pose)
        print("theta: ", theta)
        traj_opts = np.zeros((self.num_substeps, 3)) #(N, num_substeps, 3)
        
        rot_mat = np.array([[np.cos(theta), 0],
                                [np.sin(theta), 0],
                                [0, 1]])
        vel_vec = np.array([vel, rot_vel]).squeeze(-1)
        q_dot = np.matmul(rot_mat, vel_vec) + curr_pose
        waypoint = np.multiply(time, q_dot) + curr_pose
        traj_opts[0:1, :] = waypoint.T

        for k in range(1, self.num_substeps):
            theta = traj_opts[k-1:k, 2][0]
            rot_mat = np.array([[np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [0, 1]])
            q_dot = np.matmul(rot_mat, vel_vec) + traj_opts[k-1:k, :].T
            waypoint = np.multiply(time, q_dot)
            traj_opts[k:k+1, :] = traj_opts[k-1:k, :] + waypoint.T

        print("traj world: ", traj_opts)
        traj_opts[:, 0:2] = self.point_to_cell(traj_opts[:, 0:2].T).T #turn to map
        
        return traj_opts.squeeze().astype(int) #(N, self.num_substeps, 3) #MAP POINTS 

    
    def point_to_cell(self, point):
        #Convert a series of [x,y] indicies to metric points in the map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        # map origin = [-21.0, -49.25, 0.000000]
        
        map_origin = self.map_settings_dict['origin'][0:2]
        res = self.map_settings_dict['resolution']
        occ_points = point - np.tile(map_origin, (point.shape[1], 1)).T
        # print(point)
        # print(np.tile(map_origin, (point.shape[1], 1)).T)
        # print(occ_points/res)
        # print("res: ", occ_points) 
        # input()
        return np.round(occ_points/res).astype(int)

    def cell_to_point(self, point):
        #Convert a series of [x,y] metric points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        # map origin = [-21.0, -49.25, 0.000000]

        map_origin = self.map_settings_dict['origin'][0:2]
        res = self.map_settings_dict['resolution']
        world_points = point*res + np.tile(map_origin, (point.shape[1], 1)).T
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
        point_f.reshape(2)
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
            print(ang)
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
            angle_difference = points[i, 3] - points[i + 1, 3]
            total_cost += np.linalg.norm(point1 - point2) + rot_cost_coefficient * angle_difference
        return total_cost
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        path_finding = True
        check_final = 10
        check_ind = 0
        while path_finding: #Most likely need more iterations than this to complete the map!
            #Sample map space
            input() 
            point = self.sample_map_space() #world

            #Get the closest point
            closest_node_id = self.closest_node(point)
            print("point: ", point)
            print("closest node: ", self.nodes[closest_node_id].point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point) # map points, input world

            #Check for collisions
            # print("TO DO: Check for collisions and add safe points to list of nodes.")
            # traj = np.array((3, self.num_substeps)) of points
            # check collision between two lines?
            traj = trajectory_o # occupancy grid points
            print("traj: ",traj)
            print('finished getting trajectory')
            # self.bounds[0, 0] = self.map_settings_dict["origin"][0]
            # self.bounds[1, 0] = self.map_settings_dict["origin"][1]
            # self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
            # self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]
            collision = False
            if self.is_collide(traj[:, 0:2].T):
                collision = True
                print("collision true")
                continue
                    
            if collision == False:
                dist = np.sqrt((self.nodes[closest_node_id].point[0]- point[0])**2 + (self.nodes[closest_node_id].point[1] - point[1])**2)
                

                print("adding node: ", point)
                print("dist to goal: ", np.linalg.norm(traj[-1, :] - self.goal_point))
            
                #Add closest point to path:
                world_point = traj[-1]
                world_point[0:2] = self.cell_to_point(world_point[:2].T[None, :].T).T.squeeze()
                new_node = Node(world_point[None, :], closest_node_id, dist) 
                print("new_node.point ", new_node.point)
                input()

                self.window.add_point(new_node.point[0][:2])
                self.window.add_line(new_node.point[0][:2], self.nodes[closest_node_id].point.T[0][0:2])

                self.nodes.append(new_node)
                self.nodes[closest_node_id].children_ids += [len(self.nodes)]

            # ____________start trying to connect the goal:
            if check_final == check_ind:
                check_ind = 0
                point = np.zeros((3,1))
                point[:2] = self.goal_point 
                closest_node_id = self.closest_node(point)
                traj = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

                dist = np.sqrt(np.sum(np.square(traj[-1][:2] - point[:2])))
                if dist < self.stopping_dist:
                    # if new point is within target radius:
                    path_finding = False
                    break
                    
                world_point = traj[-1]
                world_point[0:2] = self.cell_to_point(world_point[:2].T[None, :].T).T.squeeze()
                new_node = Node(world_point[None, :], closest_node_id, dist) 
                print("new_node.point ", new_node.point)
                input()
                
                self.window.add_point(new_node.point[0][:2])
                self.window.add_line(new_node.point[0][:2], self.nodes[closest_node_id].point.T[0][0:2])

                self.nodes.append(new_node)
                self.nodes[closest_node_id].children_ids += [len(self.nodes)]
                
            
            # #Check if goal has been reached
            # # print("TO DO: Check if at goal point.")
            # print("dist to goal: ", np.linalg.norm(traj[-1, :] - self.goal_point))
            # print("curr nodes scatter:", [i.point for i in self.nodes])
            # if np.linalg.norm(traj[-1, :] - self.goal_point) < self.stopping_dist: # reached goal
            #     self.goal_nodes[len(self.nodes) - 1] = self.nodes[-1]
            #     self.best_goal_node_id = len(self.nodes) - 1
            #     path_finding = False
            #     break
            check_ind += 1
        
        
        points = [i.point for i in self.nodes]
        print("final points: ", points)

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

def main():
    #Set map information
    # map_filename = "willowgarageworld_05res.png"
    map_filename = "myhal.png"
    map_setings_filename = "myhal.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m WORLD POINTS
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())
    print(nodes)
    print(node_path_metric)
    #Leftover test functions
    # np.save("shortest_path.npy", node_path_metric)

if __name__ == '__main__':
    main()