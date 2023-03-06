#!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils


TRANS_GOAL_TOL = .25  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .5  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0, 0.025, 0.13, 0.26]  # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .025  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 1  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = TRANS_GOAL_TOL  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'shortest_path_rrt_star_willowgarageworld_05res.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
# TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
TEMP_HARDCODE_PATH = [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
 [ 2.34000000e+00,  -.10000000e+00,  0.00000000e+00],
 [ 6.30828802e+00, -2.34248810e-01, -0.44089210e-1],
 [ 8.56221271e+00, 1.30256975e-01, -0.44089210e-1],
 [ 1.00085646e+01, -2.37293066e-01, -0.44089210e-1],
 [ 1.08448660e+01, -9.06959354e-01, -1.3566667e+0],
 [ 1.0997248e+01, -2.50635708e+00, -1.44233333e+0],
 [ 1.16984787e+01, -4.57867633e+00, -1.42566667e+0],
 [ 1.16519847e+01, -4.96589500e+00, -1.42566667e+1],
 [ 1.15894696e+01, -7.16585592e+00, -1.42900000e+01],
 [ 1.13489173e+01, -1.14425717e+01,  -1.32300000e+01],
 [ 1.3280176e+01, -1.34378601e+01,  -1.9300000e+0],
 [ 1.4489173e+01, -1.65384838e+01,  -1.42300000e+01],
 [ 1.38918019e+01, -1.74573681e+01,  1.09200000e+01],
 [ 1.38954633e+01, -1.84578490e+01,  1.19200000e+01],
 [ 1.52512714e+01, -2.15320623e+01,  1.09200000e+01],
 [ 1.54464297e+01, -2.57275687e+01,  2.48733333e+01],
 [ 1.71020487e+01, -2.71663154e+01,  -0.5333333e+00],
 [ 1.8322647e+01, -2.75887700e+01,  -0.49333333e+00],
 [ 1.93417640e+01, -2.88240732e+01,  -0.6333333e+00],
  [ 2.10151453e+01, -2.92035771e+01,  -0.49333333e+00],
 [ 2.17139542e+01, -3.10056435e+01,  -0.6333333e+00],
 [ 2.26516745e+01, -3.2518003e+01,  2.48733333e+01],
  [ 2.55107150e+01, -3.29724653e+01,  2.48733333e+01],
 [ 2.63107150e+01, -3.36724653e+01,  2.45533333e+01],
 [ 2.66078701e+01, -3.42730464e+01, -0.93333e+00],
 [ 2.75300869e+01, -3.63047692e+01,  -1.53333e+00],
 [ 2.80782785e+01, -3.73593749e+01,  -0.53333e+00],
 [ 2.89542641e+01, -3.78241542e+01, -1.51666667e+00],
 [ 2.93180861e+01, -3.86332619e+01, -6.97666667e+00],
 [ 3.10968702e+01, -4.01629804e+01, -6.97666667e+00],
 [ 3.27396735e+01, -4.17099580e+01,  1.21333333e+01],
 [ 3.44979318e+01, -4.22654409e+01, -6.97666667e+00],
 [ 3.69385022e+01, -4.46689126e+01,  1.21333333e+01],
 [ 3.7687699e+01, -4.4691430e+01,  .1666667e+00],
 [ 4.00768994e+01, -4.40006835e+01,  1.91100000e+01],
 [ 4.05522670e+01, -4.36667181e+01,  1.27400000e+01],
 [ 4.16103166e+01, -4.38591039e+01,  -0.1666667e+00]]

DISTANCE_WEIGHT = 1
ROTATION_WEIGHT = 0.001

#Map Handling Functions
def load_map(filename):
    import matplotlib.image as mpimg
    import cv2 
    im = cv2.imread("../maps/" + filename)
    im = cv2.flip(im, 0)
    # im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    im_np = np.logical_not(im_np)     #for ros
    return im_np

class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform
        #print(self.map_odom_tf)

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        self.map = rospy.wait_for_message('/map', OccupancyGrid)
        self.map_np = np.array(self.map.data).reshape(self.map.info.height, self.map.info.width)
        self.map_resolution = round(self.map.info.resolution, 5)
        self.map_origin = -utils.se2_pose_from_pose(self.map.info.origin)  # negative because of weird way origin is stored
        #print(self.map_origin)
        self.map_nonzero_idxes = np.argwhere(self.map_np)
        #print(map)


        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        #self.path_tuples = np.load(os.path.join(cur_dir, PATH_NAME)).T
        self.path_tuples = np.array(TEMP_HARDCODE_PATH)
        print(self.path_tuples)
        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)

            #print("TO DO: Propogate the trajectory forward, storing the resulting points in local_paths!")
            # === START CUSTOM CODE === 

            #Create array to track if collision is present:
            collisions = np.zeros(self.num_opts)
            #Create array to track if target pose is within path:
            complete = np.zeros(self.num_opts)

            #print(local_paths[0,0,:])
            for t in range(1, self.horizon_timesteps + 1):
                for path_idx in range(0, self.num_opts):
                    t_vel = self.all_opts[path_idx, 0]
                    r_vel = self.all_opts[path_idx, 1]
                    # Stop propagating paths that have already collided:
                    if collisions[path_idx] == 1:
                        path_idx += 1
                        continue
                    
                    # Pass through paths that have satisified target requirements:
                    if complete[path_idx] == 1:
                        local_paths[t,path_idx,:] = local_paths[t-1,path_idx,:]
                        path_idx += 1
                        continue
                    
                    theta = local_paths[t-1,path_idx,2]
                    rot_mat = np.array([[np.cos(theta), 0],
                                        [np.sin(theta), 0],
                                        [0, 1]])
                    vel_vec = np.array([t_vel, r_vel])
                    q_dot = np.matmul(rot_mat, vel_vec) # shape (3, 1)
                    

                    local_paths[t,path_idx,:] = local_paths[t-1,path_idx,:] + q_dot.T * INTEGRATION_DT

                    # check if new path satisifies target requirements:
                    dist_from_goal = np.linalg.norm(local_paths[t,path_idx,:2] - self.cur_goal[:2])
                    abs_angle_diff = np.abs(local_paths[t,path_idx,2] - self.cur_goal[2])
                    rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
                    if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
                        complete[path_idx] = 1
                        path_idx += 1
                        continue
                    
                    # Check if new path collides:
                    new_point_pixels =  (local_paths[t,path_idx,:2] + self.map_origin[:2] ) / self.map_resolution
                    if np.sum(self.map_np[int(new_point_pixels[1]):int(new_point_pixels[1])+5,int(new_point_pixels[0]-5):int(new_point_pixels[0])+5]) > 0.65:
                        local_paths[t,path_idx,:]
                        collisions[path_idx] = 1
      
            # check all trajectory points for collisions
            '''
            # first find the closest collision point in the map to each local path point
            local_paths_pixels = (self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution
            valid_opts = range(self.num_opts)
            local_paths_lowest_collision_dist = np.ones(self.num_opts) * 50

            print("TO DO: Check the points in local_path_pixels for collisions")
            for opt in range(local_paths_pixels.shape[1]):
                for timestep in range(local_paths_pixels.shape[0]):
                    pass
            '''
            # remove trajectories that were deemed to have collisions
            #print("TO DO: Remove trajectories with collisions!")
            valid_opts = np.where(collisions == 0)[0]
            #print(valid_opts)
            #print(valid_opts)
            # calculate final cost and choose best option
            #print("TO DO: Calculate the final cost and choose the best control option!")
            final_cost = np.zeros_like(valid_opts, dtype=float)
            for n in range(0, len(valid_opts)):
                final_pos = local_paths[self.horizon_timesteps, valid_opts[n], :]
                # check euclidian distance to goal:
                final_cost[n] = DISTANCE_WEIGHT*np.sqrt(np.sum(np.square(final_pos[:2] - self.cur_goal[:2])))

                # check rotation delta to goal:
                final_cost[n] += ROTATION_WEIGHT*np.abs(final_pos[2] - self.cur_goal[2])
            # === END CUSTOM CODE === 



            if final_cost.size == 0:  # hardcoded recovery if all options have collision
                control = [-.1, 0]
                
            else:
                best_opt = valid_opts[final_cost.argmin()]
                control = self.all_opts[best_opt]
                self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, best_opt], 'map'))

            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            # print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
            #     control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass