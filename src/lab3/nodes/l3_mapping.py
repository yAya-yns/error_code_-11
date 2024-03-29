#!/usr/bin/env python3
from __future__ import division, print_function

import numpy as np
import rospy
import tf2_ros
from skimage.draw import line as ray_trace
import rospkg
import matplotlib.pyplot as plt

# msgs
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan

from utils import convert_pose_to_tf, convert_tf_to_pose, euler_from_ros_quat, \
     tf_to_tf_mat, tf_mat_to_tf


ALPHA = 1
BETA = 1
MAP_DIM = (4, 4)
CELL_SIZE = .01
NUM_PTS_OBSTACLE = 3
SCAN_DOWNSAMPLE = 1

class OccupancyGripMap:
    def __init__(self):
        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # subscribers and publishers
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb, queue_size=1)
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)

        # attributes
        width = int(MAP_DIM[0] / CELL_SIZE); height = int(MAP_DIM[1] / CELL_SIZE)
        self.log_odds = np.zeros((width, height))
        self.np_map = np.ones((width, height), dtype=np.uint8) * -1  # -1 for unknown
        self.map_msg = OccupancyGrid()
        self.map_msg.info = MapMetaData()
        self.map_msg.info.resolution = CELL_SIZE
        self.map_msg.info.width = width
        self.map_msg.info.height = height

        # transforms
        self.base_link_scan_tf = self.tf_buffer.lookup_transform('base_link', 'base_scan', rospy.Time(0),
                                                            rospy.Duration(2.0))
        odom_tf = self.tf_buffer.lookup_transform('odom', 'base_link', rospy.Time(0), rospy.Duration(2.0)).transform

        # set origin to center of map
        rob_to_mid_origin_tf_mat = np.eye(4)
        rob_to_mid_origin_tf_mat[0, 3] = -width / 2 * CELL_SIZE
        rob_to_mid_origin_tf_mat[1, 3] = -height / 2 * CELL_SIZE
        odom_tf_mat = tf_to_tf_mat(odom_tf)
        self.map_msg.info.origin = convert_tf_to_pose(tf_mat_to_tf(odom_tf_mat.dot(rob_to_mid_origin_tf_mat)))

        # map to odom broadcaster
        self.map_odom_timer = rospy.Timer(rospy.Duration(0.1), self.broadcast_map_odom)
        self.map_odom_tf = TransformStamped()
        self.map_odom_tf.header.frame_id = 'map'
        self.map_odom_tf.child_frame_id = 'odom'
        self.map_odom_tf.transform.rotation.w = 1.0

        rospy.spin()
        plt.imshow(100-self.np_map, cmap='gray', vmin=0, vmax=100)
        rospack = rospkg.RosPack()
        path = rospack.get_path("rob521_lab3")
        plt.savefig(path+"/map.png")

    def broadcast_map_odom(self, e):
        self.map_odom_tf.header.stamp = rospy.Time.now()
        self.tf_br.sendTransform(self.map_odom_tf)

    def scan_cb(self, scan_msg : LaserScan):
        # read new laser data and populate map
        # get current odometry robot pose
        # print(scan_msg)
        # rospy.loginfo(len(scan_msg.ranges))
        try:
            odom_tf = self.tf_buffer.lookup_transform('odom', 'base_scan', rospy.Time(0)).transform
        except tf2_ros.TransformException:
            rospy.logwarn('Pose from odom lookup failed. Using origin as odom.')
            odom_tf = convert_pose_to_tf(self.map_msg.info.origin)
        # rospy.loginfo(('odom_tf', odom_tf))
        # get odom in frame of map
        odom_map_tf = tf_mat_to_tf(
            np.linalg.inv(tf_to_tf_mat(convert_pose_to_tf(self.map_msg.info.origin))).dot(tf_to_tf_mat(odom_tf))
        )
        odom_map = np.zeros(3)
        odom_map[0] = odom_map_tf.translation.x
        odom_map[1] = odom_map_tf.translation.y
        odom_map[2] = euler_from_ros_quat(odom_map_tf.rotation)[2]

        print(odom_map)

        # rospy.loginfo(('odom_map', odom_map))

        # YOUR CODE HERE!!! Loop through each measurement in scan_msg to get the correct angle and
        # x_start and y_start to send to your ray_trace_update function.
        self.np_map[:, :] = -1
        # self.log_odds[:, :] = 0
        for i in range(0, len(scan_msg.ranges), 5):
        # for i in range(1):
            # rospy.loginfo(i)
            self.np_map, self.log_odds = self.ray_trace_update(self.np_map, self.log_odds, int(odom_map[1] * 100), int(odom_map[0] * 100), -odom_map[2] - i * scan_msg.angle_increment + np.pi/2, scan_msg.ranges[i])
        print(np.max(self.np_map), np.min(self.np_map))
        # self.np_map[:, self.np_map.shape[1] // 4] = 1

        # publish the message
        self.map_msg.info.map_load_time = rospy.Time.now()
        self.map_msg.data = self.np_map.flatten()
        self.map_pub.publish(self.map_msg)

    def ray_trace_update(self, map : np.ndarray, log_odds : np.ndarray, x_start : int, y_start : int, angle : float, range_mes : float):
        """
        A ray tracing grid update as described in the lab document.

        :param map: The numpy map.
        :param log_odds: The map of log odds values.
        :param x_start: The x starting point in the map coordinate frame (i.e. the x 'pixel' that the robot is in).
        :param y_start: The y starting point in the map coordinate frame (i.e. the y 'pixel' that the robot is in).
        :param angle: The ray angle relative to the x axis of the map.
        :param range_mes: The range of the measurement along the ray.
        :return: The numpy map and the log odds updated along a single ray.
        """
        # YOUR CODE HERE!!! You should modify the log_odds object and the numpy map based on the outputs from
        # ray_trace and the equations from class. Your numpy map must be an array of int8s with 0 to 100 representing
        # probability of occupancy, and -1 representing unknown.
        assert map.dtype == np.int16, map.dtype
        # print(map.shape)

        n_row, n_col = map.shape
        dir_vec = np.array([np.cos(angle), np.sin(angle)])
        start_pos = np.array([x_start, y_start])
        y_dist = n_row - y_start - 1
        x_dist = n_col - x_start - 1
        dist_to_boundary = min(np.abs(x_dist / (np.abs(np.cos(angle))+0.1)), np.abs(y_dist / (np.abs(np.sin(angle))+0.1)))
        if range_mes * 100 > dist_to_boundary:
            dist = dist_to_boundary
            dest = dist * dir_vec + start_pos
            dest = np.round(dest).astype(int)
            if not np.all(np.logical_and(map.shape - dest >= 0, dest >= 0)):
                return map, log_odds
            coord = np.array(ray_trace(x_start, y_start, dest[0], dest[1])) # (2, N)
            
            # print(coord)
            
            log_odds[coord[0, :], coord[1, :]] -= BETA
            # map[coord[0, :], coord[1, :]] = 1
        else: # TODO: duplicate code fragment
            dist = range_mes * 100
            dest = dist * dir_vec + start_pos
            dest = np.round(dest).astype(int)
            coord = np.array(ray_trace(x_start, y_start, dest[0], dest[1])) # (2, N)
            near_obs_coord = coord[:, np.where(np.linalg.norm(coord - dest.reshape(2, 1), axis=0) < NUM_PTS_OBSTACLE)[0]]
            
            # print(near_obs_coord)
            # print(coord)

            log_odds[near_obs_coord[0, :], near_obs_coord[1, :]] += (ALPHA + BETA)
            log_odds[coord[0, :], coord[1, :]] -= BETA
            # map[rr, cc] = 1

            # unknown = dist_to_boundary * dir_vec + start_pos
            # unknown = np.round(unknown).astype(int)
            # assert np.all(np.logical_and(map.shape - unknown >= 0, unknown >= 0)), unknown
            # rr, cc = ray_trace(dest[0], dest[1], unknown[0], unknown[1])
            # map[rr, cc] = -1
        # print(np.max(log_odds), np.min(log_odds))
        a = np.round(self.log_odds_to_probability(log_odds) + 0.01)
        # print(np.max(a), np.min(a))
        # map = np.round((np.round(self.log_odds_to_probability(log_odds) + 0.01) - 0.5)*2).astype(np.int16)
        map = (self.log_odds_to_probability(log_odds) * 100).astype(np.int16)
        # plt.matshow(log_odds)
        # plt.show()
        return map, log_odds

    def log_odds_to_probability(self, values):
        # print(values)
        return np.exp(values) / (1 + np.exp(values))


if __name__ == '__main__':
    try:
        rospy.init_node('mapping')
        ogm = OccupancyGripMap()
    except rospy.ROSInterruptException:
        pass