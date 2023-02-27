import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from l2_planning import PathPlanner, Node

def connect_node_to_point(num_substeps, node_i : Node, point_f : np.ndarray):
    #Given two nodes find the non-holonomic path that connects them
    #Settings
    #node is a 3 by 1 node
    #point is a 2 by 1 point

    # the following code assumes that point_f is outside of the robot's maximum turn circle
    point_f.reshape(2)
    assert point_f.shape == (2,)
    assert num_substeps >= 3

    path = np.zeros((3, num_substeps))
    min_radius = 2 #TODO: get minimum radius from velocity and maximum angular velocity

    # first check if the point is at the left or right side of the robot
    robot_xy, theta = node_i.point[0:2], node_i.point[2]
    robot_direction = np.array([np.cos(theta), np.sin(theta)])  # vector representing forward direction of robot
    robot_right_direction = np.array([np.sin(theta), -np.cos(theta)])   # vector representing right direction of robot

    # use dot product to find whether point_f is on the right side of robot, and whether it is in front of the robot
    on_right = np.sign(np.dot(robot_right_direction, point_f - robot_xy))
    in_front = np.sign(np.dot(robot_direction, point_f - robot_xy))

    print(f'on_right: {on_right}, in_front: {in_front}')

    # shit tone of math based on geometry
    circle_centre = on_right * min_radius * robot_right_direction + robot_xy
    circle_centre_to_point_f = point_f - circle_centre
    circle_centre_to_point_f_length = np.linalg.norm(circle_centre - point_f)
    alpha = np.arccos(min_radius / circle_centre_to_point_f_length) * on_right
    beta = np.arctan2(circle_centre_to_point_f[1], circle_centre_to_point_f[0])
    turning_point = (circle_centre + min_radius * np.array([np.cos(alpha + beta), np.sin(alpha + beta)]))
    theta_at_turning_point = np.arctan2(point_f[1] - turning_point[1], point_f[0] - turning_point[0])

    print(f'circle_centre: {circle_centre}')
    print(f'turning_point: {turning_point}')


    if False:
        fig = plt.figure()
        ax = fig.add_subplot()
        circle1 = patches.Circle(circle_centre, radius=min_radius)
        ax.add_patch(circle1)
        ax.axis('equal')
        ax.plot(turning_point[0], turning_point[1], color='red', marker='o')
        ax.plot(circle_centre[0], circle_centre[1], color='red', marker='o')
        ax.plot(point_f[0], point_f[1], color='red', marker='o')
        ax.plot(robot_xy[0], robot_xy[1], color='red', marker='o')
        plt.show()
        exit()

    # filling in the start, finish, and turning point
    # print(point_f.shape)
    path[:, -1] = np.concatenate([point_f, [theta_at_turning_point]], axis=0)
    # print(turning_point.shape)
    # print(theta_at_turning_point)
    path[:, -2] = np.concatenate([turning_point, [theta_at_turning_point]], axis=0)

    # filling in the remaining points in the arc
    
    start_angle = theta + on_right * np.pi/2
    circle_centre_to_turning_point = turning_point - circle_centre
    turning_point_angle = np.arctan2(circle_centre_to_turning_point[1], circle_centre_to_turning_point[0])
    delta_angle = (turning_point_angle - start_angle) / (num_substeps - 2)
    print(f'delta_angle: {delta_angle}')
    print(f'turning_point_angle: {turning_point_angle}')

    for i in range(num_substeps - 1):
        ang = start_angle + i * delta_angle
        print(ang)
        pos = (circle_centre + min_radius * np.array([np.cos(ang), np.sin(ang)]))
        path[:, i] = np.concatenate([pos, [theta - i * delta_angle]])

    if True:
        fig = plt.figure()
        ax = fig.add_subplot()
        circle1 = patches.Circle(circle_centre, radius=min_radius)
        ax.add_patch(circle1)
        ax.axis('equal')
        ax.plot(turning_point[0], turning_point[1], color='red', marker='o')
        ax.plot(circle_centre[0], circle_centre[1], color='red', marker='o')
        ax.plot(point_f[0], point_f[1], color='red', marker='o')
        ax.plot(robot_xy[0], robot_xy[1], color='red', marker='o')
        ax.plot(path[0, :], path[1, :], color='green')
        plt.show()
        exit()
    return path

def connect_node_to_point_test():
    node = Node(np.array([0, 0, np.pi/2]), None, None)
    point_f = np.array([-10, -10])
    print(f'starting pos: {node.point}, point_f: {point_f}')
    output = connect_node_to_point(10, node, point_f)
    print(output)

    # plt.plot(node.point[0:2], point_f)
    plt.plot(output[0, :], output[1, :])
    plt.show()

if __name__ == '__main__':
    connect_node_to_point_test()