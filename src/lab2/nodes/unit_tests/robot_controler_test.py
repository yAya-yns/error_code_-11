import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def robot_controller(node_i : np.ndarray, point_s : np.ndarray):
    #This controller determines the velocities that will nominally move the robot from node i to node s
    #Max velocities should be enforced
    assert node_i.shape == (3,)
    assert point_s.shape == (2,), point_s.shape

    rot_vel_max = 0.5
    vel_max = 1

    point_s = point_s.reshape(2)

    robot_xy, theta = node_i[0:2], node_i[2]
    robot_right_direction = np.array([np.sin(theta), -np.cos(theta)])   # vector representing right direction of robot
    on_right = np.sign(np.dot(robot_right_direction, point_s - robot_xy))

    start_to_end_vec = point_s - robot_xy
    radius = 0.5 * np.linalg.norm(robot_xy - point_s) / np.dot(start_to_end_vec / np.linalg.norm(start_to_end_vec), robot_right_direction * on_right)

    midpoint = (robot_xy + point_s) / 2
    circle_centre = robot_xy + robot_right_direction * on_right * radius

    ret_v = radius * rot_vel_max
    ret_w = -1 * rot_vel_max * on_right

    if ret_v > vel_max:
        factor = vel_max / ret_v
        ret_v = factor * ret_v
        ret_w = factor * ret_w

    print(f'midpoint: {midpoint}')
    print(f'circle_centre: {circle_centre}')
    fig = plt.figure()
    ax = fig.add_subplot()
    circle1 = patches.Circle(circle_centre, radius=radius)
    ax.add_patch(circle1)
    ax.axis('equal')
    # ax.plot(turning_point[0], turning_point[1], color='red', marker='o')
    ax.plot(midpoint[0], midpoint[1], color='green', marker='o')
    ax.plot(circle_centre[0], circle_centre[1], color='red', marker='o')
    ax.plot(point_f[0], point_f[1], color='red', marker='o')
    ax.plot(robot_xy[0], robot_xy[1], color='red', marker='o')
    print(ret_v, ret_w)
    plt.show()

    return ret_v, ret_w

if __name__ == '__main__':
    node_i = np.array([-1, 3, np.pi/2])
    point_f = np.array([1, 5])
    out = robot_controller(node_i, point_f)