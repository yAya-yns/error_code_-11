import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def trajectory_rollout(vel : float, rot_vel : float, curr_pose : np.ndarray) -> np.ndarray:
    # Given your chosen velocities determine the trajectory of the robot for your gtimestep
    # The returned trajectory should be a series of points to check for collisions
    #RECIEVE WORLD POINTS
    assert curr_pose.shape == (3,), curr_pose.shape
    assert type(vel) == type(rot_vel) and type(vel) in [np.float64, float], type(vel)
    assert vel > 0
    num_substeps = 10
    timestep = 0.1

    robot_xy, theta = curr_pose[0:2], curr_pose[2]
    robot_direction = np.array([np.cos(theta), np.sin(theta)])  # vector representing forward direction of robot

    if rot_vel == 0:
        temp = np.linspace([1, 1], [num_substeps, num_substeps], num_substeps).T - 1
        ret = np.zeros((3, num_substeps))
        print(temp)
        ret[0:2, :] = temp * vel * robot_direction.reshape(2, 1) * timestep + robot_xy.reshape(2, 1) # broadcast
        ret[2, :] = theta

        print(ret)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(ret[0, :], ret[1, :], color='green')
        plt.show()
        return ret

    on_right = np.sign(rot_vel) * -1
    rot_vel = np.abs(rot_vel)
    robot_right_direction = np.array([np.sin(theta), -np.cos(theta)])   # vector representing right direction of robot

    traj = np.zeros((3, num_substeps))

    radius = vel / rot_vel

    circle_centre = radius * on_right * robot_right_direction + robot_xy

    start_angle = theta + np.pi/2 * on_right
    delta_angle = rot_vel * timestep

    for i in range(num_substeps):
        ang = start_angle - i * delta_angle * on_right
        traj[2, i] = theta - i * delta_angle * on_right
        # print(ang)
        pos = (circle_centre + radius * np.array([np.cos(ang), np.sin(ang)]))
        traj[0:2, i] = pos

    fig = plt.figure()
    ax = fig.add_subplot()
    circle1 = patches.Circle(circle_centre, radius=radius)
    ax.add_patch(circle1)
    ax.axis('equal')
    # ax.plot(turning_point[0], turning_point[1], color='red', marker='o')
    # ax.plot(midpoint[0], midpoint[1], color='green', marker='o')
    ax.plot(circle_centre[0], circle_centre[1], color='red', marker='o')
    # ax.plot(point_f[0], point_f[1], color='red', marker='o')
    ax.plot(robot_xy[0], robot_xy[1], color='red', marker='o')
    ax.plot(traj[0, :], traj[1, :], color='green')
    plt.show()
    
    return traj

if __name__ == '__main__':
    node_i = np.array([-1, 3, np.pi/3])
    # out = trajectory_rollout(1.0, -0.0, node_i)
    out = trajectory_rollout(1.0, -0.4, node_i)