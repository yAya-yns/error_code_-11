from l2_planning import *
import numpy as np

def test_plot():
    #Set map information

    # TEST 1
    map_filename = "myhal.png"
    map_setings_filename = "myhal.yaml"

    #robot information
    goal_point = np.array([[7], [2]]) #m WORLD POINTS
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    path_planner.nodes.append(Node(np.array([[1],[1],[0]]), 0, np.sqrt(2))) # id = 1, parent 0
    path_planner.nodes[0].children_ids.append(1)
    path_planner.nodes.append(Node(np.array([[0],[1],[0]]), 0, 1)) # id = 2, parent 0
    path_planner.nodes[0].children_ids.append(2)
    path_planner.nodes.append(Node(np.array([[1],[0],[0]]), 0, 1)) # id = 3, parent 0
    path_planner.nodes[0].children_ids.append(3)
    path_planner.nodes.append(Node(np.array([[2],[0.5],[0]]), 3, 1)) # id = 4, parent 3
    path_planner.nodes[3].children_ids.append(4)
    path_planner.nodes.append(Node(np.array([[2],[1.5],[0]]), 3, 1)) # id = 5, parent 4
    path_planner.nodes[4].children_ids.append(5)
    path_planner.nodes.append(Node(np.array([[3],[1],[0]]), 3, 1)) # id = 6, parent 4
    path_planner.nodes[4].children_ids.append(6)

    for parent in path_planner.nodes:
        for child in parent.children_ids:
            robot_traj, vel, rot_vel = path_planner.simulate_trajectory(parent.point.squeeze(), path_planner.nodes[child].point.squeeze()[:2])
            # change to map points
            robot_traj = path_planner.point_to_cell(robot_traj[:2])
            path_planner.trajectories.append(robot_traj)

    path_planner.plot_nodes()


    # TEST 2
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[20], [-20]]) #m WORLD POINTS
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    path_planner.nodes.append(Node(np.array([[10],[-10],[0]]), 0, np.sqrt(2))) # id = 1, parent 0
    path_planner.nodes[0].children_ids.append(1)
    path_planner.nodes.append(Node(np.array([[0],[-10],[0]]), 0, 1)) # id = 2, parent 0
    path_planner.nodes[0].children_ids.append(2)
    path_planner.nodes.append(Node(np.array([[10],[0],[0]]), 0, 1)) # id = 3, parent 0
    path_planner.nodes[0].children_ids.append(3)

    path_planner.plot_nodes()

if __name__ == '__main__':
    test_plot()