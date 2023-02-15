from l2_planning import *


def main():  
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"
    goal_point = np.array([[10], [-20]]) #m WORLD POINTS
    stopping_dist = 0.5 #m
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist, display_window=False)
    

    path_planner.nodes.append(Node(np.array([[1],[1],[0]]), 0, np.sqrt(2))) # id = 1, parent 0
    path_planner.nodes[0].children_ids.append(1)
    path_planner.nodes.append(Node(np.array([[0],[1],[0]]), 0, 1)) # id = 2, parent 0
    path_planner.nodes[0].children_ids.append(2)
    path_planner.nodes.append(Node(np.array([[1],[0],[0]]), 0, 1)) # id = 3, parent 0
    path_planner.nodes[0].children_ids.append(3)


    # test 1
    trajectory_from_0_1 = np.hstack([path_planner.nodes[0].point, path_planner.nodes[1].point])
    np.testing.assert_almost_equal(np.sqrt(2), path_planner.cost_to_come(trajectory_from_0_1, rot_cost_coefficient=0))

    # test 2
    trajectory_from_0_2 = np.hstack([path_planner.nodes[0].point, path_planner.nodes[2].point])
    np.testing.assert_almost_equal(1, path_planner.cost_to_come(trajectory_from_0_2, rot_cost_coefficient=0))

    # test 3
    trajectory_from_0_3 = np.hstack([path_planner.nodes[0].point, path_planner.nodes[3].point])
    np.testing.assert_almost_equal(1, path_planner.cost_to_come(trajectory_from_0_3, rot_cost_coefficient=0))

    # # test 4
    trajectory_from_0_1_2_3 = np.hstack([path_planner.nodes[0].point,
                                          path_planner.nodes[1].point,
                                          path_planner.nodes[2].point,
                                          path_planner.nodes[3].point])
    np.testing.assert_almost_equal((1 + 2 * np.sqrt(2)), path_planner.cost_to_come(trajectory_from_0_1_2_3, rot_cost_coefficient=0))
    
    
    print("passed all tests")
    return True

if __name__ == '__main__':
    main()
