from l2_planning import *


def main():
    map_filename = "willowgarageworld_05res.png"
    # map_filename = "myhal.png"
    # map_setings_filename = "myhal.yaml"
    map_setings_filename = "willowgarageworld_05res.yaml"
    goal_point = np.array([[10], [-20]]) #m WORLD POINTS
    stopping_dist = 0.5 #m
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist, display_window=False)
    

    # test 1
    path_planner.nodes.append(Node(np.array([[1],[1],[0]]), 0, np.sqrt(2))) # id = 1, parent 0
    path_planner.nodes[0].children_ids.append(1)
    path_planner.nodes.append(Node(np.array([[0],[1],[0]]), 0, 1)) # id = 2, parent 0
    path_planner.nodes[0].children_ids.append(2)
    path_planner.nodes.append(Node(np.array([[1],[0],[0]]), 0, 1)) # id = 3, parent 0
    path_planner.nodes[0].children_ids.append(3)

    path_planner.nodes[0].cost = 10
    path_planner.update_children(0)

    # print(path_planner.nodes[0].cost)
    # print(path_planner.nodes[1].cost)
    # print(path_planner.nodes[2].cost)
    # print(path_planner.nodes[3].cost)

    np.testing.assert_almost_equal(10, path_planner.nodes[0].cost)
    np.testing.assert_almost_equal(10 + np.sqrt(2), path_planner.nodes[1].cost)
    np.testing.assert_almost_equal(11, path_planner.nodes[2].cost)
    np.testing.assert_almost_equal(11, path_planner.nodes[3].cost)


    # TODO: more chained nodes together
    
    print("passed all tests")
    return True

if __name__ == '__main__':
    main()