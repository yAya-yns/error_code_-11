from l2_planning import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

def test_myhal_map(visualize=False):
	map_filename = "myhal.png"
	map_setings_filename = "myhal.yaml"
	goal_point = np.array([[10], [-20]]) #m WORLD POINTS
	stopping_dist = 0.5 #m
	path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist, display_window=False)

	x_range = np.arange(0,len(path_planner.occupancy_map[0]))  # col
	y_range = np.arange(0,len(path_planner.occupancy_map))  # row
	robot_radius_in_cell = np.ceil(path_planner.robot_radius / path_planner.map_settings_dict['resolution'])
	
	# Test 1
	robot_positions = np.array([[10,10],[10,35]])
	if visualize:	
		map = np.array(path_planner.occupancy_map, copy=True)
		for position in robot_positions:
			robot_occupancy = (x_range[np.newaxis,:]-position[1])**2 + (y_range[:,np.newaxis]-position[0])**2 < robot_radius_in_cell**2
			map[robot_occupancy] = 0.5
		plt.figure()
		plt.pcolormesh(x_range, y_range, map)
		plt.colorbar()
		plt.show()

	assert path_planner.is_collide(robot_positions) == True

	# Test 2
	robot_positions = np.array([[10,10],[30,10]])
	if visualize:	
		map = np.array(path_planner.occupancy_map, copy=True)
		for position in robot_positions:
			robot_occupancy = (x_range[np.newaxis,:]-position[1])**2 + (y_range[:,np.newaxis]-position[0])**2 < robot_radius_in_cell**2
			map[robot_occupancy] = 0.5
		plt.figure()
		plt.pcolormesh(x_range, y_range, map)
		plt.colorbar()
		plt.show()

	assert path_planner.is_collide(robot_positions) == False

	return True


def main():
	view_map = False
	if view_map:
		map_filename = "myhal.png"
		map_setings_filename = "myhal.yaml"
		# map_filename = "willowgarageworld_05res.png"
		# map_setings_filename = "willowgarageworld_05res.yaml"
		goal_point = np.array([[10], [-20]]) #m WORLD POINTS
		stopping_dist = 0.5 #m
		path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist, display_window=False)

		x_range = np.arange(0,len(path_planner.occupancy_map[0]))  # col
		y_range = np.arange(0,len(path_planner.occupancy_map))  # row
		map = np.array(path_planner.occupancy_map, copy=True)
		plt.figure()
		plt.pcolormesh(x_range, y_range, map)
		plt.colorbar()
		plt.show()
	
	assert True == test_myhal_map(visualize=False)
	print("passed all tests")
	return True

	



if __name__ == '__main__':
	main()