import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import yaml
from l2_planning import *
import math

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

path = np.array([[0, 2.34, 4.68, 7.0, 9.3, 11.5,  11.5, 11.2, 11.3,  11.4, 11.5, 11.4,  11.2,  11.3, 11.3, 11.4,  12.5,  13.2,  13.4,  14.2, 14.9 , 15.5,  17.7 , 19.9,  20.4, 21.7, 22.5,    24.0, 25.6, 27.7, 27.9, 28.3, 30.5, 31.7, 33.2, 31.7, 31.7, 31.6, 32.8,  33.1, 34.6,  35.2 ,  35.0,   35.3 ,  35.6, 36.64,  36.67,  36.7, 36.7, 38.0, 40.1, 41.7],
                [ 0, 0,    0,    0,   0,   -0.5, -1.5, -2.09, -2.6, -4.7,  -5.9,  -7.7,  -8.0, -9.5, -11.3,-12.2, -13.6, -14.9, -17.0, -19.2, -20.4, -20.6, -20.4, -20.3, -20.3, -19.5, -18.9, -18.8, -17.8, -19.1,-19.6, -21.6, -22.4, -23.2, -25.0, -25.5, -27.3, -29.4, -31.3, -32.9, -33.5 ,-35.5, -37.8, -38.8, -40.3, -42.6, -43.0, -43.64, -44.0, -44.4, -43.5, -43.8],
                [-0.0, -0.0, -0.0, -0.0, 0.223476601140633, -1.5707963267948966, -1.1071487177940904, -1.373400766945016, -1.5232132235179132, -1.4876550949064553, -1.5152978215491797, -0.982793723247329, -1.5042281630190728, -1.5707963267948966, -1.460139105621001, -0.9048270894157867, -1.0768549578753155, -1.4758446204521403, -1.2220253232109897, -1.042721878368537, -0.3217505543966422, -0.09065988720074511, -0.04542327942157701, -0.0, 0.5516549825285469, 0.6435011087932844, 0.06656816377582381, 0.5585993153435624, 0.5543074962015513, -1.1902899496825317, -1.373400766945016, -0.348771003583907, -0.5880026035475675, -0.8760580505981934, -0.3217505543966422, -1.5707963267948966, -1.5232132235179132, -1.0074800653029286, -1.3854483767992019, -0.3805063771123649, -1.2793395323170296, -1.4840579881189115, -1.2793395323170296, -1.373400766945016, -1.142532173918242, -1.5707963267948966, -1.4940244355251187, -1.5707963267948966, -0.2984989315861793, -0.4048917862850834, -0.18534794999569476, 0.18534794999569476]])

np.save("willow_rrt_star.npy", path)

map_filename = "willowgarageworld_05res.png"
map_setings_filename = "willowgarageworld_05res.yaml"

goal_point = np.array([[42], [-44]])
occupancy_map = load_map(map_filename)
map_settings_dict = load_map_yaml(map_setings_filename)
path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, 0.5)
path[:2] = path_planner.point_to_cell(path[:2])

plt.imshow(occupancy_map)

for node in range(len(path[0])):
    # point = [[],[],[]]
    plt.plot([path[0][node]], [path[1][node]], marker='.', color='red', markersize = 3)

#get thetas
theta = []
for i in range(len(path[0])-1):
    # dx = path[0][i+1] - path[0][i]
    # dy = path[1][i+1] - path[1][i]
    # theta.append(math.atan2(dy, dx))
    ang1 = np.arctan2(path[1][i], path[0][i])
    ang2 = np.arctan2(path[1][i+1], path[0][i+1])
    theta.append((ang2 - ang1) % (2 * np.pi))

# print thetas
print(theta)

map_goal = np.array([[42], [-44]])
map_goal = path_planner.point_to_cell(map_goal)
plt.plot([map_goal[0]], [map_goal[1]], 'go')

# plot start in blue
start = np.array([[0], [0]])
start = path_planner.point_to_cell(start)
plt.plot([start[0]], [start[1]], 'bo')

# plot heading

plt.title("title")
plt.show()