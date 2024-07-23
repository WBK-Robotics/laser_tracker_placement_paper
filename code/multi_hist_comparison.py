import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from paper_experiment import CustomEnv
from paper_evaluation import DirectVisEnv, OrientationVisEnv, get_fov_from_intrinsic_matrix

kit_deep_blue = "#144466"
kit_red = "#B2372C"

kit_green = "#009682"
kit_yellow = "#EEB70D"

light_blue = "#8EB1C7"
pink = "#FFC6D9"
purble = "#A288A6"
shining_blue = "#70D6FF"

colors = [kit_deep_blue, kit_green, kit_yellow,
          light_blue, pink, purble, shining_blue]


def load_camera_data(number_of_cameras):
    optimal_states = np.load(
        "optimal_states_"+str(number_of_cameras)+"_cameras.npy")
    optimal_camera_states = np.split(
        optimal_states.flatten(), len(optimal_states.flatten())/6, axis=0)

    initial_states = np.load(
        "initial_states_"+str(number_of_cameras)+"_cameras.npy")
    initial_camera_states = np.split(
        initial_states, len(initial_states)/6, axis=0)

    focal_length = 0.1
    intrinsic_matrix = [0.0]*9
    intrinsic_matrix[0*3+0] = focal_length
    intrinsic_matrix[1*3+1] = focal_length
    intrinsic_matrix[2*3+2] = float(1)
    intrinsic_matrix = tuple(intrinsic_matrix)

    optimal_camera_configs = []
    for camera_state in optimal_camera_states:
        optimal_camera_configs.append([camera_state[:3], camera_state[3:6]])

    initial_camera_configs = []
    for i in range(len(initial_camera_states[0][0])):
        initial_camera_configs.append([
            [init_state[:3, i], init_state[3:6, i]] for init_state in initial_camera_states])

    return optimal_camera_configs, initial_camera_configs, intrinsic_matrix


def get_triangulation_data(number_of_cameras):
    optimal_camera_configs, initial_camera_configs, intrinsic_matrix = load_camera_data(
        number_of_cameras)

    env = CustomEnv(False)
    optimal_triangulation_error = env.configure_cameras([optimal_camera_configs], [[
        intrinsic_matrix]*len(optimal_camera_configs)])
    initial_triangulation_error = env.configure_cameras(initial_camera_configs, [[
        intrinsic_matrix]*len(initial_camera_configs[0])]*len(initial_camera_configs))

    env.close()
    return optimal_triangulation_error, initial_triangulation_error


def get_visibility_data(number_of_cameras):
    optimal_camera_configs, initial_camera_configs, intrinsic_matrix = load_camera_data(
        number_of_cameras)

    direction_vis_env = DirectVisEnv(False)
    direction_vis_env.configure_cameras(initial_camera_configs, [[
        intrinsic_matrix]*len(initial_camera_configs[0])]*len(initial_camera_configs))

    initial_visibility = direction_vis_env.run_simulation()

    direction_vis_env.configure_cameras([optimal_camera_configs], [[
        intrinsic_matrix]*len(optimal_camera_configs)])

    optimal_visibility = direction_vis_env.run_simulation()

    initial_visibility = np.sum(
        initial_visibility, axis=0) / len(optimal_visibility)
    optimal_visibility = np.sum(
        optimal_visibility) / len(optimal_visibility)

    direction_vis_env.close()
    #print(initial_visibility, optimal_visibility)
    return optimal_visibility, initial_visibility


def get_fov_data(number_of_cameras):
    optimal_camera_configs, initial_camera_configs, intrinsic_matrix = load_camera_data(
        number_of_cameras)

    orientation_vis_env = OrientationVisEnv(False)
    orientation_vis_env.configure_cameras(initial_camera_configs, [[
        intrinsic_matrix]*len(initial_camera_configs[0])]*len(initial_camera_configs))

    initial_angle = orientation_vis_env.run_simulation()

    orientation_vis_env.configure_cameras([optimal_camera_configs], [[
        intrinsic_matrix]*len(optimal_camera_configs)])

    optimal_angle = orientation_vis_env.run_simulation()

    orientation_vis_env.close()

    return optimal_angle, initial_angle


font = {'family': 'normal',
        'size': 12}
matplotlib.rc('font', **font)


cameras = [2, 3, 4, 5, 6, 7, 8]
markers = ["o", "s", "D", "P", "^", "*", "x"]
color_palette = [kit_deep_blue, kit_deep_blue, kit_yellow,
                 kit_yellow, kit_green, kit_green, light_blue]
dfs = []
optimal_data_list = []
for i in cameras:
    print(i)
    plt.plot(get_fov_data(i)[0], label=str(i)+" cameras",
             color=color_palette[i-2], marker=markers[i-2])

optimal_camera_configs, initial_camera_configs, intrinsic_matrix = load_camera_data(
    3)
fov = get_fov_from_intrinsic_matrix(intrinsic_matrix)


plt.axhline(y=0.5*fov, color=kit_red)
plt.ylabel("Largest FOV angle [rad]")
plt.xlabel("Timesteps")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("fov_angles.svg", dpi=300)
