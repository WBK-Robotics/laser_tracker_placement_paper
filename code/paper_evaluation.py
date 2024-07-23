from paper_experiment import CustomEnv, get_fov_from_intrinsic_matrix
import numpy as np
import matplotlib.pyplot as plt


kit_deep_blue = "#144466"
kit_green = "#009682"
kit_yellow = "#EEB70D"
kit_red = "#B2372C"


class DirectVisEnv(CustomEnv):

    def _costs(self):
        """Returns a cost matrix for each camera and marker.
        """

        # list of the visibility of each marker for each stereo camera system
        visibility = self._compute_visibility(False)

        camera_visibilities = [np.array(
            [np.sum(vis) for vis in camera_visibility]) for camera_visibility in visibility]

        return np.sum(camera_visibilities, axis=0)/(4*len(visibility[0][0]))


class OrientationVisEnv(CustomEnv):

    def _costs(self):
        """Returns a cost matrix for each camera and marker.
        """

        # list of the visibility of each marker for each stereo camera system
        visibility = self._compute_visibility(True, True)

        camera_visibilities = [np.array(
            np.max(np.abs(camera_visibility), axis=1)*np.pi) for camera_visibility in visibility]

        return np.max(
            np.max(np.abs(camera_visibilities), axis=0), axis=1)


if __name__ == "__main__":
    camera_radius = 2.5
    number_of_cameras = 8
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

    # evalute the triangulation error of the initial camera configurations
    # ----------------------------------------------------------------------------------------------

    env = CustomEnv(False)
    optimal_triangulation_error = env.configure_cameras([optimal_camera_configs], [[
        intrinsic_matrix]*len(optimal_camera_configs)])
    initial_triangulation_error = env.configure_cameras(initial_camera_configs, [[
        intrinsic_matrix]*len(initial_camera_configs[0])]*len(initial_camera_configs))

    plt.hist(initial_triangulation_error, color=kit_deep_blue)
    # highlight the optimal triangulation error
    plt.axvline(x=optimal_triangulation_error, color=kit_red)
    plt.title("Triangulation error of the camera configurations")
    plt.xlabel("Triangulation error")
    plt.ylabel("Counts")
    plt.show()
    env.close()

    # evaluate the visibility of the initial camera configurations
    # ----------------------------------------------------------------------------------------------

    direction_vis_env = DirectVisEnv(False)
    direction_vis_env.configure_cameras(initial_camera_configs, [[
        intrinsic_matrix]*len(initial_camera_configs[0])]*len(initial_camera_configs))

    initial_visibility = direction_vis_env.run_simulation()

    direction_vis_env.configure_cameras([optimal_camera_configs], [[
        intrinsic_matrix]*len(optimal_camera_configs)])

    optimal_visibility = direction_vis_env.run_simulation()

    plt.plot(initial_visibility, kit_deep_blue)
    plt.plot(optimal_visibility, kit_red)
    plt.title("Visibility of the camera configurations over the trajectory")
    plt.xlabel("Robot Trajectory steps")
    plt.ylabel("Visibility")
    plt.show()

    print(len(optimal_visibility))
    plt.hist(np.sum(initial_visibility, axis=0) /
             len(optimal_visibility), color=kit_deep_blue)
    plt.axvline(x=np.sum(optimal_visibility) /
                len(optimal_visibility), color=kit_red)
    plt.title("Total visibility of the camera configurations")
    plt.xlabel("Visibility")
    plt.ylabel("Counts")
    plt.show()
    direction_vis_env.close()


# Evalute wheter the markers are always in the field of view of the cameras
# ----------------------------------------------------------------------------------------------

    orientation_vis_env = OrientationVisEnv(False)
    orientation_vis_env.configure_cameras(initial_camera_configs, [[
        intrinsic_matrix]*len(initial_camera_configs[0])]*len(initial_camera_configs))

    initial_angle = orientation_vis_env.run_simulation()

    orientation_vis_env.configure_cameras([optimal_camera_configs], [[
        intrinsic_matrix]*len(optimal_camera_configs)])

    optimal_angle = orientation_vis_env.run_simulation()

    # highlight the fov range on the y axis
    fov = get_fov_from_intrinsic_matrix(intrinsic_matrix)
    plt.axhline(y=0.5*fov, color='g')
    plt.axhline(y=-0.5*fov, color='g')

    plt.plot(initial_angle, 'b')
    plt.plot(optimal_angle, 'r')
    plt.title(
        " Maximum Angle between any camera and the line of sight to any marker")
    plt.xlabel("Robot Trajectory steps")
    plt.ylabel("angle [rad]")
    plt.show()

    orientation_vis_env.close()
