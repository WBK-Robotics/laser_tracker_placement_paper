import os

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_industrial as pi

from copy import deepcopy
from optical_marker import OpticalMarker
from particle_optimizer import particle_optimizer, particle_optimizer_log
from rich.console import Console


camera_radius = 2.5


def quat_from_direction(direction: np.ndarray) -> np.ndarray:
    """
    Returns a quaternion that represents a coordinate system with its z-axis aligned with the given direction vector.

    Args:
        direction: A 3D direction vector.

    Returns:
        A quaternion representation of the desired coordinate system.
    """
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # If the direction is (0, 0, z), set the quaternion to (0, 0, 0, 1)
    if direction[0] == 0 and direction[1] == 0:
        return np.array([0, 0, 0, 1])

    # Calculate the rotation axis and angle
    rotation_axis = np.cross(np.array([0, 0, 1]), direction)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(direction[2])

    # Convert the rotation axis and angle to a quaternion
    s = np.sin(rotation_angle / 2)
    c = np.cos(rotation_angle / 2)
    quaternion = np.array(
        [s * rotation_axis[0], s * rotation_axis[1], s * rotation_axis[2], c])

    return quaternion


def align_with_center(particles_5d, path_center):
    """Aligns 5d particles with the center of the path

    Args:
        particles_5d ([type]): The 5d particles
        path_center ([type]): The center of the path

    Returns:
        [type]: The particles with their orientations aligned with the center of the path
    """
    center_aligned_particles = deepcopy(particles_5d)
    number_of_cameras = int(len(particles_5d)/5)
    for i in range(len(particles_5d[0])):
        for j in range(number_of_cameras):

            angles = particles_5d[j*5:j*5+2, i]
            position = spherical_to_cartesian(
                camera_radius, angles[0], angles[1])
            direction = path_center-position

            quaternion = quat_from_direction(direction)
            center_aligned_particles[j*5+2:j*5+5,
                                     i] = p.getEulerFromQuaternion(quaternion)
    return center_aligned_particles


def dummy_objective_function(particles):

    return np.zeros(len(particles[0]))


def spherical_to_cartesian(radius, phi, theta):
    cartesian_coordinates = np.zeros(3)
    cartesian_coordinates[0] = radius * \
        np.cos(phi)*np.sin(theta)
    cartesian_coordinates[1] = radius * \
        np.sin(phi)*np.sin(theta)
    cartesian_coordinates[2] = radius*np.cos(theta)
    return cartesian_coordinates


def spherical_particles_to_cartesian_particles(spherical_particles, radius):

    number_of_cameras = int(len(spherical_particles)/5)
    # divide the particles into list of camera states
    camera_particles = np.split(spherical_particles, number_of_cameras, axis=0)
    cartesian_particles = np.zeros(
        (6*number_of_cameras, len(spherical_particles[0])))

    for i in range(number_of_cameras):
        # insert the orientations into the cartesian camera states
        cartesian_particles[i*6+3:i*6+6] = camera_particles[i][2:5]

        # compute the cartesian coordinates of the cameras and insert them
        for j in range(len(spherical_particles[0])):
            cartesian_particles[i*6:i*6+3, j] = spherical_to_cartesian(
                radius, camera_particles[i][0, j], camera_particles[i][1, j])

    return cartesian_particles


def compute_extrinsic_matrix(camera_state):
    """Computes the extrinsic camera matrix from the camera state.

    Args:
        camera_state (list): A list containing the camera position and orientation.

    Returns:
        np.array: The extrinsic camera matrix.
    """

    extrinsic_matrix = np.zeros((4, 4))
    camera_pos = camera_state[0]
    camera_rot = p.getQuaternionFromEuler(camera_state[1])
    camera_rot_mat = p.getMatrixFromQuaternion(camera_rot)
    camera_rot_mat = np.array(camera_rot_mat).reshape(3, 3)
    extrinsic_matrix[0:3, 0:3] = camera_rot_mat
    extrinsic_matrix[0:3, 3] = camera_pos

    return extrinsic_matrix


def reshape(list1, list2):
    """ Resaoes the second list to the shape of the first list of lists

    Args:
        list1 (list): The list of lists
        list2 (list): The list to be reshaped

        Returns:
            list: The reshaped list
    """
    last = 0
    res = []
    for ele in list1:
        res.append(list2[last: last + len(ele)])
        last += len(ele)

    return res


def get_fov_from_intrinsic_matrix(intrinsic_matrix):
    """Function which calculates the field of view from the intrinsic matrix

    Args:
        intrinsic_matrix (list): The intrinsic matrix as a list of 9 values

    Returns:
        float: The field of view in radians
    """
    return 2*np.arctan(0.5/intrinsic_matrix[0])


class CustomEnv:
    radius = 0.6
    cylinder_center = np.array([-1, 0, 0.5])

    def __init__(self,  rendering=False, steps=20):
        if rendering is False:
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI, options='--background_color_red=1 ' +
                                            '--background_color_green=1 ' +
                                            '--background_color_blue=1')
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSolverIterations=1000)
        self._time_step_length = 0.1
        p.setTimeStep(self._time_step_length)

        self.number_of_steps = steps
        self.markers, self.paths = self._setup_cell()
        self._populate_cell()

        self.stereo_camera_states = []
        self.intrinsic_matrices = []
        self.min_singular_value = []

        self._populate_cell()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def close(self):
        """disconnects the environment from the physics server
        """
        p.disconnect(self.physics_client)

    def _compute_visibility(self, orientation=True, orientation_only=False):
        """ Function that computes the visibility of each marker for each stereo camera set."""
        flattened_states = [
            item for sublist in self.stereo_camera_states for item in sublist]
        quaternion_states = [[state[0], p.getQuaternionFromEuler(
            state[1])] for state in flattened_states]

        if orientation:
            fovs = [get_fov_from_intrinsic_matrix(intrinsic_matrix)
                    for sublist in self.intrinsic_matrices for intrinsic_matrix in sublist]
        else:
            fovs = None

        visibility = []
        for marker in self.markers:
            flattened_visibility = marker.compute_visibility(
                quaternion_states, fovs, orientation_only)
            visibility.append(
                reshape(self.stereo_camera_states, flattened_visibility))
        return visibility

    def _populate_cell(self):
        dirname = os.path.dirname(__file__)
        urdf_file = os.path.join(dirname, 'ground.urdf')
        p.loadURDF(urdf_file, [0, 0, 0], useFixedBase=True)

        cylinder_id = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.5, height=1)
        cylinder_color = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.5, length=1, rgbaColor=[196/255, 208/255, 207/255, 1], specularColor=[0, 0, 0])
        p.createMultiBody(0, cylinder_id, cylinder_color,
                          self.cylinder_center, p.getQuaternionFromEuler([np.pi/2, 0, 0]))

    def _setup_cell(self):
        """ Function that sets up the markers in the cell and the paths they follow

        Returns:
            list(OpticalMarker): A list of the markers as OpticalMarker objects
            list(ToolPath): A list of the paths as ToolPath objects
        """
        # load a robot
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = pi.RobotBase(
            "kuka_iiwa/model.urdf", [0, -0.3, 0], [0, 0, 0, 1])
        self.second_robot = pi.RobotBase(
            "kuka_iiwa/model.urdf", [0, 0.3, 0], [0, 0, 0, 1])

        for i in range(7):
            p.changeVisualShape(
                self.robot.urdf, -1+i, rgbaColor=[196/255, 208/255, 207/255, 1])
            p.changeVisualShape(
                self.second_robot.urdf, -1+i, rgbaColor=[196/255, 208/255, 207/255, 1])

        # loard a marker
        dirname = os.path.dirname(__file__)
        urdf_file = os.path.join(dirname, 'marker.urdf')
        marker = OpticalMarker(
            urdf_file, [0.2, 0.5, 0], [0, 0, 0, 1],
            [np.array([0, 0, -0.2]), np.array([0, 0.04, 0])])
        second_marker = OpticalMarker(
            urdf_file, [0.2, 0.7, 0], [0, 0, 0, 1],
            [np.array([0, 0, -0.2]), np.array([0, 0.04, 0])])
        marker.couple(self.robot)
        second_marker.couple(self.second_robot)

        # parameters of the dynamic paths
        # .........................................................................
        number_of_points = self.number_of_steps
        point_mutliplier = int(number_of_points/10)
        side_length = 0.5
        side_offset = 0.3
        # .........................................................................

        # setup a dynamic path for the first marker
        start_pos = self.cylinder_center + \
            np.array([self.radius, -side_offset, 0])
        turning_pos = self.cylinder_center + \
            np.array([self.radius*np.cos(0.5), -
                     side_offset, self.radius*np.sin(0.5)])
        end_pos = turning_pos + np.array([0, side_length, 0])

        upward = pi.circular_interpolation(
            start_pos, turning_pos, 0.5, 3*point_mutliplier, 1, False)
        sideward = pi.linear_interpolation(
            turning_pos, end_pos, 7*point_mutliplier)
        upward.append(sideward)

        upward.draw(color=[178/255, 55/255, 44/255])

        quaternion_orientation = p.getQuaternionFromEuler([0, -0.5*np.pi, 0])
        orientation = np.array([[quaternion_orientation[0]]*len(upward.positions[0]),
                                [quaternion_orientation[1]] *
                                len(upward.positions[0]),
                                [quaternion_orientation[2]] *
                                len(upward.positions[0]),
                                [quaternion_orientation[3]]*len(upward.positions[0])])
        upward.orientations = orientation

        # setup a dynamic path for the second marker
        start_pos = self.cylinder_center + \
            np.array([self.radius*np.cos(0.5),
                     side_offset, self.radius*np.sin(0.5)])
        turning_pos = self.cylinder_center + \
            np.array([self.radius, side_offset, 0])
        end_pos = turning_pos + np.array([0, -side_length, 0])

        downward = pi.circular_interpolation(
            start_pos, turning_pos, 0.5, 3*point_mutliplier, 1, True)
        sideward = pi.linear_interpolation(
            turning_pos, end_pos, 7*point_mutliplier)
        downward.append(sideward)

        downward.orientations = orientation

        downward.draw(color=[178/255, 55/255, 44/255])

        return [marker, second_marker], [upward, downward]

    def reset(self):
        """Resets the environment to the initial state
        """
        self.robot.reset_robot(*self.robot.get_world_state())
        self.second_robot.reset_robot(*self.second_robot.get_world_state())

    def _costs(self):
        """Returns a cost matrix for each camera and marker.
        """

        # list of the visibility of each marker for each stereo camera system
        visibility = self._compute_visibility(True)

        camera_visibilities = [np.array(
            [np.sum(vis) for vis in camera_visibility]) for camera_visibility in visibility]

        return -1*np.sum(camera_visibilities, axis=0) - [0.1*value for value in self.min_singular_value]

    def run_simulation(self):
        """ Runs the environment in which the markers follow the paths and the cameras observe them.
            It then returns the cost of each timestep.
        """
        costs = []
        self.reset()

        for _ in range(30):
            [marker.set_tool_pose(*path.get_start_pose())
             for marker, path in zip(self.markers, self.paths)]

        for marker_positions in zip(*self.paths):
            for i in range(len(self.markers)):
                self.markers[i].set_tool_pose(
                    marker_positions[i][0], marker_positions[i][1])
            for _ in range(30):
                p.stepSimulation()

            costs.append(self._costs())
        return costs

    def set_timestep_length(self, time_step_length):
        """Sets the timestep between subsequent step function calls.
        Args:
            time_step_length ([type]): The time step when calling the step function
        """
        self._time_step_length = time_step_length
        p.setTimeStep(time_step_length)

    def get_timestep_length(self):
        """Get the timestep between subsequent step function calls.
        Returns:
            [type]: The time step
        """
        return self._time_step_length

    def configure_cameras(self, camera_states, intrinsic_matrices):
        """Configures the cameras in the environment and
           caclulating the proection error of the camera setups.
        """
        if len(intrinsic_matrices) != len(camera_states):
            raise ValueError(
                "Number of cameras ("+str(len(camera_states))+") and number of FOVs ("+str(len(intrinsic_matrices))+") must be the same")

        self.stereo_camera_states = camera_states

        self.intrinsic_matrices = intrinsic_matrices

        min_singular_value = []
        for i, stereo_state in enumerate(self.stereo_camera_states):
            camera_matrices = []
            for j, state in enumerate(stereo_state):
                extrinsic_matrix = compute_extrinsic_matrix(state)

                intrinsic_matrix = self.intrinsic_matrices[i][j]

                intrinsic_matrix = np.array(intrinsic_matrix).reshape(3, 3)

                # add zero column to intrinsic matrix to make it 3x4 for multiplication
                padded_intrinsics = np.hstack(
                    (intrinsic_matrix, np.zeros((3, 1))))
                camera_matrices.append(
                    np.matmul(padded_intrinsics, extrinsic_matrix))

            # calculate the error component describing the overal stereo camera setup
            # \sigma_{min}(A+\Delta A)
            triangulation_matrix = np.vstack(camera_matrices)
            min_singular_value.append(np.linalg.svd(
                triangulation_matrix)[1][-1])

            self.min_singular_value = min_singular_value
        return min_singular_value


class CustomEnvNoOri(CustomEnv):

    def _costs(self):
        """Returns a cost matrix for each camera and marker.
        """

        # list of the visibility of each marker for each stereo camera system
        visibility = self._compute_visibility(False)

        camera_visibilities = [np.array(
            [np.sum(vis) for vis in camera_visibility]) for camera_visibility in visibility]

        return -1*np.sum(camera_visibilities, axis=0)


def get_5d_from_2d(states):
    """Converts a 2d array of 2d camera states to a 5d array of 5d camera states.
    Args:
        states (np.ndarray): 2d array of 2d camera states
    Returns:
        np.ndarray: 5d array of 5d camera states
    """
    states_5d = deepcopy(states)
    for i in range(int(len(states_5d)/2)):
        states_5d = np.insert(
            states_5d, 2*i+2+3*i, np.zeros((3, len(states_5d[0]))), axis=0)
    return states_5d


if __name__ == "__main__":
    # run optimization
    # ----------------------------------------------------------------------------------------------
    console = Console()

    # set up first environment with only orientation



    def run_experiment():

        def objective_function(particles):

            # divide the particles into list of camera states
            camera_particles = np.split(particles, len(particles)/6, axis=0)

            camera_states = []
            for i in range(len(camera_particles[0][0])):
                camera_state = [[state[:3, i], state[3:, i]]
                                for state in camera_particles]
                camera_states.append(camera_state)

            focal_length = 0.1
            intrinsic_matrix = [0.0]*9
            intrinsic_matrix[0*3+0] = focal_length
            intrinsic_matrix[1*3+1] = focal_length
            intrinsic_matrix[2*3+2] = float(1)
            intrinsic_matrix = tuple(intrinsic_matrix)

            env.configure_cameras(
                camera_states, [[intrinsic_matrix]*len(camera_particles)]*len(particles[0]))
            costs = env.run_simulation()
            costs = np.sum(costs, axis=0)

            # penalize the cameras that are below the ground
            for i, stereo_cameras in enumerate(camera_states):
                for camera in stereo_cameras:
                    if camera[0][2] < 0:
                        costs[i] += 1000
            return costs

        def spherical_stereo_objective_function(particles):
            radius = camera_radius

            cartesian_camera_states = spherical_particles_to_cartesian_particles(
                particles, radius)
            all_objectives = objective_function(cartesian_camera_states)

            return all_objectives

        def no_orientation_spherical_stereo_objective_function(particles):

            # add three columns of zeros every two rows to the particles
            for i in range(int(len(particles)/2)):
                particles = np.insert(
                    particles, 2*i+2+3*i, np.zeros((3, len(particles[0]))), axis=0)

            return spherical_stereo_objective_function(particles)

        env = CustomEnvNoOri()

        n_particles = 100
        initial_population = np.random.rand(
            9, n_particles)*0.3*np.pi


        np.save("initial_states.npy", spherical_particles_to_cartesian_particles(
            initial_population, camera_radius))

        start_velocities = np.random.randn(
            9, n_particles) * 0.5

        with console.status("Finding optimal camera placements"):
            particle_log, obj_log = particle_optimizer_log(no_orientation_spherical_stereo_objective_function, initial_population,
                                                           start_velocities, max_iter=100)
        env.close()



    for i in [2, 3, 4, 5, 6, 7, 8]:
        print("current camera numbers: "+str(i))
        run_experiment(i)