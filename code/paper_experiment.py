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



class LaserTrackerEnv:

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


        dirname = os.path.join(os.path.dirname(__file__), 'transformer_cell', 'Objects')
        #load Objects__________________________________________________
        plane_urdf = os.path.join(dirname,'plane.urdf')
        p.loadURDF(plane_urdf, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True, globalScaling=0.22)


        fofa_path = os.path.join(dirname, 'InRePro', 'InRePro.urdf')
        p.loadURDF(fofa_path, [0,0,0], p.getQuaternionFromEuler([np.pi/2,0,0]),
                   useFixedBase=True, globalScaling=0.001)

        ponticon_path = os.path.join(dirname,'ponticon', 'ponticon.urdf')
        p.loadURDF(ponticon_path, [0,0,0], p.getQuaternionFromEuler([np.pi/2,0,0]),
                   useFixedBase=True, globalScaling=0.001)

        dmg_path = os.path.join(dirname,'DMG', 'DMG.urdf')
        p.loadURDF(dmg_path, [0,0,0], p.getQuaternionFromEuler([np.pi/2,0,0]),
                   useFixedBase=True, globalScaling=0.001)

        mm_path = os.path.join(dirname,  'modules', 'milling_module.urdf')
        p.loadURDF(mm_path, [-0.6,-4.6,0.95], p.getQuaternionFromEuler([0,0,np.pi/2]),
                   useFixedBase=True, globalScaling=0.001)

        table_path = os.path.join(dirname,  'modules', 'table.urdf')
        p.loadURDF(table_path, [-0.6, -4.6, 0.95], p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   useFixedBase=True, globalScaling=0.001)

        #load Robots__________________________________________________
        dirname = os.path.join(os.path.dirname(__file__), 'transformer_cell')
        comau_urdf = os.path.join(dirname,'robot_descriptions', 'comau_nj290_robot.urdf')
        start_pos = np.array([-3.6353162, -0.6, 0])
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot = pi.RobotBase(comau_urdf, start_pos, start_orientation)
        start_pos2 = np.array([-0.12, -2.59, 0])
        start_orientation2 = p.getQuaternionFromEuler([0, 0, np.pi])
        self.second_robot = pi.RobotBase(comau_urdf, start_pos2, start_orientation2)


        self.marker = OpticalMarker(os.path.join(dirname, 'marker.urdf'), [0, 0, 0], [0, 0, 0, 1],
                                 [[0.1, 0.1, 0.1],
                                  [0.1, -0.1, 0.1],
                                  [-0.1, 0.1, 0.1],
                                  [-0.1, -0.1, 0.1]])
        self.marker.couple(self.robot)

        self.second_endeffector = OpticalMarker(os.path.join(dirname, 'marker.urdf'), [0, 0, 0], [0, 0, 0, 1],
                                    [[0.1, 0.1, 0.1],
                                    [0.1, -0.1, 0.1],
                                    [-0.1, 0.1, 0.1],
                                    [-0.1, -0.1, 0.1]])

        self.second_endeffector.couple(self.second_robot, [0, 0, 0], [0, 0, 0, 1])

    def reset(self):
        """Resets the environment to the initial state
        """
        self.robot.reset_robot(*self.robot.get_world_state())
        self.second_robot.reset_robot(*self.second_robot.get_world_state())

    def run_simulation(self,particles):
        """ Runs the environment  given a set of particles describing the laser tracker positions
            and marker poses. It then returns the cost for each timestep and particle.
        """
        costs = []
        self.reset()

        # implement the logic of what happens during the simulation
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


    env = CustomEnv()

    n_particles = 100
    initial_population = np.random.rand(
        9, n_particles)*0.3*np.pi


    np.save("initial_states.npy",initial_population)

    start_velocities = np.random.randn(
        9, n_particles) * 0.5

    with console.status("Finding optimal camera placements"):
        particle_log, obj_log = particle_optimizer_log(objective_function, initial_population,
                                                        start_velocities, max_iter=100)
    env.close()
    # save particle log and objective log
    np.save("particle_log.npy", particle_log)
    np.save("objective_log.npy", obj_log)




if __name__ == "__main__":
    env = LaserTrackerEnv(rendering=True)

    while True:
        p.stepSimulation()