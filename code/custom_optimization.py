""" This script offers a simple scaffolding to run your own laser tracker placement optimization for a single robot system.
For multi-robot system it can be extended as seen in the paper_experiment.py script.
By implementing every function in this script, you can run the optimization and later visualize the results. using the scripts
provided in the paper_evaluation.py and paper_visualization.py scripts.
"""

import os

import numpy as np
import pybullet as p
import pybullet_industrial as pi


from rich.console import Console

from optical_marker import ActiveMarker
from particle_optimizer import  particle_optimizer_log






class LaserTrackerEnv:

    def __init__(self,  rendering=False):
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

        self.robot, self.marker = self.setup_environment()

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def setup_environment():
        """Set up the cell environment."""

        # This function should set up the pybullet industrial environment, meaning all objects
        # including the robot.
        # This robot should be implemented as a pybullet industrial RobotBase object.
        # In this example return function it has to be named robot.


        # The marker object represents not only the marker but also the laser tracker.
        # It has to be initialized with some values but the exact values are later set by the particle optimizer.
        marker = ActiveMarker(os.path.join(os.path.dirname(__file__), 'marker.urdf'), [0, 0, 0], [0, 0, 0, 1],
                                    [[0.1, 0.1, 0.1],
                                    [0.1, -0.1, 0.1],
                                    [-0.1, 0.1, 0.1],
                                    [-0.1, -0.1, 0.1]],
                                    [[0, 0, 0, 1],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 1]],
                                    [[0.0, 0.0, 0.5],
                                    [0.0, 0.0, 0.6]],
                                    np.pi/4)

        # the maker has to be coupled to the frame of the robot on which the marker is to be placed
        # In this example this is link6 representing the endeffector of the robot
        marker.couple(robot,'link6')

        return robot, marker

    def reset(self):
        """Resets the environment to the initial state
        """
        self.robot.reset_robot(*self.robot.get_world_state())

        # it is good practice to first reset the robot to a known and predefined joint position.
        # This is because a different starting joint state might lead to different configurations
        # over the trajectory and therefore different visibility conditions.
        # A example code snippet is shown below
        '''
        for _ in range(20):
            joint_angles ={
                'q1': -0.1910351778338952,
                'q2': 0.6219868844708127,
                'q3': -2.058245174714272,
                'q4': -0.21271062205029018,
                'q5': -2.1816615644867423,
                'q6': -0.09411935083551597
            }
            self.robot.set_joint_position(joint_angles)
            self.second_robot.set_joint_position(joint_angles)
            for _ in range(50):
                p.stepSimulation()
        '''

    def run_simulation(self,particles):
        """ Runs the environment  given a set of particles describing the laser tracker positions
            and marker poses. It then returns the cost for each timestep and particle.
        """
        visibility_index = []
        distance_index = []
        self.reset()

        self.marker.set_optical_system_parameters(particles.T)

        # Here you should implement the code governing how the robot moves during the simulation
        # This could be explicitely done by following a joint or task space trajectory
        # or by using a controller.
        # For a trajectory example you can look at the paper_experiment.py script
        # It is important that at each time step you want the marker to be visible you call:
        # visibility_index.append(self.marker.compute_visibility())
        # distance_index.append(self.marker.compute_distance())
        # where the visibility_index and distance_index are lists that store the visibility and distance


        visibility_index = np.array(visibility_index)
        distance_index = np.array(distance_index)

        visibility_index = visibility_index.T
        distance_index = distance_index
        return visibility_index, distance_index

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
    console = Console()
    def objective_function(particles):

        visibility_cost,distance_cost = env.run_simulation(particles)
        visibility_cost_sum = -1*np.sum(visibility_cost,axis=1)

        # cost that keeps the tracker as close as possible to the surface of the robot
        marker_positions = particles[3:6,:]
        deviation_cost = np.linalg.norm(marker_positions, axis=0)

        # cost that keeps the distance cost as low as possible so long as it is not below 1.2

        # go through all the values in the distance cost and replace them with 1e2 if they are below 1.2
        # bore summing over the trajectory
        distance_cost = np.where(distance_cost < 1.2, 1e2, distance_cost)

        distance_cost = np.sum(distance_cost,axis=0)


        #constraint costs

        #constraints keeping the height of the laser tracker betwen 0.8 and 2
        height_constraint = np.array([ 1e2 if x < 0.8 or x > 2 else 0 for x in particles[2,:]])


        #constraint keeping the marker position within a cylinder of given radius and height
        cylinder_radius = 0.5
        cylinder_height = 0.4

        radius_constraint = np.array([ 1e2 if np.linalg.norm(marker_positions[:2,i]) > cylinder_radius else 0 for i in range(marker_positions.shape[1])])
        height_constraint = np.array([ 1e2 if marker_positions[2,i] < 0 or marker_positions[2,i] > cylinder_height else 0 for i in range(marker_positions.shape[1])])



        costs = visibility_cost_sum+ distance_cost + deviation_cost+height_constraint + radius_constraint + height_constraint

        return costs


    env = LaserTrackerEnv()

    n_particles = 500

    # Define the size and shape of the initial population here:
    # Currently the initial population is a random distribution of 500 particles
    # with the tracker positions in the range of -3.5 to -0.5 and the marker positions
    # in the range of 0 to 0.6
    initial_tracker_positions = np.random.rand(3, n_particles)*3
    initial_tracker_positions[0] += -3.5
    initial_tracker_positions[1] += -3.5
    initial_marker_positions = np.random.rand(3, n_particles)*0.6
    initial_marker_orientations = np.random.rand(3, n_particles)*2*np.pi

    initial_population = np.vstack(
        [initial_tracker_positions, initial_marker_positions, initial_marker_orientations])

    np.save(os.path.join("results","initial_states.npy"),initial_population)

    # Define the size and shape of the initial velocities here:
    # Currently the initial velocities are a random distribution of 500 particles
    # with the tracker velocities in the range of 0 to 2 and the marker velocities
    # in the range of 0 to 0.01
    # Note that the tracker generally has to move more than the markers so its starting
    # velocities should be higher
    start_tracker_velocities = np.random.rand(3, n_particles)*2
    start_marker_velocities = np.random.rand(3, n_particles)*0.01
    start_orientations_velocities = np.random.rand(3, n_particles)*0.1

    start_velocities = np.vstack(
        [start_tracker_velocities, start_marker_velocities, start_orientations_velocities])

    with console.status("Finding optimal camera placements"):
        particle_log, obj_log = particle_optimizer_log(objective_function, initial_population,
                                                        start_velocities, max_iter=100)

    # save particle log and objective log
    np.save(os.path.join("results","particle_log.npy"), particle_log)
    np.save(os.path.join("results","objective_log.npy"), obj_log)




if __name__ == "__main__":
    run_experiment()

