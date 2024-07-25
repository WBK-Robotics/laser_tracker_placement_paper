import os

import numpy as np
import pybullet as p
import pybullet_industrial as pi


from rich.console import Console

from optical_marker import ActiveMarker
from particle_optimizer import particle_optimizer, particle_optimizer_log






class LaserTrackerEnv:

    def __init__(self,  rendering=False, steps=20,path=None):
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

        if path is None:
            target_position = np.array([-2.4,-3,0.95])+np.array([-0.6,0.45,0.4])
            steps = 30
            self.path = pi.build_box_path(
                target_position, [0.5, 0.6], 0.1, [0, 0, 0, 1], steps)
            new_orientation = p.getQuaternionFromEuler([np.pi,0,0])
            #created a array with dimension steps x 4
            orientation_array = np.array([new_orientation for _ in range(steps)])
            self.path.orientations=orientation_array.T

        else:
            self.path = path


        self.path.draw()

        self.second_path = pi.linear_interpolation([-1.2, -0.6, 0.95], [-2.6, -3.4, 1.2], steps)


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
        p.loadURDF(mm_path, [-2.4,-3,0.95], p.getQuaternionFromEuler([0,0,np.pi/2]),
                   useFixedBase=True, globalScaling=0.001)

        table_path = os.path.join(dirname,  'modules', 'table.urdf')
        p.loadURDF(table_path, [-1.2, -0.6, 0.95], p.getQuaternionFromEuler([0, 0, np.pi/2]),
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


        self.marker = ActiveMarker(os.path.join(os.path.dirname(__file__), 'marker.urdf'), [0, 0, 0], [0, 0, 0, 1],
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

        self.marker.couple(self.robot,'link6')

        self.second_endeffector = pi.EndeffectorTool(os.path.join(os.path.dirname(__file__), 'marker.urdf'), [0, 0, 0], [0, 0, 0, 1])

        self.second_endeffector.couple(self.second_robot,'link6')

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def reset(self):
        """Resets the environment to the initial state
        """
        self.robot.reset_robot(*self.robot.get_world_state())
        self.second_robot.reset_robot(*self.second_robot.get_world_state())




    def run_simulation(self,particles):
        """ Runs the environment  given a set of particles describing the laser tracker positions
            and marker poses. It then returns the cost for each timestep and particle.
        """
        visibility_index = []
        distance_index = []
        self.reset()

        self.marker.set_optical_system_parameters(particles.T)

        # iterate over a path and compute the visibility matrix at each step,
        # adding it to the visibility index
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
            p.stepSimulation()


        for i in range(len(self.path.positions[0])):

            self.marker.set_tool_pose(self.path.positions[:,i],self.path.orientations[:,i])
            self.second_endeffector.set_tool_pose(self.second_path.positions[:,i],self.second_path.orientations[:,i])
            for _ in range(100):
                p.stepSimulation()
            visibility_index.append(self.marker.compute_visibility( ))
            distance_index.append(self.marker.compute_distance())

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

        print(costs)

        return costs


    env = LaserTrackerEnv()

    n_particles = 100


    initial_tracker_positions = np.random.rand(3, n_particles)*10
    initial_marker_positions = np.random.rand(3, n_particles)*0.6
    initial_marker_orientations = np.random.rand(3, n_particles)*2*np.pi

    initial_population = np.vstack(
        [initial_tracker_positions, initial_marker_positions, initial_marker_orientations])



    np.save("initial_states.npy",initial_population)

    start_tracker_velocities = np.random.rand(3, n_particles)*2
    start_marker_velocities = np.random.rand(3, n_particles)*0.01
    start_orientations_velocities = np.random.rand(3, n_particles)*0.1

    start_velocities = np.vstack(
        [start_tracker_velocities, start_marker_velocities, start_orientations_velocities])

    with console.status("Finding optimal camera placements"):
        particle_log, obj_log = particle_optimizer_log(objective_function, initial_population,
                                                        start_velocities, max_iter=100)

    # save particle log and objective log
    np.save("particle_log.npy", particle_log)
    np.save("objective_log.npy", obj_log)




if __name__ == "__main__":
    run_experiment()
    #env = LaserTrackerEnv(rendering=True)

    #while True:
    #    env.run_simulation(np.array([[-4,-1,3,0,0,0.3,0,0,0],[-3,-5,1.2,0,-0.3,0.1,0,0,0]]).T)


