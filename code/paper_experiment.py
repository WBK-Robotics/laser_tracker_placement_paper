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
            target_position = np.array([1.6, 0, 1.03])+np.array([-3.6353162, -0.6, 0])
            steps = 30
            self.path = pi.build_box_path(
                target_position, [0.5, 0.6], 0.1, [0, 0, 0, 1], steps)

        else:
            self.path = path


        self.path.draw()


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
            p.stepSimulation()


        for position,orientation,_ in self.path:
            self.marker.set_tool_pose(position,orientation)
            for _ in range(100):
                p.stepSimulation()
            visibility_index.append(self.marker.compute_visibility( ))

        visibility_index = np.array(visibility_index)

        visibility_index = visibility_index.T
        return visibility_index

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

        visibility_cost = env.run_simulation(particles)
        visibility_cost_sum = np.sum(visibility_cost,axis=1)


        #constraint costs

        #constraints keeping the height of the laser tracker betwen 0.8 and 2
        height_constraint = np.clip(np.abs(particles[2,:]-1.03)-0.8,0,-1e6)



        #constraint keeping the marker position within a cylinder of given radius and height
        cylinder_radius = 0.5
        cylinder_height = 0.4
        marker_positions = particles[3:6,:]

        radius_constraint = np.clip(np.linalg.norm(marker_positions[:2,:],axis=0)-cylinder_radius,0,-1e6)
        height_constraint = np.clip(np.abs(marker_positions[2,:])-cylinder_height,0,-1e6)






        costs = visibility_cost_sum+ height_constraint + radius_constraint + height_constraint

        print(costs)

        return costs


    env = LaserTrackerEnv()

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
    run_experiment()

