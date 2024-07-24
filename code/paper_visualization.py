from paper_experiment import LaserTrackerEnv
import pybullet as p
import pybullet_industrial as pi
import numpy as np
import os

if __name__ == "__main__":
    particle_log = np.load("particle_log.npy")
    objective_log = np.load("objective_log.npy")
    env = LaserTrackerEnv(rendering=True)

    last_particle_round = particle_log[-1]

    best_particle = last_particle_round[:,np.argmin(objective_log[-1])]

    print("Best particle: ", best_particle)

    # reshape to 9x1
    best_particle = best_particle.reshape(9,1)

    laser_tracker_position = best_particle[:3].flatten()

    laser_tracker_path = dirname = os.path.join(os.path.dirname(__file__), 'transformer_cell', 'Objects','laser_tracker.urdf')
    #load laser tracker at optimal position
    #p.loadURDF(laser_tracker_path, laser_tracker_position, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, globalScaling=0.001)

    marker_position = best_particle[3:6].flatten()
    marker_orientation = p.getQuaternionFromEuler(best_particle[6:9])

    # visualize the markers as coordinate system relative to the flange.
    print(marker_position)
    print(env.robot.urdf)
    print(env.robot._convert_endeffector('link6'))
    pi.draw_coordinate_system(marker_position,marker_orientation,
                              parent_id=env.robot.urdf,parent_index=env.robot._convert_endeffector('link6'))

    while True:
        env.run_simulation(best_particle)

