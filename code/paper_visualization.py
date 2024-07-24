from paper_experiment import LaserTrackerEnv
import pybullet as p
import pybullet_industrial as pi
import numpy as np

if __name__ == "__main__":
    particle_log = np.load("particle_log.npy")
    objective_log = np.load("objective_log.npy")
    env = LaserTrackerEnv(rendering=True)

    last_particle_round = particle_log[-1]

    best_particle = last_particle_round[:,np.argmin(objective_log[-1])]

    print("Best particle: ", best_particle)

    # reshape to 9x1
    best_particle = best_particle.reshape(9,1)

    while True:
        env.run_simulation(best_particle)

