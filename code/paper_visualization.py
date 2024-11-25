import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from paper_experiment import LaserTrackerEnv
import pybullet as p

class ParticleSelectorApp:
    def __init__(self, master, particles, objective_log, env, threshold=1):
        self.master = master
        self.particles = particles
        print(self.particles)
        self.objective_log = objective_log
        print(self.objective_log)
        self.env = env

        # Find the best objective value
        min_objective_value = np.min(objective_log)

        # Filter particles within the threshold
        self.filtered_indices = np.where(objective_log<= min_objective_value * (1 + threshold))[0]
        self.filtered_particles = particles[:, self.filtered_indices]

        self.master.title("Select Particle Solution")

        self.label = tk.Label(master, text="Select a Particle Solution:")
        self.label.pack()

        self.combobox = ttk.Combobox(master)
        self.combobox.pack()
        self.combobox['values'] = [f"Solution {i}" for i in self.filtered_indices]
        self.combobox.bind("<<ComboboxSelected>>", self.update_simulation)

        laser_tracker_path = os.path.join(os.path.dirname(__file__), 'transformer_cell', 'Objects', 'laser_tracker.urdf')
        self.laser_tracker = p.loadURDF(laser_tracker_path, [0,0,0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, globalScaling=0.001)

        self.selected_particle = None

    def update_simulation(self, event):
        selected_index = self.combobox.current()
        best_particle = self.particles[:, selected_index]

        # reshape to 9x1
        best_particle = best_particle.reshape(9, 1)

        laser_tracker_position = best_particle[:3].flatten()

        # move the laser tracker to the selected position
        p.resetBasePositionAndOrientation(self.laser_tracker, laser_tracker_position, [0, 0, 0, 1])


        self.selected_particle = best_particle
        self.env.run_simulation(best_particle)


if __name__ == "__main__":
    particle_log = np.load(os.path.join("results","particle_log.npy"))
    objective_log = np.load(os.path.join("results","objective_log.npy"))
    env = LaserTrackerEnv(rendering=True)

    last_particle_round = particle_log[-1]
    last_objective_round = objective_log[-1]

    root = tk.Tk()
    app = ParticleSelectorApp(root, last_particle_round, last_objective_round, env)
    root.mainloop()
