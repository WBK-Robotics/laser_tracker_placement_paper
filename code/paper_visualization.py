from paper_experiment import CustomEnv
import pybullet as p
import pybullet_industrial as pi
import numpy as np

if __name__ == "__main__":
    camera_radius = 2.5
    optimal_states = np.load("optimal_states_6_cameras.npy")
    optimal_camera_states = np.split(
        optimal_states.flatten(), len(optimal_states.flatten())/6, axis=0)

    initial_states = np.load("initial_states_6_cameras.npy")
    initial_camera_states = np.split(
        initial_states, len(initial_states)/6, axis=0)
    env = CustomEnv(True)

    focal_length = 0.1
    intrinsic_matrix = [0.0]*9
    intrinsic_matrix[0*3+0] = focal_length
    intrinsic_matrix[1*3+1] = focal_length
    intrinsic_matrix[2*3+2] = float(1)
    intrinsic_matrix = tuple(intrinsic_matrix)

    camera_configs = []
    for camera_state in optimal_camera_states:
        pi.draw_coordinate_system(camera_state[:3],
                                  p.getQuaternionFromEuler(camera_state[3:6]))
        camera_configs.append([camera_state[:3], camera_state[3:6]])

    env.configure_cameras([camera_configs], [[
                          intrinsic_matrix]*len(camera_configs)])

    for camera_state in initial_camera_states:
        for i in range(len(camera_state[0])):
            pi.draw_coordinate_system(camera_state[:3, i],
                                      p.getQuaternionFromEuler(
                                          camera_state[3:6, i]),
                                      length=0.05, width=1.0)
    env._compute_visibility()
    env.run_simulation()
    while True:
        p.stepSimulation()
