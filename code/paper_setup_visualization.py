from paper_experiment import CustomEnv
import pybullet as p
import pybullet_industrial as pi
import numpy as np

if __name__ == "__main__":

    env = CustomEnv(True, 200)

    focal_length = 0.1
    intrinsic_matrix = [0.0]*9
    intrinsic_matrix[0*3+0] = focal_length
    intrinsic_matrix[1*3+1] = focal_length
    intrinsic_matrix[2*3+2] = float(1)
    intrinsic_matrix = tuple(intrinsic_matrix)
    camera_state = np.array([0.0, 1, 0.6, np.pi/2, 0.0, 0.0])
    camera_configs = []
    pi.draw_coordinate_system(camera_state[:3],
                              p.getQuaternionFromEuler(camera_state[3:6]))
    camera_configs.append([camera_state[:3], camera_state[3:6]])

    env.configure_cameras([camera_configs], [[
                          intrinsic_matrix]*len(camera_configs)])

    while True:
        env.run_simulation()
