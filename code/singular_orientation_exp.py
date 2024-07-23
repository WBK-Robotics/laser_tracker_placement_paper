import numpy as np
from paper_experiment import CustomEnv


if __name__ == "__main__":
    env = CustomEnv(False)

    focal_length = 0.1
    intrinsic_matrix = [0.0]*9
    intrinsic_matrix[0*3+0] = focal_length
    intrinsic_matrix[1*3+1] = focal_length
    intrinsic_matrix[2*3+2] = float(1)
    intrinsic_matrix = tuple(intrinsic_matrix)

    base_camera_state = [[0, 0, 0], [0, 0, 0]]

    rotating_camera_state = [[1, 0, 0], [0, 0, 0]]

    steps = 100

    # create an array of angles from -pi/2 to pi/2  in n steps
    camera_angles = np.linspace(-np.pi/2, np.pi/2, steps)

    singular_values = []

    for camera_angle in camera_angles:
        rotating_camera_state[1][1] = camera_angle

        singular_value = env.configure_cameras([[base_camera_state, rotating_camera_state]], [[
            intrinsic_matrix, intrinsic_matrix]])

        singular_values.append(singular_value)

    env.close()

    import matplotlib.pyplot as plt
    plt.plot(camera_angles, singular_values)
    plt.title("Singular values of the camera configurations")
    plt.xlabel("Camera angle relative to base camera [rad]")
    plt.ylabel("Singular values")
    plt.show()
