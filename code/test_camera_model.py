import os

import numpy as np
import pybullet as p
import pybullet_industrial as pi


if __name__ == "__main__":
    # setup a pybullet simulation
    p.connect(p.GUI)

    # pybullet load marker.urdf
    dirname = os.path.dirname(__file__)
    urdf_file = os.path.join(dirname, 'marker.urdf')
    # load urdf
    p.loadURDF(urdf_file, [0.0, 0, 0], [0, 0, 0, 1], useFixedBase=True)

    camera_parameters = {}
    fov = 60
    img_width = 640
    img_height = 480
    near_plane = 0.1
    far_plane = 100
    aspect_ratio = 1
    focal_length = 10

    projection_matrix = p.computeProjectionMatrixFOV(
        fov, aspect_ratio, near_plane, far_plane)

    first_camera_pos = np.array([-0., 0, -1])
    second_camera_pos = np.array([-0., 0., -1])

    first_camera_rot = [0, 0, 0, 1]
    second_camera_rot = p.getQuaternionFromEuler([0, 0, 0])

    pi.draw_coordinate_system(first_camera_pos, first_camera_rot)
    pi.draw_coordinate_system(second_camera_pos, second_camera_rot)

    # convert quaternion to rotation matrix
    first_camera_rot_mat = p.getMatrixFromQuaternion(first_camera_rot)
    first_camera_rot_mat = np.array(first_camera_rot_mat).reshape(3, 3)
    second_camera_rot_mat = p.getMatrixFromQuaternion(second_camera_rot)
    second_camera_rot_mat = np.array(second_camera_rot_mat).reshape(3, 3)

    # calculate camera target position in world frame
    first_camera_target_pos = first_camera_pos+focal_length * \
        first_camera_rot_mat.dot(np.array([0, 0, 1]))
    second_camera_target_pos = second_camera_pos+focal_length * \
        second_camera_rot_mat.dot(np.array([0, 0, 1]))

    # Setup the view matrices for both camera
    first_camera_view_matrix = p.computeViewMatrix(
        first_camera_pos, first_camera_target_pos, np.array([0, 1, 0]))
    second_camera_view_matrix = p.computeViewMatrix(
        second_camera_pos, second_camera_target_pos, np.array([0, 1, 0]))

    first_image = p.getCameraImage(
        img_width, img_height, first_camera_view_matrix, projection_matrix)
    second_image = p.getCameraImage(
        img_width, img_height, second_camera_view_matrix, projection_matrix)

    while True:
        p.stepSimulation()
