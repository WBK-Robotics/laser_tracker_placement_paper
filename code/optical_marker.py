import os
import numpy as np
import pybullet as p
import pybullet_industrial as pi

import unittest


def compute_angle_between_vectors(v1, v2):
    """Computes the angle between two vectors

    Args:
        v1 (numpy.array): The first vector
        v2 (numpy.array): The second vector

    Returns:
        float: The angle between the two vectors
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class OpticalMarker(pi.EndeffectorTool):
    """Class which represents an optical marker
    Args:
        urdf_model (str): Path to the urdf model of the tool
        start_position (list): Position of the tool in the world frame
        start_orientation (list): Orientation of the tool in the world frame
        marker_positions (list): Positions of the markers in the tool frame
        coupled_robots (list): List of robots which are coupled to the tool
        tcp_frame (list): Position of the tool center point in the tool frame
        connector_frames (list): List of positions of the connectors in the tool frame
    """

    def __init__(self, urdf_model: str, start_position, start_orientation, marker_positions,
                 coupled_robots=None, tcp_frame=None, connector_frames=None):
        super().__init__(urdf_model, start_position, start_orientation,
                         coupled_robots, tcp_frame, connector_frames)

        self.marker_positions = marker_positions
        steps = 50
        particle_size = 0.03
        width = particle_size*500
        theta2 = np.linspace(-np.pi,  0, steps)
        phi2 = np.linspace(0,  5 * 2*np.pi, steps)

        for positions in marker_positions:
            x_coord = particle_size * \
                np.sin(theta2) * np.cos(phi2) + positions[0]
            y_coord = particle_size * \
                np.sin(theta2) * np.sin(phi2) + positions[1]
            z_coord = particle_size * \
                np.cos(theta2) + positions[2]

            path = np.array([x_coord, y_coord, z_coord])
            path_steps = len(path[0])
            for i in range(1, path_steps):
                current_point = path[:, i]
                previous_point = path[:, i-1]
                p.addUserDebugLine(current_point, previous_point,
                                   lineColorRGB=[20/255, 68/255, 102/255],
                                   lineWidth=width,
                                   lifeTime=0,
                                   parentObjectUniqueId=self.urdf,
                                   parentLinkIndex=-1)

    def compute_visibility(self, camera_states, field_of_views=None, orientation_only=False):
        """ Computes the visibility of the markers given a camera state.

        WARNING: currently only the position of the camera is considered, not the orientation.

        Args:
            camera_states (list): List of camera
            field_of_views (list): List of field of views of the cameras
                                   Defaults to None in which case only the position of the camera
                                   is considered.
            orientation_only (bool): If true, only the orientation of the camera is considered.

        Returns:
            array: A matrix of size (number of markers, number of cameras)
                   which contains the visibility of the markers for each camera

        """
        num_cameras = len(camera_states)
        num_markers = len(self.marker_positions)
        visibility = np.zeros((num_cameras, num_markers))

        current_marker_poses = self.get_marker_positions()

        ray_start_pos = []
        ray_end_pos = []
        for marker_pos in current_marker_poses:
            for camera in camera_states:
                ray_start_pos.append(marker_pos)
                ray_end_pos.append(camera[0])

        ray_intersections = p.rayTestBatch(ray_start_pos, ray_end_pos)
        for i in range(num_markers):
            for j in range(num_cameras):

                # compute the position based visibility
                # ----------------------------------------------------------------------------------
                if not orientation_only:
                    if ray_intersections[i*(num_cameras)+j][0] != -1:
                        intersection_pos = np.array(
                            ray_intersections[i*(num_cameras)+j][3])
                    else:
                        intersection_pos = ray_end_pos[i*(num_cameras)+j]

                    p.addUserDebugLine(ray_start_pos[i*(num_cameras)+j], intersection_pos,
                                       lineColorRGB=[238/255, 183/255, 13/255],
                                        lineWidth=2,
                                       lifeTime=0.1)

                    max_distance = np.linalg.norm(
                        ray_start_pos[i*(num_cameras)+j]-ray_end_pos[i*(num_cameras)+j])
                    actual_distance = np.linalg.norm(
                        ray_start_pos[i*(num_cameras)+j]-intersection_pos)

                    visibility[j][i] = int(actual_distance/max_distance)

                # compute the orientation based visibility
                # ----------------------------------------------------------------------------------
                if field_of_views is not None:
                    ray_vector = np.array(current_marker_poses[i]) - \
                        np.array(camera_states[j][0])

                    camera_rotation = p.getMatrixFromQuaternion(
                        camera_states[j][1])
                    camera_rotation = np.array(camera_rotation).reshape(3, 3)
                    camera_vector = camera_rotation[:, 2]

                    angle = compute_angle_between_vectors(
                        ray_vector, camera_vector)

                    visibility[j][i] = visibility[j][i] - angle/np.pi

        return visibility

    def get_marker_positions(self):
        """ Returns the positions of the markers in the world frame.

        Returns:
            list: List of marker positions in the world frame
        """
        marker_positions = []
        tool_pose = self.get_tool_pose()
        tool_position = tool_pose[0]
        tool_orientation = tool_pose[1]
        tool_rotation_matrix = p.getMatrixFromQuaternion(tool_orientation)
        tool_rotation_matrix = np.array(tool_rotation_matrix).reshape(3, 3)
        for positions in self.marker_positions:
            task_space_position = tool_position + \
                tool_rotation_matrix@np.array(positions)
            marker_positions.append(task_space_position)

        return marker_positions


class TestOpticalMarker(unittest.TestCase):

    def test_compute_visibility(self):
        """Tests the compute_visibility function of the OpticalMarker class.
           Specifically if the orientation of the camera is correctly taken into account.
        """
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.setRealTimeSimulation(1)

        dirname = os.path.dirname(__file__)
        urdf_file = os.path.join(dirname, 'marker.urdf')
        tool = OpticalMarker(urdf_file, [0, 0, 0], [0, 0, 0, 1],
                             [[0.1, 0.1, 0.1],
                              [0.1, -0.1, 0.1],
                              [-0.1, 0.1, 0.1],
                              [-0.1, -0.1, 0.1]])

        plane_dimensions = [0.2, 0.2, 0.01]
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=plane_dimensions),
                          p.createVisualShape(
                              p.GEOM_BOX, halfExtents=plane_dimensions),
                          [-0.2, 0, -0.0])
        p.stepSimulation()

        camera_states = [[[-0.5, 0, 0.5], [0, 0, 0, 1]],
                         [[0.0, 0, -0.8], [0, 0, 0, 1]]]

        visibility = tool.compute_visibility(camera_states)

        # check that the first camera sees all markers
        self.assertTrue(np.sum(visibility[0]) == 4)
        # check that the second camera sees only half of the markers
        self.assertTrue(np.sum(visibility[1][0:2]) == 2)
        self.assertTrue(np.sum(visibility[1][2:4]) < 2)

        # if visibility is considered the first camera should no longer see all markers
        # while the second camera should see the same amount of markers
        visibility = tool.compute_visibility(camera_states, [np.pi/2, np.pi/2])
        self.assertTrue(np.sum(visibility[0]) < 4)
        self.assertTrue(np.sum(visibility[1][0:2]) == 2)
        self.assertTrue(np.sum(visibility[1][2:4]) < 2)

        # if the field of view is decreased the second camera should see less markers
        visibility = tool.compute_visibility(
            camera_states, [np.pi/2, np.pi/100])
        self.assertTrue(np.sum(visibility[1][0:2]) < 2)

        p.disconnect()


if __name__ == "__main__":
    unittest.main()
