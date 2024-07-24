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


class ActiveMarker(pi.EndeffectorTool):
    """Class which represents an active marker for a laser tracker.
    Args:
        urdf_model (str): Path to the urdf model of the tool
        start_position (list): Position of the tool in the world frame
        start_orientation (list): Orientation of the tool in the world frame
        marker_positions (list): Positions of the markers in the tool frame
        marker_orientations (list): Orientations of the markers in the tool frame
        coupled_robots (list): List of robots which are coupled to the tool
        tcp_frame (list): Position of the tool center point in the tool frame
        connector_frames (list): List of positions of the connectors in the tool frame
    """

    def __init__(self, urdf_model: str, start_position, start_orientation, marker_positions,
                 marker_orientations, coupled_robots=None, tcp_frame=None, connector_frames=None):
        super().__init__(urdf_model, start_position, start_orientation,
                         coupled_robots, tcp_frame, connector_frames)

        self.marker_positions = marker_positions
        self.marker_orientations = marker_orientations

    def compute_visibility(self, camera_positions, field_of_views):
        """ Computes the visibility of the markers given a laser position and marker orientations.

        Args:
            camera_positions (list): List of laser tracker positions
            field_of_views (list): List of field of views of the markers

        Returns:
            array: A matrix of size (number of markers, number of cameras)
                   which contains the visibility of the markers for each camera
        """
        num_cameras = len(camera_positions)
        num_markers = len(self.marker_positions)
        visibility = np.zeros((num_cameras, num_markers))

        current_marker_poses, current_marker_orientations = self.get_marker_poses_orientations()

        ray_start_pos = []
        ray_end_pos = []
        for camera in camera_positions:
            for marker_pos in current_marker_poses:
                ray_start_pos.append(camera)
                ray_end_pos.append(marker_pos)

        ray_intersections = p.rayTestBatch(ray_start_pos, ray_end_pos)

        for i in range(num_cameras):
            for j in range(num_markers):
                ray_index = i * num_markers + j
                # Check if there is a clear line of sight
                if ray_intersections[ray_index][0] != -1:
                    continue

                ray_vector = np.array(current_marker_poses[j]) - np.array(camera_positions[i])

                marker_rotation = p.getMatrixFromQuaternion(current_marker_orientations[j])
                marker_rotation = np.array(marker_rotation).reshape(3, 3)
                marker_vector = marker_rotation[:, 2]

                angle = compute_angle_between_vectors(ray_vector, marker_vector)

                if angle < field_of_views[j] / 2:
                    visibility[i][j] = 1

        return visibility

    def get_marker_poses_orientations(self):
        """ Returns the positions and orientations of the markers in the world frame.

        Returns:
            tuple: Tuple containing list of marker positions and list of marker orientations in the world frame
        """
        marker_positions = []
        marker_orientations = []
        tool_pose = self.get_tool_pose()
        tool_position = tool_pose[0]
        tool_orientation = tool_pose[1]
        tool_rotation_matrix = p.getMatrixFromQuaternion(tool_orientation)
        tool_rotation_matrix = np.array(tool_rotation_matrix).reshape(3, 3)
        for pos, ori in zip(self.marker_positions, self.marker_orientations):
            task_space_position = tool_position + tool_rotation_matrix @ np.array(pos)
            marker_positions.append(task_space_position)

            marker_quaternion = p.multiplyTransforms(
                tool_position, tool_orientation,
                pos, ori
            )[1]
            marker_orientations.append(marker_quaternion)

        return marker_positions, marker_orientations


class TestActiveMarker(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.setRealTimeSimulation(1)

        dirname = os.path.dirname(__file__)
        urdf_file = os.path.join(dirname, 'marker.urdf')
        self.tool = ActiveMarker(urdf_file, [0, 0, 0], [0, 0, 0, 1],
                                 [[0.1, 0.1, 0.1],
                                  [0.1, -0.1, 0.1],
                                  [-0.1, 0.1, 0.1],
                                  [-0.1, -0.1, 0.1]],
                                 [[0, 0, 0, 1],
                                  [0, 0, 0, 1],
                                  [0, 0, 0, 1],
                                  [0, 0, 0, 1]])

        self.camera_positions = [[0., 0, 0.5], [0.0, 0, 0.6]]
        self.field_of_views = [np.pi , np.pi, np.pi , np.pi ]

    def tearDown(self):
        """Clean up after the test."""
        p.disconnect()

    def test_marker_within_field_of_view(self):
        """Test visibility for a marker directly in front of the laser tracker."""
        visibility = self.tool.compute_visibility(self.camera_positions, self.field_of_views)
        print(visibility)
        # The first camera should see all markers
        self.assertEqual(np.sum(visibility[0]), 4)

    def test_marker_outside_field_of_view(self):
        """Test visibility for a marker outside the field of view."""
        # Adjust marker orientation so that it is facing away from the laser tracker
        self.tool.marker_orientations = [[0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
                                         [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
                                         [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
                                         [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]]

        visibility = self.tool.compute_visibility(self.camera_positions, self.field_of_views)

        # No marker should be visible as they are all facing away
        self.assertEqual(np.sum(visibility), 0)

    def test_multiple_markers_and_cameras(self):
        """Test visibility with multiple markers and cameras with different field of views."""
        # Adjust field of views
        self.field_of_views = [np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4]

        visibility = self.tool.compute_visibility(self.camera_positions, self.field_of_views)

        # Check the first camera's visibility
        # It should see fewer markers due to the reduced field of view
        self.assertTrue(np.sum(visibility[0]) < 4)

        # Check the second camera's visibility
        # It should also see fewer markers due to the reduced field of view
        self.assertTrue(np.sum(visibility[1]) < 4)

        # Adjust marker orientation to point towards the second camera
        self.tool.marker_orientations = [[0, 0, 0, 1],
                                         [0, 0, 0, 1],
                                         [0, 0, 0, 1],
                                         [0, 0, 0, 1]]
        visibility = self.tool.compute_visibility(self.camera_positions, self.field_of_views)

        # The second camera should now see some markers
        self.assertTrue(np.sum(visibility[1]) > 0)

if __name__ == "__main__":
    unittest.main()
