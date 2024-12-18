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
        tracker_positions (list): Positions of the laser trackers
        coupled_robots (list): List of robots which are coupled to the tool
        tcp_frame (list): Position of the tool center point in the tool frame
        connector_frames (list): List of positions of the connectors in the tool frame
    """

    def __init__(self, urdf_model: str, start_position, start_orientation, marker_positions,
                 marker_orientations, tracker_positions,
                 marker_fov,coupled_robots=None, tcp_frame=None, connector_frames=None):
        super().__init__(urdf_model, start_position, start_orientation,
                         coupled_robots, tcp_frame, connector_frames)

        self.marker_positions = marker_positions
        self.marker_orientations = marker_orientations
        self.tracker_positions = tracker_positions
        self.marker_fov = marker_fov

    def compute_visibility(self):
        """ Computes the visibility of the markers given a laser position and marker orientations.

        Returns:
            list: A list which contains the visibility of each marker for its corresponding tracker
        """
        num_markers = len(self.marker_positions)
        field_of_views = [self.marker_fov]*num_markers
        visibility = np.zeros(num_markers)

        current_marker_positions, current_marker_orientations = self.get_marker_poses_orientations()

        for i in range(num_markers):
            ray_start_pos = self.tracker_positions[i]
            ray_end_pos = current_marker_positions[i]

            ray_intersection = p.rayTest(ray_start_pos, ray_end_pos)[0]

            if ray_intersection[0] != -1:
                p.addUserDebugLine(ray_start_pos, ray_end_pos, [0, 0, 1], 1)
                continue
            else:
                p.addUserDebugLine(ray_start_pos, ray_end_pos, [1, 0, 0], 1)

            ray_vector = np.array(ray_start_pos) - np.array(ray_end_pos)

            marker_rotation = p.getMatrixFromQuaternion(current_marker_orientations[i])
            marker_rotation = np.array(marker_rotation).reshape(3, 3)
            marker_vector = marker_rotation[:, 2]

            angle = compute_angle_between_vectors(ray_vector, marker_vector)

            if angle < field_of_views[i] / 2:
                visibility[i] = 1

        return visibility

    def compute_distance(self):
        """computes the distance betwen the marker and the tracker

        Returns:
            list: A list which contains the distance of each marker for its corresponding tracker
        """
        num_markers = len(self.marker_positions)
        distances = np.zeros(num_markers)
        current_marker_positions, current_marker_orientations = self.get_marker_poses_orientations()
        for i in range(num_markers):
            ray_start_pos = self.tracker_positions[i]
            ray_end_pos = current_marker_positions[i]
            distances[i] = np.linalg.norm(np.array(ray_start_pos) - np.array(ray_end_pos))
        return distances

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

    def set_optical_system_parameters(self,particles):
        """Convenient interface to set marker positions and orientations all in one go.

        """
        self.tracker_positions = [particle[:3] for particle in particles]
        self.marker_positions = [particle[3:6] for particle in particles]
        self.marker_orientations = [p.getQuaternionFromEuler(particle[6:9]) for particle in particles]

        # draw markers
        for i in range(len(self.tracker_positions)):
            pi.draw_coordinate_system(self.marker_positions[i], self.marker_orientations[i],parent_id = self.urdf)


class TestActiveMarker(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        p.connect(p.DIRECT)

        dirname = os.path.dirname(__file__)
        urdf_file = os.path.join(dirname, 'marker.urdf')
        self.tool = ActiveMarker(urdf_file, [0, 0, 0], [0, 0, 0, 1],
                                 [[0.1, 0, 0.1],
                                  [-0.1, 0, 0.1]],
                                 [[0, 0, 0, 1],
                                  [0, 0, 0, 1]],
                                  [[0.0, 0.0, 0.5],
                                   [0.0, 0.0, 0.6]],
                                   np.pi/4)


    def tearDown(self):
        """Clean up after the test."""
        p.disconnect()

    def test_marker_within_field_of_view(self):
        """Test visibility for a marker directly in front of the laser tracker."""
        self.tool.marker_orientations = [p.getQuaternionFromEuler([0, 0,0]),
                                            p.getQuaternionFromEuler([0, 0,0])]
        visibility = self.tool.compute_visibility( )
        # The first camera should see all markers
        print("visibility: ",visibility)
        self.assertEqual(np.sum(visibility), 2)

    def test_marker_outside_field_of_view(self):
        """Test visibility for a marker outside the field of view."""
        # Adjust marker orientation so that it is facing away from the laser tracker
        self.tool.marker_orientations = [p.getQuaternionFromEuler([0, np.pi,0 ]),
                                         p.getQuaternionFromEuler([0, np.pi / 4+0.1,0])]

        visibility = self.tool.compute_visibility()


        # No marker should be visible as they are all facing away
        self.assertEqual(np.sum(visibility), 0)

    def test_line_of_sight(self):
        """Test visibility for a marker with a clear line of sight."""
        # Adjust marker orientation so that it is facing towards the laser tracker
        self.tool.marker_orientations = [p.getQuaternionFromEuler([0, 0,0]),
                                         p.getQuaternionFromEuler([0, 0,0])]

        # spawn a multibody that blocks the line of sight
        plane_dimensions = [0.2, 0.2, 0.01]
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=plane_dimensions),
                          p.createVisualShape(
                              p.GEOM_BOX, halfExtents=plane_dimensions),
                          [0, 0, 0.2])

        visibility = self.tool.compute_visibility()
        # The first camera should no longer see the markers

        self.assertEqual(np.sum(visibility[0]), 0)



if __name__ == "__main__":
    unittest.main()
