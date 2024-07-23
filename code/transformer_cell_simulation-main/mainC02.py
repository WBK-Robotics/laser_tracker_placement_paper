import math
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_industrial as pi

def linear_interpolation_orn(start_point: np.array, end_point: np.array,
                             start_orn: np.array, end_orn: np.array, samples: int):
    """Performs a linear interpolation betwenn two points in 3D space

    Args:
        start_point (np.array): The start point of the interpolation
        end_point (np.array): The end point of the interpolation
        samples (int): The number of samples used to interpolate

    Returns:
        ToolPath: A ToolPath object of the interpolated path
    """
    final_path = np.linspace(start_point, end_point, num=samples)
    final_orns = np.linspace(start_orn, end_orn, num=samples)
    return pi.ToolPath(positions=final_path.transpose(), orientations= final_orns.transpose())

def move_along_path(endeffector: pi.EndeffectorTool, path: pi.ToolPath, stop=True):
    """Moving a designated endeffector along the provided path.
    Args:
        endeffector (pi.EndeffectorTool): Endeffector to be moved.
        path (pi.ToolPath): Array of points defining the path.
        stop (bool, optional): Whether or not to stop at the end of the movement.
    """
    #path.draw()
    for positions, orientations, tool_path in path:
        endeffector.set_tool_pose(positions, orientations)
        for _ in range(20):
            p.stepSimulation()
    if stop:
        for _ in range(50):
            p.stepSimulation()
def move_lin(endeffector: pi.EndeffectorTool, target_pos, target_orn=None, step_size=0.1):
    pos, orn = endeffector.get_tool_pose("tcp")
    if target_orn == None:
        target_orn = orn

    steps=round(np.linalg.norm(target_pos - pos)/step_size)
    path=linear_interpolation_orn(pos, target_pos,
                                       orn,
                                       target_orn,
                                       steps)
    move_along_path(endeffector, path)

def move_rel(endeffector: pi.EndeffectorTool, target_pos, target_orn=None, step_size=0.1):
    pos, orn = endeffector.get_tool_pose("tcp")
    if target_orn == None:
        target_orn = orn
    steps=round(np.linalg.norm(target_pos - pos)/step_size)
    path=linear_interpolation_orn(pos, pos + target_pos,
                                       orn,
                                       orn + target_orn,
                                       steps)
    move_along_path(endeffector, path)

def spawn_holder(spawn_point):

    cube_urdf = os.path.join(dirname,
                                 'objects', 'cube_small.urdf')
    p.loadURDF(cube_urdf,  spawn_point + np.array([-0.09, -0.1, -0.09]), p.getQuaternionFromEuler([math.pi/2,math.pi/6,0]),useFixedBase=True)
    p.loadURDF(cube_urdf,  spawn_point + np.array([-0.09, 0.1, -0.09]), p.getQuaternionFromEuler([math.pi/2,math.pi/6,0]),useFixedBase=True)
    p.loadURDF(cube_urdf,  spawn_point + np.array([0.09, -0.1, -0.09]), p.getQuaternionFromEuler([math.pi/2,-math.pi/6,0]),useFixedBase=True)
    p.loadURDF(cube_urdf,  spawn_point + np.array([0.09, 0.1, -0.09]), p.getQuaternionFromEuler([math.pi/2,-math.pi/6,0]),useFixedBase=True)

if __name__ == '__main__':
    screwing = True
    placing = True

    dirname = os.path.dirname(__file__)
    comau_urdf = os.path.join(dirname,
                              'robot_descriptions', 'comau_nj290_robot.urdf')
    gripper_urdf = os.path.join(dirname,
                              'robot_descriptions', 'gripper_cad.urdf')
    screwDriver_urdf = os.path.join(dirname,
                              'robot_descriptions', 'screwDriver.urdf')
    plane_urdf = os.path.join(dirname,
                                 'objects', 'plane.urdf')

    physics_client = p.connect(p.GUI, options=  '--width=1920 --height=1060 --background_color_red=0.8 --background_color_green=0.8 --background_color_blue=0.83')
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 0: no gui, 1, basic gui

    p.resetDebugVisualizerCamera(cameraDistance=6.6, cameraYaw=-7, cameraPitch=-35,
                                 cameraTargetPosition=np.array([-0.5, 0, -1]))

    p.setPhysicsEngineParameter(numSolverIterations=10000)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    path = linear_interpolation_orn(np.array([0,0,0]), np.array([0,0,1]),
                                    p.getQuaternionFromEuler([0,0,0]),
                                    p.getQuaternionFromEuler([math.pi,0,0]), 50)

    p.loadURDF(plane_urdf, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
    #load Robots__________________________________________________
    start_pos = np.array([-3.6353162, -0.6, 0])
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot = pi.RobotBase(comau_urdf, start_pos, start_orientation)
    start_pos2 = np.array([-0.12, -2.59, 0])
    start_orientation2 = p.getQuaternionFromEuler([0, 0, -np.pi/2])
    robot2 = pi.RobotBase(comau_urdf, start_pos2, start_orientation2)
    robot.set_joint_position(
        {'q2': np.deg2rad(-15.0), 'q3': np.deg2rad(-90.0)})
    robot2.set_joint_position(
        {'q2': np.deg2rad(-15.0), 'q3': np.deg2rad(-90.0)})

    fofa_path = os.path.join(dirname,
                             'Objects', 'InRePro', 'InRePro.urdf')
    p.loadURDF(fofa_path, [0,0,0], p.getQuaternionFromEuler([math.pi/2,0,0]), useFixedBase=True, globalScaling=0.001)
    ponticon_path = os.path.join(dirname,
                             'Objects', 'ponticon', 'ponticon.urdf')
    p.loadURDF(ponticon_path, [0,0,0], p.getQuaternionFromEuler([math.pi/2,0,0]), useFixedBase=True, globalScaling=0.001)
    dmg_path = os.path.join(dirname,
                             'Objects', 'DMG', 'DMG.urdf')
    p.loadURDF(dmg_path, [0,0,0], p.getQuaternionFromEuler([math.pi/2,0,0]), useFixedBase=True, globalScaling=0.001)
    mm_path = os.path.join(dirname,
                             'Objects', 'modules', 'milling_module.urdf')
    p.loadURDF(mm_path, [-0.6,-4.6,0.95], p.getQuaternionFromEuler([0,0,math.pi/2]), useFixedBase=True, globalScaling=0.001)
    table_path = os.path.join(dirname,
                           'Objects', 'modules', 'table.urdf')
    p.loadURDF(table_path, [-0.6, -4.6, 0.95], p.getQuaternionFromEuler([0, 0, math.pi/2]), useFixedBase=True,
               globalScaling=0.001)

    grinder_urdf = os.path.join(dirname,
                                 'objects', 'ws','no_name.urdf')

    spawn_point_01 = np.array([-0.4, -4.0, 1.12])
    spawn_point_02 = np.array([-1.3, -4.0, 1.12])
    spawn_point_03 = np.array([0.1, -4.0, 1.12])
    spawn_holder(spawn_point_01)
    spawn_holder(spawn_point_02)
    spawn_holder(spawn_point_03)
    """grinder_01 = p.loadURDF(grinder_urdf,  spawn_point_01 + np.array([0,0.08,0]),
                            p.getQuaternionFromEuler([math.pi/2,math.pi,math.pi]),
                            globalScaling=0.0012)
    grinder_01 = p.loadURDF(grinder_urdf,  spawn_point_03+ np.array([0,0.08,0]),
                            p.getQuaternionFromEuler([math.pi/2,math.pi,math.pi]),
                            globalScaling=0.0012)"""
    grinder_01 = p.loadURDF(grinder_urdf,  spawn_point_01 + np.array([0,0.18,0]),
                            p.getQuaternionFromEuler([math.pi/2,math.pi/2,math.pi]),
                            globalScaling=0.0015)
    housing_urdf = os.path.join(dirname,
                                 'objects', 'ws','no_name01.urdf')
    housing = p.loadURDF(housing_urdf,  spawn_point_03 + np.array([0,0.18,0]),
                            p.getQuaternionFromEuler([math.pi/2,math.pi/2,math.pi]),
                            globalScaling=0.0015, useFixedBase=True)
    axle_urdf = os.path.join(dirname,
                                 'objects', 'ws','no_name_axle.urdf')
    axle = p.loadURDF(axle_urdf,  spawn_point_03 + np.array([0,0.18,0]),
                            p.getQuaternionFromEuler([math.pi/2,math.pi/2,math.pi]),
                            globalScaling=0.0015)
    axle_contraint = p.createConstraint(axle,
                                        -1, -1, -1,
                                        p.JOINT_FIXED,
                                        [0, 0, 0],
                                        [0, 0, 0],
                                        spawn_point_03 + np.array([0,0.18,0]),
                                        None,
                                        p.getQuaternionFromEuler([math.pi/2,math.pi/2,math.pi]))

    """cube01 = p.loadURDF(cube_urdf,  spawn_point + np.array([-0.1, -0.1, -0.07]), p.getQuaternionFromEuler([math.pi/2,0,0]),useFixedBase=True)
    cube02 = p.loadURDF(cube_urdf,  spawn_point + np.array([-0.1, 0.1, -0.07]), p.getQuaternionFromEuler([math.pi/2,0,0]),useFixedBase=True)
    cube01 = p.loadURDF(cube_urdf,  spawn_point + np.array([0.1, -0.1, -0.07]), p.getQuaternionFromEuler([math.pi/2,0,0]),useFixedBase=True)
    cube02 = p.loadURDF(cube_urdf,  spawn_point + np.array([0.1, 0.1, -0.07]), p.getQuaternionFromEuler([math.pi/2,0,0]),useFixedBase=True)"""

    p.setGravity(0, 0, -10)

    for _ in range(100):
        p.stepSimulation()
    #set initial positions an couple endeffectors
    robot.set_endeffector_pose(np.array([-2,-2.0,2]), p.getQuaternionFromEuler([-math.pi/2,0,0]))
    robot2.set_endeffector_pose(np.array([-0.12,-4,2]), p.getQuaternionFromEuler([-math.pi/2,0,0]))
    for _ in range(100):
        p.stepSimulation()
    screwdriver = pi.SuctionGripper(screwDriver_urdf, np.array([-0.12,-4,2]),
                                    p.getQuaternionFromEuler([-math.pi/2,0,0]))
    screwdriver.couple(robot, 'link6')

    gripper = pi.Gripper(gripper_urdf, np.array([-2,-2.0,2]),
                         p.getQuaternionFromEuler([-math.pi/2,0,0]))
    gripper.couple(robot2, 'link6')
    gripper.actuate(0.0)
    screw_point = np.array([-2,-2,1.8])
    screwdriver.set_tool_pose(screw_point + np.array([-1, 0, 0.1]), p.getQuaternionFromEuler([0, 0, -math.pi / 4]))
    for _ in range(1000):
        p.stepSimulation()
    for _ in range(10):
        screwdriver.set_tool_pose(screw_point + np.array([-1,0,0.1]),p.getQuaternionFromEuler([0,0,0]))
        for _ in range(10):
            p.stepSimulation()

    pos, orn = gripper.get_tool_pose("tcp")
    print(p.getEulerFromQuaternion(orn))

    lID = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join(dirname, 'output.mp4'))


    if placing:
        p.resetDebugVisualizerCamera(cameraDistance=5.6, cameraYaw=118, cameraPitch=-85,
                                     cameraTargetPosition=np.array([1.8, -2, 0]))
#        p.resetDebugVisualizerCamera(cameraDistance=5.6, cameraYaw=7, cameraPitch=-20,
#                                     cameraTargetPosition=np.array([0.5, 0, 0]))
        move_lin(gripper, spawn_point_03 + np.array([0,0.24,0.7]),p.getQuaternionFromEuler([math.pi,0,math.pi/2]))
        move_rel(gripper, np.array([0,0,-0.69]))
        gripper.actuate(1.0)
        for _ in range(100):
            p.stepSimulation()
        p.removeConstraint(axle_contraint)
        move_rel(gripper, np.array([0,0,0.3]))
        #move_lin(gripper, np.array([1.5,-3.15,1.2]), p.getQuaternionFromEuler([math.pi,0,math.pi/2]))
        move_lin(gripper, np.array([2.0,-3.15,1.2]), p.getQuaternionFromEuler([math.pi/2,0,math.pi/2]))
        move_lin(gripper, np.array([2.5,-3.15,1.2]), p.getQuaternionFromEuler([math.pi/2,0,math.pi/2]))
        #gripper.actuate(0.0)
        #for _ in range(100):
        #    p.stepSimulation()
        #move_lin(gripper, np.array([1.5, -3.15, 1.2]), p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]))
        move_lin(gripper, np.array([2.0, -3.15, 1.2]), p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]))
        #for _ in range(200):
        #    p.stepSimulation()
       # move_lin(gripper, np.array([2.5,-3.15,1.2]), p.getQuaternionFromEuler([math.pi/2,0,math.pi/2]))
        #gripper.actuate(1.0)
        #for _ in range(100):
        #    p.stepSimulation()
        move_lin(gripper, np.array([0.9, -1.0, 1.8]), p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 180.0 * 160.0]))
        move_lin(gripper, np.array([1.2, -0.25, 1.8]), p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 180.0 * 160.0]))
        #move_lin(gripper, np.array([0.9, -1.0, 1.8]), p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 180.0 * 160.0]))
        #move_lin(gripper, np.array([1.2, -0.25, 1.8]), p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 180.0 * 160.0]))
        move_lin(gripper, np.array([0.9, -1.0, 1.8]), p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 180.0 * 160.0]))
        move_lin(gripper, np.array([2.0, -3.15, 1.2]), p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]))
        move_lin(gripper, np.array([2.5,-3.15,1.2]), p.getQuaternionFromEuler([math.pi/2,0,math.pi/2]))
        gripper.actuate(0.0)
        for _ in range(100):
            p.stepSimulation()
        move_lin(gripper, np.array([2.0, -3.15, 1.2]), p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]))





    p.stopStateLogging(lID)
    while True:
        p.stepSimulation()

