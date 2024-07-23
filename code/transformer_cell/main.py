import math
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_industrial as pi



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

    physics_client = p.connect(p.GUI, options=  '--width=1920 --height=1060 --background_color_red=1 --background_color_green=1 --background_color_blue=1')
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 0: no gui, 1, basic gui

    p.resetDebugVisualizerCamera(cameraDistance=6.6, cameraYaw=-7, cameraPitch=-35,
                                 cameraTargetPosition=np.array([-0.5, 0, -1]))

    p.setPhysicsEngineParameter(numSolverIterations=10000)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())


    p.loadURDF(plane_urdf, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, globalScaling=0.22)
    #load Robots__________________________________________________
    start_pos = np.array([-3.6353162, -0.6, 0])
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot = pi.RobotBase(comau_urdf, start_pos, start_orientation)
    start_pos2 = np.array([-0.12, -2.59, 0])
    start_orientation2 = p.getQuaternionFromEuler([0, 0, np.pi])
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




    while True:
        p.stepSimulation()

