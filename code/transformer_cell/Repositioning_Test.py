import math
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_industrial as pi
from pybullet_industrial import GCodeProcessor
def execute_gcode(processor, command):
    processor.gcode = processor.read_gcode(command)
    for _ in iter(processor):
        # Execute the simulation steps

        for _ in range(50):
            p.stepSimulation()
            time.sleep(0.01)
if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    comau_urdf = os.path.join(dirname,
                              'robot_descriptions', 'comau_nj290_robotNC.urdf')
    plate_urdf = os.path.join(dirname,
                                'robot_descriptions', 'plate.urdf')
    plane_urdf = os.path.join(dirname,
                                 'objects', 'plane.urdf')

    physics_client = p.connect(p.GUI, options=  '--width=1920 --height=1060 --background_color_red=0.8 --background_color_green=0.8 --background_color_blue=0.83')
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 0: no gui, 1, basic gui

    #p.resetDebugVisualizerCamera(cameraDistance=6.6, cameraYaw=-7, cameraPitch=-35,
    #                             cameraTargetPosition=np.array([-0.5, 0, -1]))

    p.setPhysicsEngineParameter(numSolverIterations=10000)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF(plane_urdf, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
    #load Robots__________________________________________________
    start_pos = np.array([-3.6353162, -0.6, 0])
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot = pi.RobotBase(comau_urdf, start_pos, start_orientation)
    start_pos2 = np.array([-0.12, -2.59, 0])
    start_orientation2 = p.getQuaternionFromEuler([0, 0, -np.pi/2])
    robot2 = pi.RobotBase(comau_urdf, start_pos2, start_orientation2)
    plate = pi.EndeffectorTool(plate_urdf, np.array([-0.12,-4,2]),
                                    p.getQuaternionFromEuler([-math.pi/2,0,0]))
    robot.set_joint_position(
        {'q2': np.deg2rad(-15.0), 'q3': np.deg2rad(-90.0)})
    robot2.set_joint_position(
        {'q1': np.deg2rad(-90.0), 'q2': np.deg2rad(0.0),
         'q3': np.deg2rad(-90.0), 'q4': np.deg2rad(0.0),
         'q5': np.deg2rad(0.0), 'q6': np.deg2rad(0.0)})
    endeffector_list = []
    endeffector_list.append(plate)
    # T-Commands have to be added in this convention
    t_commands = {
        # "0": [lambda: decouple_endeffector(test_gripper)],
        # "1": [lambda: couple_endeffector(test_gripper, test_robot, 'link6')],
        "2": [lambda: plate.couple(robot2, 'link6')],
        "3": [lambda: plate.decouple()]}
    # M-Commands have to be added in this convention
    m_commands = {}

    sinumerik = GCodeProcessor(None, robot2,
                                          endeffector_list,
                                          m_commands, t_commands,
                                          offset= np.array([[-0.12, -2.59, 0],[0.0, 0.0, 0.0]]),
                                          interpolation_precision=0.05)
    #plate.couple(robot2, 'link6')
    execute_gcode(sinumerik, f'G0 X1.5 Y-1 Z1.3 A0 B0 C0')
    execute_gcode(sinumerik, f'G0 X1.5 Y-1 Z1.3 A0 B{np.pi/2} C{np.pi}')
    execute_gcode(sinumerik, 'T2')
    fofa_path = os.path.join(dirname,
                             'Objects', 'InRePro', 'InRePro.urdf')
    p.loadURDF(fofa_path, [0,0,0], p.getQuaternionFromEuler([math.pi/2,0,0]), useFixedBase=True, globalScaling=0.001)
    ponticon_path = os.path.join(dirname,
                             'Objects', 'ponticon', 'ponticon.urdf')
    p.loadURDF(ponticon_path, [0,0,0], p.getQuaternionFromEuler([math.pi/2,0,0]), useFixedBase=True, globalScaling=0.001)
    dmg_path = os.path.join(dirname,
                             'Objects', 'DMG', 'DMG.urdf')
    p.loadURDF(dmg_path, [0,0,0], p.getQuaternionFromEuler([math.pi/2,0,0]), useFixedBase=True, globalScaling=0.001)
    for _ in range(100):
        p.stepSimulation()
    execute_gcode(sinumerik, f'G1 X1.5 Y-0.675 Z1.1 A0 B0 C{-np.pi/2}')
    execute_gcode(sinumerik, f'G1 X2.923')
    execute_gcode(sinumerik, f'G1 Z0.965')
# plate.set_tool_pose(np.array([2.923,0.675, 0.965]) + start_pos2, p.getQuaternionFromEuler([0,0,-np.pi/2]))
    while True:
        p.stepSimulation()
        time.sleep(0.01)
