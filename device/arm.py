'''
Flexiv robot interface, built upon Flexiv RDK: https://github.com/FlexivArmics/flexiv_rdk/

Author: Hongjie Fang.
'''

import time
import numpy as np

from device import flexivrdk
from utils.transformation import quat_angle


class FlexivArm:
    '''
    Flexiv Arm Interface.
    '''
    def __init__(
        self, 
        serial: str,
        **kwargs
    ) -> None:
        '''
        Initialization.
        
        Parameters:
        - serial: str, required, the serial of the robot.
        '''
        self.mode = flexivrdk.Mode
        self.robot = flexivrdk.Robot(serial)
        if self.robot.fault():
            if not self.robot.ClearFault():
                raise RuntimeError("Cannot clear faults on the robot.")
        self.robot.Enable()
        while not self.robot.operational():
            time.sleep(0.5)
        self.current_mode = None

        self.threshold_joint = kwargs.get("threshold_joint", 0.001)
        self.threshold_xyz = kwargs.get("threshold_xyz", 0.001)
        self.threshold_rot = kwargs.get("threshold_rot", 0.001)
        self.waiting_gap = kwargs.get("waiting_gap", 0.05)
        self.timeout = kwargs.get("timeout", 10)

    def get_states(self):
        """
        Get current robot states.
        """
        robot_states = self.robot.states()
        return {
            "tcp_pose": np.array(robot_states.tcp_pose, dtype = np.float32),
            "joint_pos": np.array(robot_states.q, dtype = np.float32),
            "tcp_vel": np.array(robot_states.tcp_vel, dtype = np.float32),
            "joint_vel": np.array(robot_states.dq, dtype = np.float32),
            "force_torque": np.array(robot_states.ext_wrench_in_tcp, dtype = np.float32),
        }

    def get_tcp_pose(self):
        """
        Get current tcp pose.
        """
        return np.array(self.robot.states().tcp_pose, dtype = np.float32)
    
    def get_joint_pos(self):
        """
        Get current joint position.
        """
        return np.array(self.robot.states().q, dtype = np.float32)
    
    def get_tcp_vel(self):
        """
        Get current tcp velocity.
        """
        return np.array(self.robot.states().tcp_vel, dtype = np.float32)

    def get_joint_vel(self):
        """
        Get current joint velocity.
        """
        return np.array(self.robot.states().dq, dtype = np.float32)
    
    def get_force_torque_tcp(self):
        """
        Get current tcp force/torque.
        """    
        return np.array(self.robot.states().ext_wrench_in_tcp, dtype = np.float32)
    
    def switch_mode(self, mode):
        mode = getattr(self.mode, mode)
        if mode != self.current_mode:
            self.robot.SwitchMode(mode)
            self.current_mode = mode

    def cali_sensor(self) -> None:
        """
        Calibrate sensors.
        """
        self.switch_mode("NRT_PRIMITIVE_EXECUTION")
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
    
    def set_joint_impedance(
        self,
        stiffness,
        damping_ratio = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    ) -> None:
        """
        Set joint impedance (stiffness and damping_ratio).
        """
        self.switch_mode("NRT_JOINT_IMPEDANCE")
        self.robot.SetJointImpedance(stiffness, damping_ratio)
    
    def set_cartesian_impedance(
        self,
        stiffness,
        damping_ratio = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    ) -> None:
        """
        Set cartesian impedance (stiffness and damping ratio).
        """
        self.switch_mode("NRT_CARTESIAN_MOTION_FORCE")
        self.robot.SetCartesianImpedance(stiffness, damping_ratio)

    def send_tcp_pose(
        self,
        pose,
        max_vel = 0.5,
        max_acc = 2.0,
        max_angular_vel = 1.0,
        max_angular_acc = 5.0,
        blocking: bool = False
    ) -> None:
        self.switch_mode("NRT_CARTESIAN_MOTION_FORCE")
        self.robot.SendCartesianMotionForce(
            pose, 
            max_linear_vel = max_vel,
            max_angular_vel = max_angular_vel,
            max_linear_acc = max_acc,
            max_angular_acc = max_angular_acc
        )

        if blocking:
            self.wait_for_tcp_move(pose)
    
    def send_joint_pos(
        self,
        pos,
        max_vel = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
        max_acc= [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        target_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        impedance = True,
        blocking = False
    ):  
        """
        Send joint position to the robot.

        Parameters:
        - pos: the target joint position of the robot;
        - max_vel: the maximum velocity of the robot;
        - max_acc: the maximum acceleration of the robot;
        - target_vel: the target velocity of the robot at the target joint position;
        - target_acc: the target acceleration of the robot at the target joint position;
        - impedance: whether to use impedance control;
        - blocking: whether to wait for the robot to complete movements, only available if impedance is False.
        """
        if impedance:
            self.switch_mode("NRT_JOINT_IMPEDANCE")
        else:
            self.switch_mode("NRT_JOINT_POSITION")
        
        self.robot.SendJointPosition(
            pos,
            target_vel,
            target_acc,
            max_vel,
            max_acc
        )

        if blocking:
            if impedance:
                print("[Warning] The blocking parameter is unavailable and thus ignored for impedance control.")
            else:
                self.wait_for_joint_move(pos)

    def wait_for_tcp_move(self, target_pose):
        start_time = time.time()
        has_warned = False
        while True:
            cur_pose = self.get_tcp_pose()
            if np.max(np.abs(target_pose[:3] - cur_pose[:3])) <= self.threshold_xyz and quat_angle(target_pose[3:], cur_pose[3:]) <= self.threshold_rot:
                break
            if time.time() - start_time > self.timeout and not has_warned:
                print("[Warning] Timeout for send_tcp_pose.")
                has_warned = True
            time.sleep(self.waiting_gap)

    def wait_for_joint_move(self, target_pos):
        start_time = time.time()
        has_warned = False
        while True:
            if np.max(np.abs(target_pos - self.get_joint_pos())) <= self.threshold_joint:
                break
            if time.time() - start_time > self.timeout and not has_warned:
                print("[Warning] Timeout for send_joint_pos.")
                has_warned = True
    
    def stop(self):
        self.robot.Stop()