"""
Evaluation Agent.
"""

import time
import numpy as np
from device.arm import FlexivArm
from device.camera import RealSenseRGBDCamera
from utils.transformation import xyz_rot_transform
from device.gripper import Robotiq2F85Gripper, DahuanAG95Gripper


class SingleArmAgent:
    """
    Evaluation single-arm agent with Flexiv arms, Dahuan gripper and an Intel RealSense RGB-D camera.
    
    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        robot_serial,
        gripper_port,
        camera_serial,
        max_contact_wrench = [30, 30, 30, 10, 10, 10],
        max_vel = 0.5,
        max_acc = 2.0,
        max_angular_vel = 1.0,
        max_angular_acc = 5.0,
        **kwargs
    ):
        # initialize
        self.robot = FlexivArm(robot_serial)
        self.gripper = DahuanAG95Gripper(gripper_port)
        self.camera_serial = camera_serial
        self.camera = RealSenseRGBDCamera(serial = camera_serial)
        self.intrinsics = self.camera.get_intrinsic()
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_angular_vel = max_angular_vel
        self.max_angular_acc = max_angular_acc
        
        self.last_gripper = 0

        # move to ready pose
        self.robot.send_tcp_pose(
            self.ready_pose,
            max_vel = self.max_vel,
            max_acc = self.max_acc,
            max_angular_vel = self.max_angular_vel,
            max_angular_acc = self.max_angular_acc
        )
        self.gripper.close_gripper()
        time.sleep(5)
        self.robot.cali_sensor()

        # set max contact wrench
        self.robot.robot.SetMaxContactWrench(max_contact_wrench)
    
    @property
    def ready_pose(self):
        return np.array([0.5, 0, 0.17, 0, 0, 1, 0], dtype=np.float32)

    def get_global_observation(self):
        _, colors, depths = self.camera.get_rgbd_images()
        return colors, depths
    
    def get_proprio(self, rotation_rep = "rotation_6d", rotation_rep_convention = None, with_joint = False):
        tcp_pose = self.robot.get_tcp_pose()
        tcp_pose = xyz_rot_transform(
            tcp_pose,
            from_rep = "quaternion",
            to_rep = rotation_rep,
            to_convention = rotation_rep_convention
        )
        if with_joint:
            joint_pos = self.robot.get_joint_pos()
        gripper_width = self.gripper.get_states()["width"]

        proprio = np.concatenate([tcp_pose, [gripper_width]], axis = 0)
        if with_joint:
            proprio_joint = np.concatenate([joint_pos, [gripper_width]], axis = 0)
            return proprio, proprio_joint
        else:
            return proprio

    def action(self, action, rotation_rep = "rotation_6d", rotation_rep_convention = None):
        tcp_pose = xyz_rot_transform(
            action[: 9],
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )

        self.robot.send_tcp_pose(
            tcp_pose,
            max_vel = self.max_vel,
            max_acc = self.max_acc,
            max_angular_vel = self.max_angular_vel,
            max_angular_acc = self.max_angular_acc
        )
        time.sleep(0.1)
        
        gripper_action = False
        if abs(action[9] - self.last_gripper) >= 0.01:
            self.gripper.set_width(action[9])
            self.last_gripper = action[9]        
            gripper_action = True

        if gripper_action:
            time.sleep(0.5)

    
    def stop(self):
        self.robot.stop()
        self.gripper.stop()
        self.camera.stop()


class DualArmAgent:
    """
    Evaluation dual-arm agent with Flexiv arms, Robotiq grippers and an Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        left_robot_serial,
        right_robot_serial,
        left_gripper_port,
        right_gripper_port,
        camera_serial,
        left_max_contact_wrench = [30, 30, 30, 10, 10, 10],
        right_max_contact_wrench = [30, 30, 30, 10, 10, 10],
        max_vel = 0.5,
        max_acc = 2.0,
        max_angular_vel = 1.0,
        max_angular_acc = 5.0,
        gripper_key = "width",
        **kwargs
    ): 
        # initialize
        self.left_robot = FlexivArm(left_robot_serial)
        self.right_robot = FlexivArm(right_robot_serial)
        self.left_gripper = Robotiq2F85Gripper(left_gripper_port)
        self.right_gripper = Robotiq2F85Gripper(right_gripper_port)
        self.camera_serial = camera_serial
        self.camera = RealSenseRGBDCamera(serial = camera_serial)
        self.intrinsics = self.camera.get_intrinsic()
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_angular_vel = max_angular_vel
        self.max_angular_acc = max_angular_acc

        assert gripper_key in ["width", "action"]
        self.gripper_key = gripper_key
        
        self.last_left_gripper = 0
        self.last_right_gripper = 0

        # move to ready pose
        self.left_robot.send_joint_pos(self.left_ready_pose, max_vel = [0.3] * 7, max_acc = [0.5] * 7, blocking = True)
        self.right_robot.send_joint_pos(self.right_ready_pose, max_vel = [0.3] * 7, max_acc = [0.5] * 7, blocking = True)
        self.left_gripper.close_gripper()
        self.right_gripper.close_gripper()
        time.sleep(5)
        self.left_robot.cali_sensor()
        self.right_robot.cali_sensor()

        # set max contact wrench
        tcp_left = self.left_robot.get_tcp_pose()
        tcp_right = self.right_robot.get_tcp_pose()
        self.left_robot.send_tcp_pose(
            tcp_left,
            max_vel = self.max_vel,
            max_acc = self.max_acc,
            max_angular_vel = self.max_angular_vel,
            max_angular_acc = self.max_angular_acc
        )
        self.right_robot.send_tcp_pose(
            tcp_right,
            max_vel = self.max_vel,
            max_acc = self.max_acc,
            max_angular_vel = self.max_angular_vel,
            max_angular_acc = self.max_angular_acc
        )
        self.left_robot.robot.SetMaxContactWrench(left_max_contact_wrench)
        self.right_robot.robot.SetMaxContactWrench(right_max_contact_wrench)
    
    @property
    def left_ready_pose(self):
        return np.array([
            1.5638101e+00,
            -2.1990967e+00,  
            1.1152865e+00, 
            -1.8542223e+00, 
            -5.2635312e-01,
            2.3846159e-02,  
            1.5591051e-01
        ], dtype=np.float32)

    @property
    def right_ready_pose(self):
        # return np.array([
        #     1.59702516e+00,
        #     -2.24452472e+00,  
        #     1.23052871e+00, 
        #     -1.83945274e+00, 
        #     -4.81370360e-01,
        #     1.86598241e-01,  
        #     2.09201425e-01
        # ], dtype=np.float32)
        return np.array([ 2.164233  , -1.7392905 ,  1.7191751 , -1.8739327 , -0.5582644 ,
        0.8032095 ,  0.44572696], dtype=np.float32)
        
    def get_global_observation(self):
        _, colors, depths = self.camera.get_rgbd_images()
        return colors, depths
    
    def get_proprio(self, rotation_rep = "rotation_6d", rotation_rep_convention = None, with_joint = False):
        left_tcp_pose = self.left_robot.get_tcp_pose()
        right_tcp_pose = self.right_robot.get_tcp_pose()
        left_tcp_pose = xyz_rot_transform(
            left_tcp_pose,
            from_rep = "quaternion",
            to_rep = rotation_rep,
            to_convention = rotation_rep_convention
        )
        right_tcp_pose = xyz_rot_transform(
            right_tcp_pose,
            from_rep = "quaternion",
            to_rep = rotation_rep,
            to_convention = rotation_rep_convention
        )
        if with_joint:
            left_joint_pos = self.left_robot.get_joint_pos()
            right_joint_pos = self.right_robot.get_joint_pos()
        left_gripper_width = self.left_gripper.get_states()[self.gripper_key]
        right_gripper_width = self.right_gripper.get_states()[self.gripper_key]
        
        proprio = np.concatenate([left_tcp_pose, [left_gripper_width], right_tcp_pose, [right_gripper_width]], axis = 0)
        if with_joint:
            proprio_joint = np.concatenate([left_joint_pos, [left_gripper_width], right_joint_pos, [right_gripper_width]], axis = 0)
            return proprio, proprio_joint
        else:
            return proprio

    def action(self, action, rotation_rep = "rotation_6d", rotation_rep_convention = None):
        left_tcp_pose = xyz_rot_transform(
            action[: 9],
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )
        right_tcp_pose = xyz_rot_transform(
            action[10: 19],
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )

        self.left_robot.send_tcp_pose(
            left_tcp_pose,
            max_vel = self.max_vel,
            max_acc = self.max_acc,
            max_angular_vel = self.max_angular_vel,
            max_angular_acc = self.max_angular_acc
        )
        self.right_robot.send_tcp_pose(
            right_tcp_pose,
            max_vel = self.max_vel,
            max_acc = self.max_acc,
            max_angular_vel = self.max_angular_vel,
            max_angular_acc = self.max_angular_acc
        )
        time.sleep(0.1)
        
        gripper_action = False
        if abs(action[9] - self.last_left_gripper) >= 0.01:
            self.left_gripper.set_width(action[9])
            self.last_left_gripper = action[9]        
            gripper_action = True
        if abs(action[19] - self.last_right_gripper) >= 0.01:
            self.right_gripper.set_width(action[19])
            self.last_right_gripper = action[19]   
            gripper_action = True

        if gripper_action:
            time.sleep(0.5)

    
    def stop(self):
        self.left_robot.stop()
        self.right_robot.stop()
        self.left_gripper.stop()
        self.right_gripper.stop()
        self.camera.stop()
    