import numpy as np

from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot


class ProjectorBase:
    def __init__(self, camera_pose):
        self.camera_pose = camera_pose

    def project_tcp_to_camera_coord(self, tcp, rotation_rep = "quaternion", rotation_rep_convention = None):
        tcp = mat_to_xyz_rot(
            np.linalg.inv(self.camera_pose) @ xyz_rot_to_mat(
                tcp,
                rotation_rep = rotation_rep,
                rotation_rep_convention = rotation_rep_convention
            ), 
            rotation_rep = rotation_rep,
            rotation_rep_convention = rotation_rep_convention
        )
        return tcp

    def project_tcp_to_base_coord(self, tcp, rotation_rep = "quaternion", rotation_rep_convention = None):
        tcp = mat_to_xyz_rot(
            self.camera_pose @ xyz_rot_to_mat(
                tcp, 
                rotation_rep = rotation_rep,
                rotation_rep_convention = rotation_rep_convention
            ),
            rotation_rep = rotation_rep,
            rotation_rep_convention = rotation_rep_convention
        )
        return tcp


class SingleArmProjector:
    def __init__(self, calib_path, global_cam_serial = None):
        self.calib_file = np.load(calib_path, allow_pickle = True).item()
        if global_cam_serial is None:
            global_cam_serial = self.calib_file["camera_serials_global"][0]
        else:
            assert global_cam_serial in self.calib_file["camera_serials_global"]
        self.camera_serials = {
            "global": global_cam_serial,
            "inhand": self.calib_file["camera_serial_inhand"]
        }
        self.intrinsics = self.calib_file["intrinsics"]

        self.depth_scales = {}
        for cam_serial in self.calib_file["camera_serials"]:
            # TODO: check the depth scale for your own camera
            if cam_serial[0] == 'f':   # L515 camera
                self.depth_scales[cam_serial] = 4000.
            else:
                self.depth_scales[cam_serial] = 1000.

        self.projector = ProjectorBase(np.linalg.inv(self.calib_file["camera_to_robot"][global_cam_serial]))

    def project_tcp_to_camera_coord(self, tcp, rotation_rep = "quaternion", rotation_rep_convention = None):
        tcp = self.projector.project_tcp_to_camera_coord(tcp, rotation_rep, rotation_rep_convention)
        return tcp
    
    def project_tcp_to_base_coord(self, tcp, rotation_rep = "quaternion", rotation_rep_convention = None):
        tcp = self.projector.project_tcp_to_base_coord(tcp, rotation_rep, rotation_rep_convention)
        return tcp


class DualArmProjector:
    def __init__(self, calib_path, global_cam_serial = None):
        self.calib_file = np.load(calib_path, allow_pickle = True).item()
        if global_cam_serial is None:
            global_cam_serial = self.calib_file["camera_serials_global"][0]
        else:
            assert global_cam_serial in self.calib_file["camera_serials_global"]
        self.camera_serials = {
            "global": global_cam_serial,
            "inhand_left": self.calib_file["camera_serial_inhand_left"],
            "inhand_right": self.calib_file["camera_serial_inhand_right"]
        }
        self.intrinsics = self.calib_file["intrinsics"]

        self.depth_scales = {}
        for cam_serial in self.calib_file["camera_serials"]:
            # TODO: check the depth scale for your own camera
            if cam_serial[0] == 'f':   # L515 camera
                self.depth_scales[cam_serial] = 4000.
            else:
                self.depth_scales[cam_serial] = 1000.

        self.projector_left = ProjectorBase(np.linalg.inv(self.calib_file["camera_to_robot_left"][global_cam_serial]))
        self.projector_right = ProjectorBase(np.linalg.inv(self.calib_file["camera_to_robot_right"][global_cam_serial]))

    def project_tcp_to_camera_coord(self, tcp, robot = "left", rotation_rep = "quaternion", rotation_rep_convention = None):
        assert robot in ["left", "right"]
        projector = self.projector_left if robot == "left" else self.projector_right
        tcp = projector.project_tcp_to_camera_coord(tcp, rotation_rep, rotation_rep_convention)
        return tcp
    
    def project_tcp_to_base_coord(self, tcp, robot = "left", rotation_rep = "quaternion", rotation_rep_convention = None):
        assert robot in ["left", "right"]
        projector = self.projector_left if robot == "left" else self.projector_right
        tcp = projector.project_tcp_to_base_coord(tcp, rotation_rep, rotation_rep_convention)
        return tcp
