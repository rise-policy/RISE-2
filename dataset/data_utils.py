import os
import torch
import numpy as np
import open3d as o3d
import torchvision.transforms as T

from transforms3d.quaternions import quat2mat

from utils.transformation import rot_mat_z_axis, xyz_rot_transform


def load_action(lowdim_dir, frame_id, robot_type = "dual", gripper_info_type = "command"):
    assert robot_type in ["single", "dual"]
    assert gripper_info_type in ["state", "command"]
    gripper_info_idx = 0 if gripper_info_type == "state" else 1

    lowdim_dict = np.load(
        os.path.join(lowdim_dir, "{}.npy".format(frame_id)),
        allow_pickle = True
    ).item()
    if robot_type == "single":
        tcp = lowdim_dict["robot"][0:7]
        gripper_command = lowdim_dict["gripper"][gripper_info_idx: gripper_info_idx + 1]
        action = np.concatenate([tcp, gripper_command])
    else:
        ltcp = lowdim_dict["robot_left"][0:7]
        rtcp = lowdim_dict["robot_right"][0:7]
        lgripper_command = lowdim_dict["gripper_left"][gripper_info_idx: gripper_info_idx + 1]
        rgripper_command = lowdim_dict["gripper_right"][gripper_info_idx: gripper_info_idx + 1]
        action = np.concatenate([ltcp, lgripper_command, rtcp, rgripper_command])

    return action
    
def resize_image(image_list, image_size, interpolation = T.InterpolationMode.BILINEAR):
    resize = T.Resize(image_size, interpolation)
    image_list_resized = resize(image_list)
    return image_list_resized

def vis_data(
        points,
        colors,
        action_tcps = None,
        workspace_min = None,
        workspace_max = None,
        translation_min = None,
        translation_max = None
    ):
    print(points.min(axis=0), points.max(axis=0))
    contents = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    contents.append(pcd)
    # red box stands for the workspace range
    if workspace_max is not None and workspace_max is not None:
        bbox3d_1 = o3d.geometry.AxisAlignedBoundingBox(workspace_min, workspace_max)
        bbox3d_1.color = [1, 0, 0]
        contents.append(bbox3d_1)
    # green box stands for the translation normalization range
    if translation_min is not None and translation_max is not None:
        bbox3d_2 = o3d.geometry.AxisAlignedBoundingBox(translation_min, translation_max)
        bbox3d_2.color = [0, 1, 0]
        contents.append(bbox3d_2)
    if action_tcps is not None:
        action_tcps_vis = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "matrix")
        traj = []
        for i in range(len(action_tcps)):
            action = action_tcps_vis[i]
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03).transform(action)
            traj.append(frame)
        contents += traj
    o3d.visualization.draw_geometries(contents)


class TrajLoader:
    def __init__(
        self,
        robot_type = "dual",
        sample_traj = True,
        trans_delta = 0.005,
        rot_delta = np.pi / 24,
        width_delta = 0.005,
    ):
        assert robot_type in ["single", "dual"]
        self.sample_traj = sample_traj
        self.robot_type = robot_type
        self.trans_delta = trans_delta
        self.rot_delta = rot_delta
        self.width_delta = width_delta

    def load_traj(self, demo_path):
        lowdim_dir = os.path.join(demo_path, "lowdim")
        frame_ids = [
            int(os.path.splitext(x)[0])
            for x in sorted(os.listdir(lowdim_dir))
            if x[-4:] == ".npy"
        ]

        # get data
        actions = []
        for frame_id in frame_ids:
            action = load_action(
                lowdim_dir, frame_id, self.robot_type, gripper_info_type = "command"
            )
            actions.append(action)
        actions = np.stack(actions, axis = 0)

        if self.sample_traj:
            frame_ids, actions = self.sample_frames(frame_ids, actions)

        return frame_ids, actions

    def _diff_action(self, action1, action2):
        def _diff_translation(t1, t2, delta):
            dist = np.linalg.norm(t1 - t2)
            return dist > delta
        def _diff_rotation(q1, q2, delta):
            mat1, mat2 = quat2mat(q1), quat2mat(q2)
            rot_diff = np.matmul(mat1, mat2.T)
            rot_diff = (np.diag(rot_diff).sum() - 1) / 2
            rot_diff = min(max(rot_diff, -1), 1)
            rot_diff = np.arccos(rot_diff)
            return rot_diff > delta

        tcp1, width1 = action1[0:7], action1[7:8]
        tcp2, width2 = action2[0:7], action2[7:8]
        if _diff_translation(tcp1[:3], tcp2[:3], self.trans_delta):
            return True
        if _diff_rotation(tcp1[3:7], tcp2[3:7], self.rot_delta):
            return True
        if _diff_translation(width1, width2, self.width_delta):
            return True
        
        return False

    def sample_frames(self, frame_ids, actions):
        kept_ids = [0]
        prev_id = 0

        for curr_id in range(1, len(frame_ids)):
            action1, action2 = actions[prev_id], actions[curr_id]
            if self.robot_type == "single":
                is_diff = self._diff_action(action1, action2)
            else:
                is_diff_left = self._diff_action(action1[0:8], action2[0:8])
                is_diff_right = self._diff_action(action1[8:16], action2[8:16])
                is_diff = is_diff_left | is_diff_right
            if is_diff:
                kept_ids.append(curr_id)
                prev_id = curr_id

        kept_frame_ids = [frame_ids[i] for i in kept_ids]
        kept_actions = [actions[i] for i in kept_ids]

        return kept_frame_ids, kept_actions


class ImageProcessor:
    def __init__(
            self,
            img_size,
            img_coord_size,
            voxel_size = 0.005,
            img_mean = [0.485, 0.456, 0.406],
            img_std = [0.229, 0.224, 0.225],
        ):
        self.img_size = img_size
        self.voxel_size = voxel_size
        self.img_mean = np.asarray(img_mean)
        self.img_std = np.asarray(img_std)

        # pooling to get averaged image 3D coords, pooled size is based on the output of image encoder
        self.image_coord_pooling = torch.nn.AdaptiveAvgPool2d(img_coord_size)

    def fill_depth_hole(self, grid_points, mask, kernel_size = [60, 64]):
        H, W, _ = grid_points.shape
        h, w = H // kernel_size[0], W // kernel_size[1]
        while H % h != 0:
            h += 1
        while W % w != 0:
            w += 1
        kernel_size = (H // h, W // w)
        grid_points = grid_points.reshape(h, kernel_size[0], w, kernel_size[1], 3)
        grid_points = grid_points.transpose([0, 2, 1, 3, 4]).reshape(h, w, -1, 3)
        mask = mask.reshape(h, kernel_size[0], w, kernel_size[1])
        mask = mask.transpose([0, 2, 1, 3]).reshape(h, w, -1)
        points_mean = grid_points.sum(axis=2) / (mask.astype(np.float32).sum(axis=-1, keepdims=True) + 1e-6)
        points_mean = np.tile(points_mean[:, :, np.newaxis], [1, 1, kernel_size[0]*kernel_size[1], 1])
        grid_points[~mask] = points_mean[~mask]
        grid_points = grid_points.reshape(h, w, kernel_size[0], kernel_size[1], 3).transpose([0, 2, 1, 3, 4]).reshape(H, W, 3)
        return grid_points
    
    def get_image_coordinates(self, depths, intrinsics, depth_scale, fill_hole=True):
        depths = depths.astype(np.float32)
        H, W = depths.shape
        depths = torch.from_numpy(depths[np.newaxis].astype(np.float32))
        depths = resize_image(depths, self.img_size, interpolation=T.InterpolationMode.NEAREST)[0].numpy()

        h, w = depths.shape
        rescale_factor = h / H, w / W
        fx, fy = intrinsics[0, 0] * rescale_factor[1], intrinsics[1, 1] * rescale_factor[0]
        cx, cy = intrinsics[0, 2] * rescale_factor[1], intrinsics[1, 2] * rescale_factor[0]
        depth_scale = 1000

        xmap, ymap = np.arange(w), np.arange(h)
        xmap, ymap = np.meshgrid(xmap, ymap)
        xmap, ymap = xmap.astype(np.float32), ymap.astype(np.float32)
        points_z = depths.astype(np.float32) / depth_scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        points = np.stack([points_x, points_y, points_z], axis=-1)

        if fill_hole:
            depth_mask = depths > 0
            points = self.fill_depth_hole(points, depth_mask)

        image_coords = points.astype(np.float32).transpose([2, 0, 1])

        return image_coords
    
    def preprocess_images(self, image, coords):
        image = image.astype(np.float32) / 255.0
        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image).to(torch.float32)
        image = resize_image(image, self.img_size)
        image -= torch.from_numpy(self.img_mean).view(-1, 1, 1)
        image /= torch.from_numpy(self.img_std).view(-1, 1, 1)

        coords = np.stack(coords, axis=0)
        coords /= self.voxel_size
        coords = torch.from_numpy(coords).to(torch.float32)
        coords = self.image_coord_pooling(coords)

        return image, coords