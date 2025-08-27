import yaml
import torch
import argparse
import numpy as np
import open3d as o3d
import torchvision.transforms as T

from copy import deepcopy
from easydict import EasyDict as edict

from utils.training import set_seed
from utils.ensemble import EnsembleBuffer
from remote_eval import WebsocketClientPolicy
from eval_agent import SingleArmAgent, DualArmAgent
from dataset.data_utils import resize_image, ImageProcessor
from dataset.projector import SingleArmProjector, DualArmProjector


default_args = edict({
    "type": "local",
    "calib": "calib/",
    "config": "config/dual_teleop_dino.yaml",
    "ckpt": "logs/collect_toys",
    "host": "127.0.0.1",
    "port": 8000
})


def create_point_cloud(colors, depths, intrinsics, config, depth_scale = 1000.0, rescale_factor = 1):
    """
    color, depth => point cloud
    """
    if rescale_factor != 1:
        H, W = depths.shape
        h, w = int(H * rescale_factor), int(W * rescale_factor)
        colors = colors.transpose([2, 0, 1]).astype(np.float32)
        colors = torch.from_numpy(colors)
        colors = np.ascontiguousarray(resize_image(colors, [h, w]).numpy().transpose([1, 2, 0]))
        depths = depths.astype(np.float32)
        depths = torch.from_numpy(depths[np.newaxis])
        depths = resize_image(depths, [h,w], interpolation = T.InterpolationMode.NEAREST)[0]
        depths = depths.numpy()

    # generate point cloud
    h, w = depths.shape
    fx, fy = intrinsics[0, 0] * rescale_factor, intrinsics[1, 1] * rescale_factor
    cx, cy = intrinsics[0, 2] * rescale_factor, intrinsics[1, 2] * rescale_factor
    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    # crop point cloud
    bbox3d = o3d.geometry.AxisAlignedBoundingBox(config.deploy.workspace.min, config.deploy.workspace.max)
    cloud = cloud.crop(bbox3d)
    # downsample
    cloud = cloud.voxel_down_sample(config.data.voxel_size)
    return cloud


def create_input(colors, depths, cam_intrinsics, config, depth_scale = 1000.0, rescale_factor = 1):
    """
    colors, depths => coords, points
    """
    # create point cloud
    cloud = create_point_cloud(
        colors, 
        depths, 
        cam_intrinsics, 
        config,
        depth_scale = depth_scale,
        rescale_factor = rescale_factor,
    )

    # convert to sparse tensor
    points = np.asarray(cloud.points)
    coords = np.ascontiguousarray(points / config.data.voxel_size, dtype = np.int32)

    return coords, points, cloud
    

def create_batch(coords, points):
    """
    coords, points => batch coords, batch feats
    """
    import MinkowskiEngine as ME
    input_coords = [coords]
    input_feats = [points.astype(np.float32)]
    coords_batch, feats_batch = ME.utils.sparse_collate(input_coords, input_feats)
    return coords_batch, feats_batch


def process_state(state, config, to_control = True):
    if config.robot_type == "single":
        if to_control:
            state[..., 0: 3] = (state[..., 0: 3] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 9] = (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
        else:
            state[..., 0: 3] = (state[..., 0: 3] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 9] = state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
    else:
        if to_control:
            state[..., 0: 3] = (state[..., 0: 3] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 10: 13] = (state[..., 10: 13] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 9] = (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
            state[..., 19] = (state[..., 19] + 1) / 2.0 * config.data.normalization.max_gripper_width
        else:
            state[..., 0: 3] = (state[..., 0: 3] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 10: 13] = (state[..., 10: 13] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 9] = state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
            state[..., 19] = state[..., 19] / config.data.normalization.max_gripper_width * 2.0 - 1

    return state



def evaluate(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # load config
    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader = yaml.FullLoader))
    config.data.normalization.trans_min = np.asarray(config.data.normalization.trans_min)
    config.data.normalization.trans_max = np.asarray(config.data.normalization.trans_max)

    # set seed
    set_seed(config.deploy.seed)

    # load policy for local inference
    if args.type == "local":
        from policy import RISE2
        # set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load policy
        print("Loading policy ...")
        policy = RISE2(
            num_action = config.data.num_action,
            obs_feature_dim = config.model.obs_feature_dim,
            cloud_enc_dim = config.model.cloud_enc_dim,
            image_enc_dim = config.model.image_enc_dim,
            action_dim = 10 if config.robot_type == "single" else 20,
            hidden_dim = config.model.hidden_dim,
            nheads = config.model.nheads,
            num_attn_layers = config.model.num_attn_layers,
            dim_feedforward = config.model.dim_feedforward,
            dropout = config.model.dropout,
            image_enc = config.model.image_enc,
            interp_fn_mode = config.model.interp_fn_mode,
            image_enc_finetune = config.model.image_enc_finetune,
            image_enc_dtype = config.model.image_enc_dtype
        ).to(device)

        # load checkpoint
        assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
        policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
        print("Checkpoint {} loaded.".format(args.ckpt))

        # set evaluation
        policy.eval()

    else:
        # connect to remote inference service
        print("Connecting to remote server ...")
        policy = WebsocketClientPolicy(host = args.host, port = args.port)

    # projector
    Projector = SingleArmProjector if config.robot_type == "single" else DualArmProjector
    projector = Projector(args.calib, config.deploy.agent.camera_serial)

    # image processor
    image_enc = config.model.image_enc
    if image_enc == "resnet18":
        img_size = config.data.aligner.img_size_resnet
        img_coord_size = config.data.aligner.img_coord_size_resnet
    elif image_enc.startswith("dino"):
        img_size = config.data.aligner.img_size_dinov2
        img_coord_size = config.data.aligner.img_coord_size_dinov2
    else:
        raise ValueError(f"Unknown image encoder: {image_enc}")
    
    image_processor = ImageProcessor(
        img_size = img_size,
        img_coord_size = img_coord_size,
        voxel_size = config.data.voxel_size,
        img_mean = config.data.normalization.img_mean,
        img_std = config.data.normalization.img_std
    )

    # evaluation
    Agent = SingleArmAgent if config.robot_type == "single" else DualArmAgent
    agent = Agent(**config.deploy.agent)

    # ensemble buffer
    ensemble_buffer = EnsembleBuffer(mode = config.deploy.ensemble_mode)

    # evaluation rollout
    print("Ready for rollout. Press Enter to continue...")
    input()
    
    with torch.inference_mode():
        for t in range(config.deploy.max_steps):
            if t % config.deploy.num_inference_steps == 0:
                # pre-process inputs
                colors, depths = agent.get_global_observation()
                # create cloud inputs
                coords, points, cloud = create_input(
                    colors,
                    depths,
                    cam_intrinsics = agent.intrinsics,
                    config = config,
                    depth_scale = agent.camera.depth_scale,
                    rescale_factor = 1.0
                )

                # create image inputs
                image_coords = image_processor.get_image_coordinates(depths, agent.intrinsics, agent.camera.depth_scale)        
                colors, image_coords = image_processor.preprocess_images(colors, image_coords)

                # predict action
                if args.type == "local":
                    import MinkowskiEngine as ME
                    coords_batch, feats_batch = create_batch(coords, points)
                    coords_batch, feats_batch = coords_batch.to(device), feats_batch.to(device)
                    cloud_data = ME.SparseTensor(feats_batch, coords_batch)

                    colors = colors.unsqueeze(0).to(device)
                    image_coords = image_coords.unsqueeze(0).to(device)

                    # predict
                    pred_raw_action = policy(
                        cloud_data, 
                        colors, 
                        image_coords,
                        actions = None,
                    ).squeeze(0).cpu().numpy()

                else:
                    obs_dict = {
                        "coords": coords,
                        "points": points,
                        "colors": colors.numpy(),
                        "image_coords": image_coords.numpy()
                    }

                    pred_raw_action = deepcopy(policy.infer(obs_dict)["actions"])

                # unnormalize predicted actions
                action = process_state(pred_raw_action, config, to_control = True)

                # visualization
                if config.deploy.vis:
                    tcp_vis_list = []
                    for raw_tcp in action:
                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
                        tcp_vis_list.append(tcp_vis)
                        if config.robot_type == "dual":
                            tcp_vis_r = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[10:13])
                            tcp_vis_list.append(tcp_vis_r)
                    o3d.visualization.draw_geometries([cloud, *tcp_vis_list])
                
                # project action to base coordinate
                if config.robot_type == "single":
                    action_tcp = projector.project_tcp_to_base_coord(action[..., :9], rotation_rep = "rotation_6d")
                    action = np.concatenate([action_tcp, action[..., 9:10]], axis = -1)
                else:
                    action_left_tcp = projector.project_tcp_to_base_coord(action[..., :9], "left", rotation_rep = "rotation_6d")
                    action_right_tcp = projector.project_tcp_to_base_coord(action[..., 10:19], "right", rotation_rep = "rotation_6d")
                    action = np.concatenate([action_left_tcp, action[..., 9:10], action_right_tcp, action[..., 19:20]], axis = -1)
                
                # add to ensemble buffer
                ensemble_buffer.add_action(action, t)
            
            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            
            if step_action is None:   # no action in the buffer => no movement.
                continue
            
            agent.action(step_action, rotation_rep = "rotation_6d")
    
    agent.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', action = 'store', type = str, help = 'evaluation type, choices: ["local", "remote"].', required = True, choices = ["local", "remote"])
    parser.add_argument('--calib', action = 'store', type = str, help = 'calibration path', required = True)
    parser.add_argument('--config', action = 'store', type = str, help = 'data and model config during training and deployment', required = True)
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = False, default = None)
    parser.add_argument('--host', action = 'store', type = str, help = 'server host address', required = False, default = "127.0.0.1")
    parser.add_argument('--port', action = 'store', type = int, help = 'server port', required = False, default = 8000)

    evaluate(vars(parser.parse_args()))