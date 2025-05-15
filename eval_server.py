import time
import yaml
import torch
import argparse
import numpy as np
import MinkowskiEngine as ME

from copy import deepcopy
from easydict import EasyDict as edict

from policy import RISE2
from utils.training import set_seed
from remote_eval import WebsocketPolicyServer


default_args = edict({
    "config": "config/dual_teleop.yaml",
    "ckpt": "logs/collect_toys",
    "host": "127.0.0.1",
    "port": 8000
})


def create_batch(coords, points):
    """
    coords, points => batch coords, batch feats
    """
    input_coords = [coords]
    input_feats = [points.astype(np.float32)]
    coords_batch, feats_batch = ME.utils.sparse_collate(input_coords, input_feats)
    return coords_batch, feats_batch


class PolicyInferenceAgent:
    def __init__(self, config, ckpt_path):
        # set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load policy
        print("Loading policy ...")
        self.policy = RISE2(
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
        ).to(self.device)

        # load checkpoint
        assert ckpt_path is not None, "Please provide the checkpoint to evaluate."
        self.policy.load_state_dict(torch.load(ckpt_path, map_location = self.device), strict = False)
        print("Checkpoint {} loaded.".format(ckpt_path))

        # set evaluation
        self.policy.eval()

    def infer(self, obs_dict):
        # create input from obs_dict
        tic = time.perf_counter()

        coords_batch, feats_batch = create_batch(obs_dict["coords"], obs_dict["points"])
        coords_batch, feats_batch = coords_batch.to(self.device), feats_batch.to(self.device)
        cloud_data = ME.SparseTensor(feats_batch, coords_batch)

        colors = torch.from_numpy(obs_dict["colors"]).unsqueeze(0).to(self.device)
        image_coords = torch.from_numpy(obs_dict["image_coords"]).unsqueeze(0).to(self.device)

        toc = time.perf_counter()
        print(f"1. Data Preprocess Time: {toc - tic:.6f} seconds.")
        
        tic = time.perf_counter()

        # predict
        actions = self.policy(
            cloud_data,
            colors,
            image_coords,
            actions = None
        ).squeeze(0).cpu().numpy()

        toc = time.perf_counter()
        print(f"2. Policy Inference Time: {toc - tic:.6f} seconds.")

        # return predicted actions
        return {
            "actions": actions
        }


def service(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value
    
    # load config
    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader = yaml.FullLoader))

    # set seed
    set_seed(config.deploy.seed)

    # initialize policy inference agent
    policy = PolicyInferenceAgent(config, ckpt_path = args.ckpt)

    # initialize server
    server = WebsocketPolicyServer(policy = policy, host = args.host, port = args.port)

    # start service
    server.serve_forever()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action = 'store', type = str, help = 'data and model config during training and deployment', required = True)
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = False, default = None)
    parser.add_argument('--host', action = 'store', type = str, help = 'server host address', required = False, default = "127.0.0.1")
    parser.add_argument('--port', action = 'store', type = int, help = 'server port', required = False, default = 8000)

    service(vars(parser.parse_args()))
