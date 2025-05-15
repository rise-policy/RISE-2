import os
import yaml
import torch
import argparse
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from policy import RISE2
from dataset.realworld import RealWorldDataset, collate_fn
from utils.training import plot_history, set_seed, sync_loss


default_args = edict({
    "data_path": "data/",
    "ckpt_dir": "logs/collect_toys",
    "config": "config/dual_teleop_dino.yaml",
    "resume_ckpt": None,
    "resume_step": -1
})


def train(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # load training config
    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader = yaml.FullLoader))

    # prepare distributed training
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = '0'
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = WORLD_SIZE, rank = RANK)

    # output training configs
    if RANK == 0:
        print("training config:", args)
        print("data and model config:", config)
        if config.data.sample_traj:
            print("Enabling trajectory sampling, this may take a while during dataset init.")

    # set up device
    set_seed(config.train.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset & dataloader
    if RANK == 0: print("Loading dataset ...")
    dataset = RealWorldDataset(args.data_path, config, split = 'train')
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas = WORLD_SIZE, 
        rank = RANK, 
        shuffle = True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = config.train.batch_size // WORLD_SIZE,
        num_workers = config.train.num_workers,
        collate_fn = collate_fn,
        sampler = sampler,
        drop_last = True,
    )

    # policy
    if RANK == 0: print("Loading policy ...")
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
    if RANK == 0:
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
    policy = nn.parallel.DistributedDataParallel(
        policy, 
        device_ids = [LOCAL_RANK], 
        output_device = LOCAL_RANK, 
        find_unused_parameters = True
    )

    # load checkpoint
    if args.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(args.resume_ckpt, map_location = device), strict = False)
        if RANK == 0:
            print("Checkpoint {} from step {} loaded.".format(args.resume_ckpt, args.resume_step))

    # ckpt path
    if RANK == 0 and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    # optimizer and lr scheduler
    if RANK == 0: print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr = config.train.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    num_epochs = int(np.ceil(config.train.num_steps / len(dataloader)))
    resume_epoch = int(np.floor(args.resume_step / len(dataloader)))
    cur_step = args.resume_step + 1

    if RANK == 0: print("\nStart training from epoch {} (step {}), max epoch {} (step {}).\n".format(resume_epoch + 1, args.resume_step + 1, num_epochs, config.train.num_steps))

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = config.train.num_warmup_steps,
        num_training_steps = config.train.num_steps
    )
    lr_scheduler.last_epoch = args.resume_step

    # training
    train_history = []
    steps_per_epoch = len(dataloader)

    policy.train()
    for epoch in range(resume_epoch + 1, num_epochs):
        if RANK == 0: print("Epoch {}".format(epoch)) 
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = 0
        logs = ""

        for data in pbar:
            # cloud data processing
            cloud_coords = data['cloud_coords'].to(device)
            cloud_feats = data['cloud_feats'].to(device)
            action_data = data['action_normalized'].to(device)
            image = data["image_feats"].to(device)
            image_coords = data["image_coords"].to(device)
            cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
            # forward
            loss = policy(cloud_data, image, image_coords, actions = action_data)
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()

            if (cur_step + 1) % config.train.save_steps == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(args.ckpt_dir, "policy_step_{}_seed_{}.ckpt".format(cur_step + 1, config.train.seed))
                )
                logs += "Checkpoint saved at step {}.\n".format(cur_step + 1)
            
            cur_step += 1

        avg_loss = avg_loss / steps_per_epoch
        sync_loss(avg_loss, device)
        train_history.append(avg_loss)
        plot_history(train_history, epoch, args.ckpt_dir, config.train.seed)

        logs += "# Steps: {}. Average train loss at epoch {}: {:.6f}\n".format(cur_step, epoch, avg_loss)
        if RANK == 0: print(logs)

    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.ckpt_dir, "policy_last.ckpt")
        )
        print("Final checkpoint saved at step {}.\n".format(cur_step))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action = 'store', type = str, help = 'data path', required = True)
    parser.add_argument('--ckpt_dir', action = 'store', type = str, help = 'checkpoint directory', required = True)
    parser.add_argument('--config', action = 'store', type = str, help = 'data and model config during training and deployment', required = True)
    parser.add_argument('--resume_ckpt', action = 'store', type = str, help = 'resume checkpoint file', required = False, default = None)
    parser.add_argument('--resume_step', action = 'store', type = int, help = 'resume from which step', required = False, default = -1)

    train(vars(parser.parse_args()))
