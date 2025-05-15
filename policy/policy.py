import torch
import torch.nn as nn
from torch.amp import autocast

from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy
from policy.sparse_modules import SparseEncoder, SpatialAligner
from policy.dense_modules import DINOv2Encoder, ResNetEncoder


class RISE2(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        obs_feature_dim = 512,
        cloud_enc_dim = 128,
        image_enc_dim = 128,
        action_dim = 20, 
        hidden_dim = 512,
        nheads = 8,
        num_attn_layers = 4,
        dim_feedforward = 2048, 
        dropout = 0.1,
        image_enc = "dinov2-base",  # resnetx, dinov2-x
        interp_fn_mode = 'custom',  # custom, naive
        image_enc_finetune = "lora",  # full, lora, none
        image_enc_dtype = "float32"
    ):
        super().__init__()
        num_obs = 1
        self.nheads = nheads
        self.image_enc_dtype = getattr(torch, image_enc_dtype)
        self.sparse_encoder = SparseEncoder(cloud_enc_dim)
        self.spatial_aligner = SpatialAligner(mlps = [cloud_enc_dim + image_enc_dim] * 3, out_channels = obs_feature_dim, interp_fn_mode = interp_fn_mode)
        self.transformer = Transformer(hidden_dim, nheads, num_attn_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)

        if image_enc.startswith("resnet"):
            self.dense_encoder = ResNetEncoder(image_enc, image_enc_dim, finetune = image_enc_finetune)
        elif image_enc.startswith("dino"):
            self.dense_encoder = DINOv2Encoder(image_enc, image_enc_dim, finetune = image_enc_finetune, dtype = self.image_enc_dtype)

    def generate_attn_mask(self, obs_token_len, readout_len = 1):
        mask_size = obs_token_len + readout_len
        attn_mask = torch.ones([mask_size, mask_size], dtype=bool)
        attn_mask[:obs_token_len, :obs_token_len] = False
        attn_mask[-readout_len:] = False

        return attn_mask

    def forward(self, cloud, image, image_coord, actions = None):
        with autocast(
            device_type = image.device.type, 
            dtype = self.image_enc_dtype if image.device.type == 'cuda' else torch.float32
        ):
            image_feat = self.dense_encoder(image)
        
        if self.image_enc_dtype != torch.float32:
            image_feat = image_feat.to(torch.float32)

        image_feat = image_feat.flatten(2).permute(0, 2, 1)
        image_coord = image_coord.flatten(2).permute(0, 2, 1)

        cloud_feat = self.sparse_encoder(cloud)

        src, pos, src_padding_mask = self.spatial_aligner(cloud_feat, image_feat, image_coord)

        batch_size, src_len = src.size(0), src.size(1)

        readout_pos = self.readout_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        readout = torch.zeros_like(readout_pos)
        readout_padding_mask = torch.zeros([batch_size, 1], dtype = bool).to(src.device)
        pos = torch.cat([pos, readout_pos], dim = 1)
        src = torch.cat([src, readout], dim = 1)
        src_padding_mask = torch.cat([src_padding_mask, readout_padding_mask], dim = 1)
        
        attn_mask = self.generate_attn_mask(src_len).to(src.device)
        output = self.transformer(src, attn_mask, src_padding_mask, pos)
        readout = output[:, -1]

        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred