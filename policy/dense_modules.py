import os
import torch
import torchvision

from torch import nn
from einops import rearrange
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class ResNetEncoder(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str = "resnet18", out_dim: int = 512, finetune: str = "full"):
        super().__init__()
        backbone = getattr(torchvision.models, name)(weights = "IMAGENET1K_V1", norm_layer = FrozenBatchNorm2d)

        if finetune == "none":
            backbone.requires_grad_(False)

        self.body = IntermediateLayerGetter(backbone, return_layers={'layer4': "0"})
        
        enc_out_dim = 512 if name in ('resnet18', 'resnet34') else 2048
        if enc_out_dim != out_dim:
            self.proj = nn.Conv2d(enc_out_dim, out_dim, 1)
        else:
            self.proj = nn.Identity()
        self.num_channels = out_dim

    def forward(self, img):
        feats = self.body(img)["0"]
        feats = self.proj(feats)
        return feats


class DINOv2Encoder(nn.Module):
    """DINOv2 backbone with optional LoRA fine-tuning."""
    def __init__(
        self, 
        name: str = "dinov2-base", 
        out_dim: int = 512,
        finetune: str = "lora", 
        dtype = torch.float32,
        lora_rank: int = 16, 
        lora_dropout: float = 0.1
    ):
        super().__init__()
        assert finetune in ["full", "lora", "none"], "finetune parameter should be one of [full, lora, none]."
        
        dino = AutoModel.from_pretrained(os.path.join("./weights", name), torch_dtype = dtype)

        if finetune == "lora":
            dino.requires_grad_(False)
            config = LoraConfig(
                r              = lora_rank,
                lora_alpha     = lora_rank,
                target_modules = ['projection', 'query', 'key', 'value', 'dense', 'fc1', 'fc2'],
                lora_dropout   = lora_dropout,
                bias           = 'none',
                use_rslora     = True,
            )
            dino = get_peft_model(dino, config)
            # convert LoRA parameters to float32
            for name, param in dino.named_parameters():
                if "lora_" in name:
                    param.data = param.data.float()
        elif finetune == "none":
            dino.requires_grad_(False)
            self.model = dino
        
        self.model = dino

        self.patch_size = dino.config.patch_size
        hidden_size = dino.config.hidden_size
        if hidden_size != out_dim:
            self.proj = nn.Linear(hidden_size, out_dim)
        else:
            self.proj = nn.Identity()
        self.num_channels = out_dim

    def forward(self, img):
        H, W = img.shape[-2:]
        grid_H, grid_W = H // self.patch_size, W // self.patch_size
        feats = self.model(img).last_hidden_state[:, 1:] # B, L, hidden_size
        feats = self.proj(feats)    # B, L, num_channels
        feats = feats.reshape(-1, grid_H, grid_W, self.num_channels).permute(0, 3, 1, 2)

        return feats

