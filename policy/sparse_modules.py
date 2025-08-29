import torch
from torch import nn
import MinkowskiEngine as ME

from policy.minkowski.resnet import ResNet14Max, ResNet14Mini


class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''x: (B, C, N)'''
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class SharedMLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(len(layers) - 1):
            self.blocks.append(MLPBlock(layers[i], layers[i+1]))

    def forward(self, x):
        '''x: (B, C, N)'''
        x = self.blocks(x)
        return x


class CustomWeightedInterpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_feats, selected_idxs):
        """
        src_feats: (C, m)
        selected_idxs: (n, k)
        """
        selected_idxs_expand = selected_idxs.unsqueeze(1).expand(-1, src_feats.size(0), -1) # (n, C, k)
        curr_src_feats_expand = src_feats.unsqueeze(0).expand(selected_idxs.size(0), -1, -1) # (n, C, m)
        selected_feats = torch.gather(curr_src_feats_expand, 2, selected_idxs_expand) # (n, C, k)

        ctx.save_for_backward(src_feats, selected_idxs, selected_idxs_expand)

        return selected_feats
    
    @staticmethod
    def backward(ctx, grad_out):
        src_feats, selected_idxs, selected_idxs_expand = ctx.saved_tensors

        grad_src_feats = torch.zeros_like(src_feats) # (C, m)
        grad_selected_idxs = torch.zeros_like(selected_idxs)
        selected_idxs_expand = selected_idxs_expand.permute(1, 0, 2).contiguous() # (C, n, k)
        grad_out = grad_out.permute(1, 0, 2).contiguous() # (C, n, k)

        for i in range(grad_out.size(2)):
            grad_src_feats.scatter_add_(1, selected_idxs_expand[:,:,i], grad_out[:,:,i])

        return grad_src_feats, grad_selected_idxs


class WeightedSpatialInterpolation(nn.Module):
    def __init__(self, interp_fn_mode = 'custom'):
        super().__init__()
        assert interp_fn_mode in ['custom', 'naive']
        self.interp_fn_mode = interp_fn_mode

    def forward(
        self, tgt, src, tgt_feats, src_feats, k=3
    ) -> torch.Tensor:
        """
        Args:
            tgt: (B, n, 3) tensor of the xyz positions of the target features
            src: (B, m, 3) tensor of the xyz positions of the source features
            tgt_feats: (B, C1, n) tensor of the target features
            src_feats: (B, C2, m) tensor of the source features

        Returns:
            interp_features : (B, mlp[-1], n) tensor of the features of the interpolated features
        """
        interpolated_feats = []
        for i in range(tgt.size(0)):
            all_dists = torch.linalg.norm(tgt[i].unsqueeze(1)-src[i].unsqueeze(0), dim=2) # (n, m)
            all_dists, all_idxs = torch.sort(all_dists, dim=1)

            if self.interp_fn_mode == 'naive':
                selected_idxs = all_idxs[:, :k].unsqueeze(1).expand(-1, src_feats.size(1), -1) # (n, C2, k)
                curr_src_feats = src_feats[i:i+1].expand(selected_idxs.size(0), -1, -1) # (n, C2, m)
                selected_feats = torch.gather(curr_src_feats, 2, selected_idxs) # (n, C2, k)
            else: # 'custom'
                selected_idxs = all_idxs[:, :k] # (n, k)
                selected_feats = CustomWeightedInterpFn.apply(src_feats[i], selected_idxs) # (n, C2, k)

            weight = 1.0 / (all_dists[:, :k] + 1e-6)
            norm = torch.sum(weight, dim=1, keepdim=True)
            weight = weight / norm
            selected_feats = (selected_feats * weight.unsqueeze(1)).sum(dim=2) # (n, C2)
            interpolated_feats.append(selected_feats)

        interpolated_feats = torch.stack(interpolated_feats, dim=0).permute(0, 2, 1) # (B, C2, n)
        interpolated_feats = torch.cat([interpolated_feats, tgt_feats], dim=1)  #(B, C2 + C1, n)

        return interpolated_feats


class SparsePositionalEncoding(nn.Module):
    """
    Sparse positional encoding for point tokens, similar to the standard version
    """
    def __init__(self, num_pos_feats=512, temperature=10000, max_pos=800):
        super().__init__()
        ''' max_pos: position range will be [-max_pos/2, max_pos/2) along X/Y/Z-axis.
                     remeber to keep this value fixed in your training and evaluation.
                     800 for voxel_size=0.005 in our experiments. TODO: may need point centralization?
        '''
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.max_pos = max_pos
        self.origin_pos = max_pos // 2
        self._init_position_vector()

    def _init_position_vector(self):
        x_steps = y_steps = self.num_pos_feats // 3
        z_steps = self.num_pos_feats - x_steps - y_steps
        xyz_embed = torch.arange(self.max_pos, dtype=torch.float32)[:,None]

        x_dim_t = torch.arange(x_steps, dtype=torch.float32)
        y_dim_t = torch.arange(y_steps, dtype=torch.float32)
        z_dim_t = torch.arange(z_steps, dtype=torch.float32)
        x_dim_t = self.temperature ** (2 * (x_dim_t // 2) / x_steps)
        y_dim_t = self.temperature ** (2 * (y_dim_t // 2) / y_steps)
        z_dim_t = self.temperature ** (2 * (z_dim_t // 2) / z_steps)

        pos_x_vector = xyz_embed / x_dim_t
        pos_y_vector = xyz_embed / y_dim_t
        pos_z_vector = xyz_embed / z_dim_t
        self.pos_x_vector = torch.stack([pos_x_vector[:,0::2].sin(), pos_x_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_y_vector = torch.stack([pos_y_vector[:,0::2].sin(), pos_y_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_z_vector = torch.stack([pos_z_vector[:,0::2].sin(), pos_z_vector[:,1::2].cos()], dim=2).flatten(1)

    def forward(self, coords_list):
        pos_list = []
        for coords in coords_list:
            coords = (coords[:,1:4] + self.origin_pos).long()
            coords[:,0] = torch.clamp(coords[:,0], 0, self.max_pos-1)
            coords[:,1] = torch.clamp(coords[:,1], 0, self.max_pos-1)
            coords[:,2] = torch.clamp(coords[:,2], 0, self.max_pos-1)
            pos_x = self.pos_x_vector.to(coords.device)[coords[:,0]]
            pos_y = self.pos_y_vector.to(coords.device)[coords[:,1]]
            pos_z = self.pos_z_vector.to(coords.device)[coords[:,2]]
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)
            pos_list.append(pos)
        return pos_list


class SparseEncoder(nn.Module):
    def __init__(self, cloud_enc_dim=128):
        super().__init__()
        self.cloud_enc_dim = cloud_enc_dim
        self.cloud_encoder = ResNet14Mini(in_channels=3, out_channels=cloud_enc_dim, conv1_kernel_size=3, strides=(1,1,1,2), dilations=(1,2,4,8), bn_momentum=0.02, init_pool="avg")

    def forward(self, sinput):
        soutput = self.cloud_encoder(sinput)
        return soutput


class SpatialAligner(nn.Module):
    def __init__(self, mlps, out_channels=512, interp_fn_mode='custom'):
        """
        Args:
            interp_fn_mode: str, "naive"/"custom"
        """
        super().__init__()
        self.out_channels = out_channels
        self.interp = WeightedSpatialInterpolation(interp_fn_mode = interp_fn_mode)
        self.interp_proj = SharedMLP(mlps)
        self.conv = ResNet14Max(in_channels=mlps[-1], out_channels=out_channels, conv1_kernel_size=3, strides=(4,2,2,2), dilations=(4,1,1,1), bn_momentum=0.02, init_pool=None)
        self.position_embedding = SparsePositionalEncoding(out_channels)

    def forward(self, sinput, image_feat, image_coord, max_num_token=150):
        ''' max_num_token: maximum token number for each point cloud, which can be adjusted depending on the scene density.
                           150 for voxel_size=0.005 in our experiments
        '''
        batch_size = image_feat.size(0)
        cloud_feat, cloud_coord = sinput.F, sinput.C

        cloud_feat_list = []
        for i in range(batch_size):
            cloud_mask_i = cloud_coord[:, 0] == i
            cloud_coord_i = cloud_coord[cloud_mask_i][:, 1:].unsqueeze(0)
            cloud_feat_i = cloud_feat[cloud_mask_i].permute(1, 0).unsqueeze(0)
            image_coord_i = image_coord[i:i+1]
            image_feat_i = image_feat[i].permute(1, 0).unsqueeze(0)
            cloud_feat_i = self.interp(cloud_coord_i.float(), image_coord_i.float(), cloud_feat_i, image_feat_i)
            cloud_feat_list.append(cloud_feat_i)
        cloud_feat = torch.cat(cloud_feat_list, dim=2)
        cloud_feat = self.interp_proj(cloud_feat)
        cloud_feat = cloud_feat.squeeze(0).permute(1, 0)

        sinput = ME.SparseTensor(cloud_feat, sinput.C)
        soutput = self.conv(sinput)
        feats_batch, coords_batch = soutput.F, soutput.C

        # convert to sparse tokens
        feats_list = []
        coords_list = []
        for i in range(batch_size):
            mask = (coords_batch[:,0] == i)
            feats_list.append(feats_batch[mask])
            coords_list.append(coords_batch[mask])
        pos_list = self.position_embedding(coords_list)

        tokens = torch.zeros([batch_size, max_num_token, self.out_channels], dtype=feats_batch.dtype, device=feats_batch.device)
        pos_emb = torch.zeros([batch_size, max_num_token, self.out_channels], dtype=feats_batch.dtype, device=feats_batch.device)
        token_padding_mask = torch.ones([batch_size, max_num_token], dtype=torch.bool, device=feats_batch.device)
        for i, (feats, pos) in enumerate(zip(feats_list, pos_list)):
            num_token = min(max_num_token, len(feats))
            tokens[i,:num_token] = feats[:num_token]
            pos_emb[i,:num_token] = pos[:num_token]
            token_padding_mask[i,:num_token] = False

        return tokens, pos_emb, token_padding_mask
