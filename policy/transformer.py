# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Adapted from DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                tgt_self_attn=False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                attn_mask=None,
                key_padding_mask=None,
                pos=None):
        assert len(src.shape) == 3
        # flatten NxHWxC to HWxNxC
        src = src.permute(1, 0, 2)
        pos = pos.permute(1, 0, 2)

        hs = self.decoder(
            src, src,
            tgt_attn_mask=None,
            attn_mask=attn_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=key_padding_mask,
            tgt_pos=None,
            memory_pos=pos,
        )
        hs = hs.transpose(0, 1)
        return hs


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt, memory,
        tgt_attn_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
    ):
        output = tgt

        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_attn_mask=tgt_attn_mask,
                attn_mask=attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_pos=tgt_pos,
                memory_pos=memory_pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, tgt_self_attn=False):
        super().__init__()
        self.tgt_self_attn = tgt_self_attn
        if tgt_self_attn:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_model)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, memory,
        tgt_attn_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
    ):
        if self.tgt_self_attn:
            q = k = self.with_pos_embed(tgt, tgt_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_attn_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=attn_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_pre(
        self, tgt, memory,
        tgt_attn_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
    ):
        if self.tgt_self_attn:
            tgt2 = self.norm(tgt)
            q = k = self.with_pos_embed(tgt2, tgt_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_attn_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, tgt_pos)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=attn_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self, tgt, memory,
        tgt_attn_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_attn_mask, attn_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_pos, memory_pos
            )
        return self.forward_post(
            tgt, memory, tgt_attn_mask, attn_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_pos, memory_pos
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
