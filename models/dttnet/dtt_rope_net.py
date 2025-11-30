import torch.nn as nn
import functools

import torch
from torch.utils.checkpoint import checkpoint, CheckpointPolicy, create_selective_checkpoint_contexts
from models.dttnet.modules import TFC_TDF, TFC_TDF_Res1, TFC_TDF_Res2
from models.moises_light.abstract_model import AbstractModel
from models.moises_light.batch_norm import get_norm
from models.moises_light.rope import RoPETransformer
from einops import rearrange

class DttRopeNet(AbstractModel):
    def __init__(self,
                 num_blocks=5,
                 l=3,
                 g=32, #initial dimension
                 k=3, #kernel size,
                 bn=8,  #botlleneck factor TDF FeedForward
                 bias=False,
                 bn_norm='BN',
                 block_type='TFC_TDF_Res2',
                 use_torch_checkpoint=False,
                 **kwargs):

        super(DttRopeNet, self).__init__(**kwargs)
        # self.save_hyperparameters()


        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.bn = bn
        self.bias = bias

        self.n = num_blocks // 2
        self.use_torch_checkpoint = use_torch_checkpoint
        scale = (2, 2)

        if block_type == "TFC_TDF":
            T_BLOCK = TFC_TDF
        elif block_type == "TFC_TDF_Res1":
            T_BLOCK = TFC_TDF_Res1
        elif block_type == "TFC_TDF_Res2":
            T_BLOCK = TFC_TDF_Res2
        else:
            raise ValueError(f"Unknown block type {block_type}")

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c_in, out_channels=g, kernel_size=(1, 1)),
            get_norm(bn_norm, g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

        for i in range(self.n):
            c_in = c

            self.encoding_blocks.append(T_BLOCK(c_in, c, l, f, k, bn, bn_norm, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.bottleneck_block1 = T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias)

        self.bottleneck_block2 = RoPETransformer(
            n_rope=5,
            dim=c,
            use_torch_checkpoint=self.use_torch_checkpoint,
            dim_head=c//8,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            ff_mult=4,
            flash_attn=True,
            sage_attention=False,
        )

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            # print(f"i: {i}, in channels: {c}")
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c - g),
                    nn.ReLU()
                )
            )

            f = f * 2
            c -= g

            self.decoding_blocks.append(T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c_out, kernel_size=(1, 1)),
        )

        aten = torch.ops.aten
        compute_intensive_ops = [
            aten.mm.default,
            aten.bmm.default,
            aten.addmm.default,
        ]

        def policy_fn(ctx, op, *args, **kwargs):
            if op in compute_intensive_ops:
                return CheckpointPolicy.PREFER_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        self.ac_context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)

    def forward(self, x):
        '''
        Args:
            x: (batch, c*2, 2048, 256)
        '''
        x = self.stft(x)
        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            if self.use_torch_checkpoint and i > 0:  # Checkpoint all but first encoder block
                x = checkpoint(self.encoding_blocks[i], x, use_reentrant=False, context_fn=self.ac_context_fn)
            else:
                x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        # print(f"bottleneck in: {x.shape}")
        x = self.bottleneck_block1(x)

        # Transformer
        x = rearrange(x, 'b d t f -> b t f d')
        x = self.bottleneck_block2(x)
        x = rearrange(x, 'b t f d -> b d t f')


        for i in range(self.n):
            x = self.us[i](x)
            # print(f"us{i} in: {x.shape}")
            # print(f"ds{i} out: {ds_outputs[-i - 1].shape}")
            x = x * ds_outputs[-i - 1]
            if self.use_torch_checkpoint and i < self.n - 1:  # Checkpoint all but last decoder block
                x = checkpoint(self.decoding_blocks[i], x, use_reentrant=False, context_fn=self.ac_context_fn)
            else:
                x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        x = self.istft(x)

        x = x.unsqueeze(1)

        return x