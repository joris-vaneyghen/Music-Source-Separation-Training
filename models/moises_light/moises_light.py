import functools

import torch
import torch.nn as nn
from einops import rearrange, pack, unpack
from rotary_embedding_torch import RotaryEmbedding
from torch.nn import ModuleList
from torch.utils.checkpoint import checkpoint, CheckpointPolicy, create_selective_checkpoint_contexts

from models.bs_roformer.bs_roformer import Transformer
from models.moises_light.abstract_model import AbstractModel
from models.moises_light.batch_norm import get_norm
from models.moises_light.modules import SplitModule, SplitMergeModule

# inspired by https://github.com/junyuchen-cjy/DTTNet-Pytorch


class MoisesLight(AbstractModel):
    def __init__(self,
                 n_band=4,
                 n_enc=3,
                 n_dec=1,
                 n_rope=5,
                 n_split_enc=3,
                 n_split_dec=1,
                 kernel_size=3,
                 g=48,
                 tdf_bottleneck_factor=8,
                 bias=False,
                 bn_norm='BN',
                 #params transformer
                 heads=8,
                 dim_head=24,
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 ff_mult=2,
                 flash_attn=True,
                 norm_output=False,
                 sage_attention=False,
                 skip_connection=False,
                 use_torch_checkpoint=False,
                 checkpointing_policy=None,
                 #STFT prams
                 **kwargs):

        super(MoisesLight, self).__init__(**kwargs)

        if checkpointing_policy:
            aten = torch.ops.aten
            if checkpointing_policy == 1:
                compute_intensive_ops = [
                    aten.mm.default,
                    aten.bmm.default,
                    aten.addmm.default,
                ]
            else:
                compute_intensive_ops = [
                    aten.mm.default,
                    aten.convolution.default,
                    aten.convolution_backward.default,
                    aten.bmm.default,
                    aten.addmm.default,
                    aten._scaled_dot_product_flash_attention.default,
                    aten._scaled_dot_product_efficient_attention.default,
                    aten._flash_attention_forward.default,
                    aten._efficient_attention_forward.default,
                    aten.upsample_bilinear2d.default,
                    aten._scaled_mm.default
                ]

            def policy_fn(ctx, op, *args, **kwargs):
                if op in compute_intensive_ops:
                    return CheckpointPolicy.PREFER_SAVE
                else:
                    return CheckpointPolicy.PREFER_RECOMPUTE

            self.ac_context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        else:
            self.ac_context_fn = None

        self.n_band = n_band
        self.n_band = n_band
        self.n_enc = n_enc
        self.n_dec = n_dec

        self.n_split_enc = n_split_enc
        self.n_split_dec = n_split_dec
        self.n_rope = n_rope
        self.g = g
        self.kernel_size = kernel_size
        self.tdf_bottleneck_factor = tdf_bottleneck_factor
        self.bias = bias
        self.bn_norm = bn_norm

        scale = (2, 1)

        self.first_conv = SplitModule(self.dim_c_in * self.n_band, g, 1,  self.kernel_size, bn_norm, self.n_band)

        f = self.dim_f
        c = g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

        for i in range(self.n_enc):
            encoding_block= SplitMergeModule(c,
                                             c,
                                             self.n_split_enc,
                                             f // self.n_band,
                                             self.kernel_size,
                                             self.tdf_bottleneck_factor,
                                             self.bn_norm,
                                             self.bias,
                                             self.n_band)
            self.encoding_blocks.append(encoding_block)
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c + g),
                    nn.ReLU()
                )
            )
            c += g

        self.bottleneck_block1 = SplitMergeModule(c,
                                                  c,
                                                  self.n_split_enc,
                                                  f // self.n_band,
                                                  self.kernel_size,
                                                  self.tdf_bottleneck_factor,
                                                  self.bn_norm,
                                                  self.bias,
                                                  self.n_band)
        self.bottleneck_block2 = ModuleList([])
        self.skip_connection = skip_connection
        self.use_torch_checkpoint = use_torch_checkpoint
        # todo put in config
        transformer_kwargs = dict(
            dim=c,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult = ff_mult,
            flash_attn=flash_attn,
            norm_output=norm_output,
            sage_attention=sage_attention,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)
        transformer_module_depth = 1 # Changed from 2 to 1
        for _ in range(self.n_rope):
            tran_modules = []
            tran_modules.append(
                Transformer(depth=transformer_module_depth, rotary_embed=time_rotary_embed, **transformer_kwargs)
            )
            tran_modules.append(
                Transformer(depth=transformer_module_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
            )
            self.bottleneck_block2.append(nn.ModuleList(tran_modules))

        self.us = nn.ModuleList()
        for i in range(self.n_enc):
            # print(f"i: {i}, in channels: {c}")
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c - g),
                    nn.ReLU()
                )
            )

            c -= g


        # todo impl this.n_dec > 1
        self.decoding_block = SplitMergeModule(c,
                                               c,
                                               self.n_split_dec,
                                               f // self.n_band,
                                               self.kernel_size,
                                               self.tdf_bottleneck_factor,
                                               self.bn_norm,
                                               self.bias,
                                               self.n_band)

        self.final_conv = nn.Conv2d(in_channels=c,
                                    out_channels = self.dim_c_out * self.n_band,
                                    kernel_size=(self.kernel_size, self.kernel_size) ,
                                    stride=1,
                                    padding=self.kernel_size // 2,
                                    groups=n_band)


    def forward(self, x):
        '''
        Args:
            x: (batch, c*2, 2048, 256)
        '''
        x = self.stft(x)
        x = x.transpose(-1, -2)

        B, C, T, F = x.shape
        # print('B, C, T, F ', x.shape)

        # Split frequency into n_band equal parts
        x = x.reshape(B, C, self.n_band, T, F // self.n_band)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, n_band, C, T, F//n_band)
        x = x.reshape(B, self.n_band * C, T, F // self.n_band)  # (B, C*n_band, T, F//n_band)

        # print('B, C*n_band, T, F//n_band', x.shape)

        x = self.first_conv(x)

        ds_outputs = []
        for i in range(self.n_enc):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        # print(f"bottleneck in: {x.shape}")
        x = self.bottleneck_block1(x)

        # Transformer
        x = rearrange(x, 'b d t f -> b t f d')
        store = [None] * len(self.bottleneck_block2)
        for i, transformer_block in enumerate(self.bottleneck_block2):

            time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                # Sum all previous
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            if self.use_torch_checkpoint:
                if self.ac_context_fn:
                    x = checkpoint(time_transformer, x, use_reentrant=False, context_fn=self.ac_context_fn)
                else:
                    x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            if self.use_torch_checkpoint:
                if self.ac_context_fn:
                    x = checkpoint(freq_transformer, x, use_reentrant=False, context_fn=self.ac_context_fn)
                else:
                    x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        x = rearrange(x, 'b t f d -> b d t f')

        for i in range(self.n_enc):
            x = self.us[i](x)
            # print(f"us{i} in: {x.shape}")
            # print(f"ds{i} out: {ds_outputs[-i - 1].shape}")
            # x = x * ds_outputs[-i - 1]
            # x = self.decoding_blocks[i](x)
        x = x * ds_outputs[0]
        x = self.decoding_block(x)


        x = self.final_conv(x)

        # print('B, C*n_band, T, F//n_band', x.shape)

        # Merge subbands back (reverse of the splitting operations)
        x = x.reshape(B, self.n_band, C, T, F // self.n_band)  # (B, n_band, C, T, F//n_band)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, n_band, T, F//n_band)
        x = x.reshape(B, C, T, F)  # (B, C, T, F)
        # print('B, C, T, F ', x.shape)

        x = x.transpose(-1, -2)

        x = self.istft(x)

        x = x.unsqueeze(1)

        return x