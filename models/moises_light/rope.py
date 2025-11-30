from einops import rearrange, pack, unpack
from rotary_embedding_torch import RotaryEmbedding
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint, CheckpointPolicy, create_selective_checkpoint_contexts
from models.bs_roformer.bs_roformer import Attention, FeedForward
import functools

import torch


class RoPETransformer(Module):
    def __init__(
            self,
            n_rope= 5,
            dim = 512,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            ff_mult=4,
            flash_attn=True,
            sage_attention=False,
            use_torch_checkpoint=False,
    ):
        super().__init__()
        self.use_torch_checkpoint = use_torch_checkpoint
        self.layers = ModuleList([])
        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(n_rope):
            tran_modules = []
            layer = RoPELayer(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                ff_mult=ff_mult,
                rotary_embed_time=time_rotary_embed,
                rotary_embed_freq=freq_rotary_embed,
                flash_attn=flash_attn,
                sage_attention=sage_attention,
            )

            self.layers.append(layer)

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
        for layer in self.layers:
            if self.use_torch_checkpoint:
                x = checkpoint(layer, x, use_reentrant=False, context_fn=self.ac_context_fn)
            else:
                x = layer(x)
        return x


class RoPELayer(Module):

    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            rotary_embed_time=None,
            rotary_embed_freq=None,
            flash_attn=True,
            sage_attention=False,
    ):
        super().__init__()
        self.layers = ModuleList([])

        self.attn_time = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            rotary_embed=rotary_embed_time,
            flash=flash_attn,
            sage_attention=sage_attention
        )

        self.attn_freq = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            rotary_embed=rotary_embed_freq,
            flash=flash_attn,
            sage_attention=sage_attention
        )

        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

    def forward(self, x):
        x = rearrange(x, 'b t f d -> b f t d')
        x, ps = pack([x], '* t d')

        x = self.attn_time(x) + x
        x = self.ff(x) + x

        x, = unpack(x, ps, '* t d')
        x = rearrange(x, 'b f t d -> b t f d')
        x, ps = pack([x], '* f d')

        x = self.attn_freq(x) + x
        x = self.ff2(x) + x

        x, = unpack(x, ps, '* f d')


        return x
