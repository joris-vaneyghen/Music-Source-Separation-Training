from functools import partial
import math
from collections import deque
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from rotary_embedding_torch import RotaryEmbedding
from torch.nn import Module, ModuleList

from models.bs_roformer.attend import Attend


# helper functions

def stft(x, stft_config, audio_channels):
    # B, C, L = x.shape
    # In the initial padding, ensure that the number of frames after the STFT (the length of the T dimension) is even,
    # so that the RFFT operation can be used in the separation network.
    L = x.shape[-1]

    hop_length = stft_config['hop_length']

    padding = hop_length - L % hop_length
    if (L + padding) // hop_length % 2 == 0:
        padding += hop_length
    x = F.pad(x, (0, padding))

    # STFT
    L_padded = x.shape[-1]
    x = x.reshape(-1, L_padded)
    x = torch.stft(x, **stft_config, return_complex=True)
    x = torch.view_as_real(x)
    x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // audio_channels, x.shape[3] * audio_channels,
                                      x.shape[1], x.shape[2])
    return x


def istft(x, stft_config, chunk_size, sources, audio_channels, dims):
    L = chunk_size
    hop_length = stft_config['hop_length']

    padding = hop_length - L % hop_length
    if (L + padding) // hop_length % 2 == 0:
        padding += hop_length

    B, _, Fr, T = x.shape
    n = dims[0]
    x = x.view(B, n, -1, Fr, T)

    x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
    x = torch.view_as_complex(x.contiguous().float())
    x = torch.istft(x, **stft_config)
    x = x.reshape(B, len(sources), audio_channels, -1)

    x = x[:, :, :, :-padding]
    return x



def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


class GLU(nn.Module):
    """Workaround: Op(s) not lowered: aten::glu_backward
    Gated Linear Unit (GLU) as a PyTorch module.

    This implements the GLU activation function which splits the input along a dimension
    and applies sigmoid gating to one half before element-wise multiplication.

    Args:
        dim (int): Dimension along which to split the input (default: -1, last dim).
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Forward pass of the GLU module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of GLU operation.
        """
        assert x.size(self.dim) % 2 == 0, (
            f"Dimension {self.dim} must be divisible by 2, got {x.size(self.dim)}"
        )

        a, b = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return a * torch.sigmoid(b)

    def extra_repr(self):
        return f'dim={self.dim}'


# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# attention

class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            linear_attn=False
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_embed=rotary_embed, flash=flash_attn)

            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class DualPathTran(nn.Module):
    """
    Dual-Path Transformer in Separation Network.

    Args:
        d_model (int): The number of expected features in the input (input_size).
        expand (int): Expansion factor used to calculate the hidden_size of LSTM.
        bidirectional (bool): If True, becomes a bidirectional LSTM.
    """

    def __init__(self, d_model, time_rotary_embed, freq_rotary_embed, tran_params):
        super(DualPathTran, self).__init__()

        self.d_model = d_model

        transformer_kwargs = dict(
            dim=d_model,
            heads=tran_params['heads'],
            dim_head=tran_params['dim_head'],
            attn_dropout=tran_params['attn_dropout'],
            ff_dropout=tran_params['ff_dropout'],
            flash_attn=tran_params['flash_attn']
        )
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(2)])
        self.time_layer = Transformer(depth=tran_params['depth'], rotary_embed=time_rotary_embed, **transformer_kwargs)
        self.freq_layer = Transformer(depth=tran_params['depth'], rotary_embed=freq_rotary_embed, **transformer_kwargs)

    def forward(self, x):
        B, C, F, T = x.shape

        # Process dual-path rnn
        original_x = x
        # Frequency-path
        x = self.norm_layers[0](x)
        x = x.transpose(1, 3).contiguous().view(B * T, F, C)
        # print('XXX', x.shape)
        x = self.freq_layer(x)
        x = x.view(B, T, F, C).transpose(1, 3)
        x = x + original_x

        original_x = x
        # Time-path
        x = self.norm_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(B * F, C, T).transpose(1, 2)
        # print('RRR', x.shape)
        x = self.time_layer(x)
        x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
        x = x + original_x

        return x


class SeparationNetTran(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in an encoder architecture.

    Args:
    - channels (int): Number input channels.
    - expand (int): Expansion factor used to calculate the hidden_size of LSTM.
    - num_layers (int): Number of dual-path layers.
    """

    def __init__(self, channels, expand=1, num_layers=6, tran_params=None):
        super(SeparationNetTran, self).__init__()

        self.num_layers = num_layers

        time_rotary_embed = RotaryEmbedding(dim=tran_params['rotary_embedding_dim'])
        freq_rotary_embed = RotaryEmbedding(dim=tran_params['rotary_embedding_dim'])

        modules = []
        for i in range(num_layers):
            m = DualPathTran(channels, time_rotary_embed, freq_rotary_embed, tran_params)
            modules.append(m)
        self.dp_modules = nn.ModuleList(modules)



    def forward(self, x):
        for i in range(self.num_layers):
            x = self.dp_modules[i](x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class ConvolutionModule(nn.Module):
    """
    Convolution Module in SD block.

    Args:
        channels (int): input/output channels.
        depth (int): number of layers in the residual branch. Each layer has its own
        compress (float): amount of channel compression.
        kernel (int): kernel size for the convolutions.
        """

    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        norm = lambda d: nn.GroupNorm(1, d)
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            padding = (kernel // 2)
            mods = [
                norm(channels),
                nn.Conv1d(channels, hidden_size * 2, kernel, padding=padding),
                GLU(dim=1),
                nn.Conv1d(hidden_size, hidden_size, kernel, padding=padding, groups=hidden_size),
                norm(hidden_size),
                Swish(),
                nn.Conv1d(hidden_size, channels, 1),
            ]
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    """
    A FusionLayer within the decoder.

    Args:
    - channels (int): Number of input channels.
    - kernel_size (int, optional): Kernel size for the convolutional layer, defaults to 3.
    - stride (int, optional): Stride for the convolutional layer, defaults to 1.
    - padding (int, optional): Padding for the convolutional layer, defaults to 1.
    """

    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(FusionLayer, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)
        self.glu = GLU(dim=1)

    def forward(self, x, skip=None):
        if skip is not None:
            x += skip
        x = x.repeat(1, 2, 1, 1)
        x = self.conv(x)
        x = self.glu(x)
        return x


class SDlayer(nn.Module):
    """
    Implements a Sparse Down-sample Layer for processing different frequency bands separately.

    Args:
    - channels_in (int): Input channel count.
    - channels_out (int): Output channel count.
    - band_configs (dict): A dictionary containing configuration for each frequency band.
                           Keys are 'low', 'mid', 'high' for each band, and values are
                           dictionaries with keys 'SR', 'stride', and 'kernel' for proportion,
                           stride, and kernel size, respectively.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super(SDlayer, self).__init__()

        # Initializing convolutional layers for each band
        self.convs = nn.ModuleList()
        self.strides = []
        self.kernels = []
        for config in band_configs.values():
            self.convs.append(
                nn.Conv2d(channels_in, channels_out, (config['kernel'], 1), (config['stride'], 1), (0, 0)))
            self.strides.append(config['stride'])
            self.kernels.append(config['kernel'])

        # Saving rate proportions for determining splits
        self.SR_low = band_configs['low']['SR']
        self.SR_mid = band_configs['mid']['SR']

    def forward(self, x):
        B, C, Fr, T = x.shape
        # Define splitting points based on sampling rates
        splits = [
            (0, math.ceil(Fr * self.SR_low)),
            (math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))),
            (math.ceil(Fr * (self.SR_low + self.SR_mid)), Fr)
        ]

        # Processing each band with the corresponding convolution
        outputs = []
        original_lengths = []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end - start)
            current_length = extracted.shape[2]

            # padding
            if stride == 1:
                total_padding = kernel - stride
            else:
                total_padding = (stride - current_length % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            padded = F.pad(extracted, (0, 0, pad_left, pad_right))

            output = conv(padded)
            outputs.append(output)

        return outputs, original_lengths


class SUlayer(nn.Module):
    """
    Implements a Sparse Up-sample Layer in decoder.

    Args:
    - channels_in: The number of input channels.
    - channels_out: The number of output channels.
    - convtr_configs: Dictionary containing the configurations for transposed convolutions.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super(SUlayer, self).__init__()

        # Initializing convolutional layers for each band
        self.convtrs = nn.ModuleList([
            nn.ConvTranspose2d(channels_in, channels_out, [config['kernel'], 1], [config['stride'], 1])
            for _, config in band_configs.items()
        ])

    def forward(self, x, lengths, origin_lengths):
        B, C, Fr, T = x.shape
        # Define splitting points based on input lengths
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None)
        ]
        # Processing each band with the corresponding convolution
        outputs = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            # Calculate the distance to trim the output symmetrically to original length
            current_Fr_length = out.shape[2]
            dist = abs(origin_lengths[idx] - current_Fr_length) // 2

            # Trim the output to the original length symmetrically
            trimmed_out = out[:, :, dist:dist + origin_lengths[idx], :]

            outputs.append(trimmed_out)

        # Concatenate trimmed outputs along the frequency dimension to return the final tensor
        x = torch.cat(outputs, dim=2)

        return x


class SDblock(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in encoder.

    Args:
    - channels_in (int): Number of input channels.
    - channels_out (int): Number of output channels.
    - band_config (dict): Configuration for the SDlayer specifying band splits and convolutions.
    - conv_config (dict): Configuration for convolution modules applied to each band.
    - depths (list of int): List specifying the convolution depths for low, mid, and high frequency bands.
    """

    def __init__(self, channels_in, channels_out, band_configs={}, conv_config={}, depths=[3, 2, 1], kernel_size=3):
        super(SDblock, self).__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)

        # Dynamically create convolution modules for each band based on depths
        self.conv_modules = nn.ModuleList([
            ConvolutionModule(channels_out, depth, **conv_config) for depth in depths
        ])
        # Set the kernel_size to an odd number.
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2)

    def forward(self, x):
        bands, original_lengths = self.SDlayer(x)
        # B, C, f, T = band.shape
        bands = [
            F.gelu(
                conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3]))
                .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
                .permute(0, 2, 1, 3)
            )
            for conv, band in zip(self.conv_modules, bands)

        ]
        lengths = [band.size(-2) for band in bands]
        full_band = torch.cat(bands, dim=2)
        skip = full_band

        output = self.globalconv(full_band)

        return output, skip, lengths, original_lengths


class SCNet_Tran(nn.Module):
    """
    The implementation of SCNet: Sparse Compression Network for Music Source Separation. Paper: https://arxiv.org/abs/2401.13276.pdf
    LSTM layers replaced with transformer layers

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): List of channel dimensions for each block.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band
    - conv_depths (List[int]): List specifying the number of convolution modules in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layer in convolution module.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in the dual-path RNN, default is 1.

    """

    def __init__(
            self,
            sources=('drums', 'bass', 'other', 'vocals'),
            audio_channels=2,
            # Main structure
            dims=(4, 32, 64, 128),  # dims = [4, 64, 128, 256] in SCNet-large
            # STFT
            nfft=4096,
            hop_size=1024,
            win_size=4096,
            normalized=True,
            # SD/SU layer
            band_SR=(0.175, 0.392, 0.433),
            band_stride=(1, 4, 16),
            band_kernel=(3, 4, 16),
            # Convolution Module
            conv_depths=(3, 2, 1),
            compress=4,
            conv_kernel=3,
            # Dual-path RNN
            num_dplayer=6,
            expand=1,
            tran_rotary_embedding_dim=64,
            tran_depth=1,
            tran_heads=8,
            tran_dim_head=64,
            tran_attn_dropout=0.0,
            tran_ff_dropout=0.0,
            tran_flash_attn=False,
            chunk_size=44100*11,
            input_time_domain=True,
            output_time_domain=True
    ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i
                             in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }
        self.tran_params = {
            'rotary_embedding_dim': tran_rotary_embedding_dim,
            'depth': tran_depth,
            'heads': tran_heads,
            'dim_head': tran_dim_head,
            'attn_dropout': tran_attn_dropout,
            'ff_dropout': tran_ff_dropout,
            'flash_attn': tran_flash_attn,
        }

        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            # 'window' : torch.hann_window(win_size),
            'center': True,
            'normalized': normalized,
        }

        # self.first_conv = nn.Conv2d(dims[0], dims[0], 1, 1, 0, bias=False)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblock(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayer(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNetTran(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
            tran_params=self.tran_params
        )
        self.input_time_domain = input_time_domain
        self.output_time_domain = output_time_domain
        self.chunk_size = chunk_size

    def stft(self, x):
        return stft(x, stft_config = self.stft_config, audio_channels = self.audio_channels)

    def istft(self, x):
        return istft(x, stft_config = self.stft_config,
                     chunk_size= self.chunk_size, sources= self.sources,
                     audio_channels=self.audio_channels, dims=self.dims)

    def forward(self, x):

        if self.input_time_domain:
            x = self.stft(x)

        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # separation
        x = self.separation_net(x)

        # decoder
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        # output
        if self.output_time_domain:
            x = self.istft(x)

        return  x


def collate_fn(stft_config, sources, audio_channels, input_time_domain, output_time_domain, batch):
    # batch is a list of tuples: [(y1, x1), (y2, X2), ...]
    y, x = zip(*batch)  # Unpack into two lists: y's and x's

    x_batch = torch.stack(x)
    y_batch = torch.stack(y)

    if not input_time_domain:
        x_batch = stft(x_batch, stft_config, audio_channels) # [B x C x L] -> [B x 2*C x F x T]

    if not output_time_domain:
        B = y_batch.shape[0]
        y_batch = stft(y_batch, stft_config, audio_channels) # [B x S x C x L]  -> [B*S x 2*C x F x T]
        F = y_batch.shape[-2]
        T = y_batch.shape[-1]
        y_batch = y_batch.reshape(B, 2 * len(sources) * audio_channels, F, T)

    return y_batch, x_batch

def get_collate_fn(model: SCNet_Tran) -> Callable:

    stft_config = model.stft_config
    audio_channels= model.audio_channels
    sources= model.sources
    input_time_domain= model.input_time_domain
    output_time_domain= model.output_time_domain

    """Standalone function that takes model as parameter"""

    return partial(collate_fn, stft_config, sources, audio_channels, input_time_domain, output_time_domain)
