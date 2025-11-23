import torch
import torch.nn as nn

# inspired by https://github.com/junyuchen-cjy/DTTNet-Pytorch


class AbstractModel(nn.Module):

    def __init__(self,
                 dim_f: int = 2048,
                 dim_t: int = 384,
                 n_fft: int = 6144,
                 hop_length: int = 1024,
                 overlap: int = 3072,
                 audio_ch: int = 2):
        super().__init__()
        self.dim_c_in = audio_ch * 2
        self.dim_c_out: int = audio_ch * 2
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.audio_ch = audio_ch

        self.chunk_size = hop_length * (self.dim_t - 1)
        self.inference_chunk_size = hop_length * (self.dim_t*2 - 1)
        self.overlap = overlap
        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, self.dim_c_out, self.n_bins - self.dim_f, 1]), requires_grad=False)
        self.inference_chunk_shape = (self.stft(torch.zeros([1, audio_ch, self.inference_chunk_size]))).shape



    def stft(self, x):
        """
        Args:
            x: (batch, c, 261120)
        """
        dim_b = x.shape[0]

        # (batch*c, 261120)
        x = x.reshape([dim_b * self.audio_ch, -1])

        # complex STFT → (batch*c, 3073, 256)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True,
                       return_complex=True)

        # Convert complex -> real/imag (batch*c, 3073, 256, 2)
        x = torch.view_as_real(x)

        # Reorder to your model format
        x = x.permute(0, 3, 1, 2)  # (batch*c, 2, 3073, 256)
        x = x.reshape(dim_b, self.audio_ch, 2, self.n_bins, -1).reshape(dim_b, self.audio_ch * 2, self.n_bins, -1) # (batch, c*2, 3073, 256)

        return x[:, :, :self.dim_f]  # (batch, c*2, 2048, 256)

    def istft(self, x):
        """
        Args:
            x: (batch, c*2, 2048, 256)
        """
        dim_b = x.shape[0]

        # Pad freq dimension back to 3073
        x = torch.cat(
            [x, self.freq_pad.repeat(dim_b, 1, 1, x.shape[-1])],
            dim=2
        )  # (batch, c*2, 3073, 256)

        # Reshape back to (batch*c, 2, 3073, 256)
        x = x.reshape(dim_b, self.audio_ch, 2, self.n_bins, -1)
        x = x.reshape(dim_b * self.audio_ch, 2, self.n_bins, -1)
        x = x.permute(0, 2, 3, 1)  # (batch*c, 3073, 256, 2)

        # Convert back to complex
        x = x.contiguous()
        x = torch.view_as_complex(x)

        # iSTFT → (batch*c, 261120)
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
        )

        # reshape return
        return x.reshape(dim_b, self.audio_ch, -1)  # (batch, c, 261120)

    def demix(self, mix, inf_chunk_size, batch_size=5, inf_overf=4):
        '''
        Args:
            mix: (C, L)
        Returns:
            est: (src, C, L)
        '''

        # batch_size = self.config.inference.batch_size
        #  = self.chunk_size
        # self.instruments = ['bass', 'drums', 'other', 'vocals']
        num_instruments = 1

        inf_hop = inf_chunk_size // inf_overf  # hop size
        L = mix.shape[1]
        pad_size = inf_hop - (L - inf_chunk_size) % inf_hop
        mix = torch.cat([torch.zeros(2, inf_chunk_size - inf_hop), torch.Tensor(mix), torch.zeros(2, pad_size + inf_chunk_size - inf_hop)], 1)
        mix = mix.cuda()

        chunks = []
        i = 0
        while i + inf_chunk_size <= mix.shape[1]:
            chunks.append(mix[:, i:i + inf_chunk_size])
            i += inf_hop
        chunks = torch.stack(chunks)

        batches = []
        i = 0
        while i < len(chunks):
            batches.append(chunks[i:i + batch_size])
            i = i + batch_size

        X = torch.zeros(num_instruments, 2, inf_chunk_size - inf_hop) # (src, c, t)
        X = X.cuda()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for batch in batches:
                    x = self.stft(batch)
                    x = self(x)
                    x = self.istft(x) # (batch, c, 261120)
                    # insert new axis, the model only predict 1 src so we need to add axis
                    x = x[:,None, ...] # (batch, 1, c, 261120)
                    x = x.repeat([ 1, num_instruments, 1, 1]) # (batch, src, c, 261120)
                    for w in x: # iterate over batch
                        a = X[..., :-(inf_chunk_size - inf_hop)]
                        b = X[..., -(inf_chunk_size - inf_hop):] + w[..., :(inf_chunk_size - inf_hop)]
                        c = w[..., (inf_chunk_size - inf_hop):]
                        X = torch.cat([a, b, c], -1)

        estimated_sources = X[..., inf_chunk_size - inf_hop:-(pad_size + inf_chunk_size - inf_hop)] / inf_overf

        assert L == estimated_sources.shape[-1]

        return estimated_sources
