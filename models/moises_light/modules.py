import torch.nn as nn

from models.moises_light.batch_norm import get_norm

# inspired by https://github.com/junyuchen-cjy/DTTNet-Pytorch

class SplitModule(nn.Module):
    def __init__(self, c_in, c_out, l, k, bn_norm, n_band):
        super(SplitModule, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            if i == 0:
                c_in = c_in
            else:
                c_in = c_out
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2, groups=n_band),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x



class SplitMergeModule(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, dense=False, bias=True, n_band=4):

        super(SplitMergeModule, self).__init__()

        self.n_band = n_band

        self.tfc1 = SplitModule(c_in, c_out, l, k, bn_norm, n_band)
        self.tfc2 = SplitModule(c_in, c_out, l, k, bn_norm, n_band)

        self.res = SplitModule(c_in, c_out, 1, k, bn_norm, 1)

        self.tdf = nn.Sequential(
            nn.Linear(f, f // bn, bias=bias),
            get_norm(bn_norm, c_out),
            nn.ReLU(),
            nn.Linear(f // bn, f, bias=bias),
            get_norm(bn_norm, c_out),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.res(x)
        x = self.tfc1(x)
        x = x + self.tdf(x)
        x = self.tfc2(x)
        x = x + res
        return x