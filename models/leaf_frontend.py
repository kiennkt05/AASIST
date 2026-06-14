import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

@torch.jit.script
def differentiable_ema(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # x: (B, C, T)
    # s: (C, 1) - ensure s is unsqueezed to broadcast over B and T
    B, C, T = x.shape
    M = torch.zeros_like(x)
    
    # Initialize the first frame
    M[:, :, 0] = x[:, :, 0]
    
    # Sequential EMA loop (JIT compiler will optimize this in C++)
    for t in range(1, T):
        M[:, :, t] = s * x[:, :, t] + (1.0 - s) * M[:, :, t-1]
        
    return M

class sPCEN(nn.Module):
    def __init__(self, num_filters, alpha=0.96, smooth_coef=0.04, delta=2.0, root=2.0, floor=1e-6, trainable=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(num_filters).fill_(alpha), requires_grad=trainable)
        self.delta = nn.Parameter(torch.empty(num_filters).fill_(delta), requires_grad=trainable)
        self.root = nn.Parameter(torch.empty(num_filters).fill_(root), requires_grad=trainable)
        self.s = nn.Parameter(torch.empty(num_filters).fill_(smooth_coef), requires_grad=trainable)
        self.floor = floor

    def forward(self, x):
        # x shape: (B, C, T)
        s = torch.clamp(self.s, min=1e-4, max=0.9999).view(1, -1)
        alpha = torch.clamp(self.alpha, min=0.0, max=1.0).view(1, -1, 1)
        root = torch.clamp(self.root, min=1.0).view(1, -1, 1)
        delta = torch.clamp(self.delta, min=0.0).view(1, -1, 1)

        ema = differentiable_ema(x, s)
        
        one_over_root = 1.0 / root
        out = ((x / (self.floor + ema)**alpha + delta)**one_over_root) - delta**one_over_root
        return out

class GaborConv1D(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate

        # 1. Initialize with Mel scale
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = 2595 * np.log10(1 + f / 700)
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 1)
        filbandwidthsf = 700 * (10**(filbandwidthsmel / 2595) - 1)
        
        # Normalized center frequencies [0, 0.5]
        init_freqs = filbandwidthsf[:-1] / self.sample_rate
        # Initial bandwidths
        init_bw = (filbandwidthsf[1:] - filbandwidthsf[:-1]) / self.sample_rate

        self.center_freqs = nn.Parameter(torch.tensor(init_freqs, dtype=torch.float32))
        self.bandwidths = nn.Parameter(torch.tensor(init_bw, dtype=torch.float32))

    def get_filters(self):
        # Clamp to prevent divergence
        freqs = torch.clamp(self.center_freqs, min=0.0, max=0.5)
        # sigma min = 4*sqrt(2*ln(2)) / W
        sigma_min = 4 * math.sqrt(2 * math.log(2)) / self.kernel_size
        sigmas = torch.clamp(self.bandwidths, min=sigma_min, max=0.5)

        t = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1, device=freqs.device)
        t = t.view(1, -1)
        freqs = freqs.view(-1, 1)
        sigmas = sigmas.view(-1, 1)

        # Gabor = Gaussian * Sinusoid
        gaussian = (1 / (math.sqrt(2 * math.pi) * sigmas)) * torch.exp(-0.5 * (t / sigmas)**2)
        sinusoid_real = torch.cos(2 * math.pi * freqs * t)
        sinusoid_imag = torch.sin(2 * math.pi * freqs * t)

        real_filters = gaussian * sinusoid_real
        imag_filters = gaussian * sinusoid_imag
        
        filters = torch.cat([real_filters, imag_filters], dim=0).unsqueeze(1)
        return filters

    def forward(self, x):
        filters = self.get_filters()
        out = F.conv1d(x, filters, stride=1, padding=self.kernel_size//2)
        out_real, out_imag = torch.chunk(out, 2, dim=1)
        return out_real**2 + out_imag**2 # Squared modulus
