"""
Image preprocessing operations such as denoising and background suppression.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class Denoise2D:
    """
    Apply denoising on a single-channel tensor in [0,1].
    Methods:
      - 'gaussian': separable Gaussian blur with given kernel size and sigma
      - 'median'  : median blur with given kernel size
    """
    method: str = "gaussian"
    ksize: int = 5
    sigma: float = 1.0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert t.ndim == 3 and t.shape[0] == 1, "Expected [1,H,W] float tensor"
        c, h, w = t.shape
        k = int(self.ksize)
        if k % 2 == 0 or k < 1:
            raise ValueError("ksize must be an odd positive integer")

        if self.method == "gaussian":
            return self._gaussian_blur(t, k, float(self.sigma))
        elif self.method == "median":
            return self._median_blur(t, k)
        else:
            raise ValueError(f"Unknown denoise method: {self.method}")

    @staticmethod
    def _gaussian_kernel1d(ksize: int, sigma: float, device) -> torch.Tensor:
        radius = ksize // 2
        x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    def _gaussian_blur(self, t: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
        device = t.device
        g1d = self._gaussian_kernel1d(ksize, sigma, device)

        w_h = g1d.view(1, 1, 1, -1)
        w_v = g1d.view(1, 1, -1, 1)
        x = F.pad(t.unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
        x = F.conv2d(x, w_h, padding=0)
        x = F.conv2d(x, w_v, padding=0)
        return x.squeeze(0)

    def _median_blur(self, t: torch.Tensor, ksize: int) -> torch.Tensor:
        pad = ksize // 2
        x = F.pad(t.unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
        patches = F.unfold(x, kernel_size=ksize, stride=1)
        med = patches.median(dim=1).values
        h, w = t.shape[1], t.shape[2]
        med = med.view(1, 1, h, w)
        return med.squeeze(0)


@dataclass
class ROIBackgroundStrip:
    """
    Suppress background by thresholding (keeps original tensor shape).
    Default: Otsu computed on the single-channel image in [0,1].
    """
    method: str = "otsu"   # 'otsu' or 'fixed'
    thresh: float = 0.0    # if method == 'fixed'

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        assert t.ndim == 3 and t.shape[0] == 1, "Expected [1,H,W] float tensor in [0,1]"
        x = t[0]  # [H, W]

        if self.method == "fixed":
            thr = float(self.thresh)
        elif self.method == "otsu":
            nbins = 256
            x_clamped = x.clamp(0.0, 1.0)
            hist = torch.histc(x_clamped, bins=nbins, min=0.0, max=1.0)
            if hist.sum() == 0:
                thr = 0.0
            else:
                bin_edges = torch.linspace(0.0, 1.0, steps=nbins + 1, device=x.device, dtype=torch.float32)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                w0 = torch.cumsum(hist, dim=0)
                w1 = hist.sum() - w0
                m_cum = torch.cumsum(hist * bin_centers, 0)
                muT = m_cum[-1]

                # Avoid uninitialized vars: use zeros_like(w0/w1)
                mu0 = torch.where(w0 > 0, m_cum / w0, torch.zeros_like(w0, dtype=torch.float32))
                mu1 = torch.where(w1 > 0, (muT - m_cum) / w1, torch.zeros_like(w1, dtype=torch.float32))

                sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
                idx = torch.argmax(sigma_b2)
                thr = float(bin_centers[idx].item())

        else:
            raise ValueError(f"Unknown ROI method: {self.method}")

        mask = (x > thr).to(t.dtype)
        return t * mask.unsqueeze(0)
