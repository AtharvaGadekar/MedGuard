"""Deterministic synthetic blur operators shared by evaluation and demo."""

from __future__ import annotations

import torch
import torch.nn.functional as F


KERNEL_SIZES = (3, 5, 7, 9, 11)


def blur_kernel(kind: str, size: int, device: torch.device) -> torch.Tensor:
    if kind == "defocus":
        coordinates = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        radius = (size - 1) / 2
        kernel = ((xx.square() + yy.square()) <= radius**2 + 1e-6).float()
    elif kind == "motion":
        kernel = torch.zeros((size, size), dtype=torch.float32, device=device)
        kernel[size // 2, :] = 1.0
    else:
        raise ValueError(f"Unsupported corruption: {kind}")
    return kernel / kernel.sum()


def apply_blur(images: torch.Tensor, kind: str, size: int) -> torch.Tensor:
    """Apply deterministic channel-wise blur with reflect padding to a BCHW batch."""
    kernel = blur_kernel(kind, size, images.device).view(1, 1, size, size).expand(images.shape[1], 1, size, size)
    padded = F.pad(images, (size // 2,) * 4, mode="reflect")
    return F.conv2d(padded, kernel, groups=images.shape[1])
