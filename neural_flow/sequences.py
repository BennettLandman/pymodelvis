"""Input-sequence generators for movies (all return tensors ``[T, ...]``).

    frames = pan(image, window=224, steps=48)            # camera pan across a wide image
    frames = crossfade(img_a, img_b, steps=32)           # morph between two inputs
    frames = zoom(image, start=1.0, end=0.35, steps=40)  # zoom into the centre (or a point)
    frames = occlusion_sweep(image, patch=48, stride=24) # sliding grey patch
    frames = slices_to_frames(volume, axis=-1)           # sweep through a 3-D volume as 2-D inputs
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _chw(img) -> torch.Tensor:
    t = torch.as_tensor(img)
    if t.dim() == 4:
        t = t[0]
    return t.float()


def pan(image, window: int = 224, steps: int = 48, out_size: Optional[int] = None, vertical: bool = False) -> torch.Tensor:
    """Slide a square window across a wide (or tall) image [C, H, W]."""
    x = _chw(image)
    C, H, W = x.shape
    frames = []
    if not vertical:
        for s in np.linspace(0, max(W - window, 0), steps):
            s = int(round(s))
            frames.append(x[:, (H - window) // 2:(H - window) // 2 + window, s:s + window])
    else:
        for s in np.linspace(0, max(H - window, 0), steps):
            s = int(round(s))
            frames.append(x[:, s:s + window, (W - window) // 2:(W - window) // 2 + window])
    out = torch.stack(frames)
    if out_size and out_size != window:
        out = F.interpolate(out, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return out


def crossfade(a, b, steps: int = 32, ease: bool = True) -> torch.Tensor:
    a, b = _chw(a), _chw(b)
    ts = np.linspace(0, 1, steps)
    if ease:
        ts = 0.5 - 0.5 * np.cos(np.pi * ts)
    return torch.stack([(1 - t) * a + t * b for t in ts])


def zoom(image, start: float = 1.0, end: float = 0.35, steps: int = 40, center: Optional[Tuple[float, float]] = None,
         out_size: Optional[int] = None) -> torch.Tensor:
    """Zoom from a crop of relative size ``start`` to ``end`` around ``center`` (fractions of H, W)."""
    x = _chw(image)
    C, H, W = x.shape
    cy, cx = center if center is not None else (0.5, 0.5)
    out_size = out_size or min(H, W)
    frames = []
    for s in np.geomspace(start, end, steps):
        h, w = int(round(H * s)), int(round(W * s))
        t = int(np.clip(round(cy * H - h / 2), 0, H - h))
        l = int(np.clip(round(cx * W - w / 2), 0, W - w))
        crop = x[:, t:t + h, l:l + w][None]
        frames.append(F.interpolate(crop, size=(out_size, out_size), mode="bilinear", align_corners=False)[0])
    return torch.stack(frames)


def occlusion_sweep(image, patch: int = 48, stride: int = 24, value: float = 0.0) -> torch.Tensor:
    x = _chw(image)
    C, H, W = x.shape
    frames = []
    for top in range(0, H - patch + 1, stride):
        for left in range(0, W - patch + 1, stride):
            f = x.clone()
            f[:, top:top + patch, left:left + patch] = value
            frames.append(f)
    return torch.stack(frames)


def slices_to_frames(volume, axis: int = -1, step: int = 1, channels: int = 1) -> torch.Tensor:
    """[D, H, W] or [C, D, H, W] volume → [T, C, H, W] 2-D frames along spatial ``axis`` (0, 1, 2 or negative)."""
    v = torch.as_tensor(volume).float()
    if v.dim() == 3:
        v = v[None]
    v = v.movedim(1 + (axis % 3), 1)                             # spatial axis → dim 1: [C, T, H, W]
    return v[:, ::step].transpose(0, 1).contiguous()
