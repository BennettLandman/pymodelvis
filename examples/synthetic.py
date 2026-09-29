"""Synthetic data generators for the examples (no downloads needed).

* ``cells_2d``: microscopy-like 2-D images with round bright "cells" (label 1)
  and elongated dim "debris" (label 2).
* ``brain_3d``: an MRI-like 3-D head phantom (skull, brain, ventricles, optional
  lesion) in canonical ``[X, Y, Z]`` order (x: left→right, y: posterior→anterior,
  z: inferior→superior).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _smooth_noise(shape, rng, scale=8):
    small = rng.standard_normal([max(2, s // scale) for s in shape]).astype(np.float32)
    t = torch.from_numpy(small)[None, None]
    mode = "bilinear" if len(shape) == 2 else "trilinear"
    return F.interpolate(t, size=shape, mode=mode, align_corners=False)[0, 0].numpy()


def cells_2d(n: int, size: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, size, size), np.float32)
    Y = np.zeros((n, size, size), np.int64)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for i in range(n):
        img = 0.15 + 0.08 * _smooth_noise((size, size), rng, 16)
        lab = np.zeros((size, size), np.int64)
        for _ in range(rng.integers(4, 9)):   # cells
            cx, cy = rng.uniform(10, size - 10, 2)
            r = rng.uniform(5, 11)
            m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
            img[m] = 0.8 + 0.15 * rng.standard_normal() * 0.3 + 0.1 * (1 - ((xx[m] - cx) ** 2 + (yy[m] - cy) ** 2) / r ** 2)
            lab[m] = 1
        for _ in range(rng.integers(2, 5)):   # debris
            cx, cy = rng.uniform(10, size - 10, 2)
            a, b = rng.uniform(10, 18), rng.uniform(2, 3.5)
            th = rng.uniform(0, math.pi)
            u = (xx - cx) * math.cos(th) + (yy - cy) * math.sin(th)
            v = -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
            m = (u / a) ** 2 + (v / b) ** 2 < 1
            m &= lab == 0
            img[m] = 0.5
            lab[m] = 2
        img += 0.05 * rng.standard_normal((size, size))
        X[i, 0] = img
        Y[i] = lab
    return torch.from_numpy(X), torch.from_numpy(Y)


def brain_phantom(size: int = 48, age: float = 50.0, lesion_r: float = 0.0,
                  lesion_center=(0.2, 0.15, 0.2), seed: int = 0) -> np.ndarray:
    """One MRI-like head volume [X, Y, Z] with explicit age (ventricle size) and lesion radius.

    Anatomy (head shape, texture, noise) is fixed by ``seed`` so a sequence of calls with
    changing ``age`` / ``lesion_r`` shows the *same subject* changing over time.
    """
    rng = np.random.default_rng(seed)
    g = np.linspace(-1, 1, size, dtype=np.float32)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    a = rng.uniform(0.78, 0.88)
    b = a * rng.uniform(1.08, 1.18)
    c = a * rng.uniform(0.9, 1.0)
    head = (x / a) ** 2 + (y / b) ** 2 + ((z + 0.05) / c) ** 2
    skull = (head < 1.0) & (head > 0.82)
    brain = head < 0.78
    vol = np.zeros_like(x)
    vol[skull] = 0.35
    tex = _smooth_noise((size,) * 3, rng, 6)
    vol[brain] = 0.62 + 0.06 * tex[brain]
    rim = (head < 0.78) & (head > 0.6)
    vol[rim] = 0.48 + 0.05 * tex[rim]
    vs = 0.07 + 0.12 * (age - 20) / 65
    for sx in (-1, 1):
        ven = ((x - sx * 0.12) / vs) ** 2 + ((y + 0.02) / (vs * 2.6)) ** 2 + ((z - 0.08) / (vs * 1.5)) ** 2 < 1
        vol[ven] = 0.12
    mask = np.zeros(x.shape, np.int64)
    if lesion_r > 0:
        cx, cy, cz = lesion_center
        les = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < lesion_r ** 2) & brain
        vol[les] = 0.95
        mask[les] = 1
    vol += 0.03 * rng.standard_normal(vol.shape).astype(np.float32)
    return vol.astype(np.float32), mask


def brain_3d(n: int, size: int = 64, seed: int = 0, lesion_prob: float = 0.6):
    """Returns volumes [n, 1, X, Y, Z], lesion masks [n, X, Y, Z], lesion flags [n], 'age' targets [n]."""
    rng = np.random.default_rng(seed)
    g = np.linspace(-1, 1, size, dtype=np.float32)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    V = np.zeros((n, 1, size, size, size), np.float32)
    M = np.zeros((n, size, size, size), np.int64)
    has = np.zeros(n, np.int64)
    age = np.zeros(n, np.float32)
    for i in range(n):
        a = rng.uniform(0.78, 0.88)
        b = a * rng.uniform(1.08, 1.18)
        c = a * rng.uniform(0.9, 1.0)
        head = (x / a) ** 2 + (y / b) ** 2 + ((z + 0.05) / c) ** 2
        skull = (head < 1.0) & (head > 0.82)
        brain = head < 0.78
        vol = np.zeros_like(x)
        vol[skull] = 0.35
        tex = _smooth_noise((size,) * 3, rng, 6)
        vol[brain] = 0.62 + 0.06 * tex[brain]
        # grey-matter rim
        vol[(head < 0.78) & (head > 0.6)] = 0.48 + 0.05 * tex[(head < 0.78) & (head > 0.6)]
        # ventricles grow with "age"
        ag = rng.uniform(20, 85)
        age[i] = ag
        vs = 0.07 + 0.12 * (ag - 20) / 65
        for sx in (-1, 1):
            ven = ((x - sx * 0.12) / vs) ** 2 + ((y + 0.02) / (vs * 2.6)) ** 2 + ((z - 0.08) / (vs * 1.5)) ** 2 < 1
            vol[ven] = 0.12
        if rng.uniform() < lesion_prob:
            has[i] = 1
            cx, cy, cz = rng.uniform(-0.4, 0.4), rng.uniform(-0.45, 0.45), rng.uniform(-0.2, 0.4)
            r = rng.uniform(0.15, 0.24)
            les = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < r ** 2) & brain
            vol[les] = 0.95
            M[i][les] = 1
        vol += 0.03 * rng.standard_normal(vol.shape).astype(np.float32)
        V[i, 0] = vol
    return torch.from_numpy(V), torch.from_numpy(M), torch.from_numpy(has), torch.from_numpy(age)
