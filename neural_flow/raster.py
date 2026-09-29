"""Tensor → raster rendering.

Every activation summary is turned into a single RGBA ``numpy`` image (a
:class:`Visual`).  Composing mosaics, voxel stacks and strips into one image per
stage keeps figure assembly, SVG output and animation simple: each stage is one
``imshow`` artist while text, arrows and outlines stay vector.

Includes a small emission–absorption **volume renderer** (``render_dvr``) built
on ``torch.nn.functional.grid_sample``: feature volumes become translucent,
shaded voxel blocks, low-resolution volumes keep their blocky voxel character.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from .config import FlowConfig
from .tensors import TensorSummary

RGBA = np.ndarray

STRATEGY_NAMES = {"energy": "energy (mean x²)", "variance": "variance", "spread": "robust spread (p90−p10)", "mean_abs": "mean |x|",
                  "even": "evenly spaced", "pca": "PCA representatives"}


@dataclass
class Visual:
    image: RGBA                               # [H, W, 4] float in [0,1]
    kind: str                                 # mosaic | image | strip | heatmap | volume | ortho | tokens | seg | generic
    caption: str = ""
    n_shown: int = 0
    extras: Dict[str, RGBA] = field(default_factory=dict)   # e.g. cls strip, ortho triplet, attention
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def aspect(self) -> float:
        h, w = self.image.shape[:2]
        return w / max(h, 1)


# ---------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------


def get_cmap(name: str):
    return matplotlib.colormaps.get_cmap(name)


def act_cmap(cfg: FlowConfig) -> str:
    if cfg.cmap:
        return cfg.cmap
    return "magma" if cfg.theme == "light" else "inferno"


def robust_norm(a: np.ndarray, pct=(1.0, 99.0), diverging: bool = False,
                lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, dtype=np.float32))
    if diverging:
        m = hi if hi is not None else float(np.percentile(np.abs(a), pct[1])) if a.size else 1.0
        m = m if m > 1e-12 else 1.0
        return np.clip(0.5 + 0.5 * a / m, 0, 1)
    if lo is None or hi is None:
        if a.size:
            lo, hi = np.percentile(a, pct)
        else:
            lo, hi = 0.0, 1.0
    if hi - lo < 1e-12:
        return np.zeros_like(a) + (0.0 if abs(hi) < 1e-12 else 0.5)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def colorize(a01: np.ndarray, cmap: str) -> RGBA:
    return get_cmap(cmap)(a01).astype(np.float32)


def _use_diverging(summ: TensorSummary, cfg: FlowConfig) -> bool:
    if cfg.diverging == "auto":
        return summ.stats.get("frac_neg", 0) > 0.25
    return bool(cfg.diverging)


def upsample(img: np.ndarray, min_px: int = 96, max_factor: int = 32) -> np.ndarray:
    h, w = img.shape[:2]
    f = int(min(max_factor, max(1, math.ceil(min_px / max(1, min(h, w))))))
    if f == 1:
        return img
    return np.repeat(np.repeat(img, f, axis=0), f, axis=1)


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest resize (keeps activation pixels crisp)."""
    ih, iw = img.shape[:2]
    ys = np.minimum((np.arange(h) * ih / h).astype(int), ih - 1)
    xs = np.minimum((np.arange(w) * iw / w).astype(int), iw - 1)
    return img[ys][:, xs]


def mosaic(tiles: Sequence[RGBA], cols: Optional[int] = None, gap: int = 4, pad_color=(0, 0, 0, 0)) -> RGBA:
    n = len(tiles)
    if n == 0:
        return np.zeros((8, 8, 4), np.float32)
    cols = cols or int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    out = np.zeros((rows * th + (rows - 1) * gap, cols * tw + (cols - 1) * gap, 4), np.float32)
    out[:] = pad_color
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (th + gap), c * (tw + gap)
        out[y:y + t.shape[0], x:x + t.shape[1]] = t
    return out


def alpha_over(dst: RGBA, src: RGBA, y: int, x: int) -> None:
    """Composite ``src`` over ``dst`` in place at (y, x) (straight alpha)."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    y0, x0 = max(y, 0), max(x, 0)
    y1, x1 = min(y + h, H), min(x + w, W)
    if y1 <= y0 or x1 <= x0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    sa = s[..., 3:4]
    da = d[..., 3:4]
    oa = sa + da * (1 - sa)
    rgb = (s[..., :3] * sa + d[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
    d[..., :3] = rgb
    d[..., 3:4] = oa


def to_png_bytes(img: RGBA) -> bytes:
    from PIL import Image

    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def to_data_uri(img: RGBA) -> str:
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(img)).decode("ascii")


def _frame(img: RGBA, color=(0.55, 0.55, 0.6, 1.0), width: int = 1) -> RGBA:
    out = img.copy()
    out[:width, :] = color
    out[-width:, :] = color
    out[:, :width] = color
    out[:, -width:] = color
    return out


# ---------------------------------------------------------------------------
# 2-D maps
# ---------------------------------------------------------------------------


def grid_for_channels(C: int, max_channels: int) -> int:
    """Mosaic side length: grows with channel depth (64→2×2, 128→3×3, 512→4×4)."""
    gmax = max(1, int(math.sqrt(max_channels)))
    if C <= 4:
        return 1
    g = int(math.floor(math.log2(C) / 2 - 0.5))
    return int(min(max(g, 2), gmax))


def map_tiles(maps: np.ndarray, cfg: FlowConfig, summ: TensorSummary, cmap: str, tile_px: int = 96) -> List[RGBA]:
    div = _use_diverging(summ, cfg)
    cm = "RdBu_r" if div else cmap
    lo = hi = None
    if cfg.normalize == "stage" and maps.size:
        if div:
            hi = float(np.percentile(np.abs(maps), cfg.percentiles[1]))
        else:
            lo, hi = np.percentile(maps, cfg.percentiles)
    tiles = []
    for m in maps:
        a = robust_norm(m, cfg.percentiles, div, lo, hi)
        tiles.append(upsample(colorize(a, cm), tile_px))
    return tiles


def pca_rgb(pmaps: np.ndarray, pct=(1, 99)) -> RGBA:
    k = pmaps.shape[0]
    chans = [robust_norm(pmaps[i], pct) for i in range(min(3, k))]
    while len(chans) < 3:
        chans.append(np.zeros_like(chans[0]))
    rgb = np.stack(chans, -1)
    return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], -1).astype(np.float32)


def render_image2d(summ: TensorSummary, cfg: FlowConfig, n: Optional[int] = None,
                   strategy: Optional[str] = None, style: str = "technical") -> Visual:
    strategy = strategy or cfg.channel_strategy
    g = grid_for_channels(summ.channels, cfg.max_channels)
    n = n or (summ.channels if summ.channels <= 4 else g * g)
    cmap = act_cmap(cfg)
    if strategy == "pca" and summ.pca_maps is not None and n == 1:
        tiles = [upsample(pca_rgb(summ.pca_maps, cfg.percentiles), 96)]
        ids = [-1]
    else:
        maps, ids = summ.channel_maps(strategy, n)
        if len(maps) == 0:
            return render_generic(summ, cfg)
        tiles = map_tiles(maps, cfg, summ, cmap)
    cols = len(tiles) if len(tiles) <= 3 else int(math.ceil(math.sqrt(len(tiles))))
    img = mosaic(tiles, cols=cols, gap=max(3, tiles[0].shape[0] // 12))
    return Visual(img, "mosaic", n_shown=len(tiles), meta={"channels": ids, "strategy": strategy})


def render_input_image(summ: TensorSummary, cfg: FlowConfig) -> Visual:
    im = summ.image
    C = im.shape[0]
    if C == 3:
        rgb = np.stack([robust_norm(im[i], (0.5, 99.5)) for i in range(3)], -1) if (im.min() < 0 or im.max() > 1) \
            else np.transpose(im, (1, 2, 0))
        # shared normalisation keeps colours faithful
        if im.min() < 0 or im.max() > 1:
            lo, hi = np.percentile(im, (0.5, 99.5))
            rgb = np.clip((np.transpose(im, (1, 2, 0)) - lo) / max(hi - lo, 1e-6), 0, 1)
        img = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], -1).astype(np.float32)
    elif C == 1:
        img = colorize(robust_norm(im[0], (0.5, 99.5)), "gray")
    else:
        img = mosaic([colorize(robust_norm(im[i]), "gray") for i in range(C)], gap=3)
    return Visual(upsample(img, 160), "image", n_shown=C)


# ---------------------------------------------------------------------------
# vectors / tokens / attention
# ---------------------------------------------------------------------------


def render_vector(vec: np.ndarray, cfg: FlowConfig, sort: bool = False, max_len: int = 512,
                  cols: Optional[int] = None) -> Visual:
    v = np.asarray(vec, np.float32).reshape(-1)
    n0 = v.size
    if v.size > max_len:
        f = int(math.ceil(v.size / max_len))
        pad = (-v.size) % f
        v = np.concatenate([v, np.full(pad, np.nan)]).reshape(-1, f)
        v = np.nanmean(v, 1)
    if sort:
        v = np.sort(v)[::-1]
    n = v.size
    cols = cols or (1 if n <= 16 else 2 if n <= 64 else 4 if n <= 256 else 8)
    rows = int(math.ceil(n / cols))
    pad = rows * cols - n
    frac_neg = float((v < 0).mean()) if v.size else 0
    div = (cfg.diverging is True) or (cfg.diverging == "auto" and frac_neg > 0.25)
    a = robust_norm(v, cfg.percentiles, div)
    a = np.concatenate([a, np.full(pad, np.nan)]).reshape(rows, cols, order="F")
    img = colorize(np.nan_to_num(a), "RdBu_r" if div else act_cmap(cfg))
    img[np.isnan(a)] = (0, 0, 0, 0)
    cell = max(4, int(240 / max(rows, 1)))
    img = np.repeat(np.repeat(img, cell, 0), cell, 1)
    return Visual(img, "strip", n_shown=n0, meta={"rows": rows, "cols": cols})


def render_tokens(summ: TensorSummary, cfg: FlowConfig, strategy: Optional[str] = None) -> Visual:
    mode = cfg.token_mode
    if mode == "auto":
        mode = "grid" if summ.maps is not None else "heatmap"
    if mode in ("pca", "l2", "mean") and summ.maps is not None:
        if mode == "pca" and summ.pca_maps is not None:
            img = upsample(pca_rgb(summ.pca_maps, cfg.percentiles), 160)
        elif mode == "l2" and summ.token_norm_map is not None:
            img = upsample(colorize(robust_norm(summ.token_norm_map, cfg.percentiles), act_cmap(cfg)), 160)
        else:
            m = summ.maps.mean(0)
            img = upsample(colorize(robust_norm(m, cfg.percentiles), act_cmap(cfg)), 160)
        v = Visual(img, "tokens", n_shown=1, meta={"reduction": mode})
        if summ.cls is not None and summ.cls.size:
            v.extras["cls"] = render_vector(summ.cls[0], cfg, max_len=256, cols=4).image
        return v
    if mode == "grid" and summ.maps is not None:
        v = render_image2d(summ, cfg, strategy=strategy)
        v.kind = "tokens"
        if summ.cls is not None and summ.cls.size:
            v.extras["cls"] = render_vector(summ.cls[0], cfg, max_len=256, cols=4).image
        if summ.pca_maps is not None:
            v.extras["pca"] = upsample(pca_rgb(summ.pca_maps, cfg.percentiles), 96)
        return v
    tm = summ.token_matrix
    if tm is None:
        return render_generic(summ, cfg)
    a = robust_norm(tm, cfg.percentiles)
    img = upsample(colorize(a, act_cmap(cfg)), 160)
    return Visual(img, "heatmap", n_shown=tm.shape[0], meta={"axes": ("tokens", "features")})


def render_attention(att: np.ndarray, cfg: FlowConfig, n_special: int = 0,
                     grid: Optional[Tuple[int, int]] = None) -> Dict[str, RGBA]:
    """att: [heads, N, N] or [N, N]. Returns head-average matrix, CLS→patch map, entropy map."""
    a = att if att.ndim == 3 else att[None]
    avg = a.mean(0)
    out = {"matrix": upsample(colorize(robust_norm(np.sqrt(avg), (1, 99.5)), "viridis"), 128)}
    if grid is not None and n_special >= 1 and avg.shape[1] == n_special + grid[0] * grid[1]:
        cls_row = avg[0, n_special:].reshape(grid)
        out["cls"] = upsample(colorize(robust_norm(cls_row, (1, 99.5)), "viridis"), 96)
    ent = -(np.clip(a, 1e-12, 1) * np.log(np.clip(a, 1e-12, 1))).sum(-1).mean(0)
    out["entropy_vector"] = ent
    return out


def attention_rollout(mats: Sequence[np.ndarray], residual: float = 0.5) -> np.ndarray:
    """Abnar & Zuidema attention rollout over [N, N] head-averaged matrices."""
    if not mats:
        raise ValueError("no attention matrices")
    n = mats[0].shape[-1]
    R = np.eye(n)
    for A in mats:
        if A.shape[-1] != n:
            continue
        A = residual * np.eye(n) + (1 - residual) * A
        A = A / A.sum(-1, keepdims=True)
        R = A @ R
    return R


def render_generic(summ: TensorSummary, cfg: FlowConfig) -> Visual:
    v = summ.vector
    if v is None and summ.token_matrix is not None:
        return Visual(upsample(colorize(robust_norm(summ.token_matrix), act_cmap(cfg)), 128), "heatmap")
    if v is None:
        v = np.zeros(1, np.float32)
    vis = render_vector(v, cfg)
    vis.kind = "generic"
    return vis


def render_seq1d(summ: TensorSummary, cfg: FlowConfig) -> Visual:
    tm = summ.token_matrix
    img = colorize(robust_norm(tm, cfg.percentiles), act_cmap(cfg))
    img = _resize_to(img, 160, 240)
    return Visual(img, "heatmap", meta={"axes": ("channels", "length")})


# ---------------------------------------------------------------------------
# 3-D volumes
# ---------------------------------------------------------------------------


def canonical_volume(v: np.ndarray, axes: str = "xyz") -> np.ndarray:
    """Return V[x, y, z] with x = left→right, y = posterior→anterior, z = inferior→superior.

    ``axes="xyz"`` (default) assumes nibabel/MONAI order; ``"dhw"`` assumes a
    torch/DICOM slice stack ``[D(slices, inf→sup), H(rows, ant→post), W(cols)]``.
    """
    if axes == "dhw":
        return np.ascontiguousarray(np.transpose(v, (2, 1, 0))[:, ::-1, :])
    return v


def ortho_slices(V: np.ndarray, loc: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    x, y, z = loc
    return {
        "axial": np.flipud(V[:, :, z].T),
        "coronal": np.flipud(V[:, y, :].T),
        "sagittal": np.flipud(V[x, :, :].T),
    }


def activation_peak(E: np.ndarray) -> Tuple[int, int, int]:
    """Location of strongest (smoothed) activation energy."""
    t = torch.as_tensor(np.nan_to_num(E), dtype=torch.float32)[None, None]
    f = max(1, int(math.ceil(max(E.shape) / 24)))          # work on a <=24^3 grid for speed
    small = F.avg_pool3d(t, f, stride=f, ceil_mode=True) if f > 1 else t
    mn = min(small.shape[2:])
    k = max(1, mn // 6) * 2 + 1
    if k > mn:
        k = mn if mn % 2 == 1 else mn - 1
    k = max(k, 1)
    s = F.avg_pool3d(small, k, stride=1, padding=k // 2, count_include_pad=True)[0, 0].numpy()
    peak = s >= s.max() * 0.97
    if not peak.any() or s.max() <= 0:
        return tuple(int(v) // 2 for v in E.shape)
    idx = np.argwhere(peak)
    w = s[peak]
    c = (idx * w[:, None]).sum(0) / w.sum()
    c = (c + 0.5) * f - 0.5
    return tuple(int(np.clip(round(v), 0, n - 1)) for v, n in zip(c, E.shape))


def center_of_mass(V: np.ndarray) -> Tuple[int, int, int]:
    w = np.clip(V - np.percentile(V, 50), 0, None)
    tot = w.sum()
    if tot <= 0:
        return tuple(s // 2 for s in V.shape)
    idx = np.indices(V.shape).reshape(3, -1)
    c = (idx * w.reshape(-1)).sum(1) / tot
    return tuple(int(round(v)) for v in c)


def _camera(azim: float, elev: float):
    az, el = math.radians(azim), math.radians(elev)
    d = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    fwd = -d
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def _blockify(V: np.ndarray, target: int = 36, gap: Optional[bool] = None):
    """Nearest-upsample small volumes and carve thin gaps so voxels read as discrete blocks."""
    small = max(V.shape) <= 12
    if not small:
        return V, None, 1
    f = int(math.ceil(target / max(V.shape)))
    U = np.repeat(np.repeat(np.repeat(V, f, 0), f, 1), f, 2)
    mask = None
    if gap is None or gap:
        mask = np.ones(U.shape, np.float32)
        g = max(1, f // 6)
        for ax in range(3):
            sel = (np.arange(U.shape[ax]) % f) >= f - g
            shp = [1, 1, 1]
            shp[ax] = -1
            mask = mask * np.where(sel, 0.0, 1.0).reshape(shp)
    return U, mask, f


def raycast(alpha_vol: np.ndarray, rgb_vol: np.ndarray, extent_shape, size: int = 150, azim: float = 35.0,
            elev: float = 22.0, shade: bool = True, wire: bool = True, ref_steps: float = 36.0,
            depth_cue: float = 0.25) -> RGBA:
    """Orthographic front-to-back compositing of an (alpha, rgb) volume in canonical [x, y, z] order."""
    chans = [alpha_vol[None]] + [rgb_vol[..., i][None] for i in range(3)]
    if shade:
        gx, gy, gz = np.gradient(alpha_vol)
        chans += [gx[None], gy[None], gz[None]]
    vol = torch.as_tensor(np.concatenate(chans, 0), dtype=torch.float32)[None]
    shape0 = np.array(extent_shape, np.float32)
    ext = shape0 / shape0.max()
    r = float(np.linalg.norm(ext)) * 1.02
    right, up, fwd = _camera(azim, elev)
    T = int(max(alpha_vol.shape) * 1.6)
    us = np.linspace(-r, r, size, dtype=np.float32)
    vs = np.linspace(r, -r, size, dtype=np.float32)
    ts = np.linspace(-r, r, T, dtype=np.float32)
    U, Vv, Tt = np.meshgrid(us, vs, ts, indexing="ij")
    P = U[..., None] * right + Vv[..., None] * up + Tt[..., None] * fwd
    g = P / ext
    grid = torch.as_tensor(np.stack([g[..., 2], g[..., 1], g[..., 0]], -1), dtype=torch.float32)
    grid = grid.permute(2, 1, 0, 3)[None]
    s = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    a = s[0].clamp(0, 1)
    rgb = s[1:4]
    if shade:
        n = -s[4:7]
        nn_ = (n * n).sum(0, keepdim=True).sqrt().clamp_min(1e-6)
        n = n / nn_
        light = torch.as_tensor(-fwd * 0.75 + up * 0.55 - right * 0.35, dtype=torch.float32)
        light = light / torch.linalg.norm(light)
        lam = (n * light[:, None, None, None]).sum(0).clamp(0, 1)
        has_grad = (nn_[0] > 1e-3).float()
        rgb = rgb * (0.62 + 0.38 * (lam * has_grad + (1 - has_grad)))
    depth = torch.linspace(0, 1, T)[:, None, None]
    rgb = rgb * (1.0 - depth_cue * depth)
    a = 1 - (1 - a).clamp(0, 1) ** (ref_steps / T)
    trans = torch.cumprod(torch.cat([torch.ones_like(a[:1]), 1 - a[:-1]], 0), 0)
    w = trans * a
    col = (w[None] * rgb).sum(1)
    A = w.sum(0).clamp(0, 1)
    out = np.zeros((size, size, 4), np.float32)
    out[..., :3] = (col / A.clamp_min(1e-6)).permute(1, 2, 0).numpy()
    out[..., 3] = A.numpy()
    if wire:
        _draw_box(out, ext, right, up, fwd, r, size)
    return np.clip(out, 0, 1)


def render_dvr(V: np.ndarray, cmap: str = "magma", size: int = 150, azim: float = 35.0, elev: float = 22.0,
               threshold: float = 0.3, opacity: float = 0.9, gamma: float = 1.3, voxel_gap: Optional[bool] = None,
               lo_hi: Optional[Tuple[float, float]] = None, wire: bool = True, shade: bool = True) -> RGBA:
    """Translucent emission–absorption render ("glow") of canonical V[x,y,z]."""
    V = np.asarray(V, np.float32)
    V = robust_norm(V, (1, 99.5)) if lo_hi is None else np.clip((V - lo_hi[0]) / max(lo_hi[1] - lo_hi[0], 1e-6), 0, 1)
    shape0 = V.shape
    V, mask, _ = _blockify(V, gap=voxel_gap)
    alpha = np.clip((V - threshold) / max(1 - threshold, 1e-6), 0, 1) ** gamma * opacity
    if mask is not None:
        alpha = alpha * mask
    rgb = get_cmap(cmap)(V)[..., :3].astype(np.float32)
    return raycast(alpha, rgb, shape0, size, azim, elev, shade, wire)


def cut_octant(shape, loc, azim: float = 35.0, elev: float = 22.0) -> np.ndarray:
    """Boolean mask of the octant (bounded by ``loc``) that faces the camera."""
    d = np.array([math.sin(math.radians(azim)), math.cos(math.radians(azim)), math.sin(math.radians(elev))])
    idx = np.indices(shape)
    m = np.ones(shape, bool)
    for ax in range(3):
        m &= (idx[ax] >= loc[ax]) if d[ax] >= 0 else (idx[ax] <= loc[ax])
    return m


def render_cutaway(V: np.ndarray, cmap: str = "magma", loc: Optional[Tuple[int, int, int]] = None, size: int = 150,
                   azim: float = 35.0, elev: float = 22.0, lo_hi: Optional[Tuple[float, float]] = None,
                   bg_threshold: Optional[float] = None, shell_alpha: float = 1.0, voxel_gap: Optional[bool] = None,
                   wire: bool = True) -> RGBA:
    """Opaque volume with the camera-facing octant removed: the cut faces reveal the interior.

    For anatomy (``bg_threshold`` set) only voxels above background are solid, so
    the head surface is rendered and the cut exposes internal slices.  For feature
    volumes the whole block is solid: its faces and cut planes are textured by
    activation.  Small volumes keep visible voxel blocks.
    """
    V = np.asarray(V, np.float32)
    Vn = robust_norm(V, (1, 99.5)) if lo_hi is None else np.clip((V - lo_hi[0]) / max(lo_hi[1] - lo_hi[0], 1e-6), 0, 1)
    shape0 = Vn.shape
    if loc is None:
        loc = tuple(s // 2 for s in shape0)
    cut = cut_octant(shape0, loc, azim, elev)
    solid = (~cut).astype(np.float32)
    if bg_threshold is not None:
        solid *= (Vn > bg_threshold)
    alpha0 = solid * shell_alpha
    Vb, mask, f = _blockify(Vn, gap=voxel_gap)
    if f > 1:
        alpha0 = np.repeat(np.repeat(np.repeat(alpha0, f, 0), f, 1), f, 2)
    if mask is not None:
        alpha0 = alpha0 * mask
    rgb = get_cmap(cmap)(Vb)[..., :3].astype(np.float32)
    return raycast(alpha0, rgb, shape0, size, azim, elev, shade=True, wire=wire, ref_steps=200.0, depth_cue=0.12)


def _draw_box(img: RGBA, ext, right, up, fwd, r: float, size: int, color=(0.55, 0.57, 0.62)) -> None:
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32) * ext
    uv = np.stack([corners @ right, corners @ up], 1)
    depth = corners @ fwd
    px = (uv[:, 0] + r) / (2 * r) * (size - 1)
    py = (r - uv[:, 1]) / (2 * r) * (size - 1)
    far = int(np.argmax(depth))
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    for i, j in edges:
        hidden = far in (i, j)
        _line(img, px[i], py[i], px[j], py[j], color, 0.25 if hidden else 0.7)


def _line(img, x0, y0, x1, y1, color, alpha):
    n = int(max(abs(x1 - x0), abs(y1 - y0)) * 1.5) + 2
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    H, W = img.shape[:2]
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            src = np.array([*color, alpha], np.float32)
            d = img[yi, xi]
            oa = alpha + d[3] * (1 - alpha)
            d[:3] = (src[:3] * alpha + d[:3] * d[3] * (1 - alpha)) / max(oa, 1e-6)
            d[3] = oa


def render_voxels(V: np.ndarray, cmap: str = "magma", loc: Optional[Tuple[int, int, int]] = None, size: int = 150,
                  azim: float = 35.0, elev: float = 22.0, quantile: float = 82.0, min_level: float = 0.25,
                  haze: float = 0.004, cut: bool = True, wire: bool = True, salience: str = "high") -> RGBA:
    """Feature volume as solid voxels for its strongest activations inside a faint translucent block.

    Voxels above the ``quantile``-th percentile (and ``min_level``) are opaque and
    coloured by activation; the rest of the block is a light haze that shows the
    volume's extent.  Low-resolution volumes keep discrete, separated voxel cubes;
    the camera-facing octant (at the activation peak) is optionally cut away.
    """
    V = np.asarray(V, np.float32)
    Vn = robust_norm(V, (1, 99.5))
    shape0 = Vn.shape
    # salience: departure from the channel's typical (median) level, so a channel that is
    # uniformly "on" over background does not fill the block
    sal = Vn if salience == "high" else robust_norm(np.abs(V - np.median(V)), (1, 99.5))
    if max(shape0) > 16:   # suppress speckle before thresholding high-resolution volumes
        sal = F.avg_pool3d(torch.as_tensor(sal)[None, None], 3, 1, 1, count_include_pad=False)[0, 0].numpy()
    thr = max(min_level, float(np.percentile(sal, quantile)))
    solid = (sal >= thr).astype(np.float32)
    if cut:
        if loc is None:
            loc = activation_peak(sal)
        keep = ~cut_octant(shape0, loc, azim, elev)
        solid *= keep
    Vb, mask, f = _blockify(Vn)
    if f > 1:
        solid = np.repeat(np.repeat(np.repeat(solid, f, 0), f, 1), f, 2)
    alpha = np.where(solid > 0, 1.0, haze).astype(np.float32)
    if mask is not None:
        alpha = alpha * np.where(solid > 0, mask, 1.0)
    rgb = get_cmap(cmap)(0.15 + 0.85 * Vb)[..., :3].astype(np.float32)
    rgb = np.where(solid[..., None] > 0, rgb, np.array([0.72, 0.75, 0.82], np.float32))
    return raycast(alpha, rgb.astype(np.float32), shape0, size, azim, elev, shade=True, wire=wire,
                   ref_steps=200.0, depth_cue=0.12)


def volume_stack(vols: Sequence[np.ndarray], cmap: str, size: int = 150, offset_frac: float = 0.16,
                 style: str = "cutaway", locs: Optional[Sequence] = None, **kw) -> RGBA:
    """Render several channel volumes as an offset deck of voxel blocks (front = first)."""
    n = len(vols)
    off = int(size * offset_frac)
    H = size + off * (n - 1)
    out = np.zeros((H, H, 4), np.float32)
    for i in reversed(range(n)):
        if style == "voxels":
            im = render_voxels(vols[i], cmap=cmap, size=size, loc=locs[i] if locs else None)
        elif style == "cutaway":
            im = render_cutaway(vols[i], cmap=cmap, size=size, loc=locs[i] if locs else None)
        else:
            im = render_dvr(vols[i], cmap=cmap, size=size, **kw)
        if i > 0:
            im = im.copy()
            im[..., 3] *= 0.55 if style != "glow" else 0.85
        alpha_over(out, im, off * (n - 1 - i), off * i)
    return out


def _slice_tiles(sl: Dict[str, np.ndarray], cmap: str, px: int = 96, lo_hi=None, div=False) -> List[RGBA]:
    tiles = []
    for k in ("axial", "coronal", "sagittal"):
        a = sl[k]
        if lo_hi is not None:
            a = np.clip((a - lo_hi[0]) / max(lo_hi[1] - lo_hi[0], 1e-6), 0, 1)
        else:
            a = robust_norm(a, (1, 99.5), div)
        img = colorize(a, cmap)
        tiles.append(_resize_to(img, px, int(round(px * a.shape[1] / max(a.shape[0], 1)))))
    return tiles


def _foreground_threshold(Vn: np.ndarray) -> float:
    """Otsu-like split between background and tissue on a normalised volume."""
    h, e = np.histogram(Vn, 64, (0, 1))
    c = (e[:-1] + e[1:]) / 2
    best, thr = -1, 0.1
    for i in range(1, 63):
        w0, w1 = h[:i].sum(), h[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (h[:i] * c[:i]).sum() / w0, (h[i:] * c[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, c[i]
    return float(min(thr, 0.35))


def render_volume(summ: TensorSummary, cfg: FlowConfig, mode: Optional[str] = None, axes: str = "xyz",
                  strategy: Optional[str] = None, style: str = "technical", size: int = 150,
                  n_vol: Optional[int] = None) -> Visual:
    mode = mode or cfg.volume_mode
    is_input = summ.role == "input" and summ.image is not None
    if mode == "auto":
        mode = "anatomy" if is_input else "volume"
    cmap = act_cmap(cfg)
    vstyle = getattr(cfg, "volume_style", "voxels")
    if is_input:
        V = canonical_volume(summ.image[0], axes)
        loc = center_of_mass(V)
        sl = ortho_slices(V, loc)
        lo_hi = tuple(np.percentile(V, (1, 99.5)))
        tiles = _slice_tiles(sl, "gray", 88, lo_hi)
        if mode in ("anatomy", "volume"):
            Vs = _limit(V, 96)
            Vn = np.clip((Vs - lo_hi[0]) / max(lo_hi[1] - lo_hi[0], 1e-6), 0, 1)
            sc = np.array(Vs.shape) / np.array(V.shape)
            l2 = tuple(int(round(l * s)) for l, s in zip(loc, sc))
            if vstyle in ("cutaway", "voxels"):
                cube = render_cutaway(Vn, cmap="gray", loc=l2, size=int(size * 1.33), lo_hi=(0, 1),
                                      bg_threshold=_foreground_threshold(Vn))
            else:
                cube = render_dvr(Vn, cmap="bone", size=200, threshold=0.18, opacity=0.35, gamma=1.0, lo_hi=(0, 1))
            return Visual(cube, "volume", n_shown=1, extras={"ortho": mosaic(tiles, cols=3, gap=4)},
                          meta={"loc": loc, "mode": "anatomy"})
        return _volume_mode_2d(summ, cfg, mode, V[None], axes, "gray", loc)
    g = grid_for_channels(summ.channels, cfg.max_channels)
    if n_vol is None:
        n_vol = 1 if summ.channels <= 2 else min(4, max(2, g))
    forced = summ.selections.get("_forced")
    if forced:
        ids = [i for i in forced if i in summ.channel_ids][: (n_vol if mode == "volume" else g * g)]
        maps = summ.maps[[summ.channel_ids.index(i) for i in ids]]
    else:
        maps, ids = summ.channel_maps(strategy or cfg.channel_strategy, n_vol if mode == "volume" else g * g)
    if len(maps) == 0:
        return render_generic(summ, cfg)
    vols = [canonical_volume(m, axes) for m in maps]
    E = canonical_volume(summ.energy_map, axes) if summ.energy_map is not None else np.mean(np.abs(vols), 0)
    loc = activation_peak(E)
    if mode == "volume":
        locs = [activation_peak(robust_norm(v, (1, 99.5))) for v in vols]
        loc = locs[0]
        E = vols[0]
        if vstyle in ("cutaway", "voxels"):
            stack = volume_stack(vols, cmap=cmap, size=size, style=vstyle, locs=locs)
        else:
            stack = volume_stack(vols, cmap=cmap, size=150, style="glow", threshold=0.35, opacity=0.95, gamma=1.4)
        sl = ortho_slices(E, loc)
        tiles = _slice_tiles(sl, cmap, 72)
        return Visual(stack, "volume", n_shown=len(vols), extras={"ortho": mosaic(tiles, cols=3, gap=3)},
                      meta={"channels": ids, "loc": loc, "mode": "volume"})
    return _volume_mode_2d(summ, cfg, mode, np.stack(vols), axes, cmap, loc, ids)


def _limit(V: np.ndarray, n: int) -> np.ndarray:
    if max(V.shape) <= n:
        return V
    size = tuple(max(1, int(round(s * n / max(V.shape)))) for s in V.shape)
    return F.adaptive_avg_pool3d(torch.as_tensor(V, dtype=torch.float32)[None, None], size)[0, 0].numpy()


def _volume_mode_2d(summ, cfg, mode, vols, axes, cmap, loc, ids=None) -> Visual:
    div = _use_diverging(summ, cfg)
    cm = "RdBu_r" if div else cmap
    rows = []
    for V in vols:
        if mode == "ortho":
            l = activation_peak(np.abs(V)) if summ.role != "input" else loc
            rows.append(_slice_tiles(ortho_slices(V, l), cm, 80, div=div))
        elif mode == "projection":
            op = {"max": np.max, "mean": np.mean, "meanabs": lambda a, axis: np.mean(np.abs(a), axis=axis)}[cfg.projection]
            pj = {"axial": np.flipud(op(V, axis=2).T), "coronal": np.flipud(op(V, axis=1).T),
                  "sagittal": np.flipud(op(V, axis=0).T)}
            rows.append(_slice_tiles(pj, cm, 80, div=div))
        else:  # montage: activation-driven slices along z
            prof = np.abs(V).reshape(-1, V.shape[2]).sum(0) if V.ndim == 3 else None
            k = 4
            zs = sorted(np.argsort(-prof)[: max(k * 2, 1)][:k]) if prof is not None else []
            rows.append([upsample(colorize(robust_norm(np.flipud(V[:, :, z].T), (1, 99.5), div), cm), 72) for z in zs])
    tiles = [t for r in rows for t in r]
    cols = len(rows[0]) if rows else 1
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    tiles = [_resize_to(t, th, tw) for t in tiles]
    return Visual(mosaic(tiles, cols=cols, gap=3), "ortho", n_shown=len(vols), meta={"mode": mode, "loc": loc, "channels": ids})


# ---------------------------------------------------------------------------
# segmentation overlays
# ---------------------------------------------------------------------------

LABEL_COLORS = np.array([
    [0, 0, 0, 0], [0.95, 0.30, 0.25, 1], [0.20, 0.65, 0.95, 1], [0.35, 0.85, 0.40, 1], [0.98, 0.78, 0.20, 1],
    [0.70, 0.40, 0.90, 1], [0.10, 0.85, 0.85, 1], [0.95, 0.50, 0.75, 1], [0.60, 0.60, 0.60, 1],
], np.float32)


def seg_rgba(mask: np.ndarray) -> RGBA:
    if mask.dtype.kind == "f":
        img = colorize(np.clip(mask, 0, 1), "magma")
        img[..., 3] = np.clip(mask * 1.2, 0, 1) * 0.85
        return img
    m = mask.astype(int)
    lab = np.where(m > 0, (m - 1) % (len(LABEL_COLORS) - 1) + 1, 0)
    img = LABEL_COLORS[lab].copy()
    img[..., 3] *= 0.6
    return img


def render_segmentation(mask: np.ndarray, base: Optional[np.ndarray], cfg: FlowConfig, axes: str = "xyz") -> Visual:
    """mask: [H,W] or [X,Y,Z] (labels or probabilities); base: input image [C,H,W] / [C,X,Y,Z]."""
    if mask.ndim == 2:
        H, W = mask.shape
        if base is not None and cfg.overlay_segmentation:
            b = base[0] if base.shape[0] != 3 else base.mean(0)
            b = _resize_to(b, H, W) if b.shape != (H, W) else b
            under = colorize(robust_norm(b, (0.5, 99.5)), "gray")
            under[..., :3] *= 0.85
        else:
            under = np.zeros((H, W, 4), np.float32)
            under[..., 3] = 1
            under[..., :3] = 0.08
        over = seg_rgba(mask)
        out = under.copy()
        alpha_over(out, over, 0, 0)
        return Visual(upsample(out, 160), "seg")
    # 3-D segmentation: "glass" anatomy + opaque coloured labels
    M = canonical_volume(mask, axes)
    labels = M if M.dtype.kind != "f" else (M > 0.5).astype(np.int32)
    B = None
    if base is not None:
        B = canonical_volume(base[0], axes)
        if B.shape != M.shape:
            B = F.interpolate(torch.as_tensor(B, dtype=torch.float32)[None, None], size=M.shape, mode="trilinear",
                              align_corners=False)[0, 0].numpy()
    fg = (labels > 0).astype(np.float32)
    lab_rgb = LABEL_COLORS[np.where(labels > 0, (labels - 1) % (len(LABEL_COLORS) - 1) + 1, 0)][..., :3]
    if B is not None:
        Bn = robust_norm(B, (1, 99.5))
        tissue = (Bn > _foreground_threshold(Bn)).astype(np.float32)
        alpha = np.maximum(tissue * 0.05 * (0.4 + Bn), fg * 0.9)
        rgb = np.where(fg[..., None] > 0, lab_rgb, np.repeat(Bn[..., None], 3, -1) * 0.8 + 0.1)
    else:
        alpha = fg * 0.9
        rgb = lab_rgb
    cube = raycast(alpha.astype(np.float32), rgb.astype(np.float32), fg.shape, 200, shade=True, ref_steps=64.0)
    loc = center_of_mass(fg) if fg.sum() > 0 else tuple(s // 2 for s in fg.shape)
    tiles = []
    base_sl = ortho_slices(B if B is not None else np.zeros_like(fg), loc)
    lab_sl = ortho_slices(labels, loc)
    for k in ("axial", "coronal", "sagittal"):
        u = colorize(robust_norm(base_sl[k], (1, 99.5)), "gray")
        alpha_over(u, seg_rgba(lab_sl[k]), 0, 0)
        tiles.append(_resize_to(u, 80, int(80 * u.shape[1] / u.shape[0])))
    return Visual(cube, "volume", extras={"ortho": mosaic(tiles, cols=3, gap=3)}, meta={"loc": loc})


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


def visualize_tensor(summ: TensorSummary, cfg: FlowConfig, strategy: Optional[str] = None,
                     volume_axes: str = "xyz", style: str = "technical") -> Visual:
    """Choose a visual representation from tensor kind (the core dispatcher)."""
    try:
        if summ.kind == "image2d":
            if summ.role == "input" and summ.image is not None:
                return render_input_image(summ, cfg)
            if summ.spatial and max(summ.spatial) == 1:
                vec = summ.maps[:, 0, 0] if summ.maps is not None else np.zeros(1)
                return render_vector(summ.channel_scores.get("mean", vec), cfg)
            return render_image2d(summ, cfg, strategy=strategy, style=style)
        if summ.kind == "volume3d":
            if summ.spatial and max(summ.spatial) == 1:
                return render_vector(summ.channel_scores.get("mean", np.zeros(1)), cfg)
            return render_volume(summ, cfg, axes=volume_axes, strategy=strategy, style=style)
        if summ.kind == "tokens":
            return render_tokens(summ, cfg, strategy=strategy)
        if summ.kind == "seq1d":
            return render_seq1d(summ, cfg)
        if summ.kind in ("vector", "scalar"):
            return render_vector(summ.vector if summ.vector is not None else np.zeros(1), cfg)
        if summ.kind == "attention" and summ.attention is not None:
            a = render_attention(summ.attention, cfg)
            return Visual(a["matrix"], "heatmap", meta={"axes": ("query", "key")})
    except Exception as e:  # unknown / odd tensors must never break the figure
        vis = render_generic(summ, cfg)
        vis.meta["error"] = str(e)
        return vis
    return render_generic(summ, cfg)
