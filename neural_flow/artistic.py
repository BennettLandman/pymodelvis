"""Artistic raster primitives for the cinematic style.

* :func:`render_deck` – a "book" of feature maps: each channel is a page viewed
  at an angle; pages recede up-and-left, the strongest channel is the front page.
  Returns page geometry so beams can be anchored to exact map coordinates.
* :func:`render_dots` – a latent vector as a matrix of glowing dots ("latent
  pixels"), with dot-centre geometry for contribution lines.
* :func:`glow` – bloom halo for dark backgrounds.
* :func:`vignette` – "what one unit sees": the input cropped to a unit's
  empirical receptive field.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .raster import alpha_over, colorize, get_cmap, robust_norm

_FONTS_REGISTERED = False


def register_fonts() -> str:
    """Register the bundled Inter font (OFL) with matplotlib; returns the family name."""
    global _FONTS_REGISTERED
    from matplotlib import font_manager

    if not _FONTS_REGISTERED:
        d = os.path.join(os.path.dirname(__file__), "fonts")
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".ttf"):
                try:
                    font_manager.fontManager.addfont(os.path.join(d, f))
                except Exception:
                    pass
        _FONTS_REGISTERED = True
    return "Inter"


# ---------------------------------------------------------------------------
# decks of feature maps
# ---------------------------------------------------------------------------


@dataclass
class PageGeom:
    ox: float
    oy: float
    S: float          # page width before foreshortening (px)
    Sh: float         # page height (px)
    a: float          # horizontal foreshortening
    c: float          # vertical shear (left edge higher)

    def to_px(self, u: float, v: float) -> Tuple[float, float]:
        """Normalised map coords (u: 0 left → 1 right, v: 0 top → 1 bottom) → deck image px."""
        X = self.ox + self.a * u * self.S
        Y = self.oy + v * self.Sh + self.c * u * self.S
        return X, Y

    def quad(self, u0, v0, u1, v1) -> List[Tuple[float, float]]:
        return [self.to_px(u0, v0), self.to_px(u1, v0), self.to_px(u1, v1), self.to_px(u0, v1)]


@dataclass
class Deck:
    image: np.ndarray                 # RGBA float [H, W, 4]
    pages: List[PageGeom]             # index 0 = front page (strongest channel)
    channels: List[int] = field(default_factory=list)

    @property
    def front(self) -> PageGeom:
        return self.pages[0]

    @property
    def back(self) -> PageGeom:
        return self.pages[-1]


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGBA")


def _from_pil(im: Image.Image) -> np.ndarray:
    return np.asarray(im, dtype=np.float32) / 255.0


def page_image(rgba: np.ndarray, S: int, Sh: int, a: float, c: float, border=(0.55, 0.57, 0.62, 0.9),
               shade: float = 1.0) -> np.ndarray:
    """Resize a map (nearest, crisp pixels) and warp it into a foreshortened, sheared page."""
    im = _to_pil(rgba).resize((S, Sh), Image.NEAREST)
    arr = _from_pil(im)
    if shade != 1.0:
        arr[..., :3] *= shade
    if border is not None:
        arr[:1, :] = border
        arr[-1:, :] = border
        arr[:, :1] = border
        arr[:, -1:] = border
    im = _to_pil(arr)
    W = int(math.ceil(a * S)) + 1
    H = int(math.ceil(Sh + c * S)) + 1
    # output (X, Y) -> input (u, v):  X = a u ;  Y = v + c u
    im = im.transform((W, H), Image.AFFINE, (1 / a, 0, 0, -c / a, 1, 0), resample=Image.BILINEAR)
    return _from_pil(im)


def render_deck(maps: Sequence[np.ndarray], cmap: str, S: int = 200, a: float = 0.66, c: float = 0.2,
                spacing: Optional[float] = None, ranges: Optional[Sequence[Tuple[float, float]]] = None,
                pct=(1.0, 99.5), border=(0.55, 0.57, 0.62, 0.9), rgb_first: Optional[np.ndarray] = None,
                max_width_factor: float = 1.9, dim: float = 0.6) -> Deck:
    """Compose channel maps into a receding deck of pages (front = maps[0])."""
    n = max(1, len(maps))
    h0, w0 = (rgb_first.shape[:2] if rgb_first is not None else maps[0].shape[:2])
    Sh = int(round(S * h0 / max(w0, 1)))
    page_w = a * S
    if spacing is None:
        spacing = page_w * 0.38
    if n > 1:
        spacing = min(spacing, (max_width_factor * S - page_w) / (n - 1))
    spacing = max(spacing, 3.0)
    dy = spacing * 0.42
    Wt = int(math.ceil(page_w + spacing * (n - 1))) + 2
    Ht = int(math.ceil(Sh + c * S + dy * (n - 1))) + 2
    canvas = np.zeros((Ht, Wt, 4), np.float32)
    pages: List[Optional[PageGeom]] = [None] * n
    # back pages sit up-left, front page down-right
    for i in reversed(range(n)):
        ox = (n - 1 - i) * spacing
        oy = (n - 1 - i) * dy
        if i == 0 and rgb_first is not None:
            rgba = rgb_first
        else:
            m = maps[i]
            lo_hi = ranges[i] if ranges is not None else None
            if lo_hi is not None:
                v = np.clip((m - lo_hi[0]) / max(lo_hi[1] - lo_hi[0], 1e-9), 0, 1)
            else:
                v = robust_norm(m, pct)
            rgba = colorize(v, cmap)
        shade = 1.0 - dim * (i / max(n - 1, 1)) if n > 1 else 1.0
        pg = page_image(rgba, S, Sh, a, c, border, shade)
        alpha_over(canvas, pg, int(round(oy)), int(round(ox)))
        pages[i] = PageGeom(ox, oy, S, Sh, a, c)
    return Deck(canvas, pages)


# ---------------------------------------------------------------------------
# latent vectors as dot matrices
# ---------------------------------------------------------------------------


@dataclass
class Dots:
    image: np.ndarray
    centers: np.ndarray        # [N, 2] (x, y) px of each element (after binning)
    bin_of: np.ndarray         # original index -> dot index
    cell: float


def render_dots(vec: np.ndarray, cmap: str, cell: int = 9, cols: Optional[int] = None, max_dots: int = 1024,
                pct=(1, 99.5), lo_hi: Optional[Tuple[float, float]] = None, highlight: Optional[int] = None,
                highlight_color=(1.0, 0.78, 0.2, 1.0), floor: float = 0.28) -> Dots:
    v = np.asarray(vec, np.float32).reshape(-1)
    n0 = v.size
    if n0 > max_dots:
        f = int(math.ceil(n0 / max_dots))
        pad = (-n0) % f
        vv = np.concatenate([v, np.full(pad, np.nan)]).reshape(-1, f)
        v = np.nanmax(np.abs(vv), 1) * np.sign(np.nansum(vv, 1))
        bin_of = np.arange(n0) // f
    else:
        bin_of = np.arange(n0)
    n = v.size
    if cols is None:
        cols = max(1, int(round(math.sqrt(n / 3.2))))
    rows = int(math.ceil(n / cols))
    if lo_hi is not None:
        a01 = np.clip((v - lo_hi[0]) / max(lo_hi[1] - lo_hi[0], 1e-9), 0, 1)
    else:
        a01 = robust_norm(v, pct)
    cm = get_cmap(cmap)
    W, H = cols * cell, rows * cell
    ss = 3  # supersample for smooth circles
    im = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    centers = np.zeros((n, 2), np.float32)
    for i in range(n):
        r_, c_ = i % rows, i // rows          # column-major: fills top→bottom, then next column
        cx, cy = (c_ + 0.5) * cell, (r_ + 0.5) * cell
        centers[i] = (cx, cy)
        val = float(a01[i])
        col = cm(val)
        rad = cell * (0.22 + 0.2 * val)
        alpha = floor + (1 - floor) * val
        fill = tuple(int(255 * x) for x in col[:3]) + (int(255 * alpha),)
        dr.ellipse([(cx - rad) * ss, (cy - rad) * ss, (cx + rad) * ss, (cy + rad) * ss], fill=fill)
    if highlight is not None:
        i = int(bin_of[highlight]) if highlight < len(bin_of) else None
        if i is not None:
            cx, cy = centers[i]
            rad = cell * 0.62
            dr.ellipse([(cx - rad) * ss, (cy - rad) * ss, (cx + rad) * ss, (cy + rad) * ss],
                       outline=tuple(int(255 * x) for x in highlight_color), width=max(2, ss))
    im = im.resize((W, H), Image.LANCZOS)
    return Dots(_from_pil(im), centers, bin_of, cell)


# ---------------------------------------------------------------------------
# glow & vignettes
# ---------------------------------------------------------------------------


def glow(img: np.ndarray, radius: float = 10.0, strength: float = 0.9, pad: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Bloom halo of an RGBA image; returns (halo, pad) where halo is larger by ``pad`` px per side."""
    pad = int(pad if pad is not None else radius * 2.5)
    H, W = img.shape[:2]
    big = np.zeros((H + 2 * pad, W + 2 * pad, 4), np.float32)
    big[pad:pad + H, pad:pad + W] = img
    pm = big.copy()
    pm[..., :3] *= pm[..., 3:4]
    im = Image.fromarray((np.clip(pm, 0, 1) * 255).astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(radius))
    b = np.asarray(im, dtype=np.float32) / 255.0
    a = np.clip(b[..., 3:4] * strength, 0, 1)
    rgb = b[..., :3] / np.maximum(b[..., 3:4], 1e-4)
    out = np.concatenate([np.clip(rgb * 1.15, 0, 1), a * 0.8], -1)
    return out.astype(np.float32), pad


def vignette(inp_rgb: np.ndarray, erf: np.ndarray, rf_box, size: int = 120, min_side: int = 16,
             ring=(0.4, 0.9, 1.0, 1.0)) -> np.ndarray:
    """Crop of the input around a unit's receptive field, brightness weighted by |∂unit/∂input|."""
    H, W = inp_rgb.shape[:2]
    eh, ew = erf.shape[:2]
    if (eh, ew) != (H, W):
        e = np.asarray(Image.fromarray((erf * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32) / 255
        sy, sx = H / eh, W / ew
    else:
        e = erf
        sy = sx = 1.0
    y0, x0, y1, x1 = rf_box if rf_box is not None else (0, 0, eh, ew)
    y0, y1, x0, x1 = y0 * sy, y1 * sy, x0 * sx, x1 * sx
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    side = max(y1 - y0, x1 - x0, min_side) * 1.25
    side = min(side, max(H, W))
    t, l = int(round(cy - side / 2)), int(round(cx - side / 2))
    t = int(np.clip(t, 0, max(H - side, 0)))
    l = int(np.clip(l, 0, max(W - side, 0)))
    s = int(round(side))
    crop = inp_rgb[t:t + s, l:l + s, :3]
    ecrop = e[t:t + s, l:l + s]
    if crop.size == 0:
        crop, ecrop = inp_rgb[..., :3], e
    w = 0.35 + 0.65 * np.clip(ecrop / (ecrop.max() + 1e-9), 0, 1) ** 0.5
    rgb = crop * w[..., None]
    img = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], -1)
    # tiny receptive fields are shown as crisp pixels (they really are just a few pixels)
    resample = Image.NEAREST if s * 4 <= size else (Image.BICUBIC if s < size else Image.LANCZOS)
    im = _to_pil(img).resize((size, size), resample)
    # circular mask with a thin ring
    ss = 3
    mask = Image.new("L", (size * ss, size * ss), 0)
    ImageDraw.Draw(mask).ellipse([2 * ss, 2 * ss, (size - 2) * ss, (size - 2) * ss], fill=255)
    mask = mask.resize((size, size), Image.LANCZOS)
    im.putalpha(mask)
    ring_im = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    ImageDraw.Draw(ring_im).ellipse([2 * ss, 2 * ss, (size - 2) * ss, (size - 2) * ss],
                                    outline=tuple(int(255 * x) for x in ring), width=2 * ss)
    ring_im = ring_im.resize((size, size), Image.LANCZOS)
    out = _from_pil(im)
    alpha_over(out, _from_pil(ring_im), 0, 0)
    return out


def heat_overlay(inp_rgb: np.ndarray, heat: np.ndarray, cmap: str = "inferno", alpha: float = 0.65) -> np.ndarray:
    H, W = inp_rgb.shape[:2]
    if heat.shape != (H, W):
        heat = np.asarray(Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                          np.float32) / 255
    base = inp_rgb[..., :3].mean(-1, keepdims=True).repeat(3, -1) * 0.55
    hc = get_cmap(cmap)(heat)[..., :3]
    a = (np.clip(heat, 0, 1) ** 0.8 * alpha)[..., None]
    rgb = base * (1 - a) + hc * a
    return np.concatenate([rgb, np.ones((H, W, 1))], -1).astype(np.float32)
