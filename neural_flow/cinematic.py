"""Cinematic renderer (``style="cinematic"``): presentation-grade representation flow.

Visual language
---------------
* **Books of feature maps** – each spatial stage is a deck of its most active
  channels seen at an angle.  Pages shrink as resolution drops and multiply as
  channel depth grows; the front page is the channel the other elements refer to.
* **Light beams** – from the region of the previous stage that feeds the stage's
  strongest unit (from gradients) to that unit.  A narrow beam is a local
  convolution window, a wide one is pooling / global context, and scattered rays
  are attention-like mixing.
* **What one unit sees** – below each stage, the input cropped to that unit's
  empirical receptive field (|∂unit/∂input|); the circles grow with depth.
* **Latent pixels** – vectors are dot matrices; for linear heads, lines show the
  largest weight × activation contributions to the predicted output
  (amber = pushes up, blue = pushes down).
* **Evidence** – Grad-CAM of the predicted class under the prediction card.

Works on ``theme="black"`` (default for this style), ``"dark"`` and ``"light"``.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path

from .artistic import Deck, Dots, glow, heat_overlay, register_fonts, render_deck, render_dots, vignette
from .layout import compute_layout, panel_scale
from .raster import (Visual, act_cmap, canonical_volume, colorize, pca_rgb, render_segmentation, render_volume,
                     robust_norm, upsample, visualize_tensor)
from .render import FlowFigure

CINE_THEMES = {
    "black": dict(bg="#000000", text="#f4f4f5", muted="#a1a1aa", faint="#63636b", edge="#52525b",
                  beam="#67e8f9", beam_fill="#22d3ee", accent="#fbbf24", pos="#fbbf24", neg="#60a5fa",
                  card="#0b0b0e", card_edge="#3f3f46", ring=(0.45, 0.9, 1.0, 1.0), cmap="inferno",
                  page_border=(0.42, 0.42, 0.48, 0.9), bar_bg="#27272a", skip="#a78bfa"),
    "dark": dict(bg="#0e1016", text="#e8eaef", muted="#a3aab6", faint="#6b7280", edge="#6b7280",
                 beam="#67e8f9", beam_fill="#22d3ee", accent="#fb923c", pos="#fbbf24", neg="#60a5fa",
                 card="#151821", card_edge="#2a2f3a", ring=(0.45, 0.9, 1.0, 1.0), cmap="inferno",
                 page_border=(0.45, 0.47, 0.55, 0.9), bar_bg="#2a2f3a", skip="#a78bfa"),
    "light": dict(bg="#ffffff", text="#111114", muted="#5b5f6a", faint="#9aa1ad", edge="#9ca3af",
                  beam="#0284c7", beam_fill="#0ea5e9", accent="#c2410c", pos="#d97706", neg="#2563eb",
                  card="#f7f7f8", card_edge="#e4e4e7", ring=(0.05, 0.45, 0.75, 1.0), cmap="magma",
                  page_border=(0.35, 0.35, 0.4, 0.9), bar_bg="#e5e7eb", skip="#7c3aed"),
}

P = 150.0          # raster pixels per layout unit
BAND = 1.55        # caption + "what one unit sees" band below every stage (units)
VIG = 0.78         # vignette diameter (units)


def spaced(s: str) -> str:
    return " ".join(s.upper())


@dataclass
class Node:
    key: str
    kind: str                         # page | deck | dots | volume | image | card
    image: Optional[np.ndarray] = None
    deck: Optional[Deck] = None
    dots: Optional[Dots] = None
    w: float = 1.0                    # raster width (units)
    h: float = 1.0                    # raster height (units)
    extent: Tuple[float, float, float, float] = (0, 0, 0, 0)
    caption: str = ""
    sub: str = ""
    title: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameState:
    """Per-frame overrides used by movies (fixed channels & normalisation, timeline)."""
    ranges: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    vec_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    pca_ranges: Dict[str, Any] = field(default_factory=dict)
    footer_height: float = 0.0
    footer_fill: bool = True             # grow the footer (timeline) into any spare vertical space
    draw_footer: Optional[Callable] = None
    title: Optional[str] = None
    subtitle: Optional[str] = None


# ---------------------------------------------------------------------------


def n_sheets(C: int) -> int:
    return int(np.clip(round(1.7 * math.log2(max(C, 2))), 3, 14))


def _pca_page(pmaps: np.ndarray, ranges=None) -> np.ndarray:
    """PCA of all channels as an RGB page (first three components → R, G, B)."""
    chans = []
    for i in range(3):
        m = pmaps[i]
        if ranges is not None:
            lo, hi = ranges[i]
            chans.append(np.clip((m - lo) / max(hi - lo, 1e-9), 0, 1))
        else:
            chans.append(robust_norm(m, (1, 99)))
    rgb = np.stack(chans, -1) ** 0.9
    return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], -1).astype(np.float32)


def _input_rgb(res) -> Optional[np.ndarray]:
    for st in res.graph.ordered():
        s = st.summary
        if st.kind == "input" and s is not None and s.image is not None and s.kind == "image2d":
            im = s.image
            if im.shape[0] == 3:
                lo, hi = np.percentile(im, (0.5, 99.5))
                rgb = np.clip((np.transpose(im, (1, 2, 0)) - lo) / max(hi - lo, 1e-6), 0, 1)
            else:
                g = robust_norm(im[0], (0.5, 99.5))
                rgb = np.repeat(g[..., None], 3, -1)
            return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], -1).astype(np.float32)
    return None


def _input_volume(res) -> Optional[np.ndarray]:
    for st in res.graph.ordered():
        s = st.summary
        if st.kind == "input" and s is not None and s.image is not None and s.kind == "volume3d":
            return canonical_volume(s.image[0], res.config.volume_axes)
    return None


def _strategy_channels(summ, strategy: str, n: int) -> List[int]:
    ids = summ.selections.get("_forced") or summ.selections.get(strategy) or summ.selections.get("energy") or []
    return [i for i in ids if i in summ.channel_ids][:n]


def build_nodes(res, frame: Optional[FrameState], th: dict) -> Dict[str, Node]:
    cfg = res.config
    ex = getattr(res, "explanation", None)
    cmap = cfg.cmap or th["cmap"]
    spatial = [s.summary.resolution for s in res.graph.stages.values() if s.summary is not None and s.summary.is_spatial]
    max_res = max(spatial) if spatial else 1.0
    nodes: Dict[str, Node] = {}
    heads_pred = {}
    for okey, ov in res.views.items():
        for s in res.out_nodes.get(okey, []):
            if ov.kind == "classification" and ov.vector is not None:
                heads_pred[s] = int(np.argmax(ov.vector))
    for st in res.graph.ordered():
        sm = st.summary
        if sm is None:
            continue
        nd = Node(st.key, "image")
        if st.kind == "input" and sm.kind == "image2d" and sm.image is not None:
            rgb = _input_rgb(res)
            S = 270
            nd.deck = render_deck([rgb[..., 0]], cmap, S=S, rgb_first=rgb, border=th["page_border"])
            nd.kind, nd.image = "page", nd.deck.image
            nd.caption = " × ".join(map(str, sm.shape[1:]))
        elif sm.kind in ("image2d", "tokens") and sm.maps is not None and sm.is_spatial:
            n = n_sheets(sm.channels)
            ids = _strategy_channels(sm, cfg.channel_strategy, n)
            maps = [sm.maps[sm.channel_ids.index(i)] for i in ids]
            rng = None
            if frame is not None and st.key in frame.ranges:
                rng = frame.ranges[st.key][:len(maps)]
            S = int(round(250 * panel_scale(sm.resolution, max_res)))
            rgb_first = None
            if cfg.front_page == "pca" and sm.pca_maps is not None and sm.pca_maps.shape[0] >= 3 and len(maps) > 1:
                pr = frame.pca_ranges.get(st.key) if frame is not None else None
                rgb_first = _pca_page(sm.pca_maps, pr)
                maps = [maps[0]] + maps
                if rng is not None:
                    rng = [rng[0]] + list(rng)
            nd.deck = render_deck(maps, cmap, S=S, ranges=rng, border=th["page_border"], rgb_first=rgb_first,
                                  max_width_factor=1.6 if len(maps) > 1 else 1.0)
            nd.deck.channels = ids
            nd.extras["pca_front"] = rgb_first is not None
            nd.kind, nd.image = "deck", nd.deck.image
            nd.caption = sm.shape_label()
            nd.sub = f"{len(ids)} of {sm.channels} maps" + ("  ·  front: PCA of all maps" if rgb_first is not None else "")
            if sm.kind == "tokens" and sm.cls is not None:
                nd.sub += f"  ·  +{sm.n_special} CLS"
        elif sm.kind == "volume3d" or (st.kind == "input" and sm.kind == "volume3d"):
            if sm.spatial and max(sm.spatial) > 1:
                n_vol = 1 if sm.channels <= 2 else int(np.clip(round(math.log2(sm.channels)) - 1, 2, 5))
                vis = render_volume(sm, cfg.updated(cmap=cmap), axes=cfg.volume_axes, size=210, n_vol=n_vol)
            else:
                vis = visualize_tensor(sm, cfg, volume_axes=cfg.volume_axes)
            nd.kind, nd.image = "volume", vis.image
            nd.extras["ortho"] = vis.extras.get("ortho")
            nd.caption = sm.shape_label()
            nd.sub = "volume" if st.kind == "input" else f"{vis.n_shown} of {sm.channels} channels"
        else:
            vec = None
            if sm.vector is not None:
                vec = sm.vector
            elif sm.spatial and max(sm.spatial) <= 1 and "mean" in sm.channel_scores:
                vec = sm.channel_scores["mean"]
            if vec is not None:
                lo_hi = frame.vec_ranges.get(st.key) if frame is not None else None
                hl = heads_pred.get(st.key)
                cell = int(np.clip(round(95 / math.sqrt(max(min(vec.size, 512), 1))), 9, 30))
                nd.dots = render_dots(vec, cmap, cell=cell, max_dots=512, lo_hi=lo_hi, highlight=hl,
                                      highlight_color=tuple(int(th["accent"][i:i + 2], 16) / 255 for i in (1, 3, 5)) + (1.0,))
                nd.kind, nd.image = "dots", nd.dots.image
                nd.caption = f"{vec.size}-d" if st.kind != "representation" else f"{vec.size}-d latent"
                nd.sub = "vector"
            else:
                vis = visualize_tensor(sm, cfg, volume_axes=cfg.volume_axes)
                nd.kind, nd.image = "image", vis.image
                nd.caption = sm.shape_label()
        nd.w = nd.image.shape[1] / P
        nd.h = nd.image.shape[0] / P
        nd.title = st.concept or st.label
        nodes[st.key] = nd
    return nodes


# ---------------------------------------------------------------------------


def _px_to_data(nd: Node, px: float, py: float) -> Tuple[float, float]:
    x0, x1, y0, y1 = nd.extent
    return x0 + px / P, y1 - py / P


def _frustum(ax, src_quad, dst_quad, th, z=4.2, alpha=0.16):
    """Four translucent side faces connecting two quads + edge rays."""
    arts = []
    for i in range(4):
        j = (i + 1) % 4
        poly = Polygon([src_quad[i], src_quad[j], dst_quad[j], dst_quad[i]], closed=True,
                       facecolor=th["beam_fill"], edgecolor="none", alpha=alpha, zorder=z)
        ax.add_patch(poly)
        arts.append(poly)
    for i in range(4):
        ln, = ax.plot([src_quad[i][0], dst_quad[i][0]], [src_quad[i][1], dst_quad[i][1]], color=th["beam"],
                      lw=0.6, alpha=0.55, zorder=z + 0.1, solid_capstyle="round")
        arts.append(ln)
    for q in (src_quad, dst_quad):
        pl = Polygon(q, closed=True, fill=False, edgecolor=th["beam"], lw=1.1, alpha=0.95, zorder=z + 0.2)
        ax.add_patch(pl)
        arts.append(pl)
    return arts


def _local_peaks(m: np.ndarray, k: int = 6) -> List[Tuple[int, int]]:
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(m)[None, None]
    mx = F.max_pool2d(t, 3, 1, 1)[0, 0].numpy()
    cand = np.argwhere((m == mx) & (m > 0.2 * m.max()))
    vals = m[cand[:, 0], cand[:, 1]] if len(cand) else []
    order = np.argsort(-np.asarray(vals))[:k]
    return [tuple(cand[i]) for i in order]


def _draw_beam(ax, g, nodes, uf, th) -> List:
    arts: List = []
    src = nodes.get(uf.dep_stage)
    dst = nodes.get(uf.stage)
    if src is None or dst is None or src.deck is None or dst.deck is None or uf.dep_map is None:
        return arts
    if uf.dep_map.ndim != 2 or len(uf.grid) != 2:
        return arts
    gh, gw = uf.grid
    py, px = uf.position
    back = dst.deck.back
    uq = back.quad(px / gw, py / gh, (px + 1) / gw, (py + 1) / gh)
    dst_quad = [_px_to_data(dst, *p) for p in uq]
    m = uf.dep_map
    mh, mw = m.shape
    front = src.deck.front
    idx = np.argwhere(m >= 0.15 * m.max())
    if len(idx) == 0:
        return arts
    (y0, x0), (y1, x1) = idx.min(0), idx.max(0) + 1
    frac = (y1 - y0) * (x1 - x0) / (mh * mw)
    if frac <= 0.45:
        sq = front.quad(x0 / mw, y0 / mh, x1 / mw, y1 / mh)
        src_quad = [_px_to_data(src, *p) for p in sq]
        arts += _frustum(ax, src_quad, dst_quad, th, alpha=0.13 if frac < 0.05 else 0.09)
    else:
        # diffuse dependency (pooling / attention): rays from the strongest source positions
        cx = np.mean([p[0] for p in dst_quad])
        cy = np.mean([p[1] for p in dst_quad])
        peaks = _local_peaks(m, 7)
        for (yy, xx) in peaks:
            w = float(m[yy, xx] / m.max())
            sx, sy = _px_to_data(src, *front.to_px((xx + 0.5) / mw, (yy + 0.5) / mh))
            ln, = ax.plot([sx, cx], [sy, cy], color=th["beam"], lw=0.5 + 1.2 * w, alpha=0.25 + 0.55 * w, zorder=4.3,
                          solid_capstyle="round")
            dot = ax.scatter([sx], [sy], s=10 + 25 * w, color=th["beam"], alpha=0.9, zorder=4.4, linewidths=0)
            arts += [ln, dot]
        q = Polygon(dst_quad, closed=True, fill=False, edgecolor=th["beam"], lw=1.1, zorder=4.5)
        ax.add_patch(q)
        arts.append(q)
    return arts


def _draw_funnel(ax, src: Node, dst: Node, th) -> List:
    """Spatial map collapsing into a vector (global pooling / flatten)."""
    if src.deck is None:
        return []
    fq = [_px_to_data(src, *p) for p in src.deck.front.quad(0, 0, 1, 1)]
    x0, x1, y0, y1 = dst.extent
    tgt = [(x0, y1 - 0.1), (x0, y1 - 0.1), (x0, y0 + 0.1), (x0, y0 + 0.1)]
    ymid = (y0 + y1) / 2
    tgt = [(x0 - 0.02, ymid + 0.25), (x0 - 0.02, ymid + 0.25), (x0 - 0.02, ymid - 0.25), (x0 - 0.02, ymid - 0.25)]
    arts = []
    poly = Polygon([fq[1], tgt[0], tgt[3], fq[2]], closed=True, facecolor=th["beam_fill"], edgecolor="none",
                   alpha=0.07, zorder=2.5)
    ax.add_patch(poly)
    arts.append(poly)
    for p, q in ((fq[1], tgt[0]), (fq[2], tgt[3])):
        ln, = ax.plot([p[0], q[0]], [p[1], q[1]], color=th["beam"], lw=0.6, alpha=0.35, zorder=2.6)
        arts.append(ln)
    return arts


def _draw_contrib(ax, contrib, nodes, g, views, out_boxes, th) -> List:
    arts = []
    src = nodes.get(contrib.source)
    if src is None or src.dots is None:
        return arts
    head = nodes.get(contrib.head)
    if head is not None and head.dots is not None and head.dots.bin_of.size > 1:
        i = int(head.dots.bin_of[contrib.target_index]) if contrib.target_index < head.dots.bin_of.size else 0
        tx, ty = _px_to_data(head, *head.dots.centers[i])
    else:
        return arts
    vals = np.asarray(contrib.values[:14])
    if not len(vals):
        return arts
    vmax = np.abs(vals).max() + 1e-12
    for idx, v in zip(contrib.indices[:14], vals):
        if idx >= src.dots.bin_of.size:
            continue
        di = int(src.dots.bin_of[idx])
        sx, sy = _px_to_data(src, *src.dots.centers[di])
        w = abs(v) / vmax
        col = th["pos"] if v > 0 else th["neg"]
        path = Path([(sx, sy), (sx + (tx - sx) * 0.45, sy), (tx - (tx - sx) * 0.45, ty), (tx, ty)],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        pa = FancyArrowPatch(path=path, arrowstyle="-", lw=0.4 + 2.2 * w, color=col, alpha=0.25 + 0.6 * w, zorder=4.6)
        ax.add_patch(pa)
        dot = ax.scatter([sx], [sy], s=12 + 30 * w, facecolors="none", edgecolors=col, linewidths=1.0, zorder=4.7,
                         alpha=0.9)
        arts += [pa, dot]
    return arts


def _label(ax, x, y, title, sub, th, fs, va="bottom"):
    t1 = ax.text(x, y, spaced(title), fontsize=fs * 0.86, color=th["text"], ha="center", va=va, zorder=7,
                 fontweight="semibold")
    return [t1]


# ---------------------------------------------------------------------------


def render_cinematic(res, frame: Optional[FrameState] = None, title: Optional[str] = None,
                     subtitle: Optional[str] = None) -> FlowFigure:
    cfg = res.config
    theme = cfg.theme if cfg.theme in CINE_THEMES else "black"
    th = CINE_THEMES[theme]
    fam = register_fonts()
    rc = {"font.family": [fam, "DejaVu Sans"], "font.weight": "regular"}
    with plt.rc_context(rc):
        return _render(res, frame, th, title, subtitle)


def _render(res, frame, th, title, subtitle) -> FlowFigure:
    cfg = res.config
    g = res.graph
    ex = getattr(res, "explanation", None)
    nodes = build_nodes(res, frame, th)
    inp_rgb = _input_rgb(res)
    inp_vol = _input_volume(res)
    has_vig = ex is not None and bool(ex.units) and (inp_rgb is not None or inp_vol is not None)
    band = BAND if has_vig else 0.55

    # output cards
    out_nodes = res.out_nodes
    card_nodes: Dict[str, Node] = {}
    for ok, ov in res.views.items():
        cn = Node(ok, "card")
        if ov.kind == "segmentation" and ov.mask is not None:
            base = None
            for st in g.ordered():
                if st.kind == "input" and st.summary is not None and st.summary.image is not None:
                    base = st.summary.image
                    break
            vis = render_segmentation(ov.mask, base, cfg, cfg.volume_axes)
            cn.image = vis.image
            cn.extras["ortho"] = vis.extras.get("ortho")
            cn.w, cn.h = cn.image.shape[1] / P * 0.9, cn.image.shape[0] / P * 0.9
            cn.kind = "segcard"
        else:
            n = min(len(ov.items), 5)
            cn.w = 2.25
            cn.h = 0.95 + 0.2 * n
        card_nodes[ok] = cn

    sizes = {k: (n.w, n.h + band) for k, n in nodes.items()}
    for ok, cn in card_nodes.items():
        sizes[ok] = (cn.w, cn.h + band)
    label_top, label_bottom = 0.85, 0.15
    lcfg = cfg.updated(layout="horizontal")
    gap = 0.95

    def make_layout(nb):
        lay_ = compute_layout(g, sizes, out_nodes, lcfg, label_top, label_bottom, bands=nb)
        bx = lay_.boxes
        cols_ = sorted(set(b.col for b in bx.values()))
        col_w = {c: max(b.w for b in bx.values() if b.col == c) for c in cols_}
        xs_, x_, prev_band = {}, 0.0, None
        for c in cols_:
            band_c = next(b.band for b in bx.values() if b.col == c)
            if prev_band is not None and band_c != prev_band:
                x_ = 0.0
            prev_band = band_c
            xs_[c] = x_ + col_w[c] / 2
            x_ += col_w[c] + gap
        for b in bx.values():
            b.x = xs_[b.col]
        if nb > 1:   # a little breathing room between bands for the headings
            for b in bx.values():
                b.y -= b.band * 0.2
        return lay_

    lay = make_layout(1)
    if cfg.figsize is not None or cfg.layout == "wrap":
        Wt, Ht = cfg.figsize if cfg.figsize is not None else (16.0, 9.0)
        best, best_u = 1, 0.0
        for nb in range(1, 5):
            l2 = make_layout(nb)
            bx = l2.boxes.values()
            w_ = max(b.right for b in bx) - min(b.left for b in bx) + 0.8
            h_ = max(b.top for b in bx) - min(b.bottom for b in bx) + 1.8 + label_top + \
                (frame.footer_height if frame is not None else 0.0)
            u_ = min(Wt / w_, Ht / h_)          # the rendered scale: bigger is better
            if os.environ.get("NF_DEBUG"):
                print("bands", nb, round(w_, 2), round(h_, 2), round(u_, 3), sorted(set(b.band for b in bx)))
            if u_ > best_u * 1.03:
                best, best_u = nb, u_
        if best > 1:
            lay = make_layout(best)
    boxes = lay.boxes

    # geometry of rasters inside boxes: raster on top, band below
    for k, b in boxes.items():
        nd = nodes.get(k) or card_nodes.get(k)
        if nd is None:
            continue
        top = b.top
        cy = b.y + band / 2
        nd.extent = (b.x - nd.w / 2, b.x + nd.w / 2, cy - nd.h / 2, cy + nd.h / 2)
    # per-row baseline for vignettes / captions
    row_max_half = {}
    for k, b in boxes.items():
        nd = nodes.get(k) or card_nodes.get(k)
        r = round(b.y + band / 2, 3)
        row_max_half[r] = max(row_max_half.get(r, 0), nd.h / 2 if nd else 0)

    xmin = min(b.left for b in boxes.values()) - 0.2
    xmax = max(b.right for b in boxes.values()) + 0.2
    ymin = min(b.bottom for b in boxes.values()) - 0.15
    ymax = max((nodes.get(k) or card_nodes.get(k)).extent[3] for k in boxes) + label_top + 0.15
    head_h = 1.05
    foot_h = 0.55 + (frame.footer_height if frame is not None else 0.0)
    W_units = xmax - xmin + 0.8
    H_units = ymax - ymin + head_h + foot_h
    if cfg.figsize is not None:
        W_in, H_in = cfg.figsize
        u = min(W_in / W_units, H_in / H_units)
    else:
        u = 0.95
        W_in, H_in = W_units * u, H_units * u
    fs = 10.5 * cfg.font_scale * max(0.7, min(u / 0.95, 1.6))

    fig = plt.figure(figsize=(W_in, H_in), dpi=cfg.dpi, facecolor=th["bg"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(th["bg"])
    ax.axis("off")
    cxm = (xmin + xmax) / 2
    X0 = cxm - W_in / u / 2
    Y1 = ymax + head_h + (H_in / u - H_units) / 2
    if frame is not None and frame.draw_footer is not None and frame.footer_fill:
        spare = H_in / u - H_units
        if spare > 0:
            new_fh = min(frame.footer_height + spare, 0.3 * H_in / u)
            rest = spare - (new_fh - frame.footer_height)
            frame.footer_height = new_fh
            Y1 = ymax + head_h + rest / 2
    ax.set_xlim(X0, X0 + W_in / u)
    ax.set_ylim(Y1 - H_in / u, Y1)
    ax.set_aspect("equal", adjustable="box")
    ff = FlowFigure(fig, ax, layout=lay)
    arts_of: Dict[str, List] = {k: [] for k in list(nodes) + list(card_nodes)}

    # ---- title
    t = (frame.title if frame is not None and frame.title else None) or title or cfg.title or f"{res.model_name}"
    st_ = (frame.subtitle if frame is not None and frame.subtitle is not None else None)
    if st_ is None:
        st_ = subtitle if subtitle is not None else (cfg.subtitle or "")
    ax.text(X0 + 0.45, Y1 - 0.3, t, fontsize=fs * 1.9, color=th["text"], ha="left", va="top", fontweight="light")
    if st_:
        ax.text(X0 + 0.47, Y1 - 0.3 - fs * 1.9 / 72 / u * 1.35, st_, fontsize=fs * 0.95, color=th["muted"],
                ha="left", va="top")

    # ---- glow + rasters
    dark = th["bg"] in ("#000000", "#0e1016")
    for k, nd in list(nodes.items()) + [(k, c) for k, c in card_nodes.items() if c.image is not None]:
        x0, x1, y0, y1 = nd.extent
        if dark and nd.kind in ("deck", "page", "dots", "volume", "segcard"):
            hal, pad = glow(nd.image, radius=9 if nd.kind != "dots" else 5, strength=0.9 if nd.kind != "page" else 0.5)
            pu = pad / P * (nd.w / (nd.image.shape[1] / P)) if nd.kind != "segcard" else pad / P * 0.9
            a = ax.imshow(hal, extent=(x0 - pu, x1 + pu, y0 - pu, y1 + pu), zorder=1.5, interpolation="bilinear",
                          alpha=0.85)
            arts_of[k].append(a)
        im = ax.imshow(nd.image, extent=(x0, x1, y0, y1), zorder=3, interpolation="antialiased" if nd.kind in ("dots",) else "bilinear")
        arts_of[k].append(im)

    # ---- plain flow arrows (subtle) and skip arcs
    for e in g.edges:
        a, b = nodes.get(e.src), nodes.get(e.dst)
        if a is None or b is None:
            continue
        ay = (a.extent[2] + a.extent[3]) / 2
        by = (b.extent[2] + b.extent[3]) / 2
        if boxes[e.src].band != boxes[e.dst].band:
            ym = (min(a.extent[2], a.extent[2]) + max(b.extent[3], b.extent[3])) / 2
            ym = (min(bb.bottom for bb in boxes.values() if bb.band == boxes[e.src].band) +
                  max(bb.top for bb in boxes.values() if bb.band == boxes[e.dst].band)) / 2 + 0.25
            xr, xl = a.extent[1] + 0.35, b.extent[0] - 0.35
            path = Path([(a.extent[1] + 0.08, ay), (xr, ay), (xr, ym), (xl, ym), (xl, by), (b.extent[0] - 0.08, by)],
                        [Path.MOVETO] + [Path.LINETO] * 5)
            pa = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=10, lw=1.0, color=th["edge"], alpha=0.6,
                                 zorder=2, joinstyle="round")
            ax.add_patch(pa)
            ff.edge_artists.append((e.src, e.dst, pa))
            continue
        if e.kind == "skip":
            top = max(a.extent[3], b.extent[3]) + 0.35
            path = Path([(a.extent[1] - a.w * 0.3, a.extent[3] + 0.05), ((a.extent[0] + a.extent[1]) / 2, top + 0.25),
                         ((b.extent[0] + b.extent[1]) / 2, top + 0.25), (b.extent[0] + b.w * 0.3, b.extent[3] + 0.05)],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
            pa = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=10, lw=0.9, color=th["skip"],
                                 linestyle=(0, (3, 2.5)), alpha=0.7, zorder=2)
        else:
            p0, p3 = (a.extent[1] + 0.08, ay), (b.extent[0] - 0.08, by)
            if abs(ay - by) < 1e-3:
                pa = FancyArrowPatch(p0, p3, arrowstyle="-|>", mutation_scale=10, lw=1.0, color=th["edge"], alpha=0.6,
                                     zorder=2)
            else:
                dx = (p3[0] - p0[0]) * 0.5
                pa = FancyArrowPatch(path=Path([p0, (p0[0] + dx, p0[1]), (p3[0] - dx, p3[1]), p3],
                                               [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                                     arrowstyle="-|>", mutation_scale=10, lw=1.0, color=th["edge"], alpha=0.6, zorder=2)
        ax.add_patch(pa)
        ff.edge_artists.append((e.src, e.dst, pa))
        if e.kind != "skip" and a.deck is not None and b.dots is not None:
            arts_of[e.dst] += _draw_funnel(ax, a, b, th)
    for ok, srcs in out_nodes.items():
        cn = card_nodes[ok]
        if not srcs or srcs[0] not in nodes:
            continue
        a = nodes[srcs[0]]
        ay = (a.extent[2] + a.extent[3]) / 2
        cy = (cn.extent[2] + cn.extent[3]) / 2
        p0, p3 = (a.extent[1] + 0.08, ay), (cn.extent[0] - 0.08, cy)
        dx = (p3[0] - p0[0]) * 0.5
        pa = FancyArrowPatch(path=Path([p0, (p0[0] + dx, p0[1]), (p3[0] - dx, p3[1]), p3],
                                       [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                             arrowstyle="-|>", mutation_scale=11, lw=1.2, color=th["accent"], alpha=0.8, zorder=2)
        ax.add_patch(pa)
        ff.edge_artists.append((srcs[0], ok, pa))

    # ---- beams & contribution lines
    if ex is not None:
        for key, uf in ex.units.items():
            arts_of.setdefault(key, [])
            if uf.dep_stage in boxes and key in boxes and boxes[uf.dep_stage].band != boxes[key].band:
                continue
            arts_of[key] += _draw_beam(ax, g, nodes, uf, th)
        for ok, cb in ex.contributions.items():
            arts_of.setdefault(cb.head, [])
            arts_of[cb.head] += _draw_contrib(ax, cb, nodes, g, res.views, boxes, th)

    # ---- labels, captions, vignettes
    for k, nd in nodes.items():
        st = g.stages[k]
        x0, x1, y0, y1 = nd.extent
        cx = (x0 + x1) / 2
        mod = st.label if st.kind != "input" else ""
        if mod and mod.upper() != (nd.title or "").upper():
            arts_of[k].append(ax.text(cx, y1 + 0.1, mod, fontsize=fs * 0.74, color=th["muted"],
                                      ha="center", va="bottom", zorder=7))
        r = round(boxes[k].y + band / 2, 3)
        base = r - row_max_half.get(r, nd.h / 2)
        cap_y = base - 0.12
        arts_of[k].append(ax.text(cx, cap_y, nd.caption, fontsize=fs * 0.8, color=th["muted"], ha="center", va="top",
                                  zorder=7))
        if nd.sub:
            arts_of[k].append(ax.text(cx, cap_y - fs * 1.2 / 72 / u, nd.sub, fontsize=fs * 0.66, color=th["faint"],
                                      ha="center", va="top", zorder=7))
        uf = ex.units.get(k) if ex is not None else None
        if has_vig and uf is not None and uf.erf is not None:
            vy = cap_y - 0.42 - VIG / 2
            img = None
            if inp_rgb is not None and uf.erf.ndim == 2:
                img = vignette(inp_rgb, uf.erf, uf.rf_box, size=150, ring=th["ring"])
            elif inp_vol is not None and uf.erf.ndim == 3:
                img = _vignette3d(inp_vol, uf, res.config.volume_axes, th)
            if img is not None:
                a = ax.imshow(img, extent=(cx - VIG / 2, cx + VIG / 2, vy - VIG / 2, vy + VIG / 2), zorder=5,
                              interpolation="bilinear")
                arts_of[k].append(a)
                lab = f"sees ≈ {uf.rf_px:.0f} px" if uf.erf.ndim == 2 else f"sees ≈ {uf.rf_px:.0f} vox"
                arts_of[k].append(ax.text(cx, vy - VIG / 2 - 0.05, lab, fontsize=fs * 0.64, color=th["muted"],
                                          ha="center", va="top", zorder=7))

    # ---- concept headings, grouped over consecutive stages with the same concept
    groups: List[List[str]] = []
    for st in g.ordered():
        if st.key not in nodes:
            continue
        if groups:
            last = groups[-1][-1]
            bl, bk = boxes[last], boxes[st.key]
            if (g.stages[last].concept == st.concept and abs(bl.row - bk.row) < 0.5 and bl.band == bk.band
                    and abs(bl.level - bk.level) < 0.5):
                groups[-1].append(st.key)
                continue
        groups.append([st.key])
    top_all = {}
    for grp in groups:
        ex_ = [nodes[k].extent for k in grp]
        gx0, gx1 = min(e[0] for e in ex_), max(e[1] for e in ex_)
        gtop = max(e[3] for e in ex_) + 0.1 + fs * 1.3 / 72 / u
        title_ = nodes[grp[0]].title or g.stages[grp[0]].label
        est_w = len(title_) * 2 * fs * 0.9 * 0.55 / 72 / u * 0.62     # letter-spaced caps, rough width in units
        if len(grp) == 1 and est_w > (gx1 - gx0) + gap * 0.8 and " " in title_:
            from .render import _wrap_label

            title_ = _wrap_label(title_, max_chars=max(6, len(title_) // 2))
        arts = arts_of[grp[0]]
        if len(grp) > 1:
            ln, = ax.plot([gx0 + 0.05, gx0 + 0.05, gx1 - 0.05, gx1 - 0.05], [gtop - 0.05, gtop, gtop, gtop - 0.05],
                          color=th["faint"], lw=0.8, zorder=6)
            arts.append(ln)
        arts.append(ax.text((gx0 + gx1) / 2, gtop + 0.07, "\n".join(spaced(t_) for t_ in title_.split("\n")),
                            fontsize=fs * 0.9, color=th["text"], ha="center", va="bottom", fontweight="semibold",
                            zorder=7, linespacing=1.15))

    # ---- output cards
    for ok, cn in card_nodes.items():
        ov = res.views[ok]
        x0, x1, y0, y1 = cn.extent
        cx = (x0 + x1) / 2
        arts = arts_of[ok]
        arts += [ax.text(cx, y1 + 0.16, spaced(ov.title if ov.title not in ("prediction",) else "prediction"),
                         fontsize=fs * 0.86, color=th["accent"], ha="center", va="bottom", fontweight="semibold", zorder=7)]
        if cn.kind == "segcard":
            arts.append(ax.text(cx, y0 - 0.12, ov.subline, fontsize=fs * 0.8, color=th["muted"], ha="center", va="top"))
            if cn.extras.get("ortho") is not None:
                o = cn.extras["ortho"]
                ow = cn.w
                oh = ow * o.shape[0] / o.shape[1]
                vy = y0 - 0.45 - oh / 2
                arts.append(ax.imshow(o, extent=(cx - ow / 2, cx + ow / 2, vy - oh / 2, vy + oh / 2), zorder=5))
            continue
        card = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.08",
                              facecolor=th["card"], edgecolor=th["accent"], lw=1.2, zorder=2.8)
        ax.add_patch(card)
        arts.append(card)
        head = ov.headline if len(ov.headline) <= 22 else ov.headline[:21] + "…"
        fh = fs * (1.75 if len(head) <= 12 else 1.35 if len(head) <= 18 else 1.1)
        ty = y1 - 0.14
        arts.append(ax.text(cx, ty, head, fontsize=fh, color=th["text"], ha="center", va="top", fontweight="semibold",
                            zorder=6))
        ty -= fh / 72 / u * 1.3
        arts.append(ax.text(cx, ty, ov.subline, fontsize=fs * 0.72, color=th["muted"], ha="center", va="top", zorder=6))
        ty -= fs * 0.72 / 72 / u * 1.9
        items = ov.items[:5] if ov.kind in ("classification", "multilabel", "binary", "raw") else []
        vmax = max([abs(v) for _, v in items] + [1e-9]) if ov.kind == "raw" else 1.0
        for i, (lab, v) in enumerate(items):
            yy = ty - i * 0.2 - 0.08
            lab_s = lab if len(lab) <= 18 else lab[:17] + "…"
            arts.append(ax.text(x0 + 0.14, yy, lab_s, fontsize=fs * 0.66, color=th["text"] if i == 0 else th["muted"],
                                ha="left", va="center", zorder=6))
            bx0, bw = x0 + (x1 - x0) * 0.56, (x1 - x0) * 0.26
            r0 = Rectangle((bx0, yy - 0.045), bw, 0.09, facecolor=th["bar_bg"], edgecolor="none", zorder=5)
            frac = float(np.clip(abs(v) / vmax, 0, 1))
            r1 = Rectangle((bx0, yy - 0.045), bw * frac, 0.09, facecolor=th["accent"] if i == 0 else th["muted"],
                           edgecolor="none", zorder=5.5, alpha=0.95 if i == 0 else 0.6)
            ax.add_patch(r0)
            ax.add_patch(r1)
            arts += [r0, r1, ax.text(x1 - 0.1, yy, f"{v:.2f}" if ov.kind != "raw" else f"{v:.2g}", fontsize=fs * 0.6,
                                     color=th["muted"], ha="right", va="center", zorder=6)]
        # evidence (Grad-CAM) below the card
        cam = ex.gradcam.get(ok) if ex is not None else None
        if cam is not None and inp_rgb is not None and cam.ndim == 2:
            ev = heat_overlay(inp_rgb, cam, cmap="inferno")
            r = round(boxes[ok].y + band / 2, 3)
            base = r - row_max_half.get(r, cn.h / 2)
            s_ = 1.05
            vy = base - 0.28 - s_ / 2
            arts.append(ax.imshow(ev, extent=(cx - s_ / 2, cx + s_ / 2, vy - s_ / 2, vy + s_ / 2), zorder=5))
            arts.append(ax.text(cx, vy - s_ / 2 - 0.05, "evidence (Grad-CAM)", fontsize=fs * 0.64, color=th["muted"],
                                ha="center", va="top", zorder=7))

    # ---- legend / explanation footer
    fy = Y1 - H_in / u + (frame.footer_height if frame is not None else 0) + 0.28
    parts = []
    if ex is not None and ex.units:
        parts.append("beams: the region of the previous stage that feeds each stage's strongest unit")
        if has_vig:
            parts.append("circles: what that unit sees in the input (|∂unit/∂input|)")
    if ex is not None and ex.contributions:
        parts.append("lines into the head: largest weight × activation contributions (amber ↑, blue ↓)")
    if any(n.extras.get("pca_front") for n in nodes.values()):
        parts.append("pages: most active channels; front page = PCA of all channels as RGB")
    else:
        parts.append("pages: most active channels, front = strongest")
    ax.text(X0 + 0.47, fy, "   ·   ".join(parts), fontsize=fs * 0.72, color=th["faint"], ha="left", va="bottom")
    if frame is not None and frame.draw_footer is not None:
        frame.draw_footer(fig, ax, (X0, Y1 - H_in / u, W_in / u, frame.footer_height), th, fs, u)

    ff.stage_artists = arts_of
    ff.order = [s.key for s in g.ordered() if s.key in arts_of] + list(card_nodes)
    return ff


def _vignette3d(V: np.ndarray, uf, axes: str, th) -> Optional[np.ndarray]:
    E = canonical_volume(uf.erf, axes)
    if E.shape != V.shape:
        return None
    idx = np.unravel_index(int(np.argmax(E)), E.shape)
    z = idx[2]
    sl = np.flipud(V[:, :, z].T)
    es = np.flipud(E[:, :, z].T)
    rgb = np.repeat(robust_norm(sl, (1, 99.5))[..., None], 3, -1)
    img = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], -1).astype(np.float32)
    side, box = _box2d(es)
    return vignette(img, es, box, size=150, ring=th["ring"])


def _box2d(m):
    idx = np.argwhere(m >= 0.1 * m.max())
    if len(idx) == 0:
        return 0, None
    lo, hi = idx.min(0), idx.max(0) + 1
    return float(max(hi - lo)), (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))
