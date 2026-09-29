"""Final figure assembly with matplotlib.

The whole figure uses a single axes whose data coordinates are layout units;
each stage is one ``imshow`` (its composed raster) plus vector text, arrows and
outlines, so SVG/PDF output keeps labels and connections as vector graphics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path

from .config import FlowConfig
from .graph import StageGraph
from .layout import Layout, NodeBox, compute_layout, panel_scale
from .outputs import OutputView
from .raster import Visual

THEMES = {
    "light": dict(bg="#ffffff", text="#16181d", muted="#6b7280", faint="#9aa1ad", card="#f2f3f5",
                  card_edge="#dfe2e7", edge="#3f4652", skip="#2f7fc1", merge="#8a5cc2", accent="#c2410c",
                  bar="#3f4652", bar_bg="#e5e7eb", depth="#c9ced6"),
    "dark": dict(bg="#0e1016", text="#e8eaef", muted="#a3aab6", faint="#6b7280", card="#191c24",
                 card_edge="#2a2f3a", edge="#b8bfcc", skip="#5aa9e6", merge="#b28be0", accent="#fb923c",
                 bar="#d1d5db", bar_bg="#2a2f3a", depth="#3a404c"),
}


@dataclass
class FlowFigure:
    fig: plt.Figure
    ax: plt.Axes
    stage_artists: Dict[str, List] = field(default_factory=dict)
    edge_artists: List[Tuple[str, str, object]] = field(default_factory=list)
    layout: Optional[Layout] = None
    order: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------


def node_size(vis: Optional[Visual], summ, max_res: float, kind: str, ov: Optional[OutputView] = None,
              style: str = "technical") -> Tuple[float, float]:
    w, h = _node_size(vis, summ, max_res, kind, ov, style)
    if vis is not None and "attn" in vis.extras and kind != "output":
        w += h * ATTN_FRAC + 0.06
    return w, h


ATTN_FRAC = 0.34


def _node_size(vis: Optional[Visual], summ, max_res: float, kind: str, ov: Optional[OutputView] = None,
               style: str = "technical") -> Tuple[float, float]:
    if kind == "output":
        if ov is not None and ov.kind == "segmentation" and vis is not None:
            h = 1.3
            w = min(1.7, h * _panel_aspect(vis))
            if "ortho" in vis.extras:
                o = vis.extras["ortho"]
                return max(w, 1.3), h + 1.3 * o.shape[0] / o.shape[1] + 0.05
            return max(w, 1.15), h
        n = len(ov.items) if ov is not None else 0
        h = 0.62 + 0.17 * min(n, 5) if ov is not None and ov.items else 0.72
        return 1.45, max(h, 0.72)
    if vis is None:
        return 0.6, 0.6
    res = summ.resolution if summ is not None else 0
    s = 1.15 if kind == "input" else panel_scale(res, max_res)
    if vis.kind in ("strip", "generic"):
        n = vis.meta.get("rows", 64) if isinstance(vis.meta.get("rows"), int) else 64
        h = min(0.95 if kind != "input" else 0.8, 0.14 + 0.06 * n)
        w = max(0.14, min(0.55, h * vis.aspect))
        return w, h
    if vis.kind == "heatmap":
        h = 0.9
        return min(1.5, h * max(vis.aspect, 0.6)), h
    if vis.kind == "volume":
        side = 1.25 * (s if kind != "input" else 1.1)
        extra = 0.0
        if "ortho" in vis.extras:
            o = vis.extras["ortho"]
            extra = side * o.shape[0] / o.shape[1] + 0.05
        return side, side + extra
    h = 1.2 * s
    w = h * _panel_aspect(vis)
    if w > 1.8:
        h *= 1.8 / w
        w = 1.8
    return w, h


def _panel_aspect(vis: Visual) -> float:
    a = vis.aspect
    if "cls" in vis.extras:
        a += 0.18
    return a


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def _bezier(p0, p3, horizontal=True, bend=0.45):
    (x0, y0), (x3, y3) = p0, p3
    if horizontal:
        dx = (x3 - x0) * bend
        c1, c2 = (x0 + dx, y0), (x3 - dx, y3)
    else:
        dy = (y3 - y0) * bend
        c1, c2 = (x0, y0 + dy), (x3, y3 - dy)
    return Path([p0, c1, c2, p3], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])


def _edge_lw(features: int, cfg: FlowConfig, u: float) -> float:
    base = 1.1
    if cfg.edge_width == "features" and features:
        base = 0.9 + 0.28 * math.log2(max(features, 2))
    return min(base, 3.8) * max(0.7, min(u / 1.2, 1.4))


def _imshow(ax, img, x0, x1, y0, y1, z=3):
    return ax.imshow(img, extent=(x0, x1, y0, y1), interpolation="nearest", zorder=z, origin="upper",
                     resample=False)


# ---------------------------------------------------------------------------
# main renderer
# ---------------------------------------------------------------------------


def render_flow(g: StageGraph, visuals: Dict[str, Visual], views: Dict[str, OutputView],
                out_visuals: Dict[str, Visual], out_nodes: Dict[str, List[str]], cfg: FlowConfig,
                title: Optional[str] = None, subtitle: Optional[str] = None, footnote: Optional[str] = None) -> FlowFigure:
    th = THEMES.get(cfg.theme, THEMES["light"])
    story = cfg.style == "story"
    spatial = [s.summary.resolution for s in g.stages.values() if s.summary is not None and s.summary.is_spatial]
    max_res = max(spatial) if spatial else 1.0

    sizes: Dict[str, Tuple[float, float]] = {}
    for k, st in g.stages.items():
        sizes[k] = node_size(visuals.get(k), st.summary, max_res, st.kind if st.kind == "input" else "stage", style=cfg.style)
    for ok in out_nodes:
        sizes[ok] = node_size(out_visuals.get(ok), None, max_res, "output", views.get(ok), cfg.style)

    label_top, label_bottom = (0.3, 0.2) if story else (0.3, 0.42)
    if story:
        label_top = 0.42
    head_h = 0.55 if (title or subtitle) else 0.1
    foot_h = 0.25 if footnote else 0.05
    margin = 0.25
    min_w = {k: _label_width(st, story) for k, st in g.stages.items()}
    lay = compute_layout(g, sizes, out_nodes, cfg, label_top, label_bottom, min_w=min_w)
    # wrap long chains into bands when the requested canvas is much less elongated
    if cfg.layout in ("wrap", "horizontal") and cfg.layout != "vertical":
        target = (cfg.figsize[0] / cfg.figsize[1]) if cfg.figsize else (None if cfg.layout != "wrap" else 16 / 9)
        if target:
            best = 1
            best_fill = 0.0
            for nb in range(1, 5):
                l2 = compute_layout(g, sizes, out_nodes, cfg, label_top, label_bottom, bands=nb, min_w=min_w)
                w_, h_ = l2.width + 2 * margin, l2.height + head_h + foot_h + 2 * margin
                fill = min(target / (w_ / h_), (w_ / h_) / target)
                if fill > best_fill * 1.15:
                    best, best_fill = nb, fill
            if best > 1:
                lay = compute_layout(g, sizes, out_nodes, cfg, label_top, label_bottom, bands=best, min_w=min_w)
    nat_w = lay.width + 2 * margin
    # the title / subtitle must fit (matters for narrow vertical layouts)
    nat_w = max(nat_w, 0.125 * len(title or "") * cfg.font_scale + 2 * margin,
                0.062 * len(subtitle or "") * cfg.font_scale + 2 * margin)
    nat_h = lay.height + head_h + foot_h + 2 * margin
    if cfg.figsize is not None:
        W, H = cfg.figsize
        u = min(W / nat_w, H / nat_h)
    else:
        u = 1.25 if cfg.layout != "vertical" else 1.4
        W, H = nat_w * u, nat_h * u
        if W > 30:
            u *= 30 / W
            W, H = nat_w * u, nat_h * u
    fsc = cfg.font_scale * max(0.62, min(u / 1.25, 1.5))
    fs_title, fs_label, fs_shape, fs_type = 15 * fsc, 10.5 * fsc, 8.8 * fsc, 7.6 * fsc
    if story:
        fs_label *= 1.05

    fig = plt.figure(figsize=(W, H), dpi=cfg.dpi, facecolor=th["bg"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(th["bg"])
    ax.axis("off")
    # data coordinates: units; centre content
    x0 = lay.xmin - margin - (W / u - (lay.width + 2 * margin)) / 2
    y_top_content = lay.ymin + lay.height
    y1 = y_top_content + margin + head_h + (H / u - nat_h) / 2
    ax.set_xlim(x0, x0 + W / u)
    ax.set_ylim(y1 - H / u, y1)
    ax.set_aspect("equal", adjustable="box")
    ff = FlowFigure(fig, ax, layout=lay)

    if title or subtitle:
        tx = x0 + margin
        ty = y1 - margin * 0.9
        if title:
            ax.text(tx, ty, title, fontsize=fs_title, fontweight="bold", color=th["text"], va="top", ha="left")
        if subtitle:
            ax.text(tx, ty - (0.3 if title else 0), subtitle, fontsize=fs_type * 1.15, color=th["muted"], va="top", ha="left")
    if footnote:
        ax.text(x0 + margin, y1 - H / u + 0.1, footnote, fontsize=fs_type * 0.95, color=th["faint"], va="bottom", ha="left")

    boxes = lay.boxes
    horizontal = lay.orientation == "horizontal"

    # ---- edges (behind panels)
    for src, dst, kind, feats in lay.edges:
        if src not in boxes or dst not in boxes:
            continue
        a, b = boxes[src], boxes[dst]
        lw = _edge_lw(feats, cfg, u)
        if kind == "skip":
            if not cfg.show_skips:
                continue
            art = _draw_skip(ax, a, b, lay, th, lw, horizontal)
        else:
            color = th["merge"] if kind == "merge" else th["edge"]
            if horizontal and a.band != b.band:
                art = _draw_wrap(ax, a, b, lay, color, lw)
                ff.edge_artists.append((src, dst, art))
                continue
            if horizontal:
                p0, p3 = (a.right + 0.06, a.y), (b.left - 0.06, b.y)
            else:
                p0, p3 = (a.x, a.bottom - label_bottom * 0.9), (b.x, b.top + label_top + 0.02)
            if abs(p0[1] - p3[1]) < 1e-6 or (not horizontal and abs(p0[0] - p3[0]) < 1e-6):
                art = FancyArrowPatch(p0, p3, arrowstyle="-|>", mutation_scale=9 + 2 * lw, lw=lw, color=color,
                                      shrinkA=0, shrinkB=0, zorder=2, capstyle="round")
            else:
                art = FancyArrowPatch(path=_bezier(p0, p3, horizontal), arrowstyle="-|>", mutation_scale=9 + 2 * lw,
                                      lw=lw, color=color, zorder=2, capstyle="round")
            ax.add_patch(art)
        ff.edge_artists.append((src, dst, art))

    # ---- stage panels
    ordered = [s.key for s in g.ordered()]
    for k in ordered:
        st = g.stages[k]
        b = boxes.get(k)
        vis = visuals.get(k)
        if b is None or vis is None:
            continue
        arts = _draw_stage(ax, st, b, vis, cfg, th, fs_label, fs_shape, fs_type, story, label_top, label_bottom, u)
        ff.stage_artists[k] = arts
    ff.order = [k for k in ordered if k in ff.stage_artists]

    if story:
        for grp in _concept_groups(g, boxes):
            arts = _draw_group(ax, grp, boxes, g, th, fs_label)
            for k in grp:
                ff.stage_artists.setdefault(k, []).extend(arts if k == grp[0] else [])

    # ---- output cards
    for ok in out_nodes:
        b = boxes.get(ok)
        if b is None:
            continue
        arts = _draw_output(ax, views.get(ok), out_visuals.get(ok), b, th, fs_label, fs_shape, fs_type, story, label_top, u)
        ff.stage_artists[ok] = arts
        ff.order.append(ok)
    return ff


def _draw_wrap(ax, a: NodeBox, b: NodeBox, lay: Layout, color, lw):
    """Elbow connector from the end of one band to the start of the next."""
    band_bottom = min(bx.bottom for bx in lay.boxes.values() if bx.band == a.band) - lay.label_bottom
    next_top = max(bx.top for bx in lay.boxes.values() if bx.band == b.band) + lay.label_top
    ym = (band_bottom + next_top) / 2
    xr = a.right + 0.22
    xl = b.left - 0.22
    verts = [(a.right + 0.04, a.y), (xr, a.y), (xr, ym), (xl, ym), (xl, b.y), (b.left - 0.05, b.y)]
    path = Path(verts, [Path.MOVETO] + [Path.LINETO] * 5)
    art = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=9 + 2 * lw, lw=lw, color=color, zorder=2,
                          joinstyle="round", capstyle="round")
    ax.add_patch(art)
    return art


def _draw_skip(ax, a: NodeBox, b: NodeBox, lay: Layout, th, lw, horizontal):
    if lay.unet and horizontal:
        y = a.y + a.h * 0.18
        p0, p3 = (a.right + 0.06, y), (b.left - 0.06, b.y + b.h * 0.18)
        art = FancyArrowPatch(p0, p3, arrowstyle="-|>", mutation_scale=8 + lw, lw=lw * 0.6, color=th["skip"],
                              linestyle=(0, (4, 2.5)), zorder=1.5, alpha=0.9)
        ax.add_patch(art)
        return art
    if horizontal:
        tops = [bx.top for bx in lay.boxes.values() if a.x <= bx.x <= b.x and abs(bx.row - a.row) < 0.5]
        ytop = max(tops + [a.top, b.top]) + lay.label_top + 0.18
        p0 = (a.x + a.w * 0.25, a.top + lay.label_top * 0.95)
        p3 = (b.x - b.w * 0.25, b.top + lay.label_top * 0.95)
        verts = [p0, (p0[0], ytop), (p3[0], ytop), p3]
    else:
        rights = [bx.right for bx in lay.boxes.values() if b.y <= bx.y <= a.y]
        xr = max(rights + [a.right, b.right]) + 0.25
        p0, p3 = (a.right, a.y), (b.right, b.y)
        verts = [p0, (xr, p0[1]), (xr, p3[1]), p3]
    path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    art = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=8 + lw, lw=lw * 0.6, color=th["skip"],
                          linestyle=(0, (4, 2.5)), zorder=1.5, alpha=0.9)
    ax.add_patch(art)
    return art


def _draw_stage(ax, st, b: NodeBox, vis: Visual, cfg, th, fs_label, fs_shape, fs_type, story, label_top, label_bottom, u=1.25):
    arts = []
    summ = st.summary
    x0, x1, y0, y1 = b.left, b.right, b.bottom, b.top
    # depth cue: offset "sheets" behind feature maps grow with channel depth
    if vis.kind in ("mosaic", "tokens") and summ is not None and summ.channels > 4:
        n_layers = int(np.clip(round(math.log2(summ.channels) / 2) - 1, 0, 4))
        d = 0.035
        for i in range(n_layers, 0, -1):
            r = Rectangle((x0 + i * d, y0 + i * d), x1 - x0, y1 - y0, facecolor=th["depth"],
                          edgecolor=th["bg"], lw=0.6, zorder=2.5, alpha=0.55 + 0.1 * (n_layers - i))
            ax.add_patch(r)
            arts.append(r)
    if vis.kind not in ("volume",):
        card = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.03",
                              facecolor=th["card"], edgecolor=th["card_edge"], lw=0.6, zorder=2.8)
        ax.add_patch(card)
        arts.append(card)
    main_x0 = x0
    if "cls" in vis.extras:
        cls = vis.extras["cls"]
        cw = 0.12
        ch = min(y1 - y0, cw * cls.shape[0] / max(cls.shape[1], 1))
        arts.append(_imshow(ax, cls, x0, x0 + cw, b.y - ch / 2, b.y + ch / 2))
        arts.append(ax.text(x0 + cw / 2, b.y + ch / 2 + 0.02, "CLS", fontsize=fs_type * 0.75, color=th["muted"],
                            ha="center", va="bottom", zorder=4))
        main_x0 = x0 + cw + 0.06
    if "attn" in vis.extras:
        at = vis.extras["attn"]
        sz = (y1 - y0) * ATTN_FRAC
        ax_ = x1 - sz
        arts.append(_imshow(ax, at, ax_, x1, b.y - sz / 2, b.y + sz / 2, z=4))
        tag = "rollout" if st.kind == "representation" else "attn"
        arts.append(ax.text(ax_ + sz / 2, b.y + sz / 2 + 0.02, tag, fontsize=fs_type * 0.75, color=th["muted"],
                            ha="center", va="bottom", zorder=4))
        x1 = ax_ - 0.06
    if vis.kind == "volume" and "ortho" in vis.extras:
        o = vis.extras["ortho"]
        oh = (x1 - x0) * o.shape[0] / o.shape[1]
        arts.append(_imshow(ax, vis.image, x0, x1, y0 + oh + 0.05, y1))
        arts.append(_imshow(ax, o, x0, x1, y0, y0 + oh))
    else:
        # keep image aspect inside the box
        bw, bh = x1 - main_x0, y1 - y0
        a = vis.aspect
        if bw / bh > a:
            w = bh * a
            cx = (main_x0 + x1) / 2
            arts.append(_imshow(ax, vis.image, cx - w / 2, cx + w / 2, y0, y1))
        else:
            h = bw / a
            arts.append(_imshow(ax, vis.image, main_x0, x1, b.y - h / 2, b.y + h / 2))
    # labels
    if not story:
        arts.append(ax.text(b.x, y1 + 0.07, st.label, fontsize=fs_label, fontweight="bold", color=th["text"],
                            ha="center", va="bottom", zorder=6))
    if summ is not None:
        shape_txt = summ.shape_label()
        if summ.reduced and st.kind != "input":
            shape_txt = "≈ " + shape_txt
        arts.append(ax.text(b.x, y0 - 0.06, shape_txt, fontsize=fs_shape, color=th["muted"], ha="center",
                            va="top", zorder=6))
        if not story:
            sub = st.type_name if st.kind != "input" else ("input" if not summ.kind else summ.kind.replace("image2d", "image").replace("volume3d", "volume"))
            if st.kind == "representation":
                sub = "latent vector"
            if sub:
                arts.append(ax.text(b.x, y0 - 0.06 - fs_shape / 72 * 1.45 / u, sub, fontsize=fs_type,
                                    color=th["faint"], ha="center", va="top", zorder=6, style="italic"))
    return arts


def _draw_output(ax, ov: Optional[OutputView], vis: Optional[Visual], b: NodeBox, th, fs_label, fs_shape, fs_type,
                 story, label_top, u=1.25):
    arts = []
    x0, x1, y0, y1 = b.left, b.right, b.bottom, b.top
    title = (ov.title if ov else "output") or "output"
    arts.append(ax.text(b.x, y1 + 0.07, title, fontsize=fs_label, fontweight="bold", color=th["accent"],
                        ha="center", va="bottom", zorder=6))
    if ov is None:
        return arts
    if ov.kind == "segmentation" and vis is not None:
        if "ortho" in vis.extras:
            o = vis.extras["ortho"]
            oh = (x1 - x0) * o.shape[0] / o.shape[1]
            side = min(x1 - x0, y1 - y0 - oh - 0.05)
            cx = b.x
            arts.append(_imshow(ax, vis.image, cx - side / 2, cx + side / 2, y0 + oh + 0.05, y0 + oh + 0.05 + side))
            arts.append(_imshow(ax, o, x0, x1, y0, y0 + oh))
        else:
            h = y1 - y0
            w = min(x1 - x0, h * vis.aspect)
            arts.append(_imshow(ax, vis.image, b.x - w / 2, b.x + w / 2, y0, y1))
        arts.append(ax.text(b.x, y0 - 0.06, ov.subline, fontsize=fs_shape, color=th["muted"], ha="center", va="top", zorder=6))
        return arts
    card = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.06",
                          facecolor=th["bg"], edgecolor=th["accent"], lw=1.3, zorder=2.8)
    ax.add_patch(card)
    arts.append(card)
    fs_head = fs_label * (1.55 if len(ov.headline) <= 10 else 1.2 if len(ov.headline) <= 18 else 0.95)
    head = ov.headline if len(ov.headline) <= 24 else ov.headline[:23] + "…"
    ty = y1 - 0.1
    arts.append(ax.text(b.x, ty, head, fontsize=fs_head, fontweight="bold", color=th["text"], ha="center", va="top", zorder=6))
    ty -= fs_head / 72 / u * 1.35
    if ov.subline:
        arts.append(ax.text(b.x, ty, ov.subline, fontsize=fs_type, color=th["muted"], ha="center", va="top", zorder=6))
        ty -= fs_type / 72 / u * 1.9
    if ov.items and ov.kind in ("classification", "multilabel", "binary", "raw"):
        n = min(len(ov.items), 5)
        bar_h = 0.1
        pad = 0.1
        avail = ty - (y0 + 0.08)
        step = min(0.17, avail / max(n, 1))
        vals = [v for _, v in ov.items[:n]]
        vmax = max(1e-9, max(abs(v) for v in vals)) if ov.kind == "raw" else 1.0
        for i, (lab, v) in enumerate(ov.items[:n]):
            yy = ty - i * step - step / 2
            bx0, bx1 = x0 + pad, x1 - pad
            wbar = (bx1 - bx0) * 0.5
            lab_s = lab if len(lab) <= 16 else lab[:15] + "…"
            arts.append(ax.text(bx0, yy, lab_s, fontsize=fs_type * 0.95, color=th["text"], ha="left", va="center", zorder=6))
            bxs = bx1 - wbar
            r0 = Rectangle((bxs, yy - bar_h / 2), wbar, bar_h * 0.8, facecolor=th["bar_bg"], edgecolor="none", zorder=5)
            frac = max(0.0, min(1.0, abs(v) / vmax))
            r1 = Rectangle((bxs, yy - bar_h / 2), wbar * frac, bar_h * 0.8,
                           facecolor=th["accent"] if i == 0 else th["bar"], edgecolor="none", zorder=5.5, alpha=0.9 if i == 0 else 0.55)
            ax.add_patch(r0)
            ax.add_patch(r1)
            arts += [r0, r1]
    elif vis is not None and ov.kind == "embedding":
        h = max(0.1, ty - y0 - 0.1)
        w = min(x1 - x0 - 0.2, h * vis.aspect)
        arts.append(_imshow(ax, vis.image, b.x - w / 2, b.x + w / 2, y0 + 0.08, y0 + 0.08 + h))
    return arts


def _wrap_label(text: str, max_chars: int = 12) -> str:
    if len(text) <= max_chars or " " not in text:
        return text
    words = text.split()
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        m = max(len(a), len(b))
        if best is None or m < best[0]:
            best = (m, a + "\n" + b)
    return best[1]


def _label_width(st, story: bool) -> float:
    """Approximate label width in layout units (so neighbouring labels never collide)."""
    if story:
        txt = _wrap_label(st.concept or st.label)
        per = 0.085
    else:
        txt = st.label
        per = 0.072
    shape = st.summary.shape_label() if st.summary is not None else ""
    n = max([len(t) for t in txt.split("\n")] + [len(shape) * 0.85, len(st.type_name) * 0.75 if not story else 0])
    return n * per + 0.05


def _concept_groups(g: StageGraph, boxes) -> List[List[str]]:
    groups: List[List[str]] = []
    for st in g.ordered():
        b = boxes.get(st.key)
        if b is None:
            continue
        if groups:
            last = g.stages[groups[-1][-1]]
            lb = boxes[last.key]
            if last.concept == st.concept and abs(lb.row - b.row) < 0.5 and lb.band == b.band and abs(lb.level - b.level) < 0.5:
                groups[-1].append(st.key)
                continue
        groups.append([st.key])
    return groups


def _draw_group(ax, keys, boxes, g, th, fs_label):
    bs = [boxes[k] for k in keys]
    x0 = min(b.left for b in bs)
    x1 = max(b.right for b in bs)
    top = max(b.top for b in bs) + 0.1
    arts = []
    label = g.stages[keys[0]].concept or g.stages[keys[0]].label
    if len(bs) == 1:
        label = _wrap_label(label)
    if len(bs) > 1:
        ln, = ax.plot([x0, x0, x1, x1], [top - 0.04, top, top, top - 0.04], color=th["faint"], lw=1.0, zorder=5,
                      solid_capstyle="round")
        arts.append(ln)
    fs = fs_label
    arts.append(ax.text((x0 + x1) / 2, top + 0.05, label, fontsize=fs, linespacing=1.05, fontweight="bold", color=th["text"],
                        ha="center", va="bottom", zorder=6))
    return arts


def save_figure(ff: FlowFigure, path: str, cfg: FlowConfig) -> None:
    ff.fig.savefig(path, dpi=cfg.dpi, facecolor=ff.fig.get_facecolor())
