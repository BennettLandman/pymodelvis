"""Automatic figure layout.

Nodes (inputs, stages, output cards) are placed on a grid of *columns*
(longest-path depth in the stage DAG) and *lanes* (branches).  Sibling branches
fan out symmetrically around their parent lane, so multi-head models branch
visibly and multimodal inputs merge visibly.  For encoder/decoder networks with
resolution-matched skip connections the flow optionally *dips* by resolution
level (a U-shape), which turns skip connections into short horizontal bridges.

All sizes are computed in abstract *units*; :mod:`neural_flow.render` converts
units to inches for the requested ``figsize``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .config import FlowConfig
from .graph import StageGraph


@dataclass
class NodeBox:
    key: str
    col: int
    row: float
    w: float
    h: float
    x: float = 0.0          # centre (units, y up)
    y: float = 0.0
    kind: str = "stage"     # input | stage | representation | output
    level: float = 0.0
    band: int = 0

    @property
    def left(self):
        return self.x - self.w / 2

    @property
    def right(self):
        return self.x + self.w / 2

    @property
    def top(self):
        return self.y + self.h / 2

    @property
    def bottom(self):
        return self.y - self.h / 2


@dataclass
class Layout:
    boxes: Dict[str, NodeBox]
    edges: List[Tuple[str, str, str, int]]     # (src, dst, kind, features)
    width: float
    height: float
    orientation: str
    unet: bool
    label_top: float
    label_bottom: float
    xmin: float = 0.0
    ymin: float = 0.0
    bands: int = 1


def panel_scale(res: float, max_res: float) -> float:
    """Relative panel size: shrinks (logarithmically) as spatial resolution drops."""
    if res <= 0 or max_res <= 1:
        return 0.72
    return 0.5 + 0.5 * math.log(max(res, 1.0) + 1) / math.log(max_res + 1)


def detect_unet(g: StageGraph) -> bool:
    skips = [e for e in g.edges if e.kind == "skip"]
    if not skips:
        return False
    hits = 0
    for e in skips:
        a, b = g.stages[e.src], g.stages[e.dst]
        sa = a.summary.spatial if a.summary is not None else None
        sb = b.summary.spatial if b.summary is not None else None
        if sa and sb and len(sa) >= 2 and tuple(sa) == tuple(sb) and a.order < b.order:
            # there must be a lower-resolution stage between them
            between = [s for s in g.stages.values() if a.order < s.order < b.order and s.summary is not None
                       and s.summary.spatial and len(s.summary.spatial) == len(sa)
                       and np.prod(s.summary.spatial) < np.prod(sa)]
            if between:
                hits += 1
    return hits >= 1


def compute_layout(g: StageGraph, sizes: Dict[str, Tuple[float, float]], out_nodes: Dict[str, List[str]],
                   cfg: FlowConfig, label_top: float = 0.34, label_bottom: float = 0.42, bands: int = 1,
                   min_w: Optional[Dict[str, float]] = None) -> Layout:
    """sizes: node key -> (w, h) in units; out_nodes: output-node key -> source stage keys."""
    G = nx.DiGraph()
    for k in g.stages:
        G.add_node(k)
    for k in out_nodes:
        G.add_node(k)
    edges: List[Tuple[str, str, str, int]] = []
    for e in g.edges:
        G.add_edge(e.src, e.dst, kind=e.kind)
        edges.append((e.src, e.dst, e.kind, e.features))
    for ok, srcs in out_nodes.items():
        for s in srcs[:1]:
            G.add_edge(s, ok, kind="main")
            st = g.stages.get(s)
            f = 0
            if st is not None and st.shape:
                f = st.shape[1] if len(st.shape) >= 3 and st.shape[1] != st.shape[2] else st.shape[-1]
            edges.append((s, ok, "main", min(f, 64)))
    if not nx.is_directed_acyclic_graph(G):  # pragma: no cover - defensive
        cyc = list(nx.simple_cycles(G))
        for c in cyc:
            if G.has_edge(c[-1], c[0]):
                G.remove_edge(c[-1], c[0])

    order = {k: (s.order if k in g.stages else 1e12) for k, s in list(g.stages.items()) + [(o, None) for o in out_nodes]}
    col: Dict[str, int] = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        col[n] = 0 if not preds else max(col[p] + 1 for p in preds)
    depth = dict(col)
    # right-align side chains (e.g. a clinical MLP feeding a late fusion) next to their consumer
    for n in reversed(list(nx.topological_sort(G))):
        succ = [s_ for s_ in G.successors(n) if G.edges[n, s_].get("kind", "main") != "skip"]
        if succ and n not in out_nodes:
            want = min(col[s_] for s_ in succ) - 1
            if want > col[n]:
                col[n] = want
    if out_nodes:
        last = max(col[k] for k in g.stages) + 1 if g.stages else 0
        for ok in out_nodes:
            col[ok] = max(col[ok], last)

    # primary predecessor = main edge from the most recently executed source
    def primary(n):
        preds = [p for p in G.predecessors(n) if G.edges[p, n].get("kind", "main") == "main"]
        if not preds:
            preds = list(G.predecessors(n))
        # the trunk is the deepest incoming path; ties broken by execution order
        return max(preds, key=lambda p: (depth.get(p, 0), order.get(p, 0))) if preds else None

    row: Dict[str, float] = {}
    by_col: Dict[int, List[str]] = {}
    for n, c in col.items():
        by_col.setdefault(c, []).append(n)
    inputs = sorted([k for k in g.stages if g.stages[k].kind == "input"], key=lambda k: order[k])
    for i, k in enumerate(inputs):
        row[k] = i - (len(inputs) - 1) / 2
    for c in sorted(by_col):
        nodes = sorted(by_col[c], key=lambda n: order.get(n, 0))
        groups: Dict[Optional[str], List[str]] = {}
        for n in nodes:
            if n in row:
                continue
            groups.setdefault(primary(n), []).append(n)
        for p, kids in groups.items():
            base = row.get(p, 0.0) if p is not None else 0.0
            k = len(kids)
            for i, n in enumerate(kids):
                row[n] = base + (i - (k - 1) / 2)
        # resolve collisions in this column
        placed = sorted([n for n in by_col[c]], key=lambda n: row[n])
        for i in range(1, len(placed)):
            if row[placed[i]] - row[placed[i - 1]] < 1:
                row[placed[i]] = row[placed[i - 1]] + 1

    # U-shape levels
    unet = cfg.unet_layout is True or (cfg.unet_layout == "auto" and detect_unet(g))
    level: Dict[str, float] = {}
    if unet:
        res = {k: (s.summary.resolution if s.summary is not None and s.summary.is_spatial else None) for k, s in g.stages.items()}
        rmax = max([r for r in res.values() if r] or [1])
        for k in col:
            r = res.get(k)
            if r:
                level[k] = math.log2(rmax / max(r, 1))
            elif k in out_nodes:
                level[k] = 0.0
            else:
                p = primary(k)
                level[k] = level.get(p, 0.0) if p else 0.0

    # geometry
    ncol = max(col.values()) + 1 if col else 1
    col_w = [0.0] * ncol
    for n, c in col.items():
        col_w[c] = max(col_w[c], sizes[n][0], (min_w or {}).get(n, 0.0))
    gap_x = 0.55 if cfg.layout != "vertical" else 0.45
    bands = max(1, min(bands, ncol)) if cfg.layout != "vertical" else 1
    per_band = int(math.ceil(ncol / bands))
    band_of = [c // per_band for c in range(ncol)]
    xs = []
    x = 0.0
    for c in range(ncol):
        if c > 0 and band_of[c] != band_of[c - 1]:
            x = 0.0
        xs.append(x + col_w[c] / 2)
        x += col_w[c] + gap_x

    # smallest lane pitch such that no two nodes stacked in the same column overlap
    lane_h = 0.0
    by_c: Dict[int, List[str]] = {}
    for n, c in col.items():
        by_c.setdefault(c, []).append(n)
    for c, ns in by_c.items():
        ns = sorted(ns, key=lambda n: row[n])
        for a, b in zip(ns, ns[1:]):
            dr = row[b] - row[a]
            if dr > 1e-6:
                need = (sizes[a][1] + sizes[b][1]) / 2 + label_top + label_bottom + 0.2
                lane_h = max(lane_h, need / dr)
    if lane_h == 0.0:
        lane_h = max(h for _, h in sizes.values()) + label_top + label_bottom + 0.25
    boxes: Dict[str, NodeBox] = {}
    lvl_gap = 0.0
    if unet:
        max_h = max(h for _, h in sizes.values())
        lvl_gap = max_h * 0.55 + label_top * 0.5
    for n in col:
        w, h = sizes[n]
        kind = "output" if n in out_nodes else ("input" if n in g.stages and g.stages[n].kind == "input"
                                                else "representation" if n in g.stages and g.stages[n].kind == "representation"
                                                else "stage")
        b = NodeBox(n, col[n], row[n], w, h, kind=kind, level=level.get(n, 0.0))
        b.x = xs[col[n]]
        b.y = -(row[n] * lane_h) - b.level * lvl_gap
        boxes[n] = b
    if bands > 1:
        # stack bands vertically; band height = extent of one band's lanes/levels
        span_top = max(b.top + label_top for b in boxes.values())
        span_bot = min(b.bottom - label_bottom for b in boxes.values())
        band_h = span_top - span_bot + 0.45
        for b in boxes.values():
            b.band = band_of[b.col]
            b.y -= band_of[b.col] * band_h

    if cfg.layout == "vertical":
        for b in boxes.values():
            b.x, b.y = -b.y, -b.x
            # in vertical layout columns become rows: re-space by panel heights
        # re-space vertical pitch using heights
        cols_sorted = sorted(set(b.col for b in boxes.values()))
        y = 0.0
        ypos = {}
        for c in cols_sorted:
            hmax = max(b.h for b in boxes.values() if b.col == c)
            ypos[c] = y - (label_top + hmax / 2)
            y -= hmax + label_top + label_bottom + 0.35
        lanes = sorted(set(b.row for b in boxes.values()))
        wmax = max(b.w for b in boxes.values())
        for b in boxes.values():
            b.y = ypos[b.col]
            b.x = b.row * (wmax + 0.9) + (b.level * 0.0)

    xmin = min(b.left for b in boxes.values())
    xmax = max(b.right for b in boxes.values())
    ymin = min(b.bottom - label_bottom for b in boxes.values())
    ymax = max(b.top + label_top for b in boxes.values())
    # room for skip arcs above the trunk
    if cfg.show_skips and any(e[2] == "skip" for e in edges) and not unet and cfg.layout != "vertical":
        ymax += 0.45
    return Layout(boxes, edges, xmax - xmin, ymax - ymin, "vertical" if cfg.layout == "vertical" else "horizontal",
                  unet, label_top, label_bottom, xmin, ymin, bands)
