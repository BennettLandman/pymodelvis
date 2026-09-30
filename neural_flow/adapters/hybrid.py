"""Transformer-backbone segmentation networks (UNETR, SwinUNETR, UNesT, TransUNet …).

These models run a transformer *backbone* over the image once and tap its hidden
states at several depths.  Each tap feeds a convolutional projection branch that
brings it to the resolution of one decoder level, and the decoder climbs back to
full resolution::

    input ─► patch embedding ─► transformer level 1 ─► … ─► level n ─► bottleneck
                                     │ tap                    │ tap        │
                                     ▼                        ▼            ▼
    output ◄── decoder 1 ◄──────── … ◄─────────── decoder n-1 ◄─── decoder n

The generic module-tree cut sees a dozen root children (``vit``, ``encoder1..4``,
``decoder1..5``, ``out``) and never opens the backbone, so the transformer — the
part that makes these models what they are — was drawn as one box or not at all.

This adapter recognises the pattern *from the runtime dataflow* (no model names
are hard-coded) and proposes the stages itself:

* the backbone's tapped levels / blocks (its hidden states), plus its patch
  embedding when there is room;
* the bottleneck and decoder blocks;
* the segmentation head.

The per-tap projection branches (``encoder2..4`` in UNETR) are left out by default:
they only resample a hidden state for its decoder level, and without them each
tap is drawn as a skip bridge from the transformer level to the decoder it feeds.
Add them back with ``layers=[...]`` if you want them.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

import numpy as np

from .base import Adapter

_TRANSFORMER_TYPES = re.compile(
    r"(Attention|Transformer|SwinTransformerBlock|TransformerBlock|TransformerLayer|EncoderBlock|ViTBlock|"
    r"^Block$|NestLevel|BasicLayer|MultiheadAttention|SABlock)")


def _spatial(shape) -> Optional[tuple]:
    """Spatial extent of an activation shape (tokens [B, N, C] with cubic / square N count too)."""
    if not shape:
        return None
    if len(shape) == 5:
        return tuple(shape[2:])
    if len(shape) == 4:
        return tuple(shape[2:])
    if len(shape) == 3:
        n = shape[1]
        for k in (3, 2):
            r = int(round(n ** (1.0 / k)))
            for extra in (0, 1):
                if r > 1 and r ** k == n - extra:
                    return (r,) * k
    return None


def _res(shape) -> float:
    sp = _spatial(shape)
    return float(np.prod(sp)) ** (1.0 / len(sp)) if sp else 0.0


class HybridTransformerUNetAdapter(Adapter):
    """Transformer backbone + convolutional decoder (UNETR family)."""

    name = "transformer-unet"

    def __init__(self):
        self._cache: Dict[int, Optional[dict]] = {}

    # ------------------------------------------------------------------ analysis
    def _analyse(self, trace) -> Optional[dict]:
        key = id(trace)
        if key in self._cache:
            return self._cache[key]
        info = None
        try:
            info = self._analyse_uncached(trace)
        except Exception:
            info = None
        if len(self._cache) > 32:
            self._cache.clear()
        self._cache[key] = info
        return info

    def _analyse_uncached(self, trace) -> Optional[dict]:
        from ..stages import _Tree

        if not trace.dataflow_ok or not trace.ops:
            return None
        tree = _Tree(trace)
        calls = trace.calls
        root = calls[trace.root]
        top = tree.meaningful_children(root)
        if len(top) < 4:
            return None

        def subtree_types(c) -> List[str]:
            out, todo = [], list(c.children)
            while todo:
                k = calls[todo.pop()]
                out.append(k.type_name)
                todo.extend(k.children)
            return out

        # backbone: a top-level unit that contains transformer blocks
        cands = [c for c in top if any(_TRANSFORMER_TYPES.search(t) for t in subtree_types(c) + [c.type_name])]
        if len(cands) != 1:
            return None
        B = cands[0]
        members = tree.meaningful_children(B)
        if len(members) < 2:
            return None
        member_ids = {m.id for m in members}
        others = [c for c in top if c.id != B.id]

        # owner call -> backbone member (or None); owner call -> top-level unit
        top_ids = {c.id for c in top}

        def climb(cid, stop_ids):
            while cid is not None:
                if cid in stop_ids:
                    return cid
                cid = calls[cid].parent
            return None

        member_of: Dict[int, Optional[int]] = {}
        topunit_of: Dict[int, Optional[int]] = {}
        for cid in calls:
            member_of[cid] = climb(cid, member_ids)
            topunit_of[cid] = climb(cid, top_ids)

        def sources(nodes) -> Set:
            """Backbone members / other top-level units / inputs reached by walking back from ``nodes``."""
            found: Set = set()
            seen: Set[int] = set()
            todo = list(nodes)
            while todo:
                n = todo.pop()
                if n in seen:
                    continue
                seen.add(n)
                if n in trace.input_nodes:
                    found.add(("input", trace.input_nodes[n]))
                    continue
                op = trace.ops.get(n)
                if op is None:
                    continue
                own = op.owner
                if own is not None and member_of.get(own) is not None:
                    found.add(("member", member_of[own]))
                    continue
                if own is not None and topunit_of.get(own) not in (None, B.id):
                    found.add(("unit", topunit_of[own]))
                    continue
                todo.extend(op.parents)
            return found

        consumers: Dict[int, Set] = {c.id: sources(c.in_nodes) for c in others}
        out_srcs: Set = set()
        for _, _, node in trace.outputs:
            if node is not None:
                out_srcs |= sources([node])
        taps: Set[int] = set()
        n_consumers = 0
        for cid, srcs in consumers.items():
            hit = {v for k, v in srcs if k == "member"}
            if hit:
                n_consumers += 1
            taps |= hit
        if n_consumers < 2 or not taps:
            return None                      # a plain encoder (ViT classifier …), not a tapped backbone
        taps |= {v for k, v in out_srcs if k == "member"}
        tap_calls = [m for m in members if m.id in taps]
        min_tap_res = min((_res(m.main_shape) for m in tap_calls if _res(m.main_shape) > 0), default=0.0)

        # decoder side: keep what does not merely re-project a hidden state to a higher resolution
        keep, dropped = [], []
        for c in others:
            srcs = consumers[c.id]
            from_units = any(k == "unit" for k, _ in srcs)
            r = _res(c.main_shape)
            is_projection = (not from_units) and r > 0 and min_tap_res > 0 and r > min_tap_res * 1.01
            produces_output = any(k == "unit" and v == c.id for k, v in out_srcs) or c is others[-1]
            if is_projection and not produces_output:
                dropped.append(c)
            else:
                keep.append(c)
        roles: Dict[str, str] = {}
        embed = members[0] if members[0].id not in taps else None
        for m in members:
            if m.id in taps:
                roles[m.key] = "transformer"
        if members[0].id in taps and re.search(r"embed", members[0].name + members[0].type_name, re.I):
            roles[members[0].key] = "embedding"
        for c in keep:
            srcs = consumers[c.id]
            from_units = any(k == "unit" for k, _ in srcs)
            if not from_units:
                roles[c.key] = "bottleneck"
            else:
                roles[c.key] = "decoder"
        head = keep[-1] if keep else None
        if head is not None:
            roles[head.key] = "head"
        return {"backbone": B, "members": members, "taps": tap_calls, "embed": embed, "keep": keep,
                "dropped": dropped, "roles": roles}

    # ------------------------------------------------------------------ adapter API
    def match(self, model, trace, graph=None) -> float:
        if trace is None:
            return 0.0
        return 0.95 if self._analyse(trace) is not None else 0.0

    def defaults(self, model, trace):
        return {"max_stages": 14, "unet_layout": True}

    def select(self, trace, cfg):
        info = self._analyse(trace)
        if info is None:
            return None
        taps = list(info["taps"])
        keep = list(info["keep"])
        budget = max(4, cfg.max_stages)
        room = budget - len(keep)
        if room < 2:
            return None
        if len(taps) > room:
            # sample taps evenly, always keeping the first and the last level
            idx = sorted({int(round(v)) for v in np.linspace(0, len(taps) - 1, room)})
            taps = [taps[i] for i in idx]
        chosen = list(taps)
        emb = info["embed"]
        if emb is not None and len(chosen) + len(keep) < budget and re.search(r"embed", emb.name + emb.type_name, re.I):
            chosen = [emb] + chosen
            info["roles"][emb.key] = "embedding"
        chosen += keep
        trace.meta["hybrid"] = {
            "backbone": info["backbone"].name,
            "roles": dict(info["roles"]),
            "dropped": [c.name for c in info["dropped"]],
        }
        return sorted(chosen, key=lambda c: c.start)

    def annotate(self, trace, graph) -> None:
        """Attach backbone roles to ``graph`` even when the stages were chosen explicitly (e.g. movies)."""
        if graph.meta.get("hybrid"):
            return
        info = self._analyse(trace)
        if info is None:
            return
        roles = dict(info["roles"])
        emb = info["embed"]
        if emb is not None and re.search(r"embed", emb.name + emb.type_name, re.I):
            roles[emb.key] = "embedding"
        graph.meta["hybrid"] = {"backbone": info["backbone"].name, "roles": roles,
                                "dropped": [c.name for c in info["dropped"]]}

    def concepts(self, graph):
        meta = graph.meta.get("hybrid")
        if not meta:
            return None
        roles = meta["roles"]
        names = {"embedding": "PATCH EMBEDDING", "transformer": "TRANSFORMER ENCODER", "bottleneck": "BOTTLENECK",
                 "decoder": "DECODER", "head": "SEGMENTATION HEAD"}
        out = {}
        for k, s in graph.stages.items():
            r = roles.get(k)
            if r is None:
                continue
            if r == "head" and not (s.summary is not None and s.summary.is_spatial):
                out[k] = "HEAD"
            else:
                out[k] = names[r]
        return out
