"""nnU-Net style U-Nets (``dynamic_network_architectures``: PlainConvUNet, ResidualEncoderUNet …).

nnU-Net keeps each half of its U-Net in parallel lists that run interleaved::

    encoder.stem?  encoder.stages[0 … n-1]
    decoder.transpconvs[s] → cat(skip) → decoder.stages[s] → decoder.seg_layers[s]

The generic module-tree cut sees three long lists inside ``decoder`` and runs out
of budget half way (showing upsampling layers but no decoder convolutions).  This
adapter recognises the layout *structurally* — a module with ``stages``,
``transpconvs`` and ``seg_layers`` children next to a module with ``stages`` —
and proposes one stage per resolution level:

* the stem (residual encoders) and every encoder stage, the deepest one being the
  bottleneck;
* one stage per decoder level: its ``decoder.stages[s]`` block, with the
  transposed convolution that feeds it and the skip concatenation folded into
  the incoming edges;
* the final full-resolution segmentation layer as the only head.

Deep-supervision outputs (the lower-resolution predictions used in training) are
switched off while the figure is made (see :func:`inference_view`), exactly as
nnU-Net does at inference.
"""
from __future__ import annotations

import contextlib
import re
from typing import Dict, List, Optional

import torch.nn as nn

from .base import Adapter

_IDX = re.compile(r"^(.*)\.(stages|transpconvs|seg_layers)\.(\d+)$")


@contextlib.contextmanager
def inference_view(model: nn.Module, keep_aux: bool = False):
    """Temporarily switch off deep supervision (``module.deep_supervision = False``).

    nnU-Net networks return a list of predictions at every decoder resolution
    while ``deep_supervision`` is on; at inference only the full-resolution one is
    used.  Everything is restored afterwards.
    """
    changed = []
    if not keep_aux:
        for m in model.modules():
            if getattr(m, "deep_supervision", None) is True:
                changed.append(m)
                m.deep_supervision = False
    try:
        yield bool(changed)
    finally:
        for m in changed:
            m.deep_supervision = True


class NNUNetAdapter(Adapter):
    """Encoder levels + decoder levels + final segmentation layer."""

    name = "nnunet"

    def __init__(self):
        self._cache: Dict[int, Optional[dict]] = {}

    # ------------------------------------------------------------------ analysis
    def _analyse(self, trace) -> Optional[dict]:
        key = id(trace)
        if key not in self._cache:
            try:
                info = self._analyse_uncached(trace)
            except Exception:
                info = None
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = info
        return self._cache[key]

    def _analyse_uncached(self, trace) -> Optional[dict]:
        if trace is None or not trace.calls:
            return None
        lists: Dict[str, Dict[str, Dict[int, object]]] = {}
        stems: Dict[str, object] = {}
        for c in trace.calls_in_order():
            if c.call_index or not c.name:
                continue
            m = _IDX.match(c.name)
            if m:
                lists.setdefault(m.group(1), {}).setdefault(m.group(2), {})[int(m.group(3))] = c
            elif c.name.endswith(".stem"):
                stems[c.name[:-5]] = c
        decs = [p for p, d in lists.items() if {"stages", "transpconvs", "seg_layers"} <= set(d)]
        if len(decs) != 1:
            return None
        dec = decs[0]
        parent = dec.rpartition(".")[0]
        encs = [p for p, d in lists.items() if p != dec and "stages" in d and "transpconvs" not in d
                and p.rpartition(".")[0] == parent]
        if len(encs) != 1:
            return None
        enc = encs[0]
        e_stages = [lists[enc]["stages"][i] for i in sorted(lists[enc]["stages"])]
        d_stages = [lists[dec]["stages"][i] for i in sorted(lists[dec]["stages"])]
        segs = lists[dec]["seg_layers"]
        if len(e_stages) < 2 or not d_stages or not segs:
            return None
        seg_final = segs[max(segs)]                       # the full-resolution prediction
        n = len(e_stages)
        labels, roles = {}, {}
        stem = stems.get(enc)
        if stem is not None:
            labels[stem.name] = "stem"
            roles[stem.key] = "stem"
        for i, c in enumerate(e_stages):
            last = i == n - 1
            labels[c.name] = "bottleneck" if last else f"encoder {i + 1}"
            roles[c.key] = "bottleneck" if last else "encoder"
        for s, c in enumerate(d_stages):
            level = n - 1 - s                                  # decoder.stages[s] rebuilds encoder level n-1-s
            labels[c.name] = f"decoder {level}"
            roles[c.key] = "decoder"
        labels[seg_final.name] = "segmentation"
        roles[seg_final.key] = "head"
        chosen = ([stem] if stem is not None else []) + e_stages + d_stages + [seg_final]
        return {"chosen": chosen, "labels": labels, "roles": roles, "encoder": enc, "decoder": dec,
                "aux_heads": [segs[k].name for k in sorted(segs) if segs[k] is not seg_final]}

    # ------------------------------------------------------------------ adapter API
    def match(self, model, trace, graph=None) -> float:
        return 0.97 if self._analyse(trace) is not None else 0.0

    def defaults(self, model, trace):
        info = self._analyse(trace)
        out = {"unet_layout": True, "max_stages": 16}
        if info is not None:
            out["labels"] = dict(info["labels"])
        return out

    def select(self, trace, cfg):
        info = self._analyse(trace)
        if info is None:
            return None
        trace.meta["nnunet"] = {"roles": dict(info["roles"]), "encoder": info["encoder"], "decoder": info["decoder"]}
        return sorted(info["chosen"], key=lambda c: c.start)

    def annotate(self, trace, graph) -> None:
        if graph.meta.get("nnunet"):
            return
        info = self._analyse(trace)
        if info is not None:
            graph.meta["nnunet"] = {"roles": dict(info["roles"]), "encoder": info["encoder"],
                                    "decoder": info["decoder"]}

    def concepts(self, graph):
        meta = graph.meta.get("nnunet")
        if not meta:
            return None
        names = {"stem": "STEM", "encoder": "ENCODER", "bottleneck": "BOTTLENECK", "decoder": "DECODER",
                 "head": "SEGMENTATION HEAD"}
        return {k: names[meta["roles"][k]] for k in graph.stages if k in meta["roles"]}
