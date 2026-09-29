from __future__ import annotations

import numpy as np

from .base import Adapter, _trunk


class UNetAdapter(Adapter):
    """Encoder/decoder networks with resolution-matched skip connections."""

    name = "unet"

    def match(self, model, trace, graph=None) -> float:
        if graph is None:
            return 0.0
        return 0.9 if any(e.kind == "skip" for e in graph.edges) and self._valley(graph) is not None else 0.0

    @staticmethod
    def _valley(graph):
        trunk = [s for s in _trunk(graph) if s.summary is not None and s.summary.is_spatial]
        if len(trunk) < 3:
            return None
        res = [s.summary.resolution for s in trunk]
        i = int(np.argmin(res))
        if 0 < i < len(trunk) - 1 and res[0] > res[i] and res[-1] > res[i]:
            return trunk, i
        return None

    def concepts(self, graph):
        v = self._valley(graph)
        if v is None:
            return None
        trunk, i = v
        out = {}
        for j, s in enumerate(trunk):
            if j < i:
                out[s.key] = "ENCODER"
            elif j == i:
                out[s.key] = "BOTTLENECK"
            else:
                out[s.key] = "DECODER"
        for s in graph.stages.values():
            if s.is_head and s.summary is not None and s.summary.is_spatial:
                out[s.key] = "SEGMENTATION HEAD"
        return out
