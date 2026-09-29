from __future__ import annotations

from .base import Adapter, type_count


class Medical3DAdapter(Adapter):
    """Volumetric (MRI / CT) networks: tensors ``[B, C, X, Y, Z]`` are first-class.

    Defaults: voxel-block volume rendering for feature maps, anatomical
    volume + orthogonal slices for the input, fewer channels per stage (each
    channel is a 3-D render).
    """

    name = "medical3d"

    def match(self, model, trace, graph=None) -> float:
        if type_count(model, r"(Conv3d|ConvTranspose3d|MaxPool3d|AvgPool3d)"):
            return 0.85
        if trace is not None and any(len(s) == 5 for s in trace.input_shapes.values()):
            return 0.85
        return 0.0

    def defaults(self, model, trace):
        # spatial variance favours channels with anatomical structure over channels that are
        # uniformly "on" (e.g. background detectors), which read poorly as 3-D blocks
        return {"volume_mode": "auto", "max_channels": 9, "channel_strategy": "variance"}

    def concepts(self, graph):
        return None
