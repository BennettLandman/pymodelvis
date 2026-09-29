from __future__ import annotations

from .base import Adapter, type_count


class CNNAdapter(Adapter):
    """Plain 2-D CNNs (ResNet, VGG, EfficientNet, ...): generic heuristics already fit well."""

    name = "cnn"

    def match(self, model, trace, graph=None) -> float:
        return 0.3 if type_count(model, r"^Conv2d$") else 0.0

    def defaults(self, model, trace):
        return {}
