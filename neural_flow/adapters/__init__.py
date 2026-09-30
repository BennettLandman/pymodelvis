"""Architecture adapters.

Adapters never hard-code a single model.  Each one recognises an architecture
*family* from the module types / runtime trace and contributes:

* default option tweaks (only for options the user left at their defaults),
* conceptual stage names used by ``style="story"``.

Register your own with :func:`register_adapter`.
"""
from __future__ import annotations

from typing import List

from .base import Adapter, apply_defaults, assign_concepts
from .cnn import CNNAdapter
from .hybrid import HybridTransformerUNetAdapter
from .medical3d import Medical3DAdapter
from .transformer import TransformerAdapter
from .unet import UNetAdapter

_REGISTRY: List[Adapter] = [HybridTransformerUNetAdapter(), UNetAdapter(), TransformerAdapter(), Medical3DAdapter(), CNNAdapter()]


def register_adapter(adapter: Adapter, first: bool = True) -> None:
    if first:
        _REGISTRY.insert(0, adapter)
    else:
        _REGISTRY.append(adapter)


def matching_adapters(model, trace, graph=None) -> List[Adapter]:
    scored = [(a.match(model, trace, graph), i, a) for i, a in enumerate(_REGISTRY)]
    return [a for s, i, a in sorted(scored, key=lambda t: (-t[0], t[1])) if s > 0]


__all__ = ["Adapter", "register_adapter", "matching_adapters", "apply_defaults", "assign_concepts"]
