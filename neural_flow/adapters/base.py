from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List, Optional

from ..config import FlowConfig


class Adapter:
    """Base class: override ``match``, ``defaults`` and/or ``concepts``."""

    name = "base"

    def match(self, model, trace, graph=None) -> float:  # 0 = no match
        return 0.0

    def defaults(self, model, trace) -> Dict[str, Any]:
        return {}

    def concepts(self, graph) -> Optional[Dict[str, str]]:
        return None


def has_type(model, *names: str) -> bool:
    return any(type(m).__name__ in names for m in model.modules())


def type_count(model, pattern) -> int:
    import re

    rx = re.compile(pattern)
    return sum(1 for m in model.modules() if rx.search(type(m).__name__))


def apply_defaults(cfg: FlowConfig, overrides: Dict[str, Any], user_set: set) -> FlowConfig:
    kw = {k: v for k, v in overrides.items() if k not in user_set and k in {f.name for f in fields(cfg)}}
    return cfg.updated(**kw) if kw else cfg


def _trunk(graph) -> List:
    return [s for s in graph.ordered() if s.kind == "module" and not s.is_head]


def _pretty_name(n: str) -> str:
    return n.split(".")[-1].replace("_", " ").upper()


def generic_concepts(graph) -> Dict[str, str]:
    out: Dict[str, str] = {}
    stages = graph.ordered()
    inputs = [s for s in stages if s.kind == "input"]
    for s in inputs:
        out[s.key] = "INPUT" if len(inputs) == 1 else s.label.upper()
    heads = [s for s in stages if s.is_head]
    for s in heads:
        if len(heads) == 1:
            out[s.key] = "HEAD"
        elif s.summary is not None and s.summary.is_spatial:
            out[s.key] = "SEGMENTATION" if any("seg" in o.lower() for o in s.outputs) else _pretty_name(s.outputs[0] if s.outputs else s.label)
        else:
            out[s.key] = _pretty_name(s.outputs[0] if s.outputs else s.label)
    for s in stages:
        if s.kind == "representation":
            out[s.key] = "LATENT SPACE"
    vec_inputs = {s.key for s in inputs if s.summary is not None and s.summary.kind in ("vector", "scalar")}
    rest = []
    for s in stages:
        if s.key in out or s.kind != "module":
            continue
        preds = [e for e in graph.preds(s.key) if e.kind != "skip"]
        anc_inputs = {k for k in graph.ancestors(s.key) if k.startswith("input:")}
        if len(preds) >= 2:
            out[s.key] = "FUSION"
        elif anc_inputs and anc_inputs <= vec_inputs and len(inputs) > 1:
            out[s.key] = f"{graph.stages[sorted(anc_inputs)[0]].label.upper()} FEATURES"
        elif (s.summary is not None and s.summary.is_spatial and preds
              and graph.stages[preds[0].src].summary is not None
              and graph.stages[preds[0].src].summary.is_spatial
              and s.summary.resolution > graph.stages[preds[0].src].summary.resolution * 1.2):
            out[s.key] = "DECODER"
        else:
            rest.append(s)
    spatial = [s for s in rest if s.summary is not None and s.summary.is_spatial]
    n = len(spatial)
    for i, s in enumerate(spatial):
        f = (i + 0.5) / max(n, 1)
        out[s.key] = "LOW-LEVEL FEATURES" if f < 0.34 else ("STRUCTURAL FEATURES" if f < 0.67 else "HIGH-LEVEL FEATURES")
    for s in rest:
        if s.key not in out:
            out[s.key] = "LATENT REPRESENTATION"
    return out


def assign_concepts(graph, adapters) -> None:
    concepts = generic_concepts(graph)
    for a in reversed(list(adapters)):
        c = a.concepts(graph)
        if c:
            concepts.update(c)
    for k, s in graph.stages.items():
        s.concept = concepts.get(k, s.label.upper())
