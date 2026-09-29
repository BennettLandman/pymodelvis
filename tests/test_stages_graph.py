import warnings

import pytest
import torch
import torchvision

from conftest import DataDependent, MultiInput, TinyCNN, TinyUNet, TinyViT
from neural_flow import trace_model
from neural_flow.graph import fx_stage_edges


def _module_keys(res):
    return [s.key for s in res.graph.ordered() if s.kind == "module"]


def test_auto_selection_resnet18():
    m = torchvision.models.resnet18()
    res = trace_model(m, torch.randn(1, 3, 224, 224))
    keys = _module_keys(res)
    assert keys == ["conv1", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    # trivial modules never selected
    assert not any(k in keys for k in ("bn1", "relu"))


def test_auto_selection_respects_max_stages():
    m = torchvision.models.resnet50()
    for n in (4, 6, 10):
        res = trace_model(m, torch.randn(1, 3, 128, 128), max_stages=n)
        assert 2 <= len(_module_keys(res)) <= n


def test_transformer_blocks_are_sampled():
    res = trace_model(TinyViT(depth=12), torch.randn(1, 3, 32, 32), target_stages=6)
    blocks = [k for k in _module_keys(res) if k.startswith("blocks.")]
    assert 2 <= len(blocks) < 12
    assert "blocks.0" in blocks and "blocks.11" in blocks
    # representation stage (CLS vector) inserted before the head
    assert any(s.kind == "representation" for s in res.graph.stages.values())


def test_explicit_layers_regex_and_type():
    m = TinyCNN()
    x = torch.randn(1, 3, 32, 32)
    assert _module_keys(trace_model(m, x, layers=["block1", "fc"])) == ["block1", "fc"]
    assert _module_keys(trace_model(m, x, layers=["re:^block\\d$"])) == ["block1", "block2"]
    assert _module_keys(trace_model(m, x, layers=["type:Conv2d"])) == ["stem.0", "block1.0", "block2.0"]
    assert _module_keys(trace_model(m, x, layer_selection=lambda c: c.type_name == "Linear")) == ["fc"]


def test_unet_skip_connections_detected():
    res = trace_model(TinyUNet(), torch.randn(1, 1, 32, 32))
    kinds = {(e.src, e.dst): e.kind for e in res.graph.edges}
    assert kinds.get(("enc1", "dec1")) == "skip"
    assert kinds.get(("enc2", "dec2")) == "skip"
    assert res.graph.topology_source == "runtime"
    assert "unet" in res.adapters


def test_multi_input_merge_edges():
    res = trace_model(MultiInput(), {"image": torch.randn(1, 1, 16, 16), "clinical": torch.randn(1, 6)})
    kinds = sorted(e.kind for e in res.graph.preds("fuse"))
    assert kinds == ["main", "merge"]


def test_fx_failure_falls_back():
    m = DataDependent()
    x = torch.randn(1, 3, 16, 16)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = trace_model(m, x, topology="fx")
    assert any("could not be symbolically traced" in str(i.message) for i in w)
    assert res.graph.topology_source == "sequential"
    assert len(_module_keys(res)) >= 2
    # runtime tracing still works for the same model
    res2 = trace_model(m, x)
    assert res2.graph.topology_source == "runtime"


def test_fx_topology_matches_runtime_for_traceable_model():
    m = TinyUNet()
    x = torch.randn(1, 1, 32, 32)
    res = trace_model(m, x, topology="fx")
    assert res.graph.topology_source == "fx"
    edges = {(e.src, e.dst) for e in res.graph.edges}
    assert ("enc1", "dec1") in edges and ("enc2", "dec2") in edges


def test_sequential_topology():
    res = trace_model(TinyCNN(), torch.randn(1, 3, 32, 32), topology="sequential")
    assert res.graph.topology_source == "sequential"
    keys = _module_keys(res)
    for a, b in zip(keys, keys[1:]):
        assert any(e.src == a and e.dst == b for e in res.graph.edges)
