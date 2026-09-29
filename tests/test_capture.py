import torch
import torch.nn as nn

from conftest import DictOut, MultiInput, Reuse, TinyCNN, TupleOut
from neural_flow import FlowConfig, trace_model
from neural_flow.capture import flatten_tensors, prepare_inputs, trace_metadata


def _hook_count(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_hooks_removed_and_mode_restored():
    m = TinyCNN().train()
    before = _hook_count(m)
    trace_model(m, torch.randn(1, 3, 32, 32))
    assert _hook_count(m) == before == 0
    assert m.training, "train/eval state must be restored"


def test_hooks_removed_on_exception():
    class Boom(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 3, 1)

        def forward(self, x):
            self.c(x)
            raise RuntimeError("boom")

    m = Boom()
    try:
        trace_model(m, torch.randn(1, 3, 8, 8))
    except RuntimeError:
        pass
    assert _hook_count(m) == 0


def test_model_output_unchanged():
    m = TinyCNN().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ref = m(x)
    trace_model(m, x)
    with torch.no_grad():
        assert torch.allclose(m(x), ref)
    assert not hasattr(x, "_nf_node"), "user input tensors must not be tagged"


def test_nested_modules_parent_child():
    m = TinyCNN()
    cfg = FlowConfig()
    prep = prepare_inputs(m, torch.randn(1, 3, 32, 32), cfg)
    tr = trace_metadata(m, prep, cfg)
    names = {c.name: c for c in tr.calls.values()}
    assert names["stem.0"].parent == names["stem"].id
    assert names["stem"].id in tr.calls[tr.root].children
    assert names["stem.0"].type_name == "Conv2d"
    assert names["stem.0"].out_shapes[0] == (1, 8, 32, 32)
    order = [c.name for c in tr.calls_in_order()]
    assert order.index("stem") < order.index("block1") < order.index("fc")


def test_tuple_outputs():
    res = trace_model(TupleOut(), torch.randn(1, 3, 16, 16), layers=["split", "conv"])
    st = res.graph.stages["split"]
    assert st.summary is not None and st.summary.kind == "image2d"
    assert len(res.capture.extra_outputs["split"]) == 1
    assert len(res.views) == 2


def test_dict_outputs_and_names():
    res = trace_model(DictOut(), torch.randn(1, 1, 16, 16))
    names = {v.name for v in res.views.values()}
    assert names == {"diagnosis", "age"}
    kinds = {v.name: v.kind for v in res.views.values()}
    assert kinds["diagnosis"] == "classification"
    assert kinds["age"] == "regression"


def test_multiple_inputs_dict():
    res = trace_model(MultiInput(), {"image": torch.randn(1, 1, 16, 16), "clinical": torch.randn(1, 6)})
    inputs = [s for s in res.graph.stages.values() if s.kind == "input"]
    assert {s.label for s in inputs} == {"image", "clinical"}
    fuse_preds = {e.src for e in res.graph.preds("fuse")}
    assert len(fuse_preds) == 2, fuse_preds


def test_reused_module_calls():
    res = trace_model(Reuse(), torch.randn(1, 3, 8, 8), layers=["conv"])
    keys = [s.key for s in res.graph.ordered() if s.kind == "module"]
    assert keys == ["conv", "conv#1", "conv#2"]
    # each call is chained to the previous one
    assert {e.src for e in res.graph.preds("conv#2")} == {"conv#1"}


def test_flatten_tensors_nested():
    obj = {"a": torch.zeros(1), "b": [torch.ones(2), (torch.ones(3), 5)]}
    paths = [p for p, _ in flatten_tensors(obj)]
    assert paths == ["a", "b.0", "b.1.0"]


def test_memory_limit_reduces_large_activation():
    class Big(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv3d(1, 64, 3, padding=1)
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            return self.fc(self.c(x).mean((2, 3, 4)))

    res = trace_model(Big(), torch.randn(1, 1, 64, 64, 64), max_capture_mb=4, max_channels=4)
    s = res.graph.stages["c"].summary
    assert s.reduced
    assert s.maps.nbytes < 4 * 1024 * 1024
    assert max(s.reduced_spatial) <= 48
    assert any("summarised" in n for n in res.notes)
