import os

import numpy as np
import pytest
import torch

from conftest import DictOut, MultiInput, Tiny3D, TinyCNN, TinyUNet, TinyViT
from neural_flow import FlowConfig, summarize_tensor, visualize_model, visualize_tensor
from neural_flow.raster import activation_peak, attention_rollout, render_dvr
from neural_flow.tensors import infer_patch_grid

CFG = FlowConfig()


def test_channel_selection_strategies():
    x = torch.zeros(1, 32, 16, 16)
    x[0, 7] = torch.randn(16, 16) * 10          # highest variance / energy
    x[0, 3] = 5.0                              # constant: high energy + mean_abs, zero variance
    s = summarize_tensor(x, "activation", 10**7, FlowConfig(max_channels=4))
    assert s.selections["variance"][0] == 7
    assert 3 in s.selections["energy"][:2] and 7 in s.selections["energy"][:2]
    assert s.selections["mean_abs"][0] in (3, 7)
    assert s.selections["even"] == [0, 10, 21, 31]
    assert len(s.selections["pca"]) == 4
    assert s.pca_maps is not None and s.pca_maps.shape == (3, 16, 16)


def test_deterministic_summary():
    x = torch.randn(1, 64, 20, 20)
    a = summarize_tensor(x, "activation", 10**7, CFG)
    b = summarize_tensor(x, "activation", 10**7, CFG)
    assert a.selections == b.selections
    np.testing.assert_allclose(a.pca_maps, b.pca_maps)


def test_render_2d_mosaic():
    s = summarize_tensor(torch.randn(1, 128, 28, 28), "activation", 10**7, CFG)
    v = visualize_tensor(s, CFG)
    assert v.kind == "mosaic" and v.n_shown == 9
    assert v.image.ndim == 3 and v.image.shape[2] == 4
    assert np.isfinite(v.image).all()


def test_percentile_normalisation_resists_outliers():
    m = torch.randn(1, 1, 32, 32)
    m[0, 0, 0, 0] = 1e6
    s = summarize_tensor(m, "activation", 10**7, CFG)
    v = visualize_tensor(s, FlowConfig(max_channels=1))
    lum = v.image[..., :3].mean(-1)
    assert lum.std() > 0.1, "a single outlier must not flatten the map"


def test_render_3d_volume_modes():
    x = torch.randn(1, 16, 24, 24, 24)
    s = summarize_tensor(x, "activation", 10**7, CFG)
    assert s.kind == "volume3d"
    for mode in ("volume", "ortho", "montage", "projection"):
        v = visualize_tensor(s, CFG.updated(volume_mode=mode))
        assert v.image.shape[-1] == 4 and np.isfinite(v.image).all(), mode


def test_activation_driven_slice_selection():
    E = np.zeros((20, 20, 20), np.float32)
    E[14:17, 3:6, 9:12] = 5
    assert activation_peak(E) == (15, 4, 10)


def test_dvr_small_volume_voxel_blocks():
    V = np.random.RandomState(0).rand(4, 4, 4).astype(np.float32)
    img = render_dvr(V, size=64)
    assert img.shape == (64, 64, 4) and img[..., 3].max() > 0.5


def test_transformer_tensors():
    x = torch.randn(1, 197, 64)
    s = summarize_tensor(x, "activation", 10**7, CFG)
    assert s.kind == "tokens" and s.n_special == 1 and s.spatial == (14, 14)
    assert s.cls.shape == (1, 64)
    v = visualize_tensor(s, CFG)
    assert v.kind == "tokens" and "cls" in v.extras
    s2 = summarize_tensor(torch.randn(1, 39, 16), "activation", 10**7, CFG)  # no grid -> heatmap
    assert visualize_tensor(s2, CFG).kind == "heatmap"


def test_patch_grid_inference():
    assert infer_patch_grid(197, CFG) == (1, (14, 14))
    assert infer_patch_grid(196, CFG) == (0, (14, 14))
    assert infer_patch_grid(1 + 8 * 16, CFG, aspect=0.5) == (1, (8, 16))


def test_channels_last_swin_like():
    s = summarize_tensor(torch.randn(1, 14, 14, 96), "activation", 10**7, CFG, {"type_name": "SwinTransformerBlock"})
    assert s.kind == "image2d" and s.channels_last and s.channels == 96 and s.spatial == (14, 14)


def test_attention_rollout():
    n = 5
    A = [np.full((n, n), 1 / n) for _ in range(3)]
    R = attention_rollout(A)
    np.testing.assert_allclose(R.sum(-1), 1, atol=1e-6)


def test_unknown_tensor_generic():
    s = summarize_tensor(torch.randn(1, 2, 3, 4, 5, 6), "activation", 10**6, CFG)
    assert s.kind == "generic"
    assert visualize_tensor(s, CFG).image.ndim == 3


@pytest.mark.parametrize("ext", ["png", "svg", "pdf"])
def test_output_file_creation(tmp_path, ext):
    p = tmp_path / f"flow.{ext}"
    fig = visualize_model(TinyCNN(), torch.randn(1, 3, 32, 32), output=str(p), dpi=60)
    assert p.exists() and p.stat().st_size > 1000
    assert hasattr(fig, "flow")


def test_all_model_families_render(tmp_path):
    cases = [
        (TinyUNet(), torch.randn(1, 1, 32, 32)),
        (TinyViT(), torch.randn(1, 3, 32, 32)),
        (Tiny3D(), torch.randn(1, 1, 16, 16, 16)),
        (DictOut(), torch.randn(1, 1, 16, 16)),
        (MultiInput(), {"image": torch.randn(1, 1, 16, 16), "clinical": torch.randn(1, 6)}),
    ]
    for i, (m, x) in enumerate(cases):
        for style in ("technical", "story"):
            p = tmp_path / f"f{i}_{style}.png"
            visualize_model(m, x, output=str(p), dpi=50, style=style)
            assert p.exists()


def test_vertical_and_figsize(tmp_path):
    p = tmp_path / "v.png"
    fig = visualize_model(TinyCNN(), torch.randn(1, 3, 32, 32), output=str(p), layout="vertical", dpi=50)
    w, h = fig.get_size_inches()
    assert h > w
    fig = visualize_model(TinyCNN(), torch.randn(1, 3, 32, 32), figsize=(16, 9), dpi=50)
    assert tuple(np.round(fig.get_size_inches(), 3)) == (16, 9)


def test_output_semantics_not_invented():
    import torch.nn as nn

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4, 7)

        def forward(self, x):
            return self.l1(x)

    res = visualize_model(Plain(), torch.randn(1, 4), dpi=40).flow
    v = next(iter(res.views.values()))
    assert v.kind == "raw", "unnamed [B, K] output must not be presented as probabilities"


def test_output_interpreter_hook():
    def interp(name, t):
        return {"kind": "text", "headline": "custom!", "subline": name}

    res = visualize_model(DictOut(), torch.randn(1, 1, 16, 16), output_interpreter=interp, dpi=40).flow
    assert all(v.headline == "custom!" for v in res.views.values())
