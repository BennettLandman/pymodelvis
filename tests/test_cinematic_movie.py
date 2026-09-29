import numpy as np
import torch
import torch.nn as nn

from conftest import DictOut, MultiInput, Tiny3D, TinyCNN, TinyUNet, TinyViT
from neural_flow import animate_inputs, summarize_tensor, trace_model, visualize_model
from neural_flow.capture import _small_copy
from neural_flow.config import FlowConfig
from neural_flow.sequences import crossfade, occlusion_sweep, pan, slices_to_frames, zoom


def test_cinematic_defaults_to_black_and_explains(tmp_path):
    p = tmp_path / "c.png"
    fig = visualize_model(TinyCNN(), torch.randn(1, 3, 32, 32), output=str(p), style="cinematic", dpi=40)
    res = fig.flow
    assert res.config.theme == "black"
    assert p.exists() and p.stat().st_size > 2000
    ex = res.explanation
    assert ex is not None and ex.units, "unit fields expected for spatial stages"
    assert any(k.startswith("out:") for k in ex.gradcam)
    assert ex.contributions, "linear head contributions expected"
    uf = next(iter(ex.units.values()))
    assert uf.erf.shape == (32, 32) and 0 < uf.rf_px <= 32


def test_receptive_field_grows_with_depth():
    class Deep(nn.Module):
        def __init__(self):
            super().__init__()
            self.s1 = nn.Conv2d(1, 4, 3, padding=1)
            self.s2 = nn.Sequential(nn.Conv2d(4, 4, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
            self.s3 = nn.Sequential(nn.Conv2d(4, 4, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(self.s3(self.s2(self.s1(x))).mean((2, 3)))

    torch.manual_seed(0)
    res = trace_model(Deep(), torch.rand(1, 1, 64, 64), explain=True, layers=["s1", "s2", "s3", "fc"])
    rf = [res.explanation.units[k].rf_px for k in ("s1", "s2", "s3")]
    assert rf[0] < rf[1] < rf[2], rf
    # dependency maps are defined on the predecessor's grid
    assert res.explanation.units["s2"].dep_map.shape == (64, 64)
    assert res.explanation.units["s3"].dep_map.shape == (32, 32)


def test_cinematic_all_families(tmp_path):
    cases = [
        (TinyUNet(), torch.randn(1, 1, 32, 32)),
        (TinyViT(), torch.randn(1, 3, 32, 32)),
        (Tiny3D(), torch.randn(1, 1, 16, 16, 16)),
        (DictOut(), torch.randn(1, 1, 16, 16)),
        (MultiInput(), {"image": torch.randn(1, 1, 16, 16), "clinical": torch.randn(1, 6)}),
    ]
    for i, (m, x) in enumerate(cases):
        for theme in ("black", "light"):
            p = tmp_path / f"c{i}_{theme}.png"
            visualize_model(m, x, output=str(p), style="cinematic", theme=theme, dpi=35)
            assert p.exists()


def test_cinematic_wraps_for_16x9(tmp_path):
    fig = visualize_model(TinyCNN(), torch.randn(1, 3, 32, 32), style="cinematic", figsize=(16, 9), dpi=30)
    assert tuple(np.round(fig.get_size_inches(), 2)) == (16, 9)


def test_force_channels_and_fixed_pca_basis():
    x = torch.randn(1, 32, 12, 12)
    cfg = FlowConfig()
    s1 = summarize_tensor(x, "activation", 10**7, cfg)
    basis = (s1.pca_loadings, s1.pca_mean)
    s2 = summarize_tensor(x * 1.0, "activation", 10**7, cfg, {"force": [5, 3, 9], "pca_basis": basis})
    assert s2.selections["_forced"] == [5, 3, 9] and s2.channel_ids == [3, 5, 9]
    np.testing.assert_allclose(s2.pca_maps, s1.pca_maps, rtol=1e-4, atol=1e-4)


def test_movie_gif(tmp_path):
    frames = torch.stack([torch.roll(torch.linspace(0, 1, 32 * 32).reshape(1, 32, 32).repeat(3, 1, 1), s, -1)
                          for s in range(0, 12, 4)])
    out = animate_inputs(TinyCNN(), frames, output=str(tmp_path / "m.gif"), dpi=30, figsize=(8, 4.5), progress=False,
                         frame_labels=["a", "b", "c"])
    from PIL import Image

    im = Image.open(out)
    assert im.n_frames == 3


def test_sequences_shapes():
    img = torch.rand(3, 40, 120)
    assert pan(img, window=40, steps=5).shape == (5, 3, 40, 40)
    assert crossfade(img, img * 0, steps=4).shape == (4, 3, 40, 120)
    assert zoom(img, steps=3, out_size=32).shape == (3, 3, 32, 32)
    assert occlusion_sweep(torch.rand(1, 32, 32), patch=16, stride=16).shape == (4, 1, 32, 32)
    assert slices_to_frames(torch.rand(8, 16, 16), axis=0).shape == (8, 1, 16, 16)


def test_large_dense_output_kept_as_labels():
    t = torch.randn(1, 5, 64, 64, 32)
    small = _small_copy(t, FlowConfig(), limit=1000)
    assert small.dtype == torch.int32 and small.shape == (1, 1, 64, 64, 32)
    assert torch.equal(small[0, 0], t[0].argmax(0).to(torch.int32))


def test_3d_token_grid_becomes_volume():
    s = summarize_tensor(torch.randn(1, 8 ** 3, 24), "activation", 10**7, FlowConfig(), {"input_ndim": 5})
    assert s.kind == "volume3d" and s.spatial == (8, 8, 8) and s.channels == 24
