"""3-D features: transformer-backbone U-Nets, voxel spacing, sliding windows, flat_3d, inference movies."""
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_flow import trace_model, visualize_model
from neural_flow.raster import canonical_fov, physical_fov, raycast
from neural_flow.volume3d import (crop, flatten_result, patch_grid, patch_starts, sliding_window_infer,
                                  sliding_window_movie)


class _ViT3D(nn.Module):
    """Patch embedding + 4 transformer blocks over a 4³ token grid; returns every hidden state."""

    def __init__(self, dim=32):
        super().__init__()
        self.patch_embedding = nn.Conv3d(1, dim, 8, stride=8)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, 4, 64, batch_first=True, dropout=0.0)
                                     for _ in range(4)])

    def forward(self, x):
        z = self.patch_embedding(x).flatten(2).transpose(1, 2)
        hidden = []
        for blk in self.blocks:
            z = blk(z)
            hidden.append(z)
        return hidden


class TinyTransUNet3D(nn.Module):
    """UNETR-like: transformer backbone tapped after blocks 1 and 3, convolutional decoder."""

    def __init__(self, dim=32, n_classes=3):
        super().__init__()
        self.vit = _ViT3D(dim)
        self.enc_in = nn.Sequential(nn.Conv3d(1, 8, 3, padding=1), nn.ReLU())
        self.proj1 = nn.Sequential(nn.ConvTranspose3d(dim, 16, 2, 2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose3d(dim, 16, 2, 2), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose3d(32, 8, 4, 4), nn.ReLU())
        self.out = nn.Conv3d(16, n_classes, 1)

    def forward(self, x):
        hidden = self.vit(x)
        B = x.shape[0]
        g = lambda z: z.transpose(1, 2).reshape(B, -1, 4, 4, 4)
        d2 = self.dec2(g(hidden[3]))                                    # 8³
        d1 = self.dec1(torch.cat([d2, self.proj1(g(hidden[1]))], 1))    # 32³
        return self.out(torch.cat([d1, self.enc_in(x)], 1))


def _tiny_trans_unet():
    torch.manual_seed(0)
    return TinyTransUNet3D().eval()


class TinySeg3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.Conv3d(4, 8, 3, stride=2, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose3d(8, 4, 2, 2)
        self.seg = nn.Conv3d(8, 3, 1)

    def forward(self, x):
        e = self.enc(x)
        return self.seg(torch.cat([self.up(self.down(e)), e], 1))


@pytest.fixture
def tiny3d():
    torch.manual_seed(0)
    return TinySeg3D().eval()


def test_transformer_backbone_levels_are_stages():
    net = _tiny_trans_unet()
    res = trace_model(net, torch.randn(1, 1, 32, 32, 32), output_types={"output": "segmentation"})
    names = [s.call.name for s in res.graph.ordered() if s.call is not None]
    assert "transformer-unet" in res.adapters
    assert any(n.startswith("vit.blocks.") for n in names), names
    assert "vit.blocks.1" in names and "vit.blocks.3" in names      # the two tapped blocks
    assert "proj1" not in names                                     # projection branch folded into a skip
    concepts = {s.call.name: s.concept for s in res.graph.ordered() if s.call is not None}
    assert concepts["vit.blocks.3"] == "TRANSFORMER ENCODER"
    assert concepts["dec1"] == "DECODER"
    skips = {(e.src, e.dst) for e in res.graph.edges if e.kind == "skip"}
    assert ("vit.blocks.1", "dec1") in skips


def test_monai_unetr_and_swin_unetr():
    monai = pytest.importorskip("monai")
    from monai.networks.nets import SwinUNETR, UNETR

    u = UNETR(in_channels=1, out_channels=2, img_size=(32, 32, 32), feature_size=4, hidden_size=48, mlp_dim=96,
              num_heads=4).eval()
    res = trace_model(u, torch.randn(1, 1, 32, 32, 32), output_types={"output": "segmentation"})
    names = [s.call.name for s in res.graph.ordered() if s.call is not None]
    assert {"vit.blocks.3", "vit.blocks.6", "vit.blocks.9", "vit.blocks.11"} <= set(names)
    try:
        sw = SwinUNETR(in_channels=1, out_channels=2, feature_size=12).eval()
    except TypeError:
        sw = SwinUNETR(img_size=(64, 64, 64), in_channels=1, out_channels=2, feature_size=12).eval()
    res = trace_model(sw, torch.randn(1, 1, 64, 64, 64), output_types={"output": "segmentation"}, explain=False)
    names = [s.call.name for s in res.graph.ordered() if s.call is not None]
    assert any(n.startswith("swinViT.layers4") for n in names), names
    assert res.graph.stages[[s.key for s in res.graph.ordered() if s.call is not None and
                             s.call.name.startswith("swinViT.layers2")][0]].concept == "TRANSFORMER ENCODER"


def test_plain_vit_classifier_is_not_a_transformer_unet():
    from conftest import TinyViT

    res = trace_model(TinyViT().eval(), torch.randn(1, 3, 32, 32))
    assert "transformer-unet" not in res.adapters


def test_voxel_spacing_sets_physical_fov(tiny3d):
    x = torch.randn(1, 1, 16, 16, 8)
    res = trace_model(tiny3d, x, voxel_spacing=(1.0, 1.0, 2.0), explain=False)
    assert res.config.physical_fov == (16.0, 16.0, 16.0)
    assert canonical_fov((10, 20, 30), (1, 2, 3), "dhw") == (90.0, 40.0, 10.0)
    # the renderer uses the physical extent: a 16x16x8 block with 2 mm z-spacing draws as a cube
    a = np.ones((16, 16, 8), np.float32) * 0.5
    rgb = np.ones((16, 16, 8, 3), np.float32)
    flat = raycast(a, rgb, a.shape, 64, wire=False, shade=False)
    with physical_fov((16, 16, 16)):
        cube = raycast(a, rgb, a.shape, 64, wire=False, shade=False)
    assert cube[..., 3].sum() > flat[..., 3].sum() * 1.15


def test_patch_grid_covers_volume():
    assert patch_starts(100, 48, 0.25) == [0, 36, 52]
    g = patch_grid((64, 64, 40), (32, 32, 32), 0.5)
    assert len(g) == 3 * 3 * 2
    assert crop(torch.zeros(1, 1, 20, 20, 20), (10, 10, 10), (16, 16, 16)).shape == (1, 1, 16, 16, 16)


def test_sliding_window_matches_direct_for_pointwise_model():
    net = nn.Conv3d(1, 3, 1).eval()
    x = torch.randn(1, 1, 40, 36, 30)
    st = sliding_window_infer(net, x, (16, 16, 16), 0.25)
    direct = net(x).argmax(1, keepdim=True).to(torch.int32)
    assert torch.equal(st.fused("output"), direct)


def test_sliding_window_coarse_accumulation_for_many_classes():
    net = nn.Conv3d(1, 40, 1).eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with pytest.warns(UserWarning, match="coarser grid"):
        st = sliding_window_infer(net, x, (16, 16, 16), 0.25, max_mb=1.0)
    assert st.factor["output"] > 1
    assert st.fused("output").shape == (1, 1, 32, 32, 32)


def test_sliding_window_trace_shows_whole_volume_output(tiny3d, tmp_path):
    x = torch.randn(1, 1, 24, 24, 16)
    fig = visualize_model(tiny3d, x, str(tmp_path / "sw.png"), style="cinematic", sliding_window=True,
                          roi_size=(16, 16, 16), voxel_spacing=(1, 1, 1.5), output_types={"output": "segmentation"},
                          dpi=40)
    res = fig.flow
    assert res.graph.stages["input:input"].summary.shape[2:] == (16, 16, 16)          # traced window
    ov = next(v for v in res.views.values() if v.kind == "segmentation")
    assert ov.mask.shape == (24, 24, 16)                                                # whole volume
    assert res.context["sliding_window"]["windows"] == len(patch_grid((24, 24, 16), (16, 16, 16), 0.25))
    assert "windows" in ov.subline


def test_flat_3d_turns_volumes_into_2d(tiny3d, tmp_path):
    x = torch.randn(1, 1, 16, 16, 16)
    fig = visualize_model(tiny3d, x, str(tmp_path / "flat.png"), style="cinematic", flat_3d=True,
                          output_types={"output": "segmentation"}, dpi=40)
    res = fig.flow
    kinds = {s.summary.kind for s in res.graph.stages.values() if s.summary is not None}
    assert "volume3d" not in kinds and "image2d" in kinds
    ov = next(v for v in res.views.values() if v.kind == "segmentation")
    assert ov.mask.ndim == 2
    res3 = trace_model(tiny3d, x, explain=False)
    assert any(s.summary.kind == "volume3d" for s in res3.graph.stages.values() if s.summary is not None)


def test_sliding_window_movie(tiny3d, tmp_path):
    x = torch.randn(1, 1, 24, 24, 16)
    out = sliding_window_movie(tiny3d, x, (16, 16, 16), str(tmp_path / "inf.gif"), max_windows=3, dpi=30,
                               figsize=(8, 4.5), output_types={"output": "segmentation"}, explain=False, progress=False)
    assert os.path.exists(out)


def test_cli_sliding_window_and_spacing_from_nifti(tmp_path):
    nib = pytest.importorskip("nibabel")
    from neural_flow.cli import main

    net = tmp_path / "net3d.py"
    net.write_text("import torch.nn as nn\n"
                   "class Net(nn.Module):\n"
                   "    def __init__(self):\n"
                   "        super().__init__()\n"
                   "        self.a = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU())\n"
                   "        self.b = nn.Sequential(nn.Conv3d(4, 8, 3, stride=2, padding=1), nn.ReLU())\n"
                   "        self.up = nn.ConvTranspose3d(8, 4, 2, 2)\n"
                   "        self.seg = nn.Conv3d(4, 3, 1)\n"
                   "    def forward(self, x):\n"
                   "        return self.seg(self.up(self.b(self.a(x))))\n")
    vol = np.random.rand(24, 24, 12).astype(np.float32)
    img = nib.Nifti1Image(vol, np.diag([1.0, 1.0, 2.5, 1.0]))
    p = tmp_path / "scan.nii.gz"
    nib.save(img, str(p))
    out = tmp_path / "o.png"
    rc = main(["render", f"{net}:Net", "-i", str(p), "--sliding-window", "--roi", "16", "--style", "cinematic",
               "--output-type", "output=segmentation", "-o", str(out), "--dpi", "35", "-q", "--no-explain"])
    assert rc == 0 and out.exists()
    out2 = tmp_path / "f.png"
    assert main(["render", f"{net}:Net", "-i", str(p), "--flat-3d", "-o", str(out2), "--dpi", "35", "-q",
                 "--no-explain"]) == 0
