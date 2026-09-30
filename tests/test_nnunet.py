"""nnU-Net: stage selection for dynamic_network_architectures U-Nets, results-folder loading, preprocessing."""
import glob
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

dna = pytest.importorskip("dynamic_network_architectures")
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet  # noqa: E402

from neural_flow import trace_model, visualize_model  # noqa: E402

KW = dict(input_channels=1, n_stages=4, features_per_stage=[8, 16, 32, 32], conv_op=nn.Conv3d,
          kernel_sizes=[[3, 3, 3]] * 4, strides=[[1, 1, 1]] + [[2, 2, 2]] * 3, num_classes=3, conv_bias=True,
          norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
          nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True})


def _plain(ds=False):
    torch.manual_seed(0)
    return PlainConvUNet(n_conv_per_stage=[2] * 4, n_conv_per_stage_decoder=[2] * 3, deep_supervision=ds, **KW).eval()


def _resenc():
    torch.manual_seed(0)
    return ResidualEncoderUNet(n_blocks_per_stage=[1, 2, 2, 2], n_conv_per_stage_decoder=[1] * 3,
                               deep_supervision=False, **KW).eval()


def _labels(res):
    return [s.label for s in res.stages if s.kind == "module"]


def test_plain_unet_levels_and_skips():
    res = trace_model(_plain(), torch.randn(1, 1, 32, 32, 32), explain=False)
    assert "nnunet" in res.adapters
    assert _labels(res) == ["encoder 1", "encoder 2", "encoder 3", "bottleneck",
                            "decoder 3", "decoder 2", "decoder 1", "segmentation"]
    skips = {(e.src, e.dst) for e in res.graph.edges if e.kind == "skip"}
    for i, s in ((0, 2), (1, 1), (2, 0)):
        assert (f"encoder.stages.{i}", f"decoder.stages.{s}") in skips
    concepts = {s.label: s.concept for s in res.stages}
    assert concepts["bottleneck"] == "BOTTLENECK" and concepts["decoder 2"] == "DECODER"
    assert concepts["segmentation"] == "SEGMENTATION HEAD"
    assert res.config.unet_layout


def test_deep_supervision_switched_off_and_restored():
    m = _plain(ds=True)
    res = trace_model(m, torch.randn(1, 1, 32, 32, 32), explain=False)
    assert len(res.views) == 1                              # only the full-resolution prediction
    assert _labels(res)[:4] == ["encoder 1", "encoder 2", "encoder 3", "bottleneck"]
    assert m.decoder.deep_supervision is True                # model restored
    assert any("deep supervision" in n for n in res.notes)
    aux = trace_model(m, torch.randn(1, 1, 32, 32, 32), explain=False, aux_outputs=True)
    assert len(aux.views) == 3


def test_residual_encoder_has_stem():
    res = trace_model(_resenc(), torch.randn(1, 1, 32, 32, 32), explain=False)
    labs = _labels(res)
    assert labs[0] == "stem" and labs[-1] == "segmentation"
    assert sum(l.startswith("decoder") for l in labs) == 3


# ---------------------------------------------------------------------------
# results folders
# ---------------------------------------------------------------------------

PLANS_OLD = {
    "dataset_name": "Dataset900_Tiny", "plans_name": "nnUNetPlans", "transpose_forward": [0, 1, 2],
    "transpose_backward": [0, 1, 2],
    "foreground_intensity_properties_per_channel": {"0": {"mean": 50.0, "std": 100.0, "percentile_00_5": -500.0,
                                                          "percentile_99_5": 800.0}},
    "configurations": {"3d_fullres": {
        "patch_size": [16, 32, 32], "spacing": [2.0, 1.0, 1.0], "normalization_schemes": ["CTNormalization"],
        "use_mask_for_norm": [False], "UNet_class_name": "PlainConvUNet", "UNet_base_num_features": 8,
        "n_conv_per_stage_encoder": [2, 2, 2], "n_conv_per_stage_decoder": [2, 2], "num_pool_per_axis": [1, 2, 2],
        "pool_op_kernel_sizes": [[1, 1, 1], [1, 2, 2], [2, 2, 2]], "conv_kernel_sizes": [[3, 3, 3]] * 3,
        "unet_max_num_features": 32}},
}


def _new_format(p):
    p = json.loads(json.dumps(p))
    c = p["configurations"]["3d_fullres"]
    c["architecture"] = {
        "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "arch_kwargs": {"n_stages": 3, "features_per_stage": [8, 16, 32], "conv_op": "torch.nn.modules.conv.Conv3d",
                        "kernel_sizes": [[3, 3, 3]] * 3, "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2]],
                        "n_conv_per_stage": [2, 2, 2], "n_conv_per_stage_decoder": [2, 2], "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-5, "affine": True}, "dropout_op": None,
                        "dropout_op_kwargs": None, "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True}},
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]}
    for k in ("UNet_class_name", "UNet_base_num_features", "n_conv_per_stage_encoder", "n_conv_per_stage_decoder",
              "num_pool_per_axis", "pool_op_kernel_sizes", "conv_kernel_sizes", "unet_max_num_features"):
        del c[k]
    return p


def _results(tmp_path, plans, ds=True):
    from neural_flow.nnunet import build_network

    dataset = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "liver": 1, "spleen": 2},
               "numTraining": 1, "file_ending": ".nii.gz"}
    rd = tmp_path / "Dataset900_Tiny" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    (rd / "fold_0").mkdir(parents=True)
    json.dump(plans, open(rd / "plans.json", "w"))
    json.dump(dataset, open(rd / "dataset.json", "w"))
    torch.manual_seed(1)
    net = build_network(plans, "3d_fullres", dataset, deep_supervision=ds)   # trained with deep supervision
    torch.save({"network_weights": net.state_dict(), "trainer_name": "nnUNetTrainer",
                "init_args": {"fold": 0}}, rd / "fold_0" / "checkpoint_final.pth")
    return rd, net


def _nifti(path, shape=(40, 36, 20), zooms=(1.0, 1.0, 2.0), flip_x=False):
    nib = pytest.importorskip("nibabel")
    a = np.full(shape, -1000.0, np.float32)
    a[8:32, 6:30, 4:16] = 60.0
    a[10:14, 8:12, 6:10] = 400.0                               # a bright corner to check orientation
    aff = np.diag(list(zooms) + [1.0])
    if flip_x:                                                 # stored L→R reversed (LAS)
        a = a[::-1]
        aff[0, 0] = -zooms[0]
        aff[0, 3] = zooms[0] * (shape[0] - 1)
    nib.save(nib.Nifti1Image(a, aff), str(path))
    return path


@pytest.mark.parametrize("fmt", ["old", "new"])
def test_load_results_folder(tmp_path, fmt):
    from neural_flow.nnunet import load_nnunet

    plans = PLANS_OLD if fmt == "old" else _new_format(PLANS_OLD)
    rd, net = _results(tmp_path, plans)
    for spec in (str(rd), str(rd.parent), f"{rd}:0"):
        lm = load_nnunet(spec)
        assert lm.roi == (16, 32, 32) and lm.spacing == (2.0, 1.0, 1.0) and lm.volume_axes == "zyx"
        assert lm.class_names == ["background", "liver", "spleen"]
        assert lm.sw_overlap == 0.5 and lm.info["configuration"] == "3d_fullres"
        assert lm.model.decoder.deep_supervision is False
        sd = lm.model.state_dict()
        assert all(torch.equal(sd[k], v) for k, v in net.state_dict().items())
    with pytest.raises(FileNotFoundError):
        load_nnunet(f"{rd}:3")


def test_preprocessing_orientation_and_spacing(tmp_path):
    from neural_flow.nnunet import NNUNetPreprocessor

    p = NNUNetPreprocessor(PLANS_OLD, "3d_fullres")
    a = p(str(_nifti(tmp_path / "ras.nii.gz")))
    b = p(str(_nifti(tmp_path / "las.nii.gz", flip_x=True)))
    # [C, z, y, x]; z 20 slices at 2 mm stay 20, x/y at 1 mm unchanged
    assert a["image"].shape == (1, 20, 36, 40) and a["spacing"] == (2.0, 1.0, 1.0)
    np.testing.assert_allclose(a["image"], b["image"], atol=1e-5)          # reoriented to RAS
    img = a["image"][0]
    assert img.max() == pytest.approx((400 - 50) / 100.0) and img.min() == pytest.approx((-500 - 50) / 100.0)
    z, y, x = np.unravel_index(np.argmax(img), img.shape)
    assert 10 <= x < 14 and 8 <= y < 12 and 6 <= z < 10


def test_preprocessing_resamples(tmp_path):
    from neural_flow.nnunet import NNUNetPreprocessor

    p = NNUNetPreprocessor(PLANS_OLD, "3d_fullres")
    out = p(str(_nifti(tmp_path / "fine.nii.gz", shape=(40, 36, 40), zooms=(0.5, 1.0, 1.0))))
    assert out["image"].shape == (1, 20, 36, 20)


def test_render_nnunet_sliding_window_cli(tmp_path):
    from neural_flow.cli import main

    rd, _ = _results(tmp_path, PLANS_OLD)
    vol = _nifti(tmp_path / "ct.nii.gz", shape=(70, 40, 40), zooms=(1.0, 1.0, 2.0))
    out = tmp_path / "nn.png"
    assert main(["render", f"nnunet:{rd}", "-i", str(vol), "--sliding-window", "--style", "cinematic",
                 "--dpi", "40", "--no-explain", "-o", str(out), "-q"]) == 0
    assert out.exists() and out.stat().st_size > 10_000


def test_sliding_window_trace_nnunet(tmp_path):
    from neural_flow.zoo import load_input, load_model

    rd, _ = _results(tmp_path, PLANS_OLD)
    lm = load_model(f"nnunet:{rd}")
    info = {}
    x = load_input(str(_nifti(tmp_path / "ct.nii.gz", shape=(70, 40, 40), zooms=(1.0, 1.0, 2.0))), lm,
                   info=info, whole=True)
    assert tuple(x.shape) == (1, 1, 40, 40, 70) and info["spacing"] == (2.0, 1.0, 1.0)
    res = trace_model(lm.model, x, sliding_window=True, roi_size=lm.roi, sw_overlap=lm.sw_overlap,
                      volume_axes="zyx", voxel_spacing=info["spacing"], explain=False)
    sw = res.context["sliding_window"]
    assert sw["windows"] == 4 * 2 * 4                     # 50 % overlap of a 16×32×32 patch
    ov = next(iter(res.views.values()))
    assert ov.mask.shape == (40, 40, 70)
    # canonical field of view: x = 70 mm, y = 40 mm, z = 40 slices × 2 mm
    assert res.context["seg_fov"] == pytest.approx((70.0, 40.0, 80.0))


# ---------------------------------------------------------------------------
# the real TotalSegmentator model (only when downloaded: `neural-flow fetch totalseg`)
# ---------------------------------------------------------------------------


def _real():
    from neural_flow.nnunet import find_results
    from neural_flow.zoo import find_file

    try:
        rd = find_results("totalseg", download=False)
    except FileNotFoundError:
        return None
    ct = find_file("example_ct.nii.gz") or find_file(os.path.join("totalseg", "example_ct.nii.gz"))
    ref = find_file(os.path.join("totalseg", "example_seg.nii.gz"))
    return (rd, ct, ref) if ct and ref else None


@pytest.mark.skipif(_real() is None, reason="TotalSegmentator weights / example CT not downloaded")
def test_totalsegmentator_matches_reference():
    """Our loader + preprocessing + sliding window reproduces TotalSegmentator's own reference output."""
    import nibabel as nib

    from neural_flow.nnunet import load_nnunet
    from neural_flow.volume3d import sliding_window_infer

    rd, ct, ref_p = _real()
    lm = load_nnunet(rd)
    x = torch.from_numpy(lm.preprocess({"image": ct})["image"])[None]
    seg = sliding_window_infer(lm.model, x, lm.roi, 0.5).fused("output").numpy().squeeze().astype(int)
    pred = seg.transpose(2, 1, 0)                              # [z, y, x] → RAS [x, y, z]
    full = nib.load(ct).get_fdata()
    ref = nib.load(ref_p).get_fdata().astype(int)              # reference for slices z0 … z0+30 of the example CT
    sm = nib.load(os.path.join(os.path.dirname(ref_p), "example_ct_sm.nii.gz")).get_fdata()
    z0 = next(z for z in range(full.shape[2] - sm.shape[2] + 1) if np.allclose(full[:, :, z:z + sm.shape[2]], sm))
    p = pred[:, :, z0:z0 + ref.shape[2]]
    for label in (1, 2, 3, 5):                                  # spleen, kidneys, liver
        a, b = p == label, ref == label
        assert 2 * (a & b).sum() / (a.sum() + b.sum()) > 0.9
