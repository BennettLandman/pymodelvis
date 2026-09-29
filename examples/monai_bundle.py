"""Real 3-D medical example — any MONAI bundle (default: MASI's UNesT whole-brain segmentation).

The bundle's own ``configs/inference.json`` is used to build the network
(``network_def``), load ``models/model.pt`` and run the bundle's preprocessing, so
the same script works for other MONAI Model-Zoo bundles.

1. fetch the bundle + an input on a machine with internet access::

       python tools/checkout_real_models.py --only unest [--t1 /path/to/your_T1_in_MNI.nii.gz]

2. render::

       python examples/monai_bundle.py                       # cinematic figure (black)
       python examples/monai_bundle.py --movie               # the ROI window sweeps inferior → superior
       python examples/monai_bundle.py --image my_t1.nii.gz --bundle-dir path/to/other_bundle

Requires ``pip install "monai[nibabel]"``.  UNesT: Yu et al., Medical Image Analysis 2023.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

from _common import out_path
from real_models import ROOT
from neural_flow import animate_inputs, visualize_model

DEFAULT_BUNDLE = os.path.join(ROOT, "monai_bundles", "wholeBrainSeg_Large_UNEST_segmentation")


def _config_path(bdir):
    for name in ("inference.json", "inference.yaml", "inference.yml"):
        p = os.path.join(bdir, "configs", name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no configs/inference.* in {bdir}")


def load_bundle(bdir: str):
    """Instantiate a MONAI bundle's network from its own config and load its weights."""
    from monai.bundle import ConfigParser

    sys.path.insert(0, bdir)                      # bundle-local modules (e.g. scripts/networks/…)
    parser = ConfigParser()
    parser.read_config(_config_path(bdir))
    parser["bundle_root"] = bdir
    net = parser.get_parsed_content("network_def", instantiate=True)
    wts = sorted(glob.glob(os.path.join(bdir, "models", "*.pt")) + glob.glob(os.path.join(bdir, "models", "*.pth")))
    wts = [w for w in wts if not w.endswith(".ts")]
    if wts:
        sd = torch.load(wts[0], map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"loaded {os.path.basename(wts[0])}: {len(missing)} missing / {len(unexpected)} unexpected keys")
    roi = None
    try:
        inf = parser.get("inferer")
        roi = tuple(inf.get("roi_size")) if isinstance(inf, dict) and inf.get("roi_size") else None
    except Exception:
        pass
    pre = None
    try:
        pre = parser.get_parsed_content("preprocessing", instantiate=True)
    except Exception as e:
        print("bundle preprocessing unavailable, using z-score:", str(e)[:120])
    meta = {}
    mp = os.path.join(bdir, "configs", "metadata.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    return net.eval(), roi, pre, meta


def load_volume(path: str, pre=None) -> torch.Tensor:
    """[C, X, Y, Z] tensor, through the bundle's preprocessing when possible."""
    if pre is not None:
        try:
            d = pre({"image": path})
            img = d["image"] if isinstance(d, dict) else d[0]["image"]
            return torch.as_tensor(np.asarray(img), dtype=torch.float32)
        except Exception as e:
            print("bundle preprocessing failed, using z-score:", str(e)[:160])
    import nibabel as nib

    v = np.asarray(nib.load(path).get_fdata(), np.float32)
    m = v > np.percentile(v, 20)
    v = (v - v[m].mean()) / (v[m].std() + 1e-6)
    return torch.from_numpy(v)[None]


def crop(vol: torch.Tensor, roi, center=None) -> torch.Tensor:
    """ROI-sized crop [1, C, *roi] centred on ``center`` (default: centre of mass of the head)."""
    C, *S = vol.shape
    v = vol[0].numpy()
    if center is None:
        w = np.clip(v - np.percentile(v, 50), 0, None)
        idx = np.indices(v.shape).reshape(3, -1)
        center = (idx * w.reshape(-1)).sum(1) / max(w.sum(), 1e-9)
    sl = []
    for c, r, n in zip(center, roi, S):
        s = int(np.clip(round(c - r / 2), 0, max(n - r, 0)))
        sl.append(slice(s, s + r))
    out = vol[:, sl[0], sl[1], sl[2]]
    pad = [(0, r - o) for r, o in zip(roi, out.shape[1:])]
    if any(p[1] > 0 for p in pad):
        out = torch.nn.functional.pad(out, [x for p in reversed(pad) for x in p])
    return out[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default=DEFAULT_BUNDLE)
    ap.add_argument("--image", default=None)
    ap.add_argument("--roi", type=int, default=None, help="cube size of the network input (default: bundle roi_size)")
    ap.add_argument("--movie", action="store_true")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--no-explain", action="store_true")
    args = ap.parse_args()
    if not os.path.isdir(args.bundle_dir):
        sys.exit(f"bundle not found at {args.bundle_dir} — run tools/checkout_real_models.py --only unest first")
    image = args.image
    if image is None:
        for cand in ("user_t1.nii.gz", "user_t1.nii", "mni152_t1_1mm.nii.gz"):
            if os.path.exists(os.path.join(ROOT, cand)):
                image = os.path.join(ROOT, cand)
                break
    net, roi, pre, meta = load_bundle(args.bundle_dir)
    roi = (args.roi,) * 3 if args.roi else (roi or (96, 96, 96))
    vol = load_volume(image, pre)
    name = meta.get("name") or os.path.basename(args.bundle_dir)
    kw = dict(style="cinematic", output_types={"output": "segmentation"}, explain=not args.no_explain,
              volume_axes="xyz", title=name.replace("_", " "))
    if not args.movie:
        x = crop(vol, roi)
        path = out_path("monai_bundle_cinematic.png")
        fig = visualize_model(net, x, output=path, subtitle=f"{os.path.basename(image)} · ROI {roi[0]}³ at the head's centre "
                                                            f"· {meta.get('version', '')}", **kw)
        print(fig.flow.summary_table())
        print("wrote", path)
    else:
        S = vol.shape[1:]
        zc = np.linspace(roi[2] / 2, S[2] - roi[2] / 2, args.frames)
        frames = [crop(vol, roi, center=(S[0] / 2, S[1] / 2, z)) for z in zc]
        labels = [f"ROI centre z = {z:.0f} mm" for z in zc]
        path = animate_inputs(net, frames, output=out_path("movie_monai_bundle_sweep.mp4"), fps=6,
                              frame_labels=labels, subtitle="the network's input window sweeps from inferior to superior",
                              **kw)
        print("wrote", path)


if __name__ == "__main__":
    main()
