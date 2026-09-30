"""Example — 3-D transformer segmentation networks (UNETR, SwinUNETR) on a whole head.

Both models are MONAI implementations trained here, briefly, on a synthetic
MRI-like head phantom with five labelled structures (scalp / skull, cortex,
white matter, ventricles, lesion; see ``synthetic.head_labels``).  They are
trained on 64³ windows, like real 3-D segmentation networks, so a whole head
(96 × 112 × 96 voxels of 1.6 mm) is segmented with sliding-window inference.

What the figure shows

* the transformer inside the network as real stages: UNETR's 12 blocks tapped at
  blocks 3, 6, 9, 12 (all at the 4 × 4 × 4 token grid); SwinUNETR's four
  hierarchical Swin levels (16³ → 2³, channels doubling at each level),
* the decoder climbing back to full resolution, drawn as a U with the taps as
  skip bridges,
* the one 64³ window that was traced (stages) and the whole-head segmentation
  fused from every window (output card, the traced window outlined),
* physical proportions from the voxel spacing.

Usage::

    python examples/transformer_3d.py                      # UNETR, cinematic figure
    python examples/transformer_3d.py --model swinunetr
    python examples/transformer_3d.py --flat               # optional squashed 2-D view of every stage
    python examples/transformer_3d.py --movie              # watch sliding-window inference, window by window

Training takes ~5 min (UNETR) / ~15 min (SwinUNETR) on a laptop CPU the first
time; the weights are cached in ``examples/outputs/``.

References: UNETR (Hatamizadeh et al., WACV 2022), Swin UNETR (Hatamizadeh et
al., BrainLes 2021 / Tang et al., CVPR 2022), MONAI (Cardoso et al., 2022).
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from _common import out_path
from synthetic import TISSUE_LABELS, head_labels
from neural_flow import visualize_model
from neural_flow.volume3d import sliding_window_movie

ROI = (64, 64, 64)
SPACING = (1.6, 1.6, 1.6)


def build(name: str) -> torch.nn.Module:
    from monai.networks.nets import SwinUNETR, UNETR

    if name == "unetr":
        return UNETR(in_channels=1, out_channels=len(TISSUE_LABELS), img_size=ROI, feature_size=8, hidden_size=192,
                     mlp_dim=768, num_heads=4)
    if name == "swinunetr":
        try:
            return SwinUNETR(in_channels=1, out_channels=len(TISSUE_LABELS), feature_size=12)
        except TypeError:  # MONAI < 1.5 needs img_size
            return SwinUNETR(img_size=ROI, in_channels=1, out_channels=len(TISSUE_LABELS), feature_size=12)
    raise ValueError(name)


def random_window(vol, lab, rng):
    X, Y, Z = vol.shape
    s = [int(rng.integers(0, n - r + 1)) for n, r in zip((X, Y, Z), ROI)]
    sl = tuple(slice(a, a + r) for a, r in zip(s, ROI))
    return vol[sl], lab[sl]


def train(model, steps: int, batch: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    heads = [head_labels(seed=100 + i) for i in range(12)]
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.15)
    w = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.5, 3.0])
    model.train()
    t0 = time.time()
    for s in range(steps):
        xs, ys = [], []
        for _ in range(batch):
            v, l = random_window(*heads[int(rng.integers(len(heads)))], rng)
            xs.append(v[None])
            ys.append(l)
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=w)
        p = logits.softmax(1)
        oh = F.one_hot(y, logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        dice = (2 * (p * oh).sum((0, 2, 3, 4)) + 1) / (p.sum((0, 2, 3, 4)) + oh.sum((0, 2, 3, 4)) + 1)
        loss = loss + (1 - dice[1:]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 25 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.3f}  dice {dice[1:].detach().numpy().round(2)}  ({time.time() - t0:.0f}s)")
    model.eval()


def load_or_train(name: str, steps: int) -> torch.nn.Module:
    model = build(name)
    ckpt = out_path(f"{name}_heads_{steps}steps.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
    else:
        print(f"training {name} for {steps} steps on synthetic heads (cached to {ckpt}) …")
        train(model, steps)
        # float16 halves the file; load_state_dict casts back to float32
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}, ckpt)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unetr", choices=["unetr", "swinunetr"])
    ap.add_argument("--steps", type=int, default=None, help="training steps (default 400 UNETR / 250 SwinUNETR)")
    ap.add_argument("--style", default="cinematic", choices=["cinematic", "technical", "story"])
    ap.add_argument("--flat", action="store_true", help="draw every 3-D stage as a squashed 2-D projection")
    ap.add_argument("--movie", action="store_true", help="render the sliding-window inference movie instead")
    ap.add_argument("--max-windows", type=int, default=None, help="movie: show at most this many windows")
    args = ap.parse_args()
    steps = args.steps or (400 if args.model == "unetr" else 250)
    model = load_or_train(args.model, steps)
    title = {"unetr": "UNETR", "swinunetr": "Swin UNETR"}[args.model]

    spacing = SPACING
    vol, lab = head_labels(seed=7)
    x = torch.from_numpy(vol)[None, None]

    if args.movie:
        path = out_path(f"movie_sliding_window_{args.model}.mp4")
        sliding_window_movie(model, x, ROI, output=path, overlap=0.25, max_windows=args.max_windows,
                             voxel_spacing=spacing, output_types={"output": "segmentation"},
                             class_names=TISSUE_LABELS, title=f"{title} · whole-head segmentation, one window at a time",
                             fps=3, hold_last=6)
        print("wrote", path)
        return

    suffix = ("_flat" if args.flat else "") + ("" if args.style == "cinematic" else f"_{args.style}")
    path = out_path(f"transformer3d_{args.model}{suffix}.png")
    fig = visualize_model(
        model, x, output=path, style=args.style, voxel_spacing=spacing, sliding_window=True, roi_size=ROI,
        sw_overlap=0.25, flat_3d=args.flat, output_types={"output": "segmentation"}, class_names=TISSUE_LABELS,
        title=f"{title} · synthetic head, 5 structures",
    )
    print(fig.flow.summary_table())
    print("wrote", path)


if __name__ == "__main__":
    main()
