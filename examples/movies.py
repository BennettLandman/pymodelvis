"""Movies over changing inputs (cinematic style, black background).

    python examples/movies.py pan      # real ResNet-50: camera pans across four photographs
    python examples/movies.py vit      # real ViT-B/16: the same pan through a transformer
    python examples/movies.py aging    # 3-D multi-head brain model: a synthetic subject ages, a lesion appears
    python examples/movies.py all      [--frames 48] [--fps 8] [--gif]

Real weights: ResNet-50 / ViT-B/16 are loaded from ./real_models (see
tools/checkout_real_models.py) or downloaded from GitHub releases.
"""
import argparse
import os
import sys

import numpy as np
import torch

from _common import OUT, out_path
from real_models import imagenet_classes, load_resnet50, load_vit, photo_panorama
from neural_flow import animate_inputs
from neural_flow.sequences import pan


def movie_pan(frames: int, fps: int, ext: str, which: str = "resnet"):
    if which == "vit":
        model, mean, std, name = load_vit() + ("ViT-B/16",)
    else:
        model, mean, std, name = load_resnet50() + ("ResNet-50",)
    pano = photo_panorama(height=256)
    x = torch.from_numpy(pano).permute(2, 0, 1)
    x = (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    seq = pan(x, window=256, steps=frames, out_size=224)
    return animate_inputs(model, seq, output=out_path(f"movie_pan_{which}.{ext}"), fps=fps,
                          class_names=imagenet_classes(), title=f"{name} · camera pan",
                          subtitle="a 256-px window slides across four photographs; pages keep the same channels, "
                                   "colours and scale in every frame")


def movie_aging(frames: int, fps: int, ext: str):
    from multihead import MultiTaskBrainNet, make_data, train
    from synthetic import brain_phantom

    model = MultiTaskBrainNet()
    ckpt = out_path("multihead_32_600steps.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
    else:
        train(model, 600, 32)
        torch.save(model.state_dict(), ckpt)
    model.eval()
    ages = np.linspace(25, 85, frames)
    radii = np.clip((np.arange(frames) - frames * 0.3) / (frames * 0.45), 0, 1) * 0.24
    clinical = torch.tensor([[1.0, 3.0, 0.0]])
    seq, labels = [], []
    for a, r in zip(ages, radii):
        v, _ = brain_phantom(32, float(a), float(r), seed=5)
        seq.append({"mri": torch.from_numpy(v)[None, None], "clinical": clinical})
        labels.append(f"true age {a:.0f}   ·   lesion radius {r * 16:.1f} vox")
    return animate_inputs(model, seq, output=out_path(f"movie_aging.{ext}"), fps=fps, frame_labels=labels,
                          title="Multi-head brain model · one synthetic subject over time",
                          subtitle="ventricles enlarge with age; a lesion appears and grows — "
                                   "brain age, lesion probability and the segmentation follow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["pan", "vit", "aging", "all"])
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    ext = "gif" if args.gif else "mp4"
    if args.which in ("pan", "all"):
        print(movie_pan(args.frames, args.fps, ext, "resnet"))
    if args.which in ("vit", "all"):
        print(movie_pan(args.frames, args.fps, ext, "vit"))
    if args.which in ("aging", "all"):
        print(movie_aging(max(24, args.frames * 3 // 4), args.fps, ext))


if __name__ == "__main__":
    main()
