"""Example 4 — 3-D medical imaging: a volumetric U-Net for lesion segmentation.

Input: a synthetic MRI-like head volume ``[1, 1, X, Y, Z]`` (nibabel/MONAI axis
order).  ``[B, C, X, Y, Z]`` tensors are first-class: the input is shown as a
translucent anatomical volume with orthogonal slices through its centre of
mass, feature maps become stacks of translucent voxel blocks (increasingly
coarse and abstract with depth) with activation-driven orthogonal slices, and
the output is a 3-D label render over the anatomy.

    python examples/medical_3d.py [--steps 120] [--mode volume|ortho|montage|projection]
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import out_path
from synthetic import brain_3d
from neural_flow import visualize_model


def block(i, o):
    return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.InstanceNorm3d(o, affine=True), nn.LeakyReLU(0.1, inplace=True),
                         nn.Conv3d(o, o, 3, padding=1), nn.InstanceNorm3d(o, affine=True), nn.LeakyReLU(0.1, inplace=True))


class UNet3D(nn.Module):
    def __init__(self, base=8, n_classes=2):
        super().__init__()
        b = base
        self.enc1, self.enc2, self.enc3 = block(1, b), block(b, 2 * b), block(2 * b, 4 * b)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = block(4 * b, 8 * b)
        self.up3 = nn.ConvTranspose3d(8 * b, 4 * b, 2, stride=2)
        self.dec3 = block(8 * b, 4 * b)
        self.up2 = nn.ConvTranspose3d(4 * b, 2 * b, 2, stride=2)
        self.dec2 = block(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose3d(2 * b, b, 2, stride=2)
        self.dec1 = block(2 * b, b)
        self.lesion_seg = nn.Conv3d(b, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.lesion_seg(d1)


def train(model, steps, size, seed=0):
    torch.manual_seed(seed)
    V, M, _, _ = brain_3d(24, size, seed=seed, lesion_prob=1.0)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-2, total_steps=steps)
    w = torch.tensor([0.2, 1.0])
    model.train()
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, len(V), (2,))
        logits = model(V[idx])
        loss = F.cross_entropy(logits, M[idx], weight=w)
        p = logits.softmax(1)[:, 1]
        tgt = (M[idx] == 1).float()
        loss = loss + 1 - (2 * (p * tgt).sum() + 1) / (p.sum() + tgt.sum() + 1)   # + soft Dice
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 20 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--mode", default="auto", choices=["auto", "volume", "ortho", "montage", "projection"])
    ap.add_argument("--style", default="technical", choices=["technical", "story", "cinematic"])
    ap.add_argument("--format", default="png")
    ap.add_argument("--thick", action="store_true",
                    help="thick-slice version (every 3rd slice, 1 x 1 x 3 mm voxels): shows voxel-spacing-aware rendering")
    args = ap.parse_args()
    model = UNet3D()
    ckpt = out_path(f"unet3d_{args.size}_{args.steps}steps.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        model.eval()
    elif args.steps:
        train(model, args.steps, args.size)
        torch.save(model.state_dict(), ckpt)
    V, M, _, _ = brain_3d(1, args.size, seed=7, lesion_prob=1.0)
    spacing = None
    if args.thick:
        V = V[..., ::3].contiguous()          # keep every 3rd axial slice: 1 x 1 x 3 mm voxels
        spacing = (1.0, 1.0, 3.0)
    suffix = ("" if args.mode == "auto" else f"_{args.mode}") + ("" if args.style == "technical" else f"_{args.style}") \
        + ("_thick" if args.thick else "")
    path = out_path(f"medical_3d{suffix}.{args.format}")
    shape = "×".join(map(str, V.shape[2:]))
    fig = visualize_model(
        model, V, output=path, style=args.style, volume_mode=args.mode, voxel_spacing=spacing,
        title="3-D U-Net · lesion segmentation · activation flow" + (" · thick slices" if args.thick else ""),
        subtitle=(f"synthetic MRI-like volume {shape} [B,C,X,Y,Z]" + (" with 1 × 1 × 3 mm voxels, drawn to scale"
                  if args.thick else "") + f" · trained {args.steps} steps · feature volumes rendered as "
                  f"translucent voxel blocks + activation-driven orthogonal slices"),
        output_types={"output": "segmentation"},
    )
    print(fig.flow.summary_table())
    print("wrote", path)


if __name__ == "__main__":
    main()
