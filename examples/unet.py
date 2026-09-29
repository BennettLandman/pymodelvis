"""Example 2 — 2-D U-Net segmentation (encoder, bottleneck, decoder, skip connections).

The U-Net is trained for a few hundred steps on synthetic microscopy-like
images (bright round cells vs. elongated debris) so the activations and the
segmentation are meaningful, then visualised.

    python examples/unet.py [--steps 250] [--style story]
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import out_path
from synthetic import cells_2d
from neural_flow import visualize_model


def double_conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, n_classes=3, base=16):
        super().__init__()
        b = base
        self.enc1 = double_conv(in_ch, b)
        self.enc2 = double_conv(b, 2 * b)
        self.enc3 = double_conv(2 * b, 4 * b)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(4 * b, 8 * b)
        self.up3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2)
        self.dec3 = double_conv(8 * b, 4 * b)
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.dec2 = double_conv(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.dec1 = double_conv(2 * b, b)
        self.seg_head = nn.Conv2d(b, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bn), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.seg_head(d1)


def train(model, steps, seed=0):
    torch.manual_seed(seed)
    X, Y = cells_2d(96, 128, seed=seed)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    w = torch.tensor([0.3, 1.0, 1.5])
    model.train()
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, len(X), (8,))
        loss = F.cross_entropy(model(X[idx]), Y[idx], weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 50 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--style", default="technical", choices=["technical", "story", "cinematic"])
    ap.add_argument("--format", default="png")
    args = ap.parse_args()
    model = UNet2D()
    ckpt = out_path(f"unet2d_{args.steps}steps.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        model.eval()
    elif args.steps:
        train(model, args.steps)
        torch.save(model.state_dict(), ckpt)
    x, _ = cells_2d(1, 128, seed=123)
    suffix = "" if args.style == "technical" else f"_{args.style}"
    path = out_path(f"unet2d{suffix}.{args.format}")
    fig = visualize_model(model, x, output=path, style=args.style, title="2-D U-Net · activation flow",
                          subtitle=f"synthetic microscopy · trained {args.steps} steps · labels: cell / debris")
    print(fig.flow.summary_table())
    print("wrote", path)


if __name__ == "__main__":
    main()
