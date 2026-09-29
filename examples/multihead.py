"""Example 5 — multi-input, multi-head model (visible branching).

A shared 3-D MRI encoder plus a small clinical-feature MLP are fused into one
representation that branches into three task heads:

    MRI ─→ encoder ─┬─→ decoder ─→ lesion segmentation
                    └─→ pool ─┐
    clinical ─→ MLP ──────────┴─→ fusion ─┬─→ lesion present  (sigmoid)
                                          └─→ brain age       (regression)

Inputs are passed as a dict; outputs are returned as a dict.

    python examples/multihead.py [--steps 150]
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


def conv(i, o, stride=1):
    return nn.Sequential(nn.Conv3d(i, o, 3, stride=stride, padding=1), nn.InstanceNorm3d(o, affine=True), nn.GELU())


class MultiTaskBrainNet(nn.Module):
    def __init__(self, n_clinical=3):
        super().__init__()
        self.encoder1 = conv(1, 8)
        self.encoder2 = conv(8, 16, stride=2)
        self.encoder3 = conv(16, 32, stride=2)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool3d(1), nn.Flatten())   # max: InstanceNorm makes avg-pool uninformative
        self.clinical_mlp = nn.Sequential(nn.Linear(n_clinical, 16), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(32 + 16, 32), nn.GELU())
        self.lesion_head = nn.Linear(32, 1)
        self.age_head = nn.Linear(32, 1)
        self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode="trilinear"), conv(32, 16),
                                     nn.Upsample(scale_factor=2, mode="trilinear"), conv(16, 8))
        self.seg_head = nn.Conv3d(8, 1, 1)

    def forward(self, mri, clinical):
        f = self.encoder3(self.encoder2(self.encoder1(mri)))
        z = self.fusion(torch.cat([self.global_pool(f), self.clinical_mlp(clinical)], 1))
        return {
            "lesion_seg": self.seg_head(self.decoder(f)),
            "lesion_present": self.lesion_head(z),
            "brain_age": 50 + 20 * self.age_head(z),
        }


def make_data(n, size, seed):
    V, M, has, age = brain_3d(n, size, seed=seed, lesion_prob=0.5)
    g = torch.Generator().manual_seed(seed)
    clinical = torch.stack([torch.randint(0, 2, (n,), generator=g).float(),            # sex
                            torch.randint(0, 2, (n,), generator=g).float() * 1.5 + 1.5,  # field strength (T)
                            torch.randn(n, generator=g)], 1)                              # nuisance
    return V, M, has, age, clinical


def train(model, steps, size):
    torch.manual_seed(0)
    V, M, has, age, clin = make_data(96, size, 0)
    opt = torch.optim.Adam(model.parameters(), 5e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-3, total_steps=steps)
    model.train()
    t0 = time.time()
    for s in range(steps):
        i = torch.randint(0, len(V), (4,))
        out = model(V[i], clin[i])
        l_seg = F.binary_cross_entropy_with_logits(out["lesion_seg"][:, 0], M[i].float(), pos_weight=torch.tensor(5.0))
        l_cls = F.binary_cross_entropy_with_logits(out["lesion_present"][:, 0], has[i].float())
        l_age = F.mse_loss(out["brain_age"][:, 0], age[i]) / 400
        loss = l_seg + l_cls + l_age
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 25 == 0 or s == steps - 1:
            print(f"  step {s:4d}  seg {l_seg.item():.3f}  cls {l_cls.item():.3f}  age {l_age.item():.3f}  ({time.time() - t0:.0f}s)")
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--style", default="technical", choices=["technical", "story", "cinematic"])
    ap.add_argument("--format", default="png")
    args = ap.parse_args()
    model = MultiTaskBrainNet()
    ckpt = out_path(f"multihead_{args.size}_{args.steps}steps.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        model.eval()
    elif args.steps:
        train(model, args.steps, args.size)
        torch.save(model.state_dict(), ckpt)
    V, M, has, age, clin = make_data(1, args.size, 11)
    suffix = "" if args.style == "technical" else f"_{args.style}"
    path = out_path(f"multihead{suffix}.{args.format}")
    fig = visualize_model(
        model, {"mri": V, "clinical": clin}, output=path, style=args.style,
        title="Multi-input, multi-head brain model · activation flow",
        subtitle=f"inputs: MRI volume {args.size}³ + clinical vector (sex, field strength, nuisance) · "
                 f"truth: lesion={bool(has[0])}, age={age[0]:.0f}",
    )
    print(fig.flow.summary_table())
    print("wrote", path)


if __name__ == "__main__":
    main()
