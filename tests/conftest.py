import os
import sys

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TinyCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.block1 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.block2(self.block1(self.stem(x)))
        return self.fc(torch.flatten(self.pool(x), 1))


class TupleOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.split = SplitModule()
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        a, b = self.split(self.conv(x))
        return self.head(a.mean((2, 3))), b


class SplitModule(nn.Module):
    def forward(self, x):
        return x.relu(), (-x).relu()


class DictOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cls_head = nn.Linear(8, 3)
        self.age_head = nn.Linear(8, 1)

    def forward(self, x):
        z = self.enc(x)
        return {"diagnosis": self.cls_head(z), "age": self.age_head(z)}


class MultiInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.img = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.tab = nn.Sequential(nn.Linear(6, 8), nn.ReLU())
        self.fuse = nn.Linear(16, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, image, clinical):
        z = torch.cat([self.img(image), self.tab(clinical)], 1)
        return self.head(self.fuse(z).relu())


class Reuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.inp = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        x = self.inp(x)
        for _ in range(3):
            x = self.conv(x).relu()
        return x


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.ReLU())
        self.enc1, self.enc2, self.bott = c(1, 8), c(8, 16), c(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = c(32, 16)
        self.up1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec1 = c(16, 8)
        self.seg = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.seg(d1)


class TinyViT(nn.Module):
    def __init__(self, dim=32, depth=4, n_classes=4):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 8, stride=8)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, 17, dim) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, dropout=0.0)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
        for b in self.blocks:
            t = b(t)
        return self.head(self.norm(t)[:, 0])


class Tiny3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv3d(1, 8, 3, padding=1), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc(self.pool(self.c3(self.c2(self.c1(x)))).flatten(1))


class DataDependent(nn.Module):
    """Data-dependent control flow: torch.fx.symbolic_trace fails on this."""

    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 4, 3, padding=1)
        self.b = nn.Conv2d(4, 4, 3, padding=1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        y = self.a(x)
        if y.mean() > -1e9:  # always true at runtime, untraceable symbolically
            y = self.b(y)
        return self.fc(y.mean((2, 3)))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
