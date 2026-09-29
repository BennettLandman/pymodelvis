"""Crop figures and extract movie cover frames for the slide deck (docs/deck/build_deck.js)."""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "examples", "outputs")
ASSETS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)


def load(name):
    return Image.open(os.path.join(OUT, name)).convert("RGB")


SIZES = {}


def save(im, name, max_w=2400):
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    p = os.path.join(ASSETS, name)
    im.save(p, quality=92)
    SIZES[name] = im.size
    print(name, im.size)


def trim(im, thr=14, pad=20):
    """Crop away uniform background margins (works for black and white backgrounds)."""
    a = np.asarray(im).astype(np.int16)
    bg = a[2, 2]
    diff = np.abs(a - bg).sum(-1) > thr
    ys, xs = np.where(diff)
    if len(xs) == 0:
        return im
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width),
                    min(ys.max() + pad, im.height)))


def frac_crop(im, x0, y0, x1, y1):
    W, H = im.size
    return im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))


def frame(mp4, t=0.5):
    import imageio.v2 as iio

    r = iio.get_reader(os.path.join(OUT, mp4))
    n = r.count_frames()
    f = r.get_data(int(t * (n - 1)))
    return Image.fromarray(f)


figs = {
    "resnet50_cinematic.png": "resnet_cine",
    "resnet50_technical.png": "resnet_tech",
    "resnet50_story.png": "resnet_story",
    "vit_b_16_cinematic.png": "vit_cine",
    "chest_xray_cinematic.png": "cxr_cine",
    "unet2d.png": "unet_tech",
    "unet2d_cinematic.png": "unet_cine",
    "medical_3d_cinematic.png": "med3d_cine",
    "medical_3d_projection.png": "med3d_proj",
    "multihead_cinematic.png": "multi_cine",
}
for f, tag in figs.items():
    if os.path.exists(os.path.join(OUT, f)):
        im = load(f)
        save(trim(im), f"{tag}.jpg")
        # thumbnail: the middle of the flow, 2:1
        t = trim(im)
        W, H = t.size
        cw = min(W, int(H * 2.0))
        th_ = t.crop(((W - cw) // 2, 0, (W - cw) // 2 + cw, H)).resize((600, int(600 * H / cw)), Image.LANCZOS)
        save(th_, f"{tag}_thumb.jpg", 600)

# title banner: the decks of the ResNet cat, without header / footer
if os.path.exists(os.path.join(OUT, "resnet50_cinematic.png")):
    im = load("resnet50_cinematic.png")
    save(frac_crop(im, 0.0, 0.16, 1.0, 0.93), "hero_banner.jpg", 3000)

for mp4, tag in [("movie_pan_resnet.mp4", "pan"), ("movie_pan_vit.mp4", "vitpan"), ("movie_aging.mp4", "aging"),
                 ("movie_cxr_occlusion.mp4", "cxr")]:
    if os.path.exists(os.path.join(OUT, mp4)):
        save(frame(mp4, 0.45), f"{tag}_cover.jpg", 1920)

import json

json.dump(SIZES, open(os.path.join(ASSETS, "sizes.json"), "w"), indent=1)
