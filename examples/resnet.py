"""Example 1 — ResNet-50 (real ImageNet weights) on a photograph of a cat.

image → early convolution → residual stages → global representation → classifier

    python examples/resnet.py                           # cinematic, black background (default)
    python examples/resnet.py --style technical --theme light
    python examples/resnet.py --style story
    python examples/resnet.py --image my_photo.jpg --figsize 16 9 --format pdf

Weights: torchvision IMAGENET1K_V2 from ./real_models (tools/checkout_real_models.py) or,
if absent, timm's ResNet-50 weights downloaded once from GitHub releases.
"""
import argparse

import numpy as np

from _common import out_path
from real_models import imagenet_classes, load_resnet50, photo, to_input
from neural_flow import visualize_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to an RGB image (default: scikit-image 'chelsea' cat)")
    ap.add_argument("--style", default="cinematic", choices=["cinematic", "technical", "story"])
    ap.add_argument("--theme", default=None, choices=["black", "dark", "light"])
    ap.add_argument("--format", default="png", choices=["png", "svg", "pdf"])
    ap.add_argument("--figsize", type=float, nargs=2, default=None)
    ap.add_argument("--front", default="pca", choices=["pca", "channel"])
    args = ap.parse_args()

    model, mean, std = load_resnet50()
    if args.image:
        from PIL import Image

        img = np.asarray(Image.open(args.image).convert("RGB"), np.float32) / 255
    else:
        img = photo("chelsea")
    x = to_input(img, mean, std)
    theme = args.theme or ("black" if args.style == "cinematic" else "light")
    tag = f"resnet50_{args.style}" + ("" if (args.theme is None) else f"_{theme}") + \
          ("_169" if args.figsize else "") + ("" if args.front == "pca" else "_channelfront")
    path = out_path(f"{tag}.{args.format}")
    fig = visualize_model(model, x, output=path, style=args.style, theme=theme, class_names=imagenet_classes(),
                          top_k=5, figsize=tuple(args.figsize) if args.figsize else None, front_page=args.front,
                          title="ResNet-50" if args.style == "cinematic" else "ResNet-50 · activation flow")
    print(fig.flow.summary_table())
    print("wrote", path)


if __name__ == "__main__":
    main()
