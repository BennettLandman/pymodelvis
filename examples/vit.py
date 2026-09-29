"""Example 3 — Vision Transformer (real ViT-B/16 ImageNet weights) and Swin-T.

image → patches → patch embeddings → transformer representations → CLS representation → classifier

Token tensors [B, 197, 768] are split into the CLS token and a 14×14 patch grid.
In the cinematic style, beams between blocks become scattered rays: every token
can draw on the whole image, and the "what one unit sees" circles are nearly
global from the first blocks on (contrast with the CNN's gradual growth).

    python examples/vit.py                          # cinematic (black)
    python examples/vit.py --style technical        # technical figure with attention insets
    python examples/vit.py --model swin_t           # torchvision Swin-T (untrained unless weights available)
"""
import argparse

import torchvision

from _common import load_torchvision, out_path
from real_models import imagenet_classes, load_vit, photo, to_input
from neural_flow import visualize_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_b_16", choices=["vit_b_16", "swin_t"])
    ap.add_argument("--style", default="cinematic", choices=["cinematic", "technical", "story"])
    ap.add_argument("--image", default="chelsea", help="scikit-image sample name")
    ap.add_argument("--format", default="png")
    args = ap.parse_args()
    if args.model == "vit_b_16":
        model, mean, std = load_vit()
        title, sub = "ViT-B/16", None
    else:
        model, ok = load_torchvision(torchvision.models.swin_t, torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        title, sub = "Swin-T", None if ok else "untrained weights — run tools/checkout_real_models.py or connect to the internet"
    x = to_input(photo(args.image), mean, std)
    path = out_path(f"{args.model}_{args.style}.{args.format}")
    fig = visualize_model(model.eval(), x, output=path, style=args.style, class_names=imagenet_classes(),
                          top_k=5, title=title if args.style == "cinematic" else f"{title} · activation flow",
                          subtitle=sub)
    print(fig.flow.summary_table())
    print("wrote", path)


if __name__ == "__main__":
    main()
