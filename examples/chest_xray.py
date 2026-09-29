"""Real medical example — TorchXRayVision DenseNet-121 on a public chest X-ray.

The model (Cohen et al., MIDL 2022) is trained on seven public chest X-ray
datasets and predicts 18 pathologies (calibrated probabilities).  The sample
image is ``00000001_000.png`` from the NIH ChestX-ray14 dataset (Wang et al.,
CVPR 2017), labelled *Cardiomegaly*.

    python examples/chest_xray.py                  # static cinematic figure (black)
    python examples/chest_xray.py --movie          # occlusion sweep: which regions does the decision need?
    python examples/chest_xray.py --image my.png   # your own frontal CXR

Needs ``pip install torchxrayvision`` (weights download from GitHub on first use).
"""
import argparse
import os
import urllib.request

import numpy as np
import torch

from _common import out_path
from real_models import ROOT, load_chexpert_densenet
from neural_flow import animate_inputs, visualize_model
from neural_flow.sequences import occlusion_sweep

SAMPLE_URL = "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/00000001_000.png"


def load_cxr(path=None) -> torch.Tensor:
    import skimage.io
    import torchvision
    import torchxrayvision as xrv

    if path is None:
        os.makedirs(ROOT, exist_ok=True)
        path = os.path.join(ROOT, "nih_00000001_000.png")
        if not os.path.exists(path):
            urllib.request.urlretrieve(SAMPLE_URL, path)
    img = skimage.io.imread(path)
    img = xrv.datasets.normalize(img, 255)
    if img.ndim == 3:
        img = img.mean(2)
    t = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    return torch.from_numpy(t(img[None]))[None].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--movie", action="store_true")
    ap.add_argument("--theme", default="black", choices=["black", "dark", "light"])
    args = ap.parse_args()
    model = load_chexpert_densenet()
    x = load_cxr(args.image)
    common = dict(class_names=list(model.pathologies), output_types={"output": "multilabel_probs"}, top_k=5,
                  style="cinematic", theme=args.theme)
    if not args.movie:
        path = out_path(f"chest_xray_cinematic{'' if args.theme == 'black' else '_' + args.theme}.png")
        fig = visualize_model(model, x, output=path, title="Chest X-ray · DenseNet-121 (TorchXRayVision)",
                              subtitle="NIH ChestX-ray14 00000001_000 (labelled cardiomegaly) · 18 pathologies, "
                                       "calibrated probabilities", **common)
        print(fig.flow.summary_table())
        print("wrote", path)
    else:
        frames = occlusion_sweep(x[0], patch=56, stride=28, value=float(x.mean()))
        path = animate_inputs(model, frames, output=out_path("movie_cxr_occlusion.mp4"), fps=6,
                              title="Chest X-ray · occlusion sweep",
                              subtitle="a grey patch slides over the film; watch cardiomegaly drop when it covers the heart",
                              **common)
        print("wrote", path)


if __name__ == "__main__":
    main()
