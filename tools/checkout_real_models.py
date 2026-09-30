#!/usr/bin/env python3
"""Fetch real-world pretrained models + sample data into ./real_models for neural_flow examples.

Run from the pymodelvis folder on a machine with internet access:

    python3 -m pip install torch torchvision "monai[nibabel]" einops nilearn huggingface_hub
    python3 tools/checkout_real_models.py            # everything (~0.5–1 GB)
    python3 tools/checkout_real_models.py --only unest
    python3 tools/checkout_real_models.py --t1 /path/to/subject_T1w.nii.gz   # add your own scan

What it downloads
  unest      MONAI bundle  wholeBrainSeg_Large_UNEST_segmentation  (MASI UNesT, 133-label whole-brain
             segmentation of MNI-registered T1 MRI) + the MNI152 1 mm T1 template (from nilearn) as input
  resnet50   torchvision ResNet-50 IMAGENET1K_V2 weights (fp16 copy)
  vit        torchvision ViT-B/16 IMAGENET1K_V1 weights (fp16 copy)

Everything lands in ./real_models/ with a manifest.json; nothing is uploaded anywhere.
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(HERE, "real_models")


def log(*a):
    print("[checkout]", *a, flush=True)


def size_mb(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 1e6
    return os.path.getsize(p) / 1e6


def fp16_state_dict(model):
    return {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}


def get_torchvision(manifest):
    import torch
    import torchvision

    specs = {
        "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
        "vit": (torchvision.models.vit_b_16, torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1),
    }
    for key in [k for k in specs if k in manifest["_wanted"]]:
        build, w = specs[key]
        log(f"downloading torchvision {key} ({w}) …")
        m = build(weights=w)
        out = os.path.join(DEST, f"{key}_{str(w).split('.')[-1].lower()}_fp16.pth")
        torch.save(fp16_state_dict(m), out)
        with open(os.path.join(DEST, "imagenet_classes.json"), "w") as f:
            json.dump(w.meta["categories"], f)
        manifest[key] = {"file": os.path.basename(out), "weights": str(w), "mb": round(size_mb(out), 1),
                         "builder": f"torchvision.models.{build.__name__}"}
        log(f"  → {out} ({manifest[key]['mb']} MB)")


def get_unest(manifest, t1=None):
    bundle = "wholeBrainSeg_Large_UNEST_segmentation"
    bdir = os.path.join(DEST, "monai_bundles")
    os.makedirs(bdir, exist_ok=True)
    target = os.path.join(bdir, bundle)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from neural_flow.zoo import fetch_bundle

    log(f"fetching MONAI bundle {bundle} (Hugging Face → GitHub sources + NVIDIA weights → monai.bundle) …")
    fetch_bundle(bundle, dest=target)
    # MNI152 1 mm T1 template (ships inside nilearn — the UNesT bundle expects MNI-registered T1)
    import nibabel as nib

    try:
        from nilearn import datasets

        tpl = datasets.load_mni152_template(resolution=1)
        mask = datasets.load_mni152_brain_mask(resolution=1)
        nib.save(tpl, os.path.join(DEST, "mni152_t1_1mm.nii.gz"))
        nib.save(mask, os.path.join(DEST, "mni152_brainmask_1mm.nii.gz"))
        log("  saved MNI152 T1 template + brain mask")
    except Exception as e:
        log("  nilearn template failed:", repr(e)[:200])
    if t1:
        shutil.copy(t1, os.path.join(DEST, "user_t1.nii.gz" if t1.endswith(".gz") else "user_t1.nii"))
        log("  copied your T1:", t1)
    info = {"bundle": bundle, "dir": os.path.relpath(target, DEST), "mb": round(size_mb(target), 1)}
    cfgs = []
    for root, _, files in os.walk(target):
        for f in files:
            if f.endswith((".json", ".yaml", ".py", ".md")):
                cfgs.append(os.path.relpath(os.path.join(root, f), target))
    info["text_files"] = sorted(cfgs)
    info["weights"] = sorted(os.path.relpath(os.path.join(r, f), target) for r, _, fs in os.walk(target)
                             for f in fs if f.endswith((".pt", ".pth", ".ts")))
    manifest["unest"] = info
    log(f"  bundle at {target} ({info['mb']} MB); weights: {info['weights']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=["unest", "resnet50", "vit"], choices=["unest", "resnet50", "vit"])
    ap.add_argument("--t1", default=None, help="optional: your own T1 (ideally MNI-registered, 1 mm) to include")
    args = ap.parse_args()
    os.makedirs(DEST, exist_ok=True)
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0], "_wanted": args.only}
    try:
        import torch

        manifest["torch"] = torch.__version__
    except ImportError:
        sys.exit("please `pip install torch torchvision` first")
    if "unest" in args.only:
        get_unest(manifest, args.t1)
    get_torchvision(manifest)
    manifest.pop("_wanted")
    with open(os.path.join(DEST, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"done — {round(size_mb(DEST))} MB in {DEST}")
    log("tell Claude it's finished; it will read real_models/manifest.json and build the real example.")


if __name__ == "__main__":
    main()
