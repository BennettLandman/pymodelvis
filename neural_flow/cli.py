"""Command-line interface: ``neural-flow`` (also ``python -m neural_flow``).

    neural-flow demo cat                                   # the ResNet-50 cat, cinematic
    neural-flow render resnet50 -i photo.jpg -o flow.png
    neural-flow render cxr -i sample:cxr -o cxr.png --style cinematic
    neural-flow render my_net.py:UNet --weights unet.pt -i scan.nii.gz --crop 96 -o unet.png
    neural-flow movie resnet50 --pan wide_photo.jpg -o pan.mp4
    neural-flow inspect resnet50 -i sample:cat
    neural-flow fetch all

Run ``neural-flow <command> --help`` for every option.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

EPILOG_MODELS = """model specifications:
  resnet50 | resnet18 | vit | vit_b_16 | swin_t | densenet121 | cxr | unest   built-in aliases
  torchvision:NAME[:WEIGHTS]      e.g. torchvision:efficientnet_b0, torchvision:resnet50:IMAGENET1K_V1
  timm:NAME                       e.g. timm:convnext_tiny
  xrv:WEIGHTS                     TorchXRayVision chest X-ray DenseNet (default densenet121-res224-all)
  monai:BUNDLE_DIR                a MONAI bundle directory
  path/to/file.py:ClassOrFactory  your own model (+ --model-args '{"in_ch": 1}' --weights ckpt.pt)
  package.module:ClassOrFactory   importable model class / factory
  path/to/model.pt                a whole model saved with torch.save(model)

input specifications (-i/--input):
  photo.jpg | scan.png            image; centre-cropped, resized and normalised for the model
  volume.nii.gz                   3-D volume [1,C,X,Y,Z]; --crop N takes an N³ ROI at the centre of mass
  tensor.npy | tensor.pt          a ready tensor
  random:1,3,224,224              random tensor (smoke test)
  sample:cat | sample:cxr         bundled cat photo / public NIH chest X-ray
  name=SPEC                       named input for multi-input models (repeat -i)
"""


# ---------------------------------------------------------------------------
# shared arguments
# ---------------------------------------------------------------------------


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("model", help="model specification (see below)")
    g.add_argument("--weights", help="state-dict checkpoint to load into the model")
    g.add_argument("--model-args", default=None, help="JSON keyword arguments for a model class / factory")
    g.add_argument("--allow-random-weights", action="store_true",
                   help="continue with random weights if pretrained weights cannot be downloaded")
    g.add_argument("--device", default="cpu", help="cpu | cuda | cuda:1 | mps (default: cpu)")


def _add_input_prep_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("input preprocessing")
    g.add_argument("--size", type=int, default=None, help="resize images to SIZE×SIZE (default: the model's)")
    g.add_argument("--no-center-crop", action="store_true", help="resize the whole image instead of a centre crop")
    g.add_argument("--preset", choices=["imagenet", "raw", "xray", "volume"], default=None,
                   help="override input normalisation (default: chosen from the model)")
    g.add_argument("--mean", type=float, nargs="+", default=None, help="per-channel mean for normalisation")
    g.add_argument("--std", type=float, nargs="+", default=None, help="per-channel std for normalisation")
    g.add_argument("--crop", type=int, default=None, help="3-D: crop an N³ ROI around the centre of mass")
    v = p.add_argument_group("3-D volumes")
    v.add_argument("--spacing", type=float, nargs=3, metavar=("SX", "SY", "SZ"), default=None,
                   help="voxel size per axis (default: read from the NIfTI header); draws true proportions")
    v.add_argument("--no-spacing", action="store_true", help="ignore voxel spacing (draw voxels as cubes)")
    v.add_argument("--sliding-window", action="store_true",
                   help="trace one ROI window, show the whole-volume output fused from all windows")
    v.add_argument("--roi", type=int, nargs="+", default=None, metavar="N",
                   help="sliding-window size, e.g. 96 or 96 96 64 (default: --crop, the bundle's ROI, or 96)")
    v.add_argument("--sw-overlap", type=float, default=None, help="sliding-window overlap (default 0.25)")
    v.add_argument("--roi-center", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="voxel the traced window is centred on (default: centre of the foreground)")
    v.add_argument("--flat-3d", nargs="?", const="max", choices=["max", "mean"], default=None,
                   help="draw 3-D stages as squashed 2-D projections (optional; default: true 3-D)")


def _add_view_args(p: argparse.ArgumentParser, movie: bool = False) -> None:
    g = p.add_argument_group("appearance")
    g.add_argument("--style", choices=["technical", "story", "cinematic"], default="cinematic" if movie else "technical")
    g.add_argument("--theme", choices=["light", "dark", "black"], default=None,
                   help="default: black for cinematic, light otherwise")
    g.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=None, help="inches, e.g. 16 9")
    g.add_argument("--dpi", type=int, default=None)
    g.add_argument("--layout", choices=["horizontal", "vertical", "wrap"], default=None)
    g.add_argument("--title", default=None)
    g.add_argument("--subtitle", default=None)
    g.add_argument("--font-scale", type=float, default=None)
    g.add_argument("--front-page", choices=["pca", "channel"], default=None, help="cinematic stack front page")
    s = p.add_argument_group("stages and channels")
    s.add_argument("--layers", default=None,
                   help="comma-separated stage selectors: module names, name#k, re:REGEX, type:CLASS")
    s.add_argument("--exclude", default=None, help="comma-separated selectors to leave out")
    s.add_argument("--max-stages", type=int, default=None)
    s.add_argument("--target-stages", type=int, default=None)
    s.add_argument("--channel-strategy", choices=["energy", "variance", "spread", "mean_abs", "even", "pca"],
                   default=None)
    s.add_argument("--max-channels", type=int, default=None)
    s.add_argument("--volume-mode", choices=["auto", "volume", "ortho", "montage", "projection"], default=None)
    s.add_argument("--volume-axes", choices=["xyz", "dhw"], default=None)
    o = p.add_argument_group("outputs and explanations")
    o.add_argument("--class-names", default=None,
                   help="'imagenet', a .txt (one per line) / .json file, or a comma-separated list")
    o.add_argument("--output-type", action="append", default=[], metavar="NAME=TYPE",
                   help="softmax|sigmoid|multilabel|multilabel_probs|regression|segmentation|embedding|raw (repeatable)")
    o.add_argument("--top-k", type=int, default=None)
    o.add_argument("--explain", dest="explain", action="store_true", default=None,
                   help="compute gradient explanations (default: on for cinematic)")
    o.add_argument("--no-explain", dest="explain", action="store_false", help="skip gradient explanations (faster)")
    o.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="any other FlowConfig option, e.g. --set max_capture_mb=200 --set diverging=True")


def _config_kwargs(a, lm) -> Dict[str, Any]:
    from .zoo import parse_class_names

    kw: Dict[str, Any] = {}
    for k in ("style", "theme", "dpi", "layout", "title", "subtitle", "font_scale", "front_page", "max_stages",
              "target_stages", "channel_strategy", "max_channels", "volume_mode", "volume_axes", "top_k", "explain"):
        v = getattr(a, k, None)
        if v is not None:
            kw[k] = v
    if getattr(a, "figsize", None):
        kw["figsize"] = tuple(a.figsize)
    if getattr(a, "layers", None):
        kw["layers"] = [s.strip() for s in a.layers.split(",") if s.strip()]
    if getattr(a, "exclude", None):
        kw["exclude"] = [s.strip() for s in a.exclude.split(",") if s.strip()]
    names = parse_class_names(getattr(a, "class_names", None), lm)
    if names:
        kw["class_names"] = names
    ot = dict(lm.output_types)
    for item in getattr(a, "output_type", []) or []:
        k, _, v = item.partition("=")
        ot[k] = v
    if ot:
        kw["output_types"] = ot
    if "top_k" not in kw and names and len(names) > 3:
        kw["top_k"] = 5
    if getattr(a, "_spacing", None) is not None and not getattr(a, "no_spacing", False):
        kw["voxel_spacing"] = tuple(a._spacing)
    if getattr(a, "flat_3d", None):
        kw["flat_3d"] = a.flat_3d
    if getattr(a, "sliding_window", False):
        kw["sliding_window"] = True
        kw["roi_size"] = _roi(a, lm)
        if a.sw_overlap is not None:
            kw["sw_overlap"] = a.sw_overlap
        if a.roi_center:
            kw["roi_center"] = tuple(a.roi_center)
    for item in getattr(a, "set", []) or []:
        k, _, v = item.partition("=")
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
    return kw


def _roi(a, lm):
    if getattr(a, "roi", None):
        r = tuple(a.roi)
        return r * 3 if len(r) == 1 else r
    if getattr(a, "crop", None):
        return (a.crop,) * 3
    return tuple(lm.roi) if lm.roi else (96, 96, 96)


def _load(a):
    from .zoo import load_model

    margs = json.loads(a.model_args) if a.model_args else None
    lm = load_model(a.model, weights=a.weights, model_args=margs, allow_random=a.allow_random_weights, device=a.device)
    if getattr(a, "preset", None):
        lm.preset = a.preset
    if getattr(a, "mean", None):
        lm.mean = tuple(a.mean)
    if getattr(a, "std", None):
        lm.std = tuple(a.std)
    return lm


def _inputs(a, lm):
    from .zoo import load_input

    specs = a.input or []
    if not specs:
        default = "sample:cxr" if lm.preset == "xray" else "sample:cat"
        if lm.preset == "volume":
            raise SystemExit("error: a 3-D model needs an input volume: -i scan.nii.gz")
        print(f"no --input given; using {default}", file=sys.stderr)
        specs = [default]
    named = {}
    for s in specs:
        if "=" in s and not os.path.exists(s) and not s.startswith(("random:", "sample:")):
            k, _, v = s.partition("=")
            named[k] = v
        else:
            named[f"_{len(named)}"] = s
    info: Dict[str, Any] = {}
    whole = bool(getattr(a, "sliding_window", False))
    tensors = {k: load_input(v, lm, a.size, a.crop, not a.no_center_crop, info=info, whole=whole).to(a.device)
               for k, v in named.items()}
    a._spacing = getattr(a, "spacing", None) or info.get("spacing")
    if len(tensors) == 1 and next(iter(tensors)).startswith("_"):
        return next(iter(tensors.values()))
    if all(k.startswith("_") for k in tensors):
        return tuple(tensors.values())
    return tensors


def _default_out(a, suffix):
    base = os.path.splitext(os.path.basename(a.model.split(":")[-1].rstrip("/")))[0] or "model"
    return f"{base}_{getattr(a, 'style', 'flow')}{suffix}"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_render(a) -> int:
    from . import visualize_model

    t0 = time.time()
    lm = _load(a)
    x = _inputs(a, lm)
    kw = _config_kwargs(a, lm)
    out = a.output or _default_out(a, ".png")
    fig = visualize_model(lm.model, x, output=out, interactive=a.html, **kw)
    if not a.quiet:
        print(fig.flow.summary_table())
    print(f"wrote {out}" + (f" and {os.path.splitext(out)[0]}.html" if a.html and not out.endswith('.html') else "")
          + f"  ({time.time() - t0:.1f}s)")
    return 0


def cmd_inspect(a) -> int:
    from . import trace_model

    lm = _load(a)
    x = _inputs(a, lm)
    kw = _config_kwargs(a, lm)
    kw["explain"] = False
    res = trace_model(lm.model, x, **kw)
    if a.all_modules:
        print(f"{'module call':44s} {'type':24s} out shape")
        for c in res.trace.calls_in_order():
            if c.id == res.trace.root:
                continue
            print(f"{'  ' * (c.depth - 1)}{c.key}"[:44].ljust(44), f"{c.type_name[:24]:24s}",
                  "×".join(map(str, c.main_shape[1:])) if c.main_shape else "-")
        print()
    print(res.summary_table())
    print(f"\n{len(res.trace.calls)} module calls · topology: {res.graph.topology_source} · adapters: "
          f"{', '.join(res.adapters) or '-'}")
    edges = [e for e in res.graph.edges if e.kind != "main"]
    if edges:
        print("non-sequential edges: " + ", ".join(f"{e.src} ⇢ {e.dst} ({e.kind})" for e in edges))
    print("use --layers with any names above to choose stages explicitly")
    return 0


def _movie_frames(a, lm):
    import torch

    from .sequences import crossfade, occlusion_sweep, pan, zoom
    from .zoo import image_to_tensor, load_input, read_image

    def norm(img):  # full image → normalised [C, H, W] without resizing (pan/zoom crop later)
        t = image_to_tensor(img, lm, size=None, center_crop=False)
        return t

    if a.pan:
        img = read_image(a.pan)
        H, W = img.shape[:2]
        size = a.size or lm.size
        scale = size / H
        from PIL import Image

        pil = Image.fromarray((img * 255).astype("uint8")).resize((max(size, int(round(W * scale))), size), Image.BICUBIC)
        import numpy as np

        big = np.asarray(pil, np.float32) / 255.0
        full = _normalise_full(big, lm)
        return pan(full, window=size, steps=a.steps, vertical=a.vertical), None
    if a.zoom:
        img = read_image(a.zoom)
        full = _normalise_full(img, lm)
        c = tuple(a.zoom_center) if a.zoom_center else None
        return zoom(full, start=1.0, end=a.zoom_end, steps=a.steps, center=c, out_size=a.size or lm.size), None
    if a.occlusion:
        x = load_input(a.occlusion, lm, a.size, None, not a.no_center_crop)[0]
        return occlusion_sweep(x, patch=a.patch, stride=a.stride, value=float(x.mean())), None
    if a.crossfade:
        xa = load_input(a.crossfade[0], lm, a.size, a.crop, not a.no_center_crop)[0]
        xb = load_input(a.crossfade[1], lm, a.size, a.crop, not a.no_center_crop)[0]
        return crossfade(xa, xb, steps=a.steps), None
    if a.frames:
        files = []
        for pat in a.frames:
            files += sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
        if not files:
            raise SystemExit("error: --frames matched no files")
        info: Dict[str, Any] = {}
        fr = [load_input(f, lm, a.size, a.crop, not a.no_center_crop, info=info) for f in files]
        a._spacing = a.spacing or info.get("spacing")
        return fr, [os.path.basename(f) for f in files]
    if a.volume_sweep:
        from .zoo import crop_volume, read_volume

        from .zoo import volume_spacing

        vol = read_volume(a.volume_sweep, lm)
        a._spacing = a.spacing or volume_spacing(a.volume_sweep)
        roi = (a.crop,) * 3 if a.crop else (lm.roi or (96, 96, 96))
        S = vol.shape[1:]
        zs = torch.linspace(roi[2] / 2, S[2] - roi[2] / 2, a.steps).tolist()
        return [crop_volume(vol, roi, center=(S[0] / 2, S[1] / 2, z)) for z in zs], [f"ROI centre z = {z:.0f}" for z in zs]
    raise SystemExit("error: choose a sequence: --pan, --zoom, --occlusion, --crossfade, --frames or --volume-sweep")


def _normalise_full(img, lm):
    import torch

    x = torch.from_numpy(img).permute(2, 0, 1)
    if lm.preset == "xray":
        return (x.mean(0, keepdim=True) * 2048.0 - 1024.0)
    if lm.preset == "raw":
        return x
    return (x - torch.tensor(lm.mean)[:, None, None]) / torch.tensor(lm.std)[:, None, None]


def cmd_movie(a) -> int:
    from . import animate_inputs

    t0 = time.time()
    lm = _load(a)
    if getattr(a, "inference", None):
        return _movie_inference(a, lm, t0)
    a._spacing = getattr(a, "spacing", None)
    frames, labels = _movie_frames(a, lm)
    kw = _config_kwargs(a, lm)
    kw.setdefault("figsize", (16, 9))
    kw.setdefault("dpi", 120)
    style = kw.pop("style", "cinematic")
    out = a.output or _default_out(a, ".mp4")
    if isinstance(frames, list):
        frames = [f.to(a.device) for f in frames]
    else:
        frames = frames.to(a.device)
    animate_inputs(lm.model, frames, output=out, fps=a.fps, frame_labels=labels, style=style,
                   hold_last=a.hold, progress=not a.quiet, **kw)
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


def _movie_inference(a, lm, t0) -> int:
    from .volume3d import sliding_window_movie
    from .zoo import load_input

    info: Dict[str, Any] = {}
    x = load_input(a.inference, lm, info=info, whole=True).to(a.device)
    if x.dim() != 5:
        raise SystemExit("error: --inference needs a 3-D volume (NIfTI)")
    a._spacing = a.spacing or info.get("spacing")
    kw = _config_kwargs(a, lm)
    for k in ("sliding_window", "roi_size", "sw_overlap", "roi_center"):
        kw.pop(k, None)
    kw.setdefault("figsize", (16, 9))
    kw.setdefault("dpi", 120)
    style = kw.pop("style", "cinematic")
    out = a.output or _default_out(a, "_inference.mp4")
    sliding_window_movie(lm.model, x, _roi(a, lm), output=out, overlap=a.sw_overlap if a.sw_overlap is not None else 0.25,
                         max_windows=a.max_windows, fps=a.fps if a.fps != 8 else 3, style=style, hold_last=a.hold,
                         progress=not a.quiet, **kw)
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


DEMOS = {
    "cat": ("resnet50", "sample:cat", "ResNet-50 looks at a cat"),
    "vit": ("vit", "sample:cat", "ViT-B/16 looks at a cat"),
    "cxr": ("cxr", "sample:cxr", "Chest X-ray · DenseNet-121 (TorchXRayVision)"),
    "technical": ("resnet50", "sample:cat", "ResNet-50 · activation flow"),
}


def cmd_demo(a) -> int:
    from . import visualize_model
    from .zoo import load_input, load_model

    os.makedirs(a.outdir, exist_ok=True)
    names = list(DEMOS) + ["movie"] if a.name == "all" else [a.name]
    for n in names:
        t0 = time.time()
        if n == "movie":
            from . import animate_inputs
            from .sequences import pan

            lm = load_model("resnet50")
            from .zoo import sample_image_path, read_image

            img = read_image(sample_image_path())
            from PIL import Image
            import numpy as np

            big = np.asarray(Image.fromarray((img * 255).astype("uint8")).resize((374, 249)), np.float32) / 255
            full = _normalise_full(big, lm)
            out = os.path.join(a.outdir, "demo_movie_pan.mp4")
            animate_inputs(lm.model, pan(full, window=249, steps=a.steps, out_size=224), output=out,
                           class_names=lm.class_names, title="ResNet-50 · a pan across the cat",
                           progress=not a.quiet)
            print(f"wrote {out}  ({time.time() - t0:.0f}s)")
            continue
        spec, inp, title = DEMOS[n]
        lm = load_model(spec)
        x = load_input(inp, lm)
        style = "technical" if n == "technical" else "cinematic"
        out = os.path.join(a.outdir, f"demo_{n}.png")
        visualize_model(lm.model, x, output=out, style=style, class_names=lm.class_names, top_k=5,
                        output_types=lm.output_types or None, title=title)
        print(f"wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


def cmd_fetch(a) -> int:
    from . import zoo

    what = ["resnet50", "vit", "cxr", "unest"] if "all" in a.what else a.what
    for w in what:
        if w == "resnet50":
            print(zoo.fetch("resnet50_ram-a26f946b.pth"))
        elif w == "vit":
            print(zoo.fetch("vit_base_patch16_224_jx.pth"))
        elif w == "cxr":
            zoo.load_model("cxr")
            print(zoo.cxr_sample_path())
        elif w == "unest":
            dest = zoo.fetch_bundle(zoo.UNEST_BUNDLE)
            print(dest)
            try:
                import nibabel as nib
                from nilearn import datasets

                p = os.path.join(zoo.cache_dir(), "mni152_t1_1mm.nii.gz")
                if not os.path.exists(p):
                    nib.save(datasets.load_mni152_template(resolution=1), p)
                print(p)
            except Exception as e:
                print(f"(MNI152 template skipped: {e}; `pip install nilearn` to get it)", file=sys.stderr)
    print(f"cache: {zoo.cache_dir()}")
    return 0


def cmd_models(a) -> int:
    from .zoo import ALIASES, search_dirs

    print("built-in aliases:")
    for k, v in ALIASES.items():
        print(f"  {k:12s} → {v}")
    print("\nweights are searched in:")
    for d in search_dirs():
        print(f"  {d}{'  (exists)' if os.path.isdir(d) else ''}")
    print("\n" + EPILOG_MODELS)
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    fmt = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(prog="neural-flow", formatter_class=fmt,
                                description="See what a PyTorch network sees: representation-flow figures and movies.",
                                epilog="commands:\n  demo      render the built-in demos (no arguments needed)\n"
                                       "  render    one figure for a model and an input\n"
                                       "  movie     a movie over a sequence of inputs\n"
                                       "  inspect   list the stages (and module calls) that would be shown\n"
                                       "  fetch     download pretrained weights / bundles into the cache\n"
                                       "  models    list model aliases and where weights are looked up\n\n"
                                       "examples:\n  neural-flow demo cat\n"
                                       "  neural-flow render resnet50 -i photo.jpg -o flow.png --style cinematic\n"
                                       "  neural-flow movie resnet50 --pan panorama.jpg -o pan.mp4\n"
                                       "  neural-flow inspect my_net.py:Net --weights ckpt.pt -i random:1,1,64,64,64 --all-modules")
    p.add_argument("--version", action="version", version=f"neural_flow {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    r = sub.add_parser("render", help="one figure for a model and an input", formatter_class=fmt, epilog=EPILOG_MODELS,
                       description="Render the representation flow of MODEL on INPUT.")
    _add_model_args(r)
    r.add_argument("-i", "--input", action="append", help="input specification (repeat for multiple inputs)")
    r.add_argument("-o", "--output", help="output file: .png .svg .pdf or .html (default: <model>_<style>.png)")
    r.add_argument("--html", action="store_true", help="also write the interactive HTML explorer")
    r.add_argument("-q", "--quiet", action="store_true", help="do not print the stage table")
    _add_input_prep_args(r)
    _add_view_args(r)
    r.set_defaults(func=cmd_render)

    m = sub.add_parser("movie", help="a movie over a sequence of inputs", formatter_class=fmt, epilog=EPILOG_MODELS,
                       description="Render a movie; stages, channels, colours and scales stay fixed across frames.")
    _add_model_args(m)
    seq = m.add_argument_group("input sequence (choose one)")
    seq.add_argument("--pan", metavar="IMAGE", help="slide a square window across a wide image")
    seq.add_argument("--vertical", action="store_true", help="pan top → bottom instead of left → right")
    seq.add_argument("--zoom", metavar="IMAGE", help="zoom into the image")
    seq.add_argument("--zoom-end", type=float, default=0.35, help="final crop fraction (default 0.35)")
    seq.add_argument("--zoom-center", type=float, nargs=2, metavar=("FY", "FX"), help="zoom target (fractions)")
    seq.add_argument("--occlusion", metavar="INPUT", help="slide an occluding patch over the input")
    seq.add_argument("--patch", type=int, default=56, help="occlusion patch size (px)")
    seq.add_argument("--stride", type=int, default=28, help="occlusion stride (px)")
    seq.add_argument("--crossfade", nargs=2, metavar=("A", "B"), help="morph from input A to input B")
    seq.add_argument("--frames", nargs="+", metavar="FILE_OR_GLOB", help="explicit frames, e.g. 'scans/*.nii.gz'")
    seq.add_argument("--volume-sweep", metavar="VOLUME", help="3-D: move the ROI window inferior → superior")
    seq.add_argument("--inference", metavar="VOLUME",
                     help="3-D: watch sliding-window inference over the whole volume, one window per frame")
    m.add_argument("--max-windows", type=int, default=None, help="--inference: show at most N windows")
    m.add_argument("--steps", type=int, default=36, help="number of frames for generated sequences (default 36)")
    m.add_argument("--fps", type=int, default=8)
    m.add_argument("--hold", type=int, default=6, help="repeat the last frame N times")
    m.add_argument("-o", "--output", help="output .mp4 (needs imageio-ffmpeg) or .gif")
    m.add_argument("-q", "--quiet", action="store_true")
    _add_input_prep_args(m)
    _add_view_args(m, movie=True)
    m.set_defaults(func=cmd_movie)

    i = sub.add_parser("inspect", help="list the stages that would be shown", formatter_class=fmt, epilog=EPILOG_MODELS)
    _add_model_args(i)
    i.add_argument("-i", "--input", action="append")
    i.add_argument("--all-modules", action="store_true", help="also list every module call with its output shape")
    _add_input_prep_args(i)
    _add_view_args(i)
    i.set_defaults(func=cmd_inspect)

    d = sub.add_parser("demo", help="render the built-in demos", formatter_class=fmt,
                       description="Render ready-made examples with real pretrained models (downloaded on first use).")
    d.add_argument("name", nargs="?", default="cat", choices=list(DEMOS) + ["movie", "all"])
    d.add_argument("-o", "--outdir", default="neural_flow_demos")
    d.add_argument("--steps", type=int, default=24, help="frames for the movie demo")
    d.add_argument("-q", "--quiet", action="store_true")
    d.set_defaults(func=cmd_demo)

    f = sub.add_parser("fetch", help="download weights / bundles into the cache")
    f.add_argument("what", nargs="+", choices=["resnet50", "vit", "cxr", "unest", "all"])
    f.set_defaults(func=cmd_fetch)

    ml = sub.add_parser("models", help="list model aliases and weight locations")
    ml.set_defaults(func=cmd_models)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "command", None):
        p.print_help()
        return 1
    if a.command in ("render", "movie", "inspect") and getattr(a, "theme", None) is None and \
            getattr(a, "style", None) != "cinematic":
        a.theme = None
    try:
        return a.func(a) or 0
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as e:
        if os.environ.get("NEURAL_FLOW_DEBUG"):
            raise
        print(f"error: {e}", file=sys.stderr)
        print("(set NEURAL_FLOW_DEBUG=1 for the full traceback)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
