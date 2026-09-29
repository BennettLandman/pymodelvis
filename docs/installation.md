# Installation

## Requirements

* Python 3.9 – 3.12
* PyTorch ≥ 2.1 (CPU or CUDA; Apple-silicon MPS models also work, since tensors are summarized on
  the model's own device)
* Core dependencies, installed automatically: `numpy`, `matplotlib`, `pillow`, `networkx`

## Install from source

```bash
git clone https://github.com/BennettLandman/pymodelvis.git
cd pymodelvis
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .
```

The distribution is named `pymodelvis` and the import name is `neural_flow`:

```python
import neural_flow
print(neural_flow.__version__)
```

It also installs the `neural-flow` command (see the [command-line guide](cli.md)):

```bash
neural-flow --version
neural-flow demo cat      # writes neural_flow_demos/demo_cat.png
```

If your shell can't find `neural-flow`, the virtual environment isn't active or pip's script
folder isn't on `PATH`. `python -m neural_flow …` always works.

## Optional extras

| extra | installs | needed for |
|---|---|---|
| `examples` | `torchvision`, `timm`, `scikit-image`, `torchxrayvision` | the example scripts (ResNet, ViT, chest X-ray, sample photos) |
| `animation` | `imageio`, `imageio-ffmpeg` | MP4 output (GIF works without it) |
| `medical` | `monai`, `nibabel`, `einops`, `nilearn` | MONAI bundles (UNesT), NIfTI I/O, the MNI152 template |
| `dev` | `pytest`, `torchvision` | running the tests |
| `all` | everything above | |

```bash
pip install -e ".[examples,animation]"
pip install -e ".[all]"
```

## CPU-only PyTorch (smaller download on Linux)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

## Verify

```bash
pytest -q                 # CPU only, about a minute
neural-flow demo cat      # or: python examples/resnet.py → examples/outputs/resnet50_cinematic.png
```

The first run downloads ResNet-50 weights (~100 MB) once. The command-line tool caches them in
`~/.cache/neural_flow` (or `$NEURAL_FLOW_HOME`); the example scripts use `real_models/`, which is
git-ignored.

## Fonts

The cinematic style uses the Inter typeface, which ships inside the package (`neural_flow/fonts`,
SIL Open Font License). Nothing needs to be installed system-wide.
