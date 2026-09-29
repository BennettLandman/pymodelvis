import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from neural_flow.cli import build_parser, main
from neural_flow.zoo import LoadedModel, load_input, load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PY = '''
import torch, torch.nn as nn
class Net(nn.Module):
    def __init__(self, n_classes=3, in_ch=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 8, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, n_classes)
    def forward(self, x):
        return self.head(self.pool(self.down(self.stem(x))).flatten(1))
'''


@pytest.fixture
def user_model(tmp_path):
    p = tmp_path / "my_net.py"
    p.write_text(MODEL_PY)
    return str(p)


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for cmd in ("render", "movie", "inspect", "demo", "fetch", "models"):
        assert cmd in out


def test_render_user_model_file(tmp_path, user_model):
    out = tmp_path / "f.png"
    rc = main(["render", f"{user_model}:Net", "--model-args", '{"n_classes": 4}', "-i", "random:1,3,32,32",
               "-o", str(out), "--class-names", "a,b,c,d", "-q", "--dpi", "40"])
    assert rc == 0 and out.exists()


def test_render_with_weights_and_cinematic(tmp_path, user_model):
    sys.path.insert(0, str(tmp_path))
    import importlib

    Net = importlib.import_module("my_net").Net
    ck = tmp_path / "ckpt.pt"
    torch.save({"state_dict": Net().state_dict()}, ck)
    out = tmp_path / "c.png"
    rc = main(["render", f"{user_model}:Net", "--weights", str(ck), "-i", "random:1,3,32,32", "-o", str(out),
               "--style", "cinematic", "--dpi", "35", "-q", "--html"])
    assert rc == 0 and out.exists() and (tmp_path / "c.html").exists()


def test_render_whole_saved_model_and_npy(tmp_path, user_model):
    sys.path.insert(0, str(tmp_path))
    import importlib

    m = importlib.import_module("my_net").Net()
    mp = tmp_path / "whole.pt"
    torch.save(m, mp)
    xp = tmp_path / "x.npy"
    np.save(xp, np.random.rand(3, 32, 32).astype(np.float32))
    out = tmp_path / "w.svg"
    assert main(["render", str(mp), "-i", str(xp), "-o", str(out), "-q", "--dpi", "40"]) == 0
    assert out.exists()


def test_inspect_all_modules(capsys, user_model):
    assert main(["inspect", f"{user_model}:Net", "-i", "random:1,3,32,32", "--all-modules"]) == 0
    out = capsys.readouterr().out
    assert "stem.0" in out and "topology: runtime" in out


def test_movie_gif(tmp_path, user_model):
    img = tmp_path / "wide.png"
    from PIL import Image

    Image.fromarray((np.random.rand(40, 120, 3) * 255).astype(np.uint8)).save(img)
    out = tmp_path / "m.gif"
    rc = main(["movie", f"{user_model}:Net", "--pan", str(img), "--steps", "3", "--size", "32", "--preset", "raw",
               "-o", str(out), "--dpi", "30", "--figsize", "8", "4.5", "-q", "--hold", "0"])
    assert rc == 0 and out.exists()


def test_input_loading_image_sample():
    lm = LoadedModel(torch.nn.Identity(), "id")
    x = load_input("sample:cat", lm, size=64)
    assert x.shape == (1, 3, 64, 64)
    lm.preset = "xray"
    x = load_input("sample:cat", lm, size=64)
    assert x.shape == (1, 1, 64, 64) and x.min() >= -1024 and x.max() <= 1024


def test_bad_model_spec_is_a_clean_error(capsys):
    assert main(["render", "not_a_model", "-i", "random:1,3,8,8"]) == 2
    assert "unknown model spec" in capsys.readouterr().err


def test_python_m_entrypoint():
    r = subprocess.run([sys.executable, "-m", "neural_flow", "--version"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "neural_flow" in r.stdout
