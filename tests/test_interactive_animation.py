import json
import re

import torch

from conftest import TinyCNN, TinyViT
from neural_flow import animate_model, visualize_model


def test_interactive_html(tmp_path):
    p = tmp_path / "flow.html"
    visualize_model(TinyViT(), torch.randn(1, 3, 32, 32), output=str(p), dpi=40)
    html = p.read_text()
    assert html.startswith("<!doctype html>")
    data = json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1))
    assert len(data["stages"]) >= 3 and data["outputs"]
    tok = [s for s in data["stages"] if s["tensor_kind"] == "tokens"]
    assert tok and set(tok[0]["strategies"]) >= {"energy", "variance", "pca"}
    assert tok[0]["heads"], "attention heads should be captured for nn.TransformerEncoderLayer"
    assert "http://" not in html and "https://" not in html, "HTML must be self-contained"


def test_interactive_flag_writes_both(tmp_path):
    p = tmp_path / "flow.png"
    visualize_model(TinyCNN(), torch.randn(1, 3, 32, 32), output=str(p), interactive=True, dpi=40)
    assert p.exists() and (tmp_path / "flow.html").exists()


def test_animation_gif(tmp_path):
    out = animate_model(TinyCNN(), torch.randn(1, 3, 32, 32), output=str(tmp_path / "a.gif"),
                        frames_per_stage=1, hold_frames=1, dpi=30)
    from PIL import Image

    im = Image.open(out)
    assert im.n_frames >= 4
