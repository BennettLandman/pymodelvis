#!/usr/bin/env python
"""Assemble and build the project website (GitHub Pages).

The site is made from the files that already live in the repository, so it never drifts:

* ``docs/*.md``            → the documentation pages
* ``examples/outputs/*``   → web-sized figures, movie posters, movies, the interactive explorer
* ``docs/deck/``           → the example slide deck
* ``CHANGELOG.md``, ``CONTRIBUTING.md`` → pages of their own
* ``website/pages``        → the landing page and site-only pages

Usage (from the repository root)::

    pip install -r website/requirements.txt
    python website/build.py            # → website/_site/  (static site, open index.html)
    python website/build.py --serve    # live preview at http://127.0.0.1:8000/pymodelvis/
    python website/build.py --no-build # only assemble website/_src/ (mkdocs input)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "_src")
OUT = os.path.join(HERE, "_site")
OUTPUTS = os.path.join(ROOT, "examples", "outputs")
REPO = "https://github.com/MASILab/pymodelvis"

# figures shown on the site: (file, max width for the page, thumbnail width)
FIGURES = [
    "resnet50_cinematic.png", "resnet50_cinematic_169.png", "resnet50_cinematic_light.png",
    "resnet50_technical.png", "resnet50_story.png",
    "vit_b_16_cinematic.png", "vit_b_16_technical.png",
    "chest_xray_cinematic.png", "chest_xray_cinematic_light.png",
    "unet2d_cinematic.png", "unet2d.png", "unet2d_story.png",
    "medical_3d_cinematic.png", "medical_3d.png", "medical_3d_story.png", "medical_3d_projection.png",
    "multihead_cinematic.png", "multihead.png", "multihead_story.png",
    "medical_3d_cinematic_thick.png",
    "transformer3d_unetr.png", "transformer3d_unetr_flat.png", "transformer3d_swinunetr.png",
    "monai_bundle_cinematic.png",
]
MOVIES = ["movie_pan_resnet.mp4", "movie_pan_vit.mp4", "movie_aging.mp4", "movie_cxr_occlusion.mp4",
          "movie_sliding_window_unetr.mp4"]

# docs that are not pages on the site (the landing page replaces the docs index)
SKIP_DOCS = {"README.md"}


def log(msg: str) -> None:
    print(f"  {msg}")


def web_images() -> None:
    """Write web-sized WebP versions (full width + thumbnail) of the example figures."""
    from PIL import Image

    dst = os.path.join(SRC, "media")
    os.makedirs(dst, exist_ok=True)
    for name in FIGURES:
        src = os.path.join(OUTPUTS, name)
        if not os.path.exists(src):
            log(f"skip {name} (missing)")
            continue
        stem = os.path.splitext(name)[0]
        im = Image.open(src).convert("RGB")
        for suffix, width in (("", 2400), ("_thumb", 960)):
            out = os.path.join(dst, f"{stem}{suffix}.webp")
            if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
                continue
            w = im.copy()
            if w.width > width:
                w = w.resize((width, round(w.height * width / w.width)), Image.LANCZOS)
            w.save(out, "WEBP", quality=86 if not suffix else 80, method=6)
    # social card (1200 × 630) from the 16:9 cat figure
    card = os.path.join(dst, "social_card.jpg")
    src = os.path.join(OUTPUTS, "resnet50_cinematic_169.png")
    if os.path.exists(src) and not os.path.exists(card):
        im = Image.open(src).convert("RGB")
        tw, th = 1200, 630
        s = max(tw / im.width, th / im.height)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        l, t = (im.width - tw) // 2, (im.height - th) // 2
        im.crop((l, t, l + tw, t + th)).save(card, quality=88)


def movies() -> None:
    """Copy the movies and extract a poster frame for each."""
    dst = os.path.join(SRC, "media")
    os.makedirs(dst, exist_ok=True)
    for name in MOVIES:
        src = os.path.join(OUTPUTS, name)
        if not os.path.exists(src):
            log(f"skip {name} (missing)")
            continue
        shutil.copy2(src, os.path.join(dst, name))
        poster = os.path.join(dst, os.path.splitext(name)[0] + "_poster.webp")
        if os.path.exists(poster):
            continue
        try:
            import imageio.v2 as imageio
            from PIL import Image

            r = imageio.get_reader(src)
            n = r.count_frames()
            frame = r.get_data(min(n - 1, max(0, n // 3)))
            r.close()
            im = Image.fromarray(frame)
            if im.width > 1600:
                im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
            im.save(poster, "WEBP", quality=80)
        except Exception as e:  # imageio-ffmpeg not installed: the video still works without a poster
            log(f"no poster for {name} ({type(e).__name__})")


def rewrite_links(text: str) -> str:
    """Point repository-relative links at site media or at GitHub."""
    # figures → web-sized WebP
    text = re.sub(r"\]\(\.\./examples/outputs/([\w\-]+)\.png\)", r"](media/\1.webp)", text)
    # anything else outside docs/ → the file on GitHub
    text = re.sub(r"\]\(\.\./([^)#\s]+)\)", lambda m: f"]({REPO}/blob/main/{m.group(1)})", text)
    return text


def copy_docs() -> None:
    docs = os.path.join(ROOT, "docs")
    for dirpath, dirnames, filenames in os.walk(docs):
        rel = os.path.relpath(dirpath, docs)
        if rel.startswith("deck") and "assets" in rel.split(os.sep):
            continue
        for f in filenames:
            src = os.path.join(dirpath, f)
            relf = os.path.normpath(os.path.join(rel, f))
            if relf in SKIP_DOCS or f.endswith((".js", ".py")) or f == ".DS_Store":
                continue
            dst = os.path.join(SRC, relf)
            if relf == os.path.join("deck", "README.md"):
                dst = os.path.join(SRC, "deck", "index.md")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if f.endswith(".md"):
                with open(src, encoding="utf-8") as fh:
                    text = rewrite_links(fh.read())
                text = text.replace("](deck/README.md)", "](deck/index.md)")
                with open(dst, "w", encoding="utf-8") as fh:
                    fh.write(text)
            else:
                shutil.copy2(src, dst)
    # root documents that become pages
    for name, title in (("CHANGELOG.md", None), ("CONTRIBUTING.md", None)):
        with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
            text = fh.read()
        text = re.sub(r"\]\((?!http|#)([^)\s]+)\)", lambda m: f"]({REPO}/blob/main/{m.group(1)})", text)
        with open(os.path.join(SRC, name.lower()), "w", encoding="utf-8") as fh:
            fh.write(text)


def copy_site_pages() -> None:
    pages = os.path.join(HERE, "pages")
    for dirpath, _, filenames in os.walk(pages):
        for f in filenames:
            src = os.path.join(dirpath, f)
            dst = os.path.join(SRC, os.path.relpath(src, pages))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    explorer = os.path.join(OUTPUTS, "resnet50_interactive.html")
    if os.path.exists(explorer):
        os.makedirs(os.path.join(SRC, "media", "explorer"), exist_ok=True)
        shutil.copy2(explorer, os.path.join(SRC, "media", "explorer", "resnet50.html"))


def assemble() -> None:
    print("assembling website/_src")
    media = os.path.join(SRC, "media")
    keep = None
    if os.path.isdir(media):  # keep generated media between runs (it is the slow part)
        keep = os.path.join(HERE, "_media_cache")
        shutil.rmtree(keep, ignore_errors=True)
        shutil.move(media, keep)
    shutil.rmtree(SRC, ignore_errors=True)
    os.makedirs(SRC)
    if keep:
        shutil.move(keep, media)
    copy_docs()
    copy_site_pages()
    web_images()
    movies()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="assemble, then run `mkdocs serve`")
    ap.add_argument("--no-build", action="store_true", help="only assemble website/_src")
    ap.add_argument("--strict", action="store_true", help="fail on mkdocs warnings (used in CI)")
    a = ap.parse_args()
    assemble()
    cfg = os.path.join(HERE, "mkdocs.yml")
    if a.no_build:
        return 0
    cmd = [sys.executable, "-m", "mkdocs", "serve" if a.serve else "build", "-f", cfg]
    if a.strict and not a.serve:
        cmd.append("--strict")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
