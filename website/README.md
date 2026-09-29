# Project website

Source for **https://bennettlandman.github.io/pymodelvis/**, built with
[MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

The site has no content of its own apart from the landing page. Everything else is taken from the
repository at build time, so the website can't drift from the docs:

| on the site | comes from |
|---|---|
| landing page | `website/overrides/home.html` |
| Gallery page | `website/pages/gallery.md` |
| documentation pages | `docs/*.md` (links into `examples/outputs/` are rewritten to web images) |
| figures | `examples/outputs/*.png` → resized WebP in `media/` |
| movies, explorer | `examples/outputs/*.mp4`, `resnet50_interactive.html` |
| slide deck | `docs/deck/` |
| Changelog, Contributing | `CHANGELOG.md`, `CONTRIBUTING.md` |
| styling, lightbox, logo | `website/pages/assets/` |

## Preview locally

```bash
pip install -r website/requirements.txt
python website/build.py --serve      # http://127.0.0.1:8000/pymodelvis/ , reloads when website/ changes
python website/build.py              # static build → website/_site/
```

`--serve` watches the assembled copy, so after editing `docs/` rerun the command. The generated
folders `website/_src/` and `website/_site/` are git-ignored.

## Publishing

`.github/workflows/pages.yml` builds and deploys the site on every push to `main` that touches
`docs/`, `examples/outputs/`, `website/`, `CHANGELOG.md` or `CONTRIBUTING.md`, and can be started
by hand from the Actions tab. One-time setup: repository **Settings → Pages → Build and deployment
→ Source: GitHub Actions**.

## Notes

* MkDocs is pinned below 2.0: MkDocs 2.0 removes the theme and plugin system this site uses.
* To add a page, put it in `docs/` and add it to `nav:` in `website/mkdocs.yml`.
* To show a new figure, add its file name to `FIGURES` in `website/build.py`.
