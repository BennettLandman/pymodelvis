# Example slide deck

[`neural_flow_deck.pptx`](neural_flow_deck.pptx) is a 15-slide, 16:9 PowerPoint deck built
entirely from the package's example outputs. It is included as an example of what can be made for
a talk. It has four embedded movies, speaker notes on every slide, and a black design that matches
the cinematic figures.

![deck preview](deck_preview.jpg)

| # | slide |
|---|---|
| 1 | Title (the ResNet-50 cat banner) |
| 2 | Project scoping: problem, goal, in / out of scope |
| 3 | Design criteria: six rules |
| 4 | Summary of the examples |
| 5 | ResNet-50 looks at a cat |
| 6 | The same trace as technical and story figures |
| 7 | ViT-B/16 |
| 8 | Chest X-ray (TorchXRayVision DenseNet-121) |
| 9 | 2-D U-Net |
| 10 | 3-D U-Net (`[B, C, X, Y, Z]`) |
| 11 | Multi-input, multi-head |
| 12 | Movies: camera pan (ResNet-50, ViT) |
| 13 | Movies: ageing subject, occlusion sweep |
| 14 | User instructions |
| 15 | References |

## Rebuilding

After re-rendering the examples (`bash examples/run_all.sh`):

```bash
pip install -e ".[all]"            # imageio for movie cover frames
npm install pptxgenjs              # Node.js ≥ 18
python docs/deck/prepare_assets.py docs/deck/assets        # crop figures, extract movie covers
node docs/deck/build_deck.js docs/deck/assets docs/deck/neural_flow_deck.pptx
```

`build_deck.js` holds all slide text and layout. Edit it to adapt the deck to your own models.
