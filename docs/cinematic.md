# The cinematic style

`style="cinematic"` is designed for talks, posters and movies. It uses the whitespace between
stages to show *how* information moves, not just *what* each stage holds. Everything drawn comes
from the actual input: activations from the capture pass, and beams, circles, lines and evidence
from one extra gradient pass (`neural_flow/explain.py`).

## Books of feature maps

Each spatial stage is drawn as a deck of pages seen at an angle (`artistic.render_deck`).

* **Pages are channels.** The deck shows the stage's most active channels (`channel_strategy`,
  default energy; variance for 3-D, robust p90−p10 spread for transformers). The number of pages
  grows with log₂(channels): 3 to 14.
* **Page size tracks spatial resolution** (logarithmically), so books shrink as the network
  downsamples.
* **Front page = PCA of all channels** (`front_page="pca"`). The first three principal components
  of the stage's channel-by-position matrix are mapped to R, G and B. Regions with the same colour
  have the same feature signature, which makes a whole layer readable at a glance. With trained
  networks, deep layers segment objects and parts.
* Each page is normalised to its own 1st–99.5th percentile range. Movies use one fixed range per
  channel across all frames.

## Beams: the region of the previous stage that feeds each stage's strongest unit

For stage *s*, take its front channel *c* and the position *p* of its maximum. The unit
u = A_s[c, p] is backpropagated to the **previous stage's** output A_{s−1}:

    D(q) = Σ_k | ∂u / ∂A_{s−1}[k, q] |

D is shown over the previous stage's grid.

* If D is **compact** (≤ 45 % of the grid above 15 % of its maximum), a translucent **frustum**
  joins its bounding box on the previous stage's front page to the unit's cell on this stage.
  This is the local window of a convolution, and it widens after downsampling.
* If D is **diffuse** (global pooling, attention), **rays** run from the strongest source
  positions to the unit. In a ViT, rays gather from all over the image from the first blocks on.

## Circles: what that unit sees in the input

The same unit backpropagated to the **input**, |∂u/∂x| summed over input channels, is its
empirical (effective) receptive field (Luo et al., 2016). Each circle shows the input cropped to
the region above 10 % of the maximum, brightness-weighted by the gradient. "sees ≈ N px" is that
region's extent. For a ResNet-50 on a 224-px image, the extent goes 8 → 22 → 37 → 68 → 224 px.
For 3-D inputs, the circle shows an axial slice through the receptive-field peak.

## Latent pixels and contribution lines

Vectors (pooled features, fusion layers, logits) are dot matrices: one dot per unit, brightness =
activation (binned beyond 512 units). For a **linear head** producing output *k*, the
contribution of latent unit *j* is

    c_j = W[k, j] · z_j          (Σ_j c_j + b_k = logit_k)

The 14 largest |c_j| are drawn as lines from their dots to the predicted output's dot. Amber
pushes the prediction up; blue pushes it down.

## Evidence

Grad-CAM (Selvaraju et al., 2017) for the predicted class is computed on the last spatial stage
that actually carries gradient to the output. A ViT's final patch tokens do not reach the CLS
output, so its Grad-CAM falls back one block. It is shown as a heat overlay under the prediction
card.

## Layout, typography and themes

* Columns follow the stage DAG. Branches fan out, merges converge, and encoder/decoder nets keep
  their skip arcs. With `figsize=` the flow wraps into as many bands as maximise the rendered
  scale.
* Headings are letter-spaced concept names (LOW-LEVEL FEATURES, ENCODER, FUSION, …), grouped
  over consecutive stages. Module names, shapes and "n of C maps" sit in muted type below.
* `theme="black"` (default) adds a soft bloom behind every raster, so activations read as
  illuminated pixels. `dark` and `light` work too.
* The typeface is Inter (bundled, SIL Open Font License).

## Movies

`animate_inputs` renders one cinematic frame per input.

1. Stages are chosen on frame 0 and reused.
2. Channels are ranked by their activity summed over **all** frames, so every frame shows the
   same channels.
3. The PCA basis is fitted on the middle frame and reused, so colours mean the same thing in
   every frame.
4. Per-channel display ranges are taken over all frames.
5. Beams, circles, contribution lines, Grad-CAM and outputs are recomputed per frame. They move.
6. The footer shows a film strip of the inputs and the time course of every output
   (probabilities, regressed values, segmented %), with a cursor at the current frame.
