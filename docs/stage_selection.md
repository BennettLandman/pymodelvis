# Stage-selection heuristics

`neural_flow` shows a **representation flow**, not a computational graph. A stage is a module
*call* whose output is worth looking at. Everything here lives in `neural_flow/stages.py`; the
inputs are the `ModuleCall` records from the metadata pass (name, type, call index, parent,
children, input/output shapes, parameter count, execution order).

## 1. What is never a stage (trivial modules)

Activations (`ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Softmax`, …), `Dropout`/`DropPath`/`StochasticDepth`,
normalisation (`BatchNorm*`, `LayerNorm`, `GroupNorm`, `InstanceNorm*`, `RMSNorm`), and pure shape
helpers (`Identity`, `Flatten`, `Unflatten`, `Permute`, padding, pixel shuffle). Modules without a
tensor output are skipped as well.

A **wrapper** (a container with exactly one meaningful child) is looked through: expanding
`encoder` in torchvision's ViT goes straight to its 12 blocks rather than stopping at
`encoder.layers`.

## 2. Greedy module-tree cut (`layer_selection="auto"`)

1. **Frontier** = meaningful children of the root, in execution order.
2. **Expand** the frontier unit whose children reveal the most *new tensor shapes*. The shape
   signature drops the batch dimension and size-1 dims, so resolution, channel count and rank
   changes all count as new information. Examples:
   * ResNet `layer2` → two `BasicBlock`s at 128×28×28: nothing new, not expanded.
   * U-Net `root` → enc1…dec1: every resolution level is new.
3. Units are also expanded without new shapes while the frontier is smaller than `target_stages`
   (8 technical / 6 story). This is how a ViT's single `encoder` becomes blocks.
4. **Residual blocks and repeated blocks are natural units.** A unit whose input shape equals its
   output shape (and is not a plain `Sequential`/`ModuleList`), or which has a sibling of the same
   type and shape in the frontier, is not expanded for "new shapes". A transformer block's MLP
   hidden width is an implementation detail.
5. **Sampling on overflow.** When an expansion would exceed the budget (`max_stages` for
   informative expansions, `target_stages` otherwise), the children are sampled: shape-changing
   children, the first and the last are always kept, and the rest are spread evenly across depth.
6. **Run compression.** Runs of ≥ 3 homogeneous stages (same type and output shape; e.g. 12
   transformer blocks listed directly under the root) are resampled toward the target, keeping
   the first and last.
7. **Absorption.** If the frontier is still above target, connector leaves are dropped when the
   following *block* has the same spatial resolution: pooling, upsampling, padding,
   `PatchMerging`, `ConvTranspose`. U-Net `pool → enc2` becomes `enc2`; Swin's `PatchMerging` is
   folded into the next stage.
8. **Pruning.** Anything left above `max_stages` is removed by lowest importance score. The score
   is +3 for blocks/containers, +2 for parameterised leaves, +1 otherwise; +2 for the first stage;
   +3 for heads and the last stage; +2 for a rank/shape change; +1 for a resolution change; −1.5
   for a leaf followed by a same-resolution stage.

Results on reference models (defaults):

| model | stages |
|---|---|
| ResNet-18 | conv1, maxpool, layer1–4, avgpool, fc |
| ViT-B/16 | conv_proj, blocks 0/2/4/7/9/11, *representation* (CLS), heads |
| Swin-T | patch embed, 4 stages (PatchMerging absorbed), avgpool, head |
| 2-D / 3-D U-Net | enc1–3, bottleneck, dec3–1, seg head (pool / up-conv absorbed) |

## 2b. Architecture proposals (transformer U-Nets)

Before the generic cut, an architecture adapter may propose the stages itself. The one that does is
the *transformer U-Net* adapter (UNETR, Swin UNETR, UNesT, TransUNet-like models). It recognises, from
the runtime dataflow, a top-level unit containing transformer blocks whose hidden states are read by
two or more other top-level units, and then shows:

1. the backbone's **tapped** levels / blocks (those whose outputs leave the backbone), sampled evenly
   if there are more than fit, plus its patch embedding when there is room;
2. the other top-level units, **except** projection branches that only take a hidden state (or the
   input) to a higher resolution for one decoder level; these become skip bridges;
3. roles for the story / cinematic headings: PATCH EMBEDDING, TRANSFORMER ENCODER, BOTTLENECK,
   DECODER, SEGMENTATION HEAD.

It also raises the stage budget to 14 and turns the U layout on. A plain ViT classifier (one
consumer of the backbone) is not affected.

The *nnU-Net* adapter proposes the stages of U-Nets built by `dynamic_network_architectures`
(`PlainConvUNet`, `ResidualEncoderUNet`), recognised by structure: a module with `stages`,
`transpconvs` and `seg_layers` lists next to a module with `stages`. It shows the stem (residual
encoders), every encoder stage (the last one as BOTTLENECK), one stage per decoder level
(`decoder.stages[s]`; the transposed convolution and skip concatenation are folded into its edges),
and the final full-resolution `seg_layers` entry as the head. See [nnU-Net](nnunet.md).

## 3. Inserted representation stages

If a head consumes a *vector* but its predecessor stage is still spatial or token-shaped (ViT:
`x[:, 0]` between the last block and `heads`), a **representation** stage is inserted. It
captures the head's input, so the figure shows the spatial → latent → decision transition. Heads
that share the same input tensor share one representation stage.

## 4. Heads, branches, inputs

Heads are the stages whose outputs reach a model output in the runtime dataflow graph (see
`topology.md`). Several outputs produced from different stages make the figure branch. Model
inputs become input stages, so multimodal models show separate input lanes merging at the fusion
stage.

## 5. Explicit control

```python
layers=["encoder.stage1", "bottleneck", "decoder.stage2", "head"]   # exact names
layers=["re:^encoder\\.blocks\\.\\d+$"]                             # regex over module names
layers=["type:Conv3d"]                                               # by module type (regex)
layers=["conv#2"]                                                    # 3rd call of a reused module
layer_selection=lambda call: call.type_name.endswith("Block")        # predicate over ModuleCall
exclude=["type:Upsample"]
labels={"encoder.stage1": "stem"}
```

## 6. Story mode

`style="story"` lowers the target to 6. Architecture adapters then assign conceptual names:
INPUT, LOW-LEVEL / STRUCTURAL / HIGH-LEVEL FEATURES, LATENT SPACE, HEAD. U-Nets get
ENCODER / BOTTLENECK / DECODER / SEGMENTATION HEAD. Transformers get PATCH EMBEDDING and
LOCAL / CONTEXT / GLOBAL mixing. Consecutive stages with the same concept share one bracketed
label, and module names and types are hidden.
