// Build the neural_flow slide deck.
//   python docs/deck/prepare_assets.py docs/deck/assets     (crops figures, extracts movie covers)
//   node docs/deck/build_deck.js docs/deck/assets neural_flow_deck.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ASSETS = process.argv[2] || path.join(__dirname, "assets");
const OUTFILE = process.argv[3] || "neural_flow_deck.pptx";
const MOVIES = process.argv[4] || path.join(__dirname, "..", "..", "examples", "outputs");
const SIZES = JSON.parse(fs.readFileSync(path.join(ASSETS, "sizes.json"), "utf8"));

const C = {
  bg: "09090B", card: "16161B", card2: "1E1E25", line: "2E2E36",
  text: "F4F4F5", body: "D4D4D8", muted: "A1A1AA", faint: "71717A",
  amber: "FBBF24", cyan: "67E8F9", blue: "60A5FA",
};
const HEAD = "Arial", BODY = "Calibri", MONO = "Courier New";
const W = 13.333, H = 7.5, MX = 0.6;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "MASI Lab, Vanderbilt University";
pres.title = "neural_flow — representation-flow visualisation for PyTorch";

const asset = (n) => path.join(ASSETS, n);
const has = (n) => fs.existsSync(asset(n));

function base(kicker, title) {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  if (kicker) {
    s.addShape(pres.shapes.OVAL, { x: MX, y: 0.47, w: 0.11, h: 0.11, fill: { color: C.cyan }, line: { color: C.cyan } });
    s.addText(kicker.toUpperCase(), { x: MX + 0.2, y: 0.36, w: 9, h: 0.32, fontFace: HEAD, fontSize: 11, bold: true,
      color: C.amber, charSpacing: 4, margin: 0, isTextBox: true });
  }
  if (title) {
    s.addText(title, { x: MX, y: 0.72, w: W - 2 * MX, h: 0.75, fontFace: HEAD, fontSize: 30, bold: true,
      color: C.text, margin: 0, valign: "top", isTextBox: true });
  }
  return s;
}

// place an image inside a box, preserving its aspect ratio (centred)
function fit(s, name, x, y, w, h, opts = {}) {
  if (!has(name)) return null;
  const [pw, ph] = SIZES[name];
  const ar = pw / ph;
  let iw = w, ih = w / ar;
  if (ih > h) { ih = h; iw = h * ar; }
  const ix = opts.align === "left" ? x : x + (w - iw) / 2;
  const iy = opts.valign === "top" ? y : y + (h - ih) / 2;
  s.addImage({ path: asset(name), x: ix, y: iy, w: iw, h: ih });
  return { x: ix, y: iy, w: iw, h: ih };
}

function callouts(s, items, y, h = 1.35) {
  const n = items.length, gap = 0.3;
  const cw = (W - 2 * MX - gap * (n - 1)) / n;
  items.forEach((it, i) => {
    const x = MX + i * (cw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h, fill: { color: C.card }, line: { color: C.line, width: 0.75 },
      rectRadius: 0.08 });
    s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: y + 0.2, w: 0.34, h: 0.34, fill: { color: it.color || C.amber },
      line: { color: it.color || C.amber } });
    s.addText(String(i + 1), { x: x + 0.18, y: y + 0.2, w: 0.34, h: 0.34, fontFace: HEAD, fontSize: 12, bold: true,
      color: C.bg, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(it.t, { x: x + 0.62, y: y + 0.15, w: cw - 0.75, h: 0.45, fontFace: HEAD, fontSize: 13, bold: true,
      color: C.text, valign: "middle", margin: 0, isTextBox: true });
    s.addText(it.b, { x: x + 0.2, y: y + 0.66, w: cw - 0.4, h: h - 0.76, fontFace: BODY, fontSize: 12, color: C.muted,
      valign: "top", margin: 0, isTextBox: true });
  });
}

function caption(s, text, x, y, w) {
  s.addText(text, { x, y, w, h: 0.3, fontFace: BODY, fontSize: 11, italic: true, color: C.faint, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 1. title
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  fit(s, "hero_banner.jpg", 0, 2.55, W, 3.75);
  s.addText("neural_flow", { x: MX, y: 0.62, w: 9, h: 0.95, fontFace: HEAD, fontSize: 48, bold: true, color: C.text,
    margin: 0, isTextBox: true });
  s.addText("Seeing what a network sees: representation-flow visualisation for PyTorch", { x: MX, y: 1.55, w: 11.5, h: 0.5,
    fontFace: BODY, fontSize: 20, color: C.body, margin: 0, isTextBox: true });
  s.addText("pymodelvis  ·  MASI Lab / VALIANT, Vanderbilt University  ·  September 2026", { x: MX, y: 6.62, w: 10, h: 0.35,
    fontFace: BODY, fontSize: 13, color: C.muted, margin: 0, isTextBox: true });
  s.addText("ResNet-50 (ImageNet weights) looking at a cat", { x: W - MX - 5, y: 6.62, w: 5, h: 0.35, fontFace: BODY,
    fontSize: 11, italic: true, color: C.faint, align: "right", margin: 0, isTextBox: true });
  s.addNotes("neural_flow shows what one real input becomes as it moves through a PyTorch network, and why the network decides what it decides. The banner is the ResNet-50 example: books of feature maps, beams between stages, and what one unit in each stage sees.");
}

// ---------------------------------------------------------------- 2. scoping
{
  const s = base("Project scoping", "From boxes-and-arrows to representations");
  s.addText([
    { text: "The problem", options: { bold: true, color: C.amber, fontSize: 14, breakLine: true } },
    { text: "Graph viewers such as Netron and TensorBoard graphs draw operations as boxes. They don't show what a specific input becomes, which region drives a decision, or how representations change as the input changes. That is what we need for teaching, debugging, papers and talks, and especially for 3-D medical models.", options: { color: C.body, fontSize: 15, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "The goal", options: { bold: true, color: C.amber, fontSize: 14, breakLine: true } },
    { text: "For any PyTorch model and a real input, show the flow INPUT → FEATURES → LATENT → HEAD(S) → OUTPUT using the actual activations, at presentation quality, and as movies over changing inputs.", options: { color: C.body, fontSize: 15 } },
  ], { x: MX, y: 1.75, w: 5.9, h: 4.6, fontFace: BODY, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true });

  const cards = [
    { t: "In scope", c: C.cyan, items: ["Arbitrary PyTorch models: CNN, U-Net, ViT/Swin, 3-D, multi-input / multi-head",
      "Automatic stage selection with explicit overrides",
      "Static PNG / SVG / PDF, interactive HTML, movies",
      "Gradient explanations: receptive fields, contributions, Grad-CAM",
      "Memory-safe capture of very large 3-D activations"] },
    { t: "Out of scope", c: C.faint, items: ["Training or fine-tuning models",
      "Editing or exporting computational graphs",
      "Feature visualisation by optimisation (DeepDream-style)",
      "Non-PyTorch frameworks; clinical validation"] },
  ];
  let y = 1.75;
  cards.forEach((cd) => {
    const h = cd.items.length * 0.39 + 0.6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y, w: 5.73, h, fill: { color: C.card }, line: { color: C.line, width: 0.75 }, rectRadius: 0.08 });
    s.addText(cd.t, { x: 7.25, y: y + 0.14, w: 5.2, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: cd.c, margin: 0, isTextBox: true });
    s.addText(cd.items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < cd.items.length - 1 } })),
      { x: 7.25, y: y + 0.52, w: 5.3, h: h - 0.6, fontFace: BODY, fontSize: 13, color: C.body, valign: "top", margin: 0, paraSpaceAfter: 5, isTextBox: true });
    y += h + 0.25;
  });
  s.addText("Delivered: Python package · 50 CPU tests · 4 design docs · 9 examples on 5 real and 3 trained demo models · 4 movies · UNesT checkout",
    { x: MX, y: 6.7, w: W - 2 * MX, h: 0.35, fontFace: BODY, fontSize: 12, color: C.muted, margin: 0, isTextBox: true });
  s.addNotes("Scope: representation flow for one real input (or a sequence of inputs), not graph drawing. Deliberately out of scope: training, graph editing, optimisation-based feature visualisation, and any claim of clinical validity for the demo models.");
}

// ---------------------------------------------------------------- 3. design criteria
{
  const s = base("Design criteria", "Six rules the package is built around");
  const crit = [
    ["Representation, not computation", "5–12 meaningful stages chosen automatically. ReLUs, norms, dropout and reshapes are folded away."],
    ["Architecture-agnostic", "Forward hooks plus runtime dataflow tracing recover skips, merges and branches. torch.fx is optional; failures fall back gracefully."],
    ["Non-invasive and memory-safe", "eval() + no_grad(); every hook removed; user tensors untouched. Large tensors are reduced on-device within max_capture_mb."],
    ["Faithful", "Every pixel comes from the actual input. Percentile normalisation; outputs are never presented as probabilities unless they are; ≈ marks summaries."],
    ["Explanatory", "Beams: which region feeds the strongest unit. Circles: what that unit sees. Lines: W·x contributions. Grad-CAM evidence."],
    ["Presentation-grade", "Black cinematic theme, Inter typeface, 16:9 wrapping, vector SVG/PDF. Movies keep channels, colours and scale fixed across frames."],
  ];
  const cw = (W - 2 * MX - 2 * 0.3) / 3, ch = 2.35;
  crit.forEach(([t, b], i) => {
    const x = MX + (i % 3) * (cw + 0.3), y = 1.75 + Math.floor(i / 3) * (ch + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, fill: { color: C.card }, line: { color: C.line, width: 0.75 }, rectRadius: 0.08 });
    s.addText(String(i + 1).padStart(2, "0"), { x: x + 0.25, y: y + 0.2, w: 1, h: 0.5, fontFace: HEAD, fontSize: 26, bold: true, color: i % 2 ? C.cyan : C.amber, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.25, y: y + 0.78, w: cw - 0.5, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: C.text, margin: 0, isTextBox: true });
    s.addText(b, { x: x + 0.25, y: y + 1.2, w: cw - 0.5, h: ch - 1.3, fontFace: BODY, fontSize: 12.5, color: C.muted, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("These criteria came from the original specification: representation-flow rather than a computational graph, arbitrary architectures, memory safety for large 3-D tensors, faithful rendering, and presentation quality. Explanations and movies were added in the second round.");
}

// ---------------------------------------------------------------- 4. summary of examples
{
  const s = base("Summary of examples", "Nine examples, five with real pretrained weights");
  const rows = [
    ["resnet_cine_thumb.jpg", "ResNet-50 · the cat", "Real ImageNet weights (timm / torchvision)", "Hierarchy of features, receptive fields 8 → 224 px, Grad-CAM"],
    ["vit_cine_thumb.jpg", "ViT-B/16", "Real ImageNet weights", "Tokens as a 14×14 grid; attention globalises by block 4"],
    ["cxr_cine_thumb.jpg", "Chest X-ray DenseNet-121", "Real: TorchXRayVision, NIH ChestX-ray14 image", "Cardiomegaly evidence on the heart; ± contributions"],
    ["unet_cine_thumb.jpg", "2-D U-Net", "Trained on synthetic microscopy", "Skip connections, U layout, segmentation overlay"],
    ["med3d_cine_thumb.jpg", "3-D U-Net", "Trained on synthetic MRI volumes", "[B,C,X,Y,Z] as voxel blocks; ortho slices; 3-D segmentation"],
    ["multi_cine_thumb.jpg", "Multi-input, multi-head", "Trained on synthetic MRI + clinical", "Fusion and branching into 3 heads"],
    ["pan_cover.jpg", "Movies", "ResNet-50, ViT, DenseNet, multi-head", "Camera pan, subject ageing, occlusion sweep"],
  ];
  const y0 = 1.72, rh = 0.66, tw = 1.25;
  const cols = [MX + tw + 0.25, 4.85, 8.3];
  const widths = [2.85, 3.3, 4.4];
  ["Example", "Model / weights", "What it shows"].forEach((h, i) => s.addText(h.toUpperCase(),
    { x: cols[i], y: y0 - 0.02, w: widths[i], h: 0.3, fontFace: HEAD, fontSize: 10, bold: true, color: C.faint, charSpacing: 2, margin: 0, isTextBox: true }));
  rows.forEach((r, i) => {
    const y = y0 + 0.38 + i * rh;
    if (i % 2 === 0) s.addShape(pres.shapes.RECTANGLE, { x: MX, y: y - 0.04, w: W - 2 * MX, h: rh - 0.04, fill: { color: C.card }, line: { color: C.card } });
    fit(s, r[0], MX + 0.05, y, tw, rh - 0.12);
    s.addText(r[1], { x: cols[0], y, w: widths[0], h: rh - 0.12, fontFace: HEAD, fontSize: 13, bold: true, color: C.text, valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[2], { x: cols[1], y, w: widths[1], h: rh - 0.12, fontFace: BODY, fontSize: 12, color: C.body, valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[3], { x: cols[2], y, w: widths[2], h: rh - 0.12, fontFace: BODY, fontSize: 12, color: C.muted, valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("Ready to run on your machine: MASI UNesT whole-brain segmentation (MONAI bundle) via tools/checkout_real_models.py",
    { x: MX, y: 6.85, w: W - 2 * MX, h: 0.32, fontFace: BODY, fontSize: 12, italic: true, color: C.amber, margin: 0, isTextBox: true });
  s.addNotes("Real weights: ResNet-50, ViT-B/16 and the TorchXRayVision DenseNet. The U-Net, 3-D U-Net and multi-head models were trained briefly on built-in synthetic data, so their activations are meaningful. They demonstrate the tool, not clinical performance.");
}

// ---------------------------------------------------------------- example slides (wide figure + callouts)
function figureSlide(kicker, title, img, items, notes, capText) {
  const s = base(kicker, title);
  const box = fit(s, img, MX, 1.6, W - 2 * MX, 3.55);
  if (capText && box) caption(s, capText, MX, box.y + box.h + 0.02, W - 2 * MX);
  callouts(s, items, 5.55, 1.55);
  s.addNotes(notes);
  return s;
}

figureSlide("Example 1 · real weights", "ResNet-50 looks at a cat", "resnet_cine.jpg", [
  { t: "Front page = PCA of all maps", b: "Three principal components of every channel as RGB: the whole layer at a glance. Deep layers separate the face." },
  { t: "Beams between stages", b: "The region of the previous stage that feeds each stage's strongest unit: a small window, then wider after downsampling.", color: C.cyan },
  { t: "What one unit sees", b: "|∂unit/∂input| crops: the receptive field grows 8 → 22 → 37 → 68 → 224 px." },
  { t: "Why 'Egyptian cat'", b: "Amber and blue lines are the largest weight × activation terms. Grad-CAM evidence lies on the face.", color: C.cyan },
], "The hero example, kept from the first round. Everything is computed from this one input: activations from the capture pass; beams, circles, lines and Grad-CAM from one gradient pass.",
"style=\"cinematic\" · theme=\"black\" · 7 stages chosen automatically from 183 module calls");

{
  const s = base("Example 1 · same trace, other renderers", "Technical figures for papers, story mode for teaching");
  const b1 = fit(s, "resnet_tech.jpg", MX, 1.62, W - 2 * MX, 2.55);
  caption(s, "style=\"technical\": channel mosaics, module names and types, exact shapes; channel depth shown as offset sheets", MX, b1 ? b1.y + b1.h + 0.02 : 4.2, W - 2 * MX);
  const b2 = fit(s, "resnet_story.jpg", MX, 4.6, W - 2 * MX, 2.3);
  caption(s, "style=\"story\": conceptual stages with grouped labels (LOW-LEVEL → STRUCTURAL → HIGH-LEVEL → LATENT → HEAD)", MX, b2 ? b2.y + b2.h + 0.02 : 6.95, W - 2 * MX);
  s.addNotes("The same FlowResult can be rendered three ways. The technical style targets publications (SVG/PDF keep text and arrows as vectors); the story style targets teaching.");
}

figureSlide("Example 2 · real weights", "A Vision Transformer mixes the whole image early", "vit_cine.jpg", [
  { t: "Tokens become a grid", b: "197 tokens × 768: CLS separated, 196 patch tokens reshaped to 14 × 14 pages." },
  { t: "Rays instead of windows", b: "Dependencies between blocks are scattered: every token can draw on the whole image.", color: C.cyan },
  { t: "Receptive fields go global", b: "17 px at the patch embedding, 42 px at block 2, ≈ 170 px by block 4. Compare the CNN's gradual growth." },
  { t: "Robust channel ranking", b: "Transformers carry a few near-constant 'massive' channels, so pages are ranked by p90 − p10 spread.", color: C.cyan },
], "ViT-B/16 (ImageNet-1k, JAX port via timm). Same input as ResNet-50: Egyptian cat, p = 0.98.",
"timm vit_base_patch16_224 · blocks sampled 0, 2, 4, 7, 9, 11 · Egyptian cat p = 0.98");

figureSlide("Example 3 · real medical model", "Chest X-ray: where the Cardiomegaly evidence lies", "cxr_cine.jpg", [
  { t: "Real model, public image", b: "TorchXRayVision DenseNet-121 (7 datasets, 18 pathologies) on NIH ChestX-ray14 00000001_000." },
  { t: "Evidence on the heart", b: "Grad-CAM for Cardiomegaly (p = 0.62) concentrates on the cardiac silhouette.", color: C.cyan },
  { t: "Arguments for and against", b: "Amber latent units push Cardiomegaly up; blue ones push it down. Both are shown." },
  { t: "Honest outputs", b: "output_types = multilabel_probs: calibrated probabilities, top-5 listed, 7 labels above 0.5.", color: C.cyan },
], "Real clinical-style model on a real, public chest radiograph. The occlusion movie (later) tests whether the prediction depends on the heart region.",
"DenseNet-121 dense blocks 1–4 · 1024-d latent · 18 calibrated outputs");

{
  const s = base("Example 4 · trained on synthetic data", "2-D U-Net: skip connections as bridges");
  const b = fit(s, "unet_cine.jpg", MX, 1.6, 8.3, 5.3);
  const x = MX + 8.6, w = W - MX - x;
  s.addText([
    { text: "Topology from the actual execution", options: { bold: true, color: C.text, breakLine: true } },
    { text: "A TorchFunctionMode records every tensor op; walking back from each stage's inputs finds enc1 → dec1, enc2 → dec2 … even through torch.cat in forward().", options: { color: C.muted, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "U layout, automatically", options: { bold: true, color: C.text, breakLine: true } },
    { text: "Resolution-matched skips trigger a layout that dips by resolution level, so skips become horizontal bridges.", options: { color: C.muted, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Segmentation overlay", options: { bold: true, color: C.text, breakLine: true } },
    { text: "Cells (red) and debris (blue) are overlaid on the input; connector layers such as pooling and up-convs are absorbed into their blocks.", options: { color: C.muted } },
  ], { x, y: 1.75, w, h: 5.0, fontFace: BODY, fontSize: 13, valign: "top", margin: 0, paraSpaceAfter: 3, isTextBox: true });
  s.addNotes("A small U-Net trained for 200 steps on built-in synthetic microscopy images. Skip connections come from runtime dataflow, not from names.");
}

{
  const s = base("Example 5 · [B, C, X, Y, Z] as a first-class tensor", "3-D U-Net: from anatomy to voxel blocks");
  const b1 = fit(s, "med3d_cine.jpg", MX, 1.6, 7.6, 5.35);
  const b2 = fit(s, "med3d_proj.jpg", 8.55, 1.75, 4.2, 2.6);
  caption(s, "volume_mode=\"projection\" (max-intensity per axis)", 8.55, b2 ? b2.y + b2.h + 0.02 : 4.4, 4.2);
  s.addText([
    { text: "Input: cut-away anatomy with orthogonal slices through the centre of mass.", options: { bullet: true, breakLine: true } },
    { text: "Features: solid voxels for the strongest activations inside a translucent block; small volumes keep discrete cubes.", options: { bullet: true, breakLine: true } },
    { text: "Slices through each stage's activation peak, not the geometric centre.", options: { bullet: true, breakLine: true } },
    { text: "Output: 'glass' anatomy with an opaque lesion (Dice 0.89 on the test volume).", options: { bullet: true } },
  ], { x: 8.55, y: 4.8, w: 4.2, h: 2.3, fontFace: BODY, fontSize: 12, color: C.body, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true });
  s.addNotes("The 3-D U-Net is trained on synthetic MRI-like head phantoms with lesions. The renderer is a small orthographic ray caster on grid_sample. Tensors are nibabel/MONAI [X, Y, Z] by default.");
}

{
  const s = base("Example 6 · multi-input, multi-head", "MRI + clinical vector → segmentation, lesion and brain age");
  fit(s, "multi_cine.jpg", MX, 1.55, 7.1, 5.55);
  const x = 8.05, w = W - MX - x;
  const pts = [
    ["Dict inputs, dict outputs", "{'mri', 'clinical'} in; {'lesion_seg', 'lesion_present', 'brain_age'} out, each with its own card."],
    ["Fusion and branching", "The clinical MLP sits next to its late fusion; heads fan out from the shared representation."],
    ["Conceptual labels", "ENCODER, DECODER, FUSION, CLINICAL FEATURES and head names are inferred from the graph."],
    ["Conservative semantics", "Sigmoid probability for lesion_present, regression for brain_age, a 3-D mask for lesion_seg."],
  ];
  pts.forEach(([t, b], i) => {
    const y = 1.75 + i * 1.3;
    s.addShape(pres.shapes.OVAL, { x, y: y + 0.02, w: 0.34, h: 0.34, fill: { color: i % 2 ? C.cyan : C.amber }, line: { color: i % 2 ? C.cyan : C.amber } });
    s.addText(String(i + 1), { x, y: y + 0.02, w: 0.34, h: 0.34, fontFace: HEAD, fontSize: 12, bold: true, color: C.bg, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.5, y, w: w - 0.5, h: 0.38, fontFace: HEAD, fontSize: 14, bold: true, color: C.text, valign: "middle", margin: 0, isTextBox: true });
    s.addText(b, { x: x + 0.5, y: y + 0.42, w: w - 0.5, h: 0.8, fontFace: BODY, fontSize: 12.5, color: C.muted, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("A small multi-task network trained for 600 steps on synthetic data. It tracks age (predicted 29 → 80 for true 25 → 85) and detects lesions above about 2 voxels in radius.");
}

// ---------------------------------------------------------------- movie slides (embedded video)
function movieSlide(kicker, title, vids, notes) {
  const s = base(kicker, title);
  const n = vids.length, gap = 0.35;
  const vw = (W - 2 * MX - gap * (n - 1)) / n, vh = vw * 9 / 16;
  vids.forEach((v, i) => {
    const x = MX + i * (vw + gap), y = 1.7;
    const mp4 = path.join(MOVIES, v.file);
    if (fs.existsSync(mp4)) {
      s.addMedia({ type: "video", path: mp4, x, y, w: vw, h: vh, cover: has(v.cover) ? "image/jpeg;base64," + fs.readFileSync(asset(v.cover)).toString("base64") : undefined });
    } else if (has(v.cover)) {
      s.addImage({ path: asset(v.cover), x, y, w: vw, h: vh });
    }
    s.addText(v.t, { x, y: y + vh + 0.15, w: vw, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: C.text, margin: 0, isTextBox: true });
    s.addText(v.b, { x, y: y + vh + 0.52, w: vw, h: 1.3, fontFace: BODY, fontSize: 12, color: C.muted, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes(notes);
}

movieSlide("Movies over changing inputs", "A camera pans across four photographs", [
  { file: "movie_pan_resnet.mp4", cover: "pan_cover.jpg", t: "ResNet-50", b: "Egyptian cat → espresso → drilling platform / crane → go-kart (the astronaut). The timeline and film strip track the top classes." },
  { file: "movie_pan_vit.mp4", cover: "vitpan_cover.jpg", t: "ViT-B/16", b: "The same pan through a transformer: rays gather from across the image, and predictions switch more sharply." },
], "Movies hold stages, channels, per-channel scale and the PCA colour basis fixed across frames, so any change you see is a change in activation. Beams, receptive fields, contributions and Grad-CAM move with the input.");

movieSlide("Movies where the network makes sense", "A subject ages; a patch hides the heart", [
  { file: "movie_aging.mp4", cover: "aging_cover.jpg", t: "One synthetic subject over time (3-D)", b: "Ventricles enlarge while a lesion appears and grows. Predicted brain age rises (29 → 72), lesion probability switches 0 → 1, and the segmented volume grows." },
  { file: "movie_cxr_occlusion.mp4", cover: "cxr_cover.jpg", t: "Occlusion sweep on a chest X-ray", b: "A grey patch slides over the film. Cardiomegaly falls when the heart is covered (and the model then prefers Hernia), a direct test of reliance." },
], "Both movies are in-distribution sweeps where the expected behaviour is known, so the movie checks the model rather than just decorating it.");

// ---------------------------------------------------------------- user instructions
{
  const s = base("User instructions", "Install, render, explore, animate");
  const blocks = [
    ["Install", "cd ~/dev/pymodelvis\npython -m venv .venv && source .venv/bin/activate\npip install -e \".[all]\"\npytest -q                       # 50 tests"],
    ["One figure", "from neural_flow import visualize_model\nvisualize_model(model, x, output=\"flow.png\",\n                style=\"cinematic\")   # black\n# style=\"technical\" | \"story\";  theme=\"light\"\n# figsize=(16, 9);  output=\"flow.svg\" | \".html\""],
    ["A movie", "from neural_flow import animate_inputs\nfrom neural_flow.sequences import pan\nframes = pan(img, window=256, steps=48)\nanimate_inputs(model, frames,\n               output=\"pan.mp4\", class_names=names)"],
    ["Examples and real models", "bash examples/run_all.sh        # every figure + movie\npython tools/checkout_real_models.py\npython examples/monai_bundle.py  # MASI UNesT\npython examples/monai_bundle.py --movie"],
  ];
  const cw = (W - 2 * MX - 0.3) / 2, ch = 2.45;
  blocks.forEach(([t, code], i) => {
    const x = MX + (i % 2) * (cw + 0.3), y = 1.7 + Math.floor(i / 2) * (ch + 0.3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, fill: { color: C.card }, line: { color: C.line, width: 0.75 }, rectRadius: 0.08 });
    s.addText(t, { x: x + 0.25, y: y + 0.15, w: cw - 0.5, h: 0.38, fontFace: HEAD, fontSize: 14, bold: true, color: i % 2 ? C.cyan : C.amber, margin: 0, isTextBox: true });
    s.addText(code, { x: x + 0.25, y: y + 0.62, w: cw - 0.5, h: ch - 0.75, fontFace: MONO, fontSize: 11.5, color: C.body, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("Outputs land in examples/outputs. The checkout script fetches the UNesT MONAI bundle, the MNI152 template and torchvision weights into real_models/. monai_bundle.py builds any MONAI bundle from its own config.");
}

// ---------------------------------------------------------------- references
{
  const s = base("Citations and references", "References");
  const refs = [
    "He K, et al. Deep residual learning for image recognition. CVPR 2016.",
    "Dosovitskiy A, et al. An image is worth 16×16 words: Transformers for image recognition at scale. ICLR 2021.",
    "Ronneberger O, et al. U-Net: Convolutional networks for biomedical image segmentation. MICCAI 2015.",
    "Çiçek Ö, et al. 3D U-Net: Learning dense volumetric segmentation from sparse annotation. MICCAI 2016.",
    "Selvaraju RR, et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV 2017.",
    "Luo W, et al. Understanding the effective receptive field in deep convolutional neural networks. NeurIPS 2016.",
    "Abnar S, Zuidema W. Quantifying attention flow in transformers. ACL 2020.",
    "Zeiler MD, Fergus R. Visualizing and understanding convolutional networks. ECCV 2014.",
    "Olah C, Mordvintsev A, Schubert L. Feature visualization. Distill 2017.",
    "Cohen JP, et al. TorchXRayVision: A library of chest X-ray datasets and models. MIDL 2022.",
    "Wang X, et al. ChestX-ray8: Hospital-scale chest X-ray database and benchmarks. CVPR 2017.",
    "Yu X, et al. UNesT: Local spatial representation learning with hierarchical transformer for efficient medical segmentation. Medical Image Analysis 2023.",
    "Cardoso MJ, et al. MONAI: An open-source framework for deep learning in healthcare. arXiv 2211.02701, 2022.",
    "Wightman R. PyTorch Image Models (timm). GitHub, 2019.",
    "Fonov V, et al. Unbiased average age-appropriate atlases for pediatric studies (MNI152 2009). NeuroImage 2011.",
    "van der Walt S, et al. scikit-image: Image processing in Python. PeerJ 2014.",
    "Roeder L. Netron: visualizer for neural network models. GitHub.",
  ];
  const half = Math.ceil(refs.length / 2);
  [refs.slice(0, half), refs.slice(half)].forEach((col, j) => {
    s.addText(col.map((r, i) => ({ text: r, options: { bullet: { type: "number", startAt: j * half + 1 }, breakLine: i < col.length - 1 } })),
      { x: MX + j * 6.2, y: 1.65, w: 5.95, h: 5.4, fontFace: BODY, fontSize: 11.5, color: C.body, valign: "top", margin: 0, paraSpaceAfter: 5, isTextBox: true });
  });
  s.addNotes("Primary references for the architectures, the explanation methods (Grad-CAM, effective receptive field, attention rollout), the real models and data, and the tooling.");
}

pres.writeFile({ fileName: OUTFILE }).then((f) => console.log("wrote " + f));
