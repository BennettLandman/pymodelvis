# neural_flow documentation

**Start here**

1. [Installation](installation.md)
2. [Command-line guide](cli.md): figures and movies without writing Python
3. [User guide](user_guide.md): the Python API, concepts and recipes
4. [Examples](examples.md): the gallery and how each figure was made

**Reference**

* [API reference](api.md): every function and every option
* [Movies](movies.md): input sequences, what is held fixed, cost
* [3-D models](volumes_3d.md): voxel spacing, UNETR / Swin UNETR / UNesT, sliding windows, 3-D inference movies
* [Real models](real_models.md): pretrained weights, MONAI bundles, MASI UNesT
* [Troubleshooting](troubleshooting.md)

**How it works**

* [Cinematic style](cinematic.md): the visual language and what beams, circles and lines mean
* [Stage selection](stage_selection.md): how the 5–12 stages are chosen
* [Tensor rendering](tensor_rendering.md): 2-D, 3-D, token, vector and attention tensors → pictures
* [Topology](topology.md): hooks, runtime dataflow tracing, the `torch.fx` findings
* [Architecture](architecture.md): package layout, data flow, design rules, tests

**Other**

* [Slide deck](deck/README.md): the example PowerPoint and how to rebuild it
* [References](references.md): the papers behind every demo model and method
* [About](about.md): why this project exists and how it was built
