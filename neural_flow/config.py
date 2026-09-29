"""Configuration object shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

LayerSpec = Union[str, Sequence[str], Callable[..., bool], None]


@dataclass
class FlowConfig:
    # ---------------- stage selection ----------------
    layer_selection: Union[str, Callable[..., bool]] = "auto"
    layers: Optional[Sequence[str]] = None          # explicit names, "re:<regex>" or "type:<ClassName>"
    exclude: Optional[Sequence[str]] = None          # same selector syntax; removed from auto selection
    max_stages: int = 10
    target_stages: Optional[int] = None              # None -> 8 (technical) / 6 (story) / 7 (cinematic)
    labels: Optional[Dict[str, str]] = None          # module name -> display label

    # ---------------- tensor -> visual ----------------
    max_channels: int = 16
    channel_strategy: str = "energy"                 # energy | variance | spread | mean_abs | even | pca
    normalize: str = "channel"                       # channel | stage
    percentiles: Tuple[float, float] = (1.0, 99.0)
    diverging: Union[bool, str] = False              # True | False | "auto"
    cmap: Optional[str] = None                       # activation colormap (theme default if None)
    volume_mode: str = "auto"                        # auto | volume | ortho | montage | projection
    projection: str = "max"                          # max | mean | meanabs
    volume_style: str = "voxels"                     # voxels (solid strong voxels in a translucent block) | cutaway | glow
    volume_axes: str = "xyz"                         # xyz (nibabel/MONAI [X,Y,Z]) | dhw (torch slice stack [D,H,W])
    token_mode: str = "auto"                         # auto | grid | heatmap | pca | l2 | mean
    cls_tokens: Optional[int] = None                 # number of leading special tokens (None = infer)
    patch_grid: Optional[Tuple[int, int]] = None     # (rows, cols) of patch tokens (None = infer)
    channels_last: Optional[bool] = None             # force interpretation of 4-D tensors
    capture_attention: bool = True

    # ---------------- outputs ----------------
    output_interpreter: Optional[Callable[..., Any]] = None
    class_names: Optional[Union[Sequence[str], Dict[str, Sequence[str]]]] = None
    output_types: Optional[Dict[str, str]] = None    # output name -> softmax|sigmoid|multilabel|multilabel_probs|regression|segmentation|embedding|raw
    top_k: int = 3
    input_names: Optional[Sequence[str]] = None
    batch_index: int = 0

    # ---------------- capture / memory ----------------
    capture_device: str = "cpu"
    max_capture_mb: float = 500.0
    max_spatial_2d: int = 128
    max_spatial_3d: int = 48
    max_input_2d: int = 256
    max_input_3d: int = 128
    topology: str = "auto"                           # auto | runtime | fx | sequential
    random_state: int = 42                           # reserved: all reductions are deterministic (no sampling)

    # ---------------- layout / rendering ----------------
    layout: str = "horizontal"                       # horizontal | vertical | wrap
    style: str = "technical"                         # technical | story | cinematic
    front_page: str = "pca"                          # cinematic decks: pca (layer summary in RGB) | channel
    explain: Optional[bool] = None                   # gradient explanations (default: on for cinematic)
    force_channels: Optional[Dict[str, List[int]]] = None   # stage key -> channel ids to show (movies)
    force_pca: Optional[Dict[str, Any]] = None       # stage key -> (loadings [3,C], mean [C]) fixed PCA basis
    theme: str = "light"                             # light | dark | black (cinematic defaults to black)
    figsize: Optional[Tuple[float, float]] = None
    dpi: int = 200
    title: Optional[str] = None
    subtitle: Optional[str] = None
    show_skips: bool = True
    edge_width: str = "features"                     # features | constant
    unet_layout: Union[bool, str] = "auto"           # dip the flow by resolution level for encoder/decoder nets
    font_scale: float = 1.0
    overlay_segmentation: bool = True

    def resolved_target(self) -> int:
        if self.target_stages is not None:
            return max(2, min(self.target_stages, self.max_stages))
        return min(self.max_stages, {"story": 6, "cinematic": 7}.get(self.style, 8))

    def updated(self, **kw) -> "FlowConfig":
        valid = {f.name for f in fields(self)}
        bad = set(kw) - valid
        if bad:
            raise TypeError(f"Unknown neural_flow option(s): {sorted(bad)}")
        return replace(self, **kw)


def make_config(config: Optional[FlowConfig] = None, **kw) -> FlowConfig:
    base = config or FlowConfig()
    kw = {k: v for k, v in kw.items() if v is not None}
    return base.updated(**kw) if kw else base
