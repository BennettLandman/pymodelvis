"""neural_flow — representation-flow visualisation for PyTorch models.

    from neural_flow import visualize_model
    fig = visualize_model(model, input_tensor, output="activation_flow.png")
"""
from .api import FlowResult, draw, trace_model, visualize_model
from .config import FlowConfig
from .outputs import OutputView
from .raster import Visual, visualize_tensor
from .tensors import TensorSummary, summarize_tensor

__version__ = "0.1.0"


def animate_inputs(*args, **kwargs):
    """Movie of the activation flow over a sequence of inputs (see :mod:`neural_flow.movie`)."""
    from .movie import animate_inputs as _a

    return _a(*args, **kwargs)


def animate_model(*args, **kwargs):
    from .animate import animate_model as _a

    return _a(*args, **kwargs)


__all__ = ["visualize_model", "trace_model", "draw", "animate_model", "animate_inputs", "FlowConfig", "FlowResult",
           "OutputView", "TensorSummary", "Visual", "summarize_tensor", "visualize_tensor", "__version__"]
