"""
3Dream: Layered Image to 3D Gaussian Splats Pipeline.
"""

from .gaussians import Gaussians3D, SceneMetadata, extract_layer_gaussians, combine_gaussians
from .decompose import decompose_image, composite_layers
from .depth import load_moge_model, unload_moge_model, predict_depth, get_focal_px
from .rendering import render_spiral_video, TrajectoryParams, TrajectoryType
from .io_utils import save_gaussians_ply
from .infer import process_layered_image

__all__ = [
    # Gaussians
    "Gaussians3D",
    "SceneMetadata",
    "extract_layer_gaussians",
    "combine_gaussians",
    # Decomposition
    "decompose_image",
    "composite_layers",
    # Depth
    "load_moge_model",
    "unload_moge_model",
    "predict_depth",
    "get_focal_px",
    # Rendering
    "render_spiral_video",
    "TrajectoryParams",
    "TrajectoryType",
    # I/O
    "save_gaussians_ply",
    # Pipeline
    "process_layered_image",
]
