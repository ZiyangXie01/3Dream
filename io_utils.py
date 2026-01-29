"""
I/O utilities for PLY export and file operations.
"""

import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

from gaussians import Gaussians3D


def save_gaussians_ply(
    gaussians: Gaussians3D,
    output_path: str,
    verbose: bool = True,
) -> None:
    """
    Save Gaussians to a PLY file compatible with 3D Gaussian Splatting viewers.
    
    Args:
        gaussians: The Gaussians to save
        output_path: Path to save the PLY file
        verbose: Whether to print save message
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy
    xyz = gaussians.means.cpu().numpy()
    scales = gaussians.scales.cpu().numpy()
    quats = gaussians.quats.cpu().numpy()
    colors = gaussians.colors.cpu().numpy()
    opacities = gaussians.opacities.cpu().numpy()
    
    # Convert scales to log space (standard 3DGS format)
    scale_logits = np.log(np.clip(scales, 1e-8, None))
    
    # Convert colors to spherical harmonics DC component
    # SH0 coefficient: c = (rgb - 0.5) / sqrt(1/(4*pi))
    sh_coeff = np.sqrt(1.0 / (4.0 * np.pi))
    sh_dc = (colors - 0.5) / sh_coeff
    
    # Convert opacity to logit space
    opacities_clamped = np.clip(opacities, 1e-6, 1 - 1e-6)
    opacity_logits = np.log(opacities_clamped / (1 - opacities_clamped))
    
    # Build structured array
    num_points = len(xyz)
    dtype_full = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    
    elements = np.empty(num_points, dtype=dtype_full)
    attributes = np.concatenate([
        xyz,
        sh_dc,
        opacity_logits[:, None],
        scale_logits,
        quats,
    ], axis=1)
    elements[:] = list(map(tuple, attributes))
    
    vertex_element = PlyElement.describe(elements, "vertex")
    plydata = PlyData([vertex_element])
    plydata.write(str(output_path))
    
    if verbose:
        print(f"Saved {num_points} Gaussians to: {output_path}")
