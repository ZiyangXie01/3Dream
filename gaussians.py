"""
Gaussian splat data structures and operations.
"""

import torch
import numpy as np
from typing import List, NamedTuple, Tuple
from dataclasses import dataclass


class Gaussians3D(NamedTuple):
    """Represents a collection of 3D Gaussians for rendering."""
    means: torch.Tensor          # (N, 3) xyz positions
    scales: torch.Tensor         # (N, 3) scale in each axis
    quats: torch.Tensor          # (N, 4) quaternions (wxyz)
    colors: torch.Tensor         # (N, 3) RGB colors [0, 1]
    opacities: torch.Tensor      # (N,) opacity values [0, 1]

    def to(self, device: torch.device) -> "Gaussians3D":
        return Gaussians3D(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats=self.quats.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
        )


@dataclass
class SceneMetadata:
    """Metadata for the reconstructed scene."""
    resolution: Tuple[int, int]  # (width, height)
    focal_px: float              # focal length in pixels


def extract_layer_gaussians(
    points: torch.Tensor,
    depth: torch.Tensor,
    colors: torch.Tensor,
    mask: torch.Tensor,
    focal: float,
    scale_multiplier: float = 1.0,
    conf_threshold: float = 0.0,
) -> Gaussians3D:
    """
    Convert a layer's 3D points to Gaussian splats.
    
    Args:
        points: (H, W, 3) 3D points in camera space
        depth: (H, W) depth map
        colors: (H, W, 3) RGB colors [0, 1]
        mask: (H, W) boolean mask for valid pixels
        focal: Focal length in pixels
        scale_multiplier: Multiplier for Gaussian scale
        conf_threshold: Minimum depth threshold
        
    Returns:
        Gaussians3D for this layer
    """
    device = points.device
    H, W = depth.shape
    
    # Flatten
    points_flat = points.reshape(H * W, 3)
    depths_flat = depth.reshape(H * W)
    colors_flat = colors.reshape(H * W, 3)
    mask_flat = mask.reshape(H * W)
    
    # Filter by mask and valid depth
    valid_mask = mask_flat & (depths_flat > conf_threshold) & torch.isfinite(depths_flat)
    valid_mask = valid_mask & torch.isfinite(points_flat).all(dim=-1)
    
    means = points_flat[valid_mask]
    rgb = colors_flat[valid_mask]
    d = depths_flat[valid_mask]
    
    if means.shape[0] == 0:
        return Gaussians3D(
            means=torch.empty(0, 3, device=device),
            scales=torch.empty(0, 3, device=device),
            quats=torch.empty(0, 4, device=device),
            colors=torch.empty(0, 3, device=device),
            opacities=torch.empty(0, device=device),
        )
    
    # Clamp RGB to valid range
    rgb = rgb.clamp(0.0, 1.0)
    
    # Scale from depth / focal (pixel size in world space)
    pixel_size = d / focal * scale_multiplier
    pixel_size = torch.clamp(pixel_size, min=1e-6, max=100.0)
    scales = pixel_size.unsqueeze(-1).expand(-1, 3)  # Isotropic
    
    # Identity quaternions (spherical Gaussians)
    quats = torch.zeros(means.shape[0], 4, device=device)
    quats[:, 0] = 1.0  # w=1, x=y=z=0
    
    # Full opacity for layer pixels
    opacities = torch.ones(means.shape[0], device=device) * 0.95
    
    return Gaussians3D(
        means=means.float(),
        scales=scales.float(),
        quats=quats.float(),
        colors=rgb.float(),
        opacities=opacities.float(),
    )


def combine_gaussians(gaussians_list: List[Gaussians3D]) -> Gaussians3D:
    """Combine multiple Gaussians3D into one."""
    if not gaussians_list:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return Gaussians3D(
            means=torch.empty(0, 3, device=device),
            scales=torch.empty(0, 3, device=device),
            quats=torch.empty(0, 4, device=device),
            colors=torch.empty(0, 3, device=device),
            opacities=torch.empty(0, device=device),
        )
    
    # Filter out empty Gaussians
    non_empty = [g for g in gaussians_list if g.means.shape[0] > 0]
    
    if not non_empty:
        device = gaussians_list[0].means.device
        return Gaussians3D(
            means=torch.empty(0, 3, device=device),
            scales=torch.empty(0, 3, device=device),
            quats=torch.empty(0, 4, device=device),
            colors=torch.empty(0, 3, device=device),
            opacities=torch.empty(0, device=device),
        )
    
    return Gaussians3D(
        means=torch.cat([g.means for g in non_empty], dim=0),
        scales=torch.cat([g.scales for g in non_empty], dim=0),
        quats=torch.cat([g.quats for g in non_empty], dim=0),
        colors=torch.cat([g.colors for g in non_empty], dim=0),
        opacities=torch.cat([g.opacities for g in non_empty], dim=0),
    )
