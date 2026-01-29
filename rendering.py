"""
Spiral rendering pipeline for Gaussian splats.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
import imageio.v2 as iio

from gaussians import Gaussians3D, SceneMetadata


TrajectoryType = Literal["rotate", "rotate_forward", "swipe", "shake"]


@dataclass
class TrajectoryParams:
    """Parameters for camera trajectory."""
    type: TrajectoryType = "rotate_forward"
    max_disparity: float = 0.08
    max_zoom: float = 0.15
    distance: float = 0.0
    num_steps: int = 60
    num_repeats: int = 1


def compute_depth_quantiles(
    points: torch.Tensor,
    q_values: List[float] = [0.001, 0.1, 0.999],
) -> Tuple[float, float, float]:
    """Compute depth quantiles from point cloud."""
    depth_values = points[:, 2]
    depth_values = depth_values[depth_values > 0]
    
    if len(depth_values) == 0:
        return (1.0, 2.0, 10.0)
    
    quantiles = torch.quantile(
        depth_values, 
        torch.tensor(q_values, device=points.device)
    )
    return (quantiles[0].item(), quantiles[1].item(), quantiles[2].item())


def compute_max_offset(
    gaussians: Gaussians3D,
    params: TrajectoryParams,
    resolution: Tuple[int, int],
    focal_px: float,
) -> np.ndarray:
    """Compute maximum camera offset based on scene geometry."""
    min_depth, _, _ = compute_depth_quantiles(gaussians.means)
    
    width, height = resolution
    diagonal = np.sqrt((width / focal_px) ** 2 + (height / focal_px) ** 2)
    max_lateral_offset = params.max_disparity * diagonal * min_depth
    max_medial_offset = params.max_zoom * min_depth
    
    return np.array([max_lateral_offset, max_lateral_offset, max_medial_offset])


def create_trajectory(
    gaussians: Gaussians3D,
    params: TrajectoryParams,
    resolution: Tuple[int, int],
    focal_px: float,
) -> List[torch.Tensor]:
    """Create eye trajectory for rendering."""
    max_offset = compute_max_offset(gaussians, params, resolution, focal_px)
    offset_x, offset_y, offset_z = max_offset
    num_steps_total = params.num_steps * params.num_repeats
    
    eye_positions = []
    
    if params.type == "rotate":
        for t in np.linspace(0, params.num_repeats, num_steps_total):
            eye_positions.append(torch.tensor([
                offset_x * np.sin(2 * np.pi * t),
                offset_y * np.cos(2 * np.pi * t),
                params.distance,
            ], dtype=torch.float32))
            
    elif params.type == "rotate_forward":
        for t in np.linspace(0, params.num_repeats, num_steps_total):
            eye_positions.append(torch.tensor([
                offset_x * np.sin(2 * np.pi * t),
                0.0,
                params.distance + offset_z * (1.0 - np.cos(2 * np.pi * t)) / 2,
            ], dtype=torch.float32))
            
    elif params.type == "swipe":
        for x in np.linspace(-offset_x, offset_x, params.num_steps):
            eye_positions.append(torch.tensor(
                [x, 0, params.distance], dtype=torch.float32
            ))
        eye_positions = eye_positions * params.num_repeats
        
    elif params.type == "shake":
        num_h = num_steps_total // 2
        num_v = num_steps_total - num_h
        for t in np.linspace(0, params.num_repeats, num_h):
            eye_positions.append(torch.tensor([
                offset_x * np.sin(2 * np.pi * t), 0.0, params.distance
            ], dtype=torch.float32))
        for t in np.linspace(0, params.num_repeats, num_v):
            eye_positions.append(torch.tensor([
                0.0, offset_y * np.sin(2 * np.pi * t), params.distance
            ], dtype=torch.float32))
    
    return eye_positions


def create_camera_matrix(
    position: torch.Tensor,
    look_at: torch.Tensor,
    world_up: torch.Tensor,
) -> torch.Tensor:
    """Create camera extrinsics matrix (world-to-camera)."""
    device = position.device
    
    forward = look_at - position
    forward = forward / forward.norm()
    
    right = torch.cross(forward, world_up)
    right = right / right.norm()
    
    down = torch.cross(forward, right)
    
    rotation = torch.stack([right, down, forward], dim=-1)
    
    extrinsics = torch.eye(4, device=device)
    extrinsics[:3, :3] = rotation.T
    extrinsics[:3, 3] = -rotation.T @ position
    
    return extrinsics


def render_gaussians_gsplat(
    gaussians: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render Gaussians using gsplat.
    
    Returns:
        color: (H, W, 3) RGB image
        depth: (H, W) depth map
    """
    import gsplat
    
    device = gaussians.means.device
    bg = torch.tensor(background, device=device, dtype=torch.float32)
    
    if gaussians.means.shape[0] == 0:
        color = bg.view(1, 1, 3).expand(height, width, 3)
        depth = torch.zeros(height, width, device=device)
        return color, depth
    
    colors_with_depth, alphas, _ = gsplat.rasterization(
        means=gaussians.means,
        quats=gaussians.quats,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=gaussians.colors,
        viewmats=extrinsics.unsqueeze(0),
        Ks=intrinsics[:3, :3].unsqueeze(0),
        width=width,
        height=height,
        render_mode="RGB+D",
    )
    
    color = colors_with_depth[0, :, :, :3]
    depth = colors_with_depth[0, :, :, 3]
    alpha = alphas[0]
    
    # Handle NaN/Inf
    color = torch.nan_to_num(color, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Composite with background
    color = color + (1.0 - alpha) * bg
    
    return color, depth


def render_spiral_video(
    gaussians: Gaussians3D,
    metadata: SceneMetadata,
    output_path: Path,
    params: Optional[TrajectoryParams] = None,
    fps: float = 30.0,
) -> None:
    """
    Render spiral trajectory video of Gaussians.
    
    Args:
        gaussians: The Gaussians to render
        metadata: Scene metadata (resolution, focal length)
        output_path: Path to save video
        params: Trajectory parameters
        fps: Video frame rate
    """
    if params is None:
        params = TrajectoryParams()
    
    device = gaussians.means.device
    width, height = metadata.resolution
    focal_px = metadata.focal_px
    
    # Ensure even dimensions for video encoding
    width = width + (width % 2)
    height = height + (height % 2)
    
    # Build intrinsics matrix
    intrinsics = torch.tensor([
        [focal_px, 0, (width - 1) / 2.0, 0],
        [0, focal_px, (height - 1) / 2.0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], device=device, dtype=torch.float32)
    
    # Create trajectory
    trajectory = create_trajectory(gaussians, params, (width, height), focal_px)
    
    # Compute focus point
    _, focus_depth, _ = compute_depth_quantiles(gaussians.means)
    focus_depth = max(2.0, focus_depth)
    look_at = torch.tensor([0.0, 0.0, focus_depth], device=device)
    world_up = torch.tensor([0.0, -1.0, 0.0], device=device)
    
    # Initialize video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(str(output_path), fps=fps)
    
    print(f"Rendering {len(trajectory)} frames...")
    
    for i, eye_pos in enumerate(trajectory):
        eye_pos = eye_pos.to(device)
        extrinsics = create_camera_matrix(eye_pos, look_at, world_up)
        
        color, _ = render_gaussians_gsplat(
            gaussians, extrinsics, intrinsics, width, height
        )
        
        frame = (color.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        writer.append_data(frame)
        
        if (i + 1) % 10 == 0 or i == len(trajectory) - 1:
            print(f"  Rendered {i + 1}/{len(trajectory)} frames")
    
    writer.close()
    print(f"Saved video to: {output_path}")
