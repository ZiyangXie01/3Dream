"""
Layered Image to 3D Gaussian Splats Pipeline.

This script:
1. Decomposes an image into multiple layers
2. Predicts depth for each layer progressively (back to front)
3. Generates point clouds for each layer's visible region
4. Combines all point clouds into Gaussian splats
5. Exports as PLY file and renders spiral video
"""

import torch
import numpy as np
import gc
from pathlib import Path
from typing import Literal

from gaussians import (
    Gaussians3D,
    SceneMetadata,
    extract_layer_gaussians,
    combine_gaussians,
)
from decompose import decompose_image, composite_layers
from depth import load_moge_model, unload_moge_model, predict_depth, get_focal_px
from rendering import render_spiral_video, TrajectoryParams, TrajectoryType
from io_utils import save_gaussians_ply


def process_layered_image(
    image_path: str,
    output_dir: str = "output",
    num_layers: int = 4,
    resolution: int = 640,
    device: str = "cuda",
    use_fp16: bool = True,
    scale_multiplier: float = 1.0,
    resolution_level: int = 9,
    num_inference_steps: int = 30,
    save_intermediate: bool = False,
    render_video: bool = True,
    trajectory_type: TrajectoryType = "rotate_forward",
    num_frames: int = 60,
    fps: float = 30.0,
    use_offload: bool = False,
) -> Gaussians3D:
    """
    Main pipeline: Decompose image -> predict depth per layer -> combine Gaussians.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs (will be created)
        num_layers: Number of layers to decompose into
        resolution: Resolution for layer decomposition
        device: Device to use
        use_fp16: Whether to use fp16 for depth prediction
        scale_multiplier: Multiplier for Gaussian scale
        resolution_level: MoGe resolution level (0-9)
        num_inference_steps: Diffusion steps (lower = faster, 30 recommended)
        save_intermediate: Whether to save intermediate layer images
        render_video: Whether to render spiral video
        trajectory_type: Type of camera trajectory
        num_frames: Number of frames in video
        fps: Video frame rate
        
    Returns:
        Combined Gaussians3D
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
    # =========================================================================
    # Step 1: Decompose image into layers
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Decomposing image into layers")
    print("=" * 60)
    
    layers = decompose_image(
        image_path,
        num_layers=num_layers,
        resolution=resolution,
        device=device,
        num_inference_steps=num_inference_steps,
        use_offload=use_offload,
    )
    
    # Get dimensions
    first_layer = np.array(layers[0])
    img_height, img_width = first_layer.shape[:2]
    print(f"Layer dimensions: {img_width}x{img_height}")
    
    # Always save layers to output
    layers_path = output_path / "layers"
    layers_path.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(layers):
        layer.save(layers_path / f"layer_{i}.png")
        print(f"  Saved layer_{i}.png")
    
    # Also save composite of all layers
    composite_all = composite_layers(layers, len(layers) - 1)
    composite_all.save(layers_path / "composite_all.png")
    print(f"  Saved composite_all.png")
    
    # =========================================================================
    # Step 2: Load depth model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Loading depth estimation model")
    print("=" * 60)
    
    moge_model = load_moge_model(device, use_fp16)
    
    # =========================================================================
    # Step 3: Process each layer from back to front
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Predicting depth for each layer")
    print("=" * 60)
    
    all_gaussians = []
    last_focal_px = None
    
    for layer_idx in range(num_layers):
        print(f"\n--- Layer {layer_idx}/{num_layers - 1} ---")
        
        # Get layer alpha mask
        layer_rgba = np.array(layers[layer_idx])
        layer_alpha = layer_rgba[:, :, 3]
        
        # Skip fully transparent layers
        max_alpha = layer_alpha.max()
        if max_alpha < 10:
            print(f"  Skipping (fully transparent, max_alpha={max_alpha})")
            continue
        
        layer_mask = layer_alpha > 128
        visible_pixels = np.sum(layer_mask)
        print(f"  max_alpha={max_alpha}, visible_pixels={visible_pixels}")
        
        if visible_pixels == 0:
            print(f"  Skipping (no visible pixels)")
            continue
        
        # Composite layers up to this index
        composite = composite_layers(layers, layer_idx)
        
        # Save composite
        composite.save(output_path / "layers" / f"composite_{layer_idx}.png")
        
        # Predict depth
        print(f"  Predicting depth for composite 0-{layer_idx}...")
        depth_output = predict_depth(
            moge_model,
            composite,
            device,
            use_fp16,
            resolution_level,
        )
        
        # Extract results
        points = depth_output['points']
        depth = depth_output['depth']
        intrinsics = depth_output['intrinsics']
        
        # Get focal length in pixels
        focal_px = get_focal_px(intrinsics, img_width, img_height)
        last_focal_px = focal_px
        
        # Log depth stats
        mask_tensor = torch.from_numpy(layer_mask).to(device)
        valid_depth = depth[mask_tensor]
        print(f"  Depth: [{valid_depth.min().item():.3f}, {valid_depth.max().item():.3f}]")
        print(f"  Intrinsics fx={intrinsics[0,0].item():.4f}, fy={intrinsics[1,1].item():.4f} (normalized)")
        print(f"  Focal: {focal_px:.2f} px (from {img_width}x{img_height} image)")
        
        # Get colors from layer (not composite)
        layer_rgb = torch.from_numpy(
            layer_rgba[:, :, :3].astype(np.float32) / 255.0
        ).to(device)
        
        # Apply MoGe mask if available
        if 'mask' in depth_output and depth_output['mask'] is not None:
            mask_tensor = mask_tensor & depth_output['mask']
        
        # Extract Gaussians
        layer_gaussians = extract_layer_gaussians(
            points=points,
            depth=depth,
            colors=layer_rgb,
            mask=mask_tensor,
            focal=focal_px,
            scale_multiplier=scale_multiplier,
        )
        
        print(f"  Created {layer_gaussians.means.shape[0]} Gaussians")
        all_gaussians.append(layer_gaussians)
    
    # Clean up depth model
    unload_moge_model()
    gc.collect()
    
    # =========================================================================
    # Step 4: Combine all Gaussians
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Combining Gaussians")
    print("=" * 60)
    
    combined = combine_gaussians(all_gaussians)
    print(f"Total Gaussians: {combined.means.shape[0]}")
    
    # =========================================================================
    # Step 5: Export PLY
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Exporting PLY")
    print("=" * 60)
    
    ply_path = output_path / "gaussians.ply"
    save_gaussians_ply(combined, str(ply_path))
    
    # =========================================================================
    # Step 6: Render video
    # =========================================================================
    if render_video and combined.means.shape[0] > 0:
        print("\n" + "=" * 60)
        print("Step 6: Rendering spiral video")
        print("=" * 60)
        
        metadata = SceneMetadata(
            resolution=(img_width, img_height),
            focal_px=last_focal_px if last_focal_px else img_width,
        )
        
        params = TrajectoryParams(
            type=trajectory_type,
            num_steps=num_frames,
            num_repeats=1,
        )
        
        video_path = output_path / "spiral.mp4"
        render_spiral_video(combined, metadata, video_path, params, fps)
    
    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"Done! Output saved to: {output_path.absolute()}")
    print("=" * 60)
    
    return combined


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert layered image to 3D Gaussian Splats"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--layers", "-l", type=int, default=4,
        help="Number of layers (default: 4)"
    )
    parser.add_argument(
        "--resolution", "-r", type=int, default=640,
        help="Resolution for decomposition (default: 640)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cuda)"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Gaussian scale multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--resolution-level", type=int, default=9,
        help="MoGe resolution level 0-9 (default: 9)"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Diffusion inference steps (default: 30, lower=faster)"
    )
    parser.add_argument(
        "--save-intermediate", action="store_true",
        help="Save intermediate layer images"
    )
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable fp16 precision"
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip video rendering"
    )
    parser.add_argument(
        "--trajectory", type=str, default="rotate_forward",
        choices=["rotate", "rotate_forward", "swipe", "shake"],
        help="Camera trajectory type (default: rotate_forward)"
    )
    parser.add_argument(
        "--num-frames", type=int, default=60,
        help="Number of video frames (default: 60)"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Video frame rate (default: 30.0)"
    )
    parser.add_argument(
        "--offload", action="store_true",
        help="Use CPU offloading for limited GPU memory"
    )
    
    args = parser.parse_args()
    
    process_layered_image(
        image_path=args.input,
        output_dir=args.output,
        num_layers=args.layers,
        resolution=args.resolution,
        device=args.device,
        use_fp16=not args.no_fp16,
        scale_multiplier=args.scale,
        resolution_level=args.resolution_level,
        num_inference_steps=args.steps,
        save_intermediate=args.save_intermediate,
        render_video=not args.no_video,
        trajectory_type=args.trajectory,
        num_frames=args.num_frames,
        fps=args.fps,
        use_offload=args.offload,
    )


if __name__ == "__main__":
    main()
