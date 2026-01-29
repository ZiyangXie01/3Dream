"""
Depth prediction using MoGe model.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict


_moge_model = None


def load_moge_model(device: str = "cuda", use_fp16: bool = True):
    """Load MoGe depth estimation model (cached)."""
    global _moge_model
    
    if _moge_model is not None:
        return _moge_model
    
    from moge.model import import_model_class_by_version
    
    print("Loading MoGe model (v2)...")
    model = import_model_class_by_version("v2").from_pretrained(
        "Ruicheng/moge-2-vitl-normal"
    ).to(device).eval()
    
    if use_fp16:
        model.half()
    
    _moge_model = model
    return model


def unload_moge_model():
    """Unload MoGe model to free memory."""
    global _moge_model
    if _moge_model is not None:
        del _moge_model
        _moge_model = None
        torch.cuda.empty_cache()


def predict_depth(
    model,
    image: Image.Image,
    device: str = "cuda",
    use_fp16: bool = True,
    resolution_level: int = 9,
) -> Dict[str, torch.Tensor]:
    """
    Predict depth for an image.
    
    Args:
        model: MoGe model
        image: Input RGBA or RGB image
        device: Device to use
        use_fp16: Whether to use fp16
        resolution_level: MoGe resolution level (0-9)
        
    Returns:
        Dictionary with 'points', 'depth', 'intrinsics', 'mask'
    """
    # Convert to RGB
    rgb_image = image.convert("RGB")
    img_np = np.array(rgb_image).astype(np.float32) / 255.0
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)
    
    # Run inference
    output = model.infer(
        img_tensor,
        resolution_level=resolution_level,
        use_fp16=use_fp16,
    )
    
    return output


def get_focal_px(intrinsics: torch.Tensor, image_width: int, image_height: int) -> float:
    """
    Extract focal length in pixels from MoGe intrinsics matrix.
    
    MoGe returns normalized intrinsics where:
    - fx, fy are normalized by image width/height respectively
    - cx, cy are 0.5 (principal point at center)
    
    Args:
        intrinsics: (3, 3) intrinsics matrix from MoGe
        image_width: Image width
        image_height: Image height
        
    Returns:
        Focal length in pixels (average of fx and fy)
    """
    if intrinsics.dim() == 2:
        # MoGe intrinsics are normalized: fx is relative to width, fy to height
        fx_normalized = intrinsics[0, 0].item()
        fy_normalized = intrinsics[1, 1].item()
        
        # Convert to pixel units
        fx_px = fx_normalized * image_width
        fy_px = fy_normalized * image_height
        
        # Return average (they should be very close for square pixels)
        return (fx_px + fy_px) / 2.0
    
    # Fallback for scalar
    return float(intrinsics) * image_width
