"""
Image decomposition into layers using QwenImageLayeredPipeline.
"""

import torch
import gc
import os
from PIL import Image
from typing import List
from pathlib import Path


def _reset_cuda():
    """Reset CUDA state to recover from errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def decompose_image(
    image_path: str,
    num_layers: int = 4,
    resolution: int = 640,
    device: str = "cuda",
    num_inference_steps: int = 50,
    use_compile: bool = True,
    use_offload: bool = False,
) -> List[Image.Image]:
    """
    Decompose an image into multiple layers using QwenImageLayeredPipeline.
    
    Args:
        image_path: Path to input image
        num_layers: Number of layers to generate
        resolution: Resolution for processing (640 recommended)
        device: Device to use
        num_inference_steps: Number of diffusion steps (lower = faster)
        use_compile: Whether to use torch.compile for acceleration
        use_offload: Whether to use CPU offloading (use if GPU memory is limited)
        
    Returns:
        List of PIL Images (RGBA), ordered from back (0) to front (N-1)
    """
    from diffusers import QwenImageLayeredPipeline
    
    print(f"Loading QwenImageLayeredPipeline...")
    
    # Reset CUDA state first
    _reset_cuda()
    
    # Initialize CUDA explicitly to catch errors early
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.init()
            capability = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if capability >= 8 else torch.float16
            print(f"  CUDA initialized, compute capability: {capability}")
        except RuntimeError as e:
            print(f"  CUDA init failed: {e}, falling back to CPU")
            device = "cpu"
            dtype = torch.float32
    else:
        dtype = torch.float32
    
    pipeline = QwenImageLayeredPipeline.from_pretrained(
        "Qwen/Qwen-Image-Layered",
        torch_dtype=dtype if device == "cuda" else torch.float32,
    )
    
    # Move to device
    if device == "cuda":
        if use_offload:
            # CPU offloading for limited GPU memory - but generator must still be cuda
            pipeline.enable_model_cpu_offload()
            print(f"  Loaded pipeline with CPU offloading (dtype {dtype})")
        else:
            pipeline = pipeline.to(device, dtype)
            print(f"  Loaded pipeline on {device} with dtype {dtype}")
    else:
        pipeline = pipeline.to(device)
        print(f"  Loaded pipeline on {device}")
    
    # Enable memory efficient attention if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("  Enabled xformers memory efficient attention")
    except Exception:
        try:
            pipeline.enable_attention_slicing(slice_size="auto")
            print("  Enabled attention slicing")
        except Exception:
            pass
    
    # Compile for speed (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        try:
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
            print("  Compiled UNet with torch.compile")
        except Exception as e:
            print(f"  torch.compile not available: {e}")
    
    pipeline.set_progress_bar_config(disable=None)
    
    # Load and prepare image
    image = Image.open(image_path).convert("RGBA")
    print(f"Input image size: {image.size}")
    
    # Generator device must match pipeline device (cuda)
    gen_device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
    
    inputs = {
        "image": image,
        "generator": torch.Generator(device=gen_device).manual_seed(777),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": 1,
        "layers": num_layers,
        "resolution": resolution,
        "cfg_normalize": True,
        "use_en_prompt": True,
    }
    
    print(f"Decomposing image into {num_layers} layers ({num_inference_steps} steps)...")
    
    with torch.inference_mode():
        output = pipeline(**inputs)
        layers = output.images[0]
    
    print(f"Generated {len(layers)} layers")
    
    # Debug: Check layer values and fix NaN issues
    import numpy as np
    fixed_layers = []
    for i, layer in enumerate(layers):
        arr = np.array(layer)
        
        # Check for NaN/inf values and fix them
        if not np.isfinite(arr).all():
            print(f"  Layer {i}: WARNING - contains NaN/inf values, fixing...")
            arr = np.nan_to_num(arr, nan=0, posinf=255, neginf=0)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            layer = Image.fromarray(arr, mode='RGBA')
        
        print(f"  Layer {i}: shape={arr.shape}, dtype={arr.dtype}, "
              f"alpha range=[{arr[:,:,3].min()}, {arr[:,:,3].max()}], "
              f"rgb range=[{arr[:,:,:3].min()}, {arr[:,:,:3].max()}]")
        fixed_layers.append(layer)
    
    # Clean up to free memory
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
    return fixed_layers


def composite_layers(
    layers: List[Image.Image],
    up_to_idx: int,
) -> Image.Image:
    """
    Composite layers from 0 to up_to_idx (inclusive).
    Layer 0 is the background, higher indices are in front.
    
    Args:
        layers: List of RGBA PIL Images
        up_to_idx: Index of the topmost layer to include
        
    Returns:
        Composited RGBA image
    """
    if up_to_idx < 0 or up_to_idx >= len(layers):
        raise ValueError(f"Invalid layer index: {up_to_idx}")
    
    result = layers[0].copy()
    for i in range(1, up_to_idx + 1):
        result = Image.alpha_composite(result, layers[i])
    
    return result
