# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Stub module for inference-only usage

import torch
from typing import List, Tuple, Union

def rle_encode(masks: torch.Tensor, return_areas: bool = False) -> Union[List, Tuple[List, List]]:
    """
    Run-length encode binary masks.
    
    Args:
        masks: Binary masks of shape (N, H, W) or (H, W)
        return_areas: Whether to return mask areas along with RLE
        
    Returns:
        If return_areas=False: List of RLE dicts
        If return_areas=True: Tuple of (RLE list, areas list)
    """
    try:
        import pycocotools.mask as mask_utils
        import numpy as np
        
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        
        # Convert to fortran-order uint8 numpy array as required by pycocotools
        masks_np = masks.cpu().numpy().astype(np.uint8)
        
        rles = []
        areas = []
        for mask in masks_np:
            # pycocotools expects fortran order
            mask_f = np.asfortranarray(mask)
            rle = mask_utils.encode(mask_f)
            rles.append(rle)
            if return_areas:
                areas.append(int(mask_utils.area(rle)))
        
        if return_areas:
            return rles, areas
        return rles
        
    except ImportError:
        # Fallback: return empty lists if pycocotools not available
        n = masks.shape[0] if masks.dim() == 3 else 1
        if return_areas:
            return [{}] * n, [0] * n
        return [{}] * n

__all__ = ["rle_encode"]
