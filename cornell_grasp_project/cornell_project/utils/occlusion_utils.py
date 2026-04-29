"""
utils/occlusion_utils.py
-------------------------
Occlusion map computation and synthetic occlusion augmentation.

Two types of occlusion map:
    'binary'   : 1 where depth == 0 (missing pixels), 0 elsewhere
    'variance' : local depth variance map — high where geometry is
                 discontinuous (edges/occlusion boundaries)

Synthetic occlusion:
    Randomly zeros out rectangular blocks in the depth image to
    simulate real sensor occlusion. Used for training-time augmentation
    and for occlusion-level evaluation sweeps.
"""

import numpy as np
from scipy.ndimage import uniform_filter
import random


# ─────────────────────────────────────────────────────────────
# Occlusion map computation
# ─────────────────────────────────────────────────────────────

def compute_occlusion_map(depth, mode='variance', window=15):
    """
    Compute a per-pixel occlusion severity map from a depth image.

    Args:
        depth  (np.ndarray): H×W float32 depth image (0 = missing)
        mode   (str):        'binary' or 'variance'
        window (int):        local window size for variance mode

    Returns:
        occ_map (np.ndarray): H×W float32 in [0, 1]
            0 = fully observed / certain
            1 = missing / occluded / uncertain
    """
    if mode == 'binary':
        return _binary_occlusion_map(depth)
    elif mode == 'variance':
        return _variance_occlusion_map(depth, window)
    else:
        raise ValueError(f"Unknown occlusion map mode: {mode}. "
                         f"Use 'binary' or 'variance'.")


def _binary_occlusion_map(depth):
    """
    Simple binary map: 1 where depth is missing (== 0), else 0.

    This directly captures sensor drop-outs and true occlusions.
    """
    occ = (depth == 0).astype(np.float32)
    return occ


def _variance_occlusion_map(depth, window=15):
    """
    Local depth variance map.

    High variance indicates depth discontinuities which occur at
    occlusion boundaries and edges. This is a richer signal than
    binary missing-pixels.

    Steps:
        1. Compute local mean of depth in a (window × window) box
        2. Compute local mean of depth² in same box
        3. variance = E[x²] - E[x]²
        4. Normalise to [0, 1]
        5. Zero out pixels where depth itself is missing

    Args:
        depth  (np.ndarray): H×W float32
        window (int):        local box size in pixels

    Returns:
        occ_map (np.ndarray): H×W float32 in [0, 1]
    """
    # Replace missing pixels with local mean for smoother variance
    d = depth.copy()
    missing = (d == 0)

    # Local statistics
    mean_d   = uniform_filter(d,    size=window, mode='reflect')
    mean_d2  = uniform_filter(d**2, size=window, mode='reflect')
    variance = np.maximum(mean_d2 - mean_d**2, 0.0)

    # Normalise variance to [0, 1]
    v_max = variance.max()
    if v_max > 0:
        variance = variance / v_max

    # Missing pixels get maximum occlusion score
    variance[missing] = 1.0

    return variance.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Synthetic occlusion augmentation
# ─────────────────────────────────────────────────────────────

def apply_random_occlusion(depth, num_blocks=3,
                            block_size_range=(20, 80)):
    """
    Randomly zero out rectangular blocks in a depth image.

    Simulates real occlusion caused by nearby objects blocking
    part of the sensor's field of view.

    Args:
        depth            (np.ndarray): H×W float32 depth image
        num_blocks       (int):        number of blocks to zero out
        block_size_range (tuple):      (min_px, max_px) block edge length

    Returns:
        occluded (np.ndarray): H×W float32 — copy with blocks zeroed
    """
    H, W = depth.shape
    occluded = depth.copy()
    lo, hi = block_size_range

    for _ in range(num_blocks):
        bh = random.randint(lo, min(hi, H))
        bw = random.randint(lo, min(hi, W))
        r0 = random.randint(0, H - bh)
        c0 = random.randint(0, W - bw)
        occluded[r0:r0 + bh, c0:c0 + bw] = 0.0

    return occluded


def apply_occlusion_fraction(depth, fraction):
    """
    Apply occlusion to cover approximately `fraction` of valid pixels.

    Used for occlusion-level evaluation sweeps:
        fraction=0.0 → no occlusion
        fraction=0.3 → ~30% of pixels zeroed
        fraction=0.6 → ~60% of pixels zeroed

    Strategy: repeatedly add random blocks until target coverage is met.

    Args:
        depth    (np.ndarray): H×W float32 depth image
        fraction (float):      target fraction of pixels to occlude [0, 1]

    Returns:
        occluded (np.ndarray): H×W float32
    """
    if fraction <= 0.0:
        return depth.copy()

    H, W    = depth.shape
    total   = H * W
    target  = int(fraction * total)
    occluded = depth.copy()
    covered  = 0

    attempts = 0
    while covered < target and attempts < 100:
        bh = random.randint(20, min(80, H))
        bw = random.randint(20, min(80, W))
        r0 = random.randint(0, H - bh)
        c0 = random.randint(0, W - bw)

        # Count newly occluded pixels
        newly = (occluded[r0:r0+bh, c0:c0+bw] != 0).sum()
        occluded[r0:r0+bh, c0:c0+bw] = 0.0
        covered += newly
        attempts += 1

    return occluded


# ─────────────────────────────────────────────────────────────
# Occlusion severity score per image (for analysis)
# ─────────────────────────────────────────────────────────────

def image_occlusion_fraction(depth):
    """
    Measure what fraction of pixels are missing (depth == 0).

    Args:
        depth (np.ndarray): H×W

    Returns:
        fraction (float): in [0, 1]
    """
    return float((depth == 0).mean())
