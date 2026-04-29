"""
utils/cornell_loader.py
------------------------
Full Cornell Grasping Dataset loader.

Loads ALL samples from the dataset — no artificial subsetting.
Each sample returns:
    depth   : (C, H, W) tensor  — C=1 (baseline) or C=2 (robust: depth+occ)
    quality : (1, H, W) tensor  — 1 inside valid grasp regions
    angle   : (1, H, W) tensor  — grasp angle in radians
    width   : (1, H, W) tensor  — normalised gripper width

Cornell folder structure expected:
    data/raw/cornell/
        pcd0100r.png      RGB image
        pcd0100d.tiff     depth image (float32, mm)
        pcd0100cpos.txt   positive grasp rectangles
        pcd0100cneg.txt   negative grasp rectangles (optional)
        ...
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import imageio
import random

from utils.grasp_utils import load_grasp_rectangles, build_pixel_maps
from utils.occlusion_utils import (
    compute_occlusion_map, apply_random_occlusion
)


# ─────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────

class CornellDataset(Dataset):
    """
    Full Cornell Grasping Dataset.

    Discovers all samples automatically from the data root.
    Applies data augmentation and occlusion simulation for the
    robust training condition.

    Args:
        root         (str):   path to data/raw/cornell/
        split        (str):   'train' or 'val'
        split_ratio  (float): fraction of data used for training (e.g. 0.9)
        image_size   (tuple): (H, W) — resize target, or None to keep original
        use_depth_only (bool): True → 1-channel depth; False → 2-channel depth+occ
        normalize_depth (bool): normalise depth to [0,1]
        augment      (bool):  random horizontal flip + rotation
        occlusion_aug (bool): apply synthetic occlusion blocks
        occ_aug_prob  (float): probability of applying occlusion per sample
        occ_block_size (list): [min, max] random block size in pixels
        occ_num_blocks (list): [min, max] number of blocks
        occ_map_type  (str):  'binary' or 'variance'
        variance_window (int): window size for local variance map
        seed         (int):   random seed for train/val split
    """

    def __init__(
        self,
        root,
        split='train',
        split_ratio=0.9,
        image_size=(480, 640),
        use_depth_only=True,
        normalize_depth=True,
        augment=False,
        occlusion_aug=False,
        occ_aug_prob=0.5,
        occ_block_size=(20, 80),
        occ_num_blocks=(1, 5),
        occ_map_type='variance',
        variance_window=15,
        seed=42,
    ):
        self.root           = root
        self.split          = split
        self.image_size     = image_size
        self.use_depth_only = use_depth_only
        self.normalize_depth= normalize_depth
        self.augment        = augment
        self.occlusion_aug  = occlusion_aug
        self.occ_aug_prob   = occ_aug_prob
        self.occ_block_size = occ_block_size
        self.occ_num_blocks = occ_num_blocks
        self.occ_map_type   = occ_map_type
        self.variance_window= variance_window

        # ── Discover all sample IDs ───────────────────────────
        self.samples = self._find_samples()

        # ── Train / val split ─────────────────────────────────
        random.seed(seed)
        random.shuffle(self.samples)
        n_train = int(len(self.samples) * split_ratio)
        if split == 'train':
            self.samples = self.samples[:n_train]
        else:
            self.samples = self.samples[n_train:]

        print(f"[CornellDataset] split={split} | "
              f"samples={len(self.samples)} | "
              f"mode={'depth+occ' if not use_depth_only else 'depth_only'} | "
              f"occ_aug={occlusion_aug}")

    def _find_samples(self):
        """
        Find all valid samples = those with a depth file AND a cpos label file.
        Searches the root directory recursively.
        """
        # Cornell depth files are .tiff
        depth_files = sorted(glob.glob(
            os.path.join(self.root, '**', '*d.tiff'), recursive=True
        ) + glob.glob(
            os.path.join(self.root, '**', '*d.png'), recursive=True
        ))

        samples = []
        for depth_path in depth_files:
            # Derive the sample prefix
            base = depth_path
            if base.endswith('d.tiff'):
                prefix = base[:-6]   # strip 'd.tiff'
            elif base.endswith('d.png'):
                prefix = base[:-5]   # strip 'd.png'
            else:
                continue

            label_path = prefix + 'cpos.txt'
            if os.path.exists(label_path):
                samples.append({
                    'depth_path': depth_path,
                    'label_path': label_path,
                    'prefix':     prefix,
                })

        if len(samples) == 0:
            raise FileNotFoundError(
                f"No valid Cornell samples found in {self.root}\n"
                f"Expected files like: pcd0100d.tiff + pcd0100cpos.txt\n"
                f"Run: bash scripts/download_cornell.sh"
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ── Load depth ────────────────────────────────────────
        depth = self._load_depth(sample['depth_path'])  # H×W float32

        # ── Apply synthetic occlusion augmentation ────────────
        if self.occlusion_aug and random.random() < self.occ_aug_prob:
            n_blocks = random.randint(*self.occ_num_blocks)
            depth = apply_random_occlusion(
                depth,
                num_blocks=n_blocks,
                block_size_range=self.occ_block_size,
            )

        # ── Resize ────────────────────────────────────────────
        if self.image_size is not None:
            depth = self._resize(depth, self.image_size)

        H, W = depth.shape

        # ── Normalise depth ───────────────────────────────────
        if self.normalize_depth:
            depth_norm = self._normalize(depth)
        else:
            depth_norm = depth.copy()

        # ── Occlusion map ─────────────────────────────────────
        occ_map = compute_occlusion_map(depth, mode=self.occ_map_type,
                                         window=self.variance_window)

        # ── Load grasp labels ─────────────────────────────────
        rectangles = load_grasp_rectangles(sample['label_path'])
        quality, angle, width = build_pixel_maps(rectangles, shape=(H, W))

        # ── Data augmentation (train only) ────────────────────
        if self.augment:
            depth_norm, occ_map, quality, angle, width = \
                self._augment(depth_norm, occ_map, quality, angle, width)

        # ── Build input tensor ────────────────────────────────
        if self.use_depth_only:
            x = torch.FloatTensor(depth_norm[None])       # (1, H, W)
        else:
            x = torch.FloatTensor(
                np.stack([depth_norm, occ_map], axis=0))  # (2, H, W)

        return {
            'input':   x,
            'quality': torch.FloatTensor(quality[None]),  # (1, H, W)
            'angle':   torch.FloatTensor(angle[None]),    # (1, H, W)
            'width':   torch.FloatTensor(width[None]),    # (1, H, W)
            'depth_raw': torch.FloatTensor(depth[None]),  # (1, H, W) raw
            'idx':     idx,
        }

    # ── Private helpers ───────────────────────────────────────

    def _load_depth(self, path):
        """Load a depth image as float32 numpy array (H, W)."""
        if path.endswith('.tiff') or path.endswith('.tif'):
            depth = np.array(imageio.imread(path), dtype=np.float32)
        else:
            depth = np.array(Image.open(path), dtype=np.float32)

        # Handle multi-channel depth (take first channel)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # Replace NaN / inf with 0
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return depth

    def _resize(self, img, target_size):
        """Resize (H, W) numpy array using PIL."""
        H, W = target_size
        pil  = Image.fromarray(img)
        pil  = pil.resize((W, H), Image.BILINEAR)
        return np.array(pil, dtype=np.float32)

    def _normalize(self, depth):
        """Normalise depth to [0, 1], ignoring zero (invalid) pixels."""
        valid = depth[depth > 0]
        if len(valid) == 0:
            return depth
        d_min, d_max = valid.min(), valid.max()
        if d_max == d_min:
            return np.zeros_like(depth)
        norm = (depth - d_min) / (d_max - d_min + 1e-8)
        norm[depth == 0] = 0.0   # keep invalid pixels at 0
        return norm.astype(np.float32)

    def _augment(self, depth, occ_map, quality, angle, width):
        """Apply random horizontal flip and 90° rotation."""
        # Random horizontal flip
        if random.random() > 0.5:
            depth   = np.fliplr(depth).copy()
            occ_map = np.fliplr(occ_map).copy()
            quality = np.fliplr(quality).copy()
            angle   = -np.fliplr(angle).copy()   # flip angle sign
            width   = np.fliplr(width).copy()

        # Random rotation: 0°, 90°, 180°, 270°
        k = random.randint(0, 3)
        if k > 0:
            depth   = np.rot90(depth,   k).copy()
            occ_map = np.rot90(occ_map, k).copy()
            quality = np.rot90(quality, k).copy()
            angle   = np.rot90(angle,   k).copy() + k * (np.pi / 2)
            width   = np.rot90(width,   k).copy()
            # Wrap angle into (-pi/2, pi/2)
            angle   = ((angle + np.pi/2) % np.pi) - np.pi/2

        return depth, occ_map, quality, angle, width


# ─────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────

def build_dataloaders(cfg):
    """
    Build train and val DataLoaders from a config dict.

    Args:
        cfg (dict): full config loaded from YAML

    Returns:
        train_loader, val_loader
    """
    data_cfg = cfg['data']

    common = dict(
        root           = data_cfg['root'],
        split_ratio    = data_cfg.get('split_ratio', [0.9, 0.1])[0],
        image_size     = tuple(data_cfg.get('image_size', [480, 640])),
        use_depth_only = data_cfg.get('use_depth_only', True),
        normalize_depth= data_cfg.get('normalize_depth', True),
        occ_aug_prob   = data_cfg.get('occlusion_aug_prob', 0.5),
        occ_block_size = tuple(data_cfg.get('occlusion_block_size', [20, 80])),
        occ_num_blocks = tuple(data_cfg.get('occlusion_num_blocks', [1, 5])),
        occ_map_type   = data_cfg.get('occlusion_map_type', 'variance'),
        variance_window= data_cfg.get('variance_window', 15),
    )

    train_ds = CornellDataset(
        split='train',
        augment=data_cfg.get('augment_train', True),
        occlusion_aug=data_cfg.get('occlusion_aug', False),
        **common,
    )
    val_ds = CornellDataset(
        split='val',
        augment=False,
        occlusion_aug=False,   # never augment validation
        **common,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,           # one at a time for evaluation
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
    )
    return train_loader, val_loader
