#!/bin/bash
# scripts/setup_colab.sh
# -----------------------
# One-click environment setup for Google Colab.
#
# Run this at the top of any notebook:
#     import os
#     os.system('bash scripts/setup_colab.sh')
#
# What this does:
#   1. Installs all Python dependencies
#   2. Downloads the Cornell dataset automatically (~500 MB)
#   3. Verifies the data structure
#   4. Checks GPU availability
#   5. Runs a quick import smoke test

set -e

echo "================================================="
echo "  Occlusion-Robust Grasp: Colab Setup"
echo "================================================="

# ── 1. Dependencies ───────────────────────────────────────────
echo ""
echo "[1/4] Installing dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -q numpy scipy matplotlib seaborn tqdm pyyaml \
               scikit-learn pandas tensorboard Pillow imageio \
               scikit-image opencv-python

# ── 2. Directories ────────────────────────────────────────────
echo "[2/4] Creating directories..."
mkdir -p data/raw/cornell
mkdir -p checkpoints/{baseline,robust}
mkdir -p logs/{baseline,robust}
mkdir -p results/{figures,baseline,robust,paper_figures}

# ── 3. Dataset ────────────────────────────────────────────────
echo "[3/4] Downloading Cornell Grasping Dataset..."
bash scripts/download_cornell.sh

# ── 4. Verify ─────────────────────────────────────────────────
echo "[4/4] Checking GPU and imports..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU. Go to Runtime > Change runtime type > T4 GPU')

import sys
sys.path.insert(0, '.')
from utils.cornell_loader import CornellDataset
from utils.grasp_utils import GraspRectangle
from utils.occlusion_utils import compute_occlusion_map
from models.baseline_model import BaselineGraspCNN
from models.robust_model import RobustGraspCNN
print('  Imports: OK')

ds = CornellDataset('data/raw/cornell', split='train')
sample = ds[0]
print(f'  Dataset: {len(ds)} training samples')
print(f'  Input shape: {sample[\"input\"].shape}')
print()
print('  Setup complete! Run:')
print('    python train.py --config configs/baseline.yaml')
"

echo "================================================="
