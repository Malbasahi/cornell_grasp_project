# Occlusion-Robust 2D Grasp Detection on Cornell Dataset

> **AI7102: Introduction to Deep Learning — Fall 2025**
> Mohamed Bin Zayed University of Artificial Intelligence

---

## Project Overview

This project investigates **occlusion-robust grasp detection** using the Cornell
Grasping Dataset. We train a fully-convolutional network that predicts per-pixel
grasp quality, angle, and width from depth images, then study how performance
degrades under synthetic occlusion and propose an occlusion-aware extension.

### Contributions
1. **Baseline CNN grasp detector** — encoder–decoder with three prediction heads
2. **Synthetic occlusion augmentation** — random block masking + depth corruption
3. **Occlusion map injection** — concatenated local-variance map for robustness
4. **Occlusion-stratified evaluation** — accuracy vs. occlusion severity curves

---

## Repository Structure

```
cornell_grasp_project/
├── configs/
│   ├── baseline.yaml          # baseline training config
│   └── robust.yaml            # occlusion-robust training config
├── data/
│   └── raw/cornell/           # place downloaded Cornell data here
│       ├── pcd0100r.png       # RGB images
│       ├── pcd0100d.tiff      # depth images
│       └── pcd0100cpos.txt    # grasp rectangle labels
├── models/
│   ├── baseline_model.py      # CNN encoder-decoder + 3 heads
│   └── robust_model.py        # baseline + occlusion map input
├── utils/
│   ├── cornell_loader.py      # dataset class + rectangle parsing
│   ├── grasp_utils.py         # rectangle ↔ pixel-map conversion
│   ├── occlusion_utils.py     # occlusion map generation + augmentation
│   └── visualization.py       # result plotting
├── train.py                   # main training script
├── evaluate.py                # main evaluation script
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_occlusion_experiments.ipynb
│   └── 04_results_analysis.ipynb
├── scripts/
│   ├── download_cornell.sh    # one-command data download
│   └── setup_colab.sh         # one-click Colab environment
├── tests/
│   └── test_pipeline.py       # unit tests (no data required)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Download data
```bash
bash scripts/download_cornell.sh
```

### 2. Setup environment (Colab)
```bash
bash scripts/setup_colab.sh
```

### 3. Train baseline
```bash
python train.py --config configs/baseline.yaml
```

### 4. Train occlusion-robust model
```bash
python train.py --config configs/robust.yaml
```

### 5. Evaluate both models
```bash
python evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/best.pth
python evaluate.py --config configs/robust.yaml   --checkpoint checkpoints/robust/best.pth
```

---

## Cornell Dataset

**Download** (no registration required):
```
bash scripts/download_cornell.sh
```

~500 MB. Contains 885 images of 240 household objects.

Each sample:
- `pcd####r.png` — RGB image (640×480)
- `pcd####d.tiff` — depth image (640×480, float32 mm)
- `pcd####cpos.txt` — positive grasp rectangles (4 corner points each)
- `pcd####cneg.txt` — negative grasp rectangles

A predicted grasp is **correct** if:
- Angle error < 30°
- IoU with ground-truth rectangle > 0.25

---

## Results

| Model              | Accuracy (all) | Accuracy (occluded 30%) | Accuracy (occluded 60%) |
|--------------------|----------------|--------------------------|--------------------------|
| Baseline           | —              | —                        | —                        |
| Robust (ours)      | —              | —                        | —                        |

*(fill in after training)*

---

## Team

| Member              | Role                                      |
|---------------------|-------------------------------------------|
| Marwah              | Data pipeline, evaluation, OSI metric     |
| Rawda               | Model training, augmentation experiments  |
| Ayah                | Occlusion head, report, visualisations    |

---

## References

1. Jiang et al., *Efficient Grasping from RGBD Images* (ICRA 2011) — Cornell dataset
2. Redmon & Angelova, *Real-Time Grasp Detection* (ICRA 2015)
3. Morrison et al., *GG-CNN: Generative Grasping CNN* (RSS 2018)
