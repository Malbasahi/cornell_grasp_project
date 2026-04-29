````markdown
# Occlusion-Robust 2D Grasp Detection on the Cornell Grasping Dataset

**AI7102: Introduction to Deep Learning — Fall 2025**  
Mohamed Bin Zayed University of Artificial Intelligence

---

## Project Overview

This project studies **occlusion-robust 2D grasp detection** using the **Cornell Grasping Dataset**. The core task is to predict dense grasp representations from depth images, including:

- **grasp quality**
- **grasp angle**
- **grasp width**

We first train a lightweight fully convolutional baseline, then extend it with an **occlusion-aware input representation** and **synthetic occlusion augmentation** to test whether the model becomes more robust under degraded observations.

This repository is designed for a **small, practical, reproducible pipeline**, in contrast to heavier grasping benchmarks that require large-scale 3D infrastructure.

---

## Main Contributions

- **Baseline 2D grasp detector**  
  A fully convolutional encoder-decoder network with separate prediction heads for grasp quality, angle, and width.

- **Synthetic occlusion augmentation**  
  Random masking and depth corruption to simulate missing or unreliable observations during training.

- **Occlusion-aware robust model**  
  An extended input representation that injects an occlusion cue, such as local depth variance or missing-depth structure, alongside the depth image.

- **Occlusion-stratified evaluation**  
  Evaluation under increasing occlusion severity, including robustness curves and accuracy breakdowns.

---

## Repository Structure

```text
cornell_grasp_project/
├── checkpoints/                  # saved model weights after training
│   ├── baseline/
│   └── robust/
├── configs/
│   ├── baseline.yaml            # baseline training config
│   └── robust.yaml              # robust-model training config
├── data/
│   ├── raw/
│   │   └── cornell/             # place Cornell dataset here
│   ├── processed/               # optional cached maps / preprocessed tensors
│   └── splits/                  # train/val/test split files
├── models/
│   ├── baseline_model.py        # CNN encoder-decoder + prediction heads
│   └── robust_model.py          # baseline + occlusion-aware input
├── utils/
│   ├── cornell_loader.py        # dataset class + annotation parsing
│   ├── grasp_utils.py           # rectangle ↔ dense map conversion
│   ├── occlusion_utils.py       # occlusion map generation + augmentation
│   └── visualization.py         # plotting and visual analysis
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_occlusion_experiments.ipynb
│   └── 04_results_analysis.ipynb
├── scripts/
│   ├── download_cornell.sh      # Cornell download helper
│   └── setup_colab.sh           # Colab / environment setup
├── tests/
│   └── test_pipeline.py         # pipeline sanity checks
├── train.py                     # main training entry point
├── evaluate.py                  # main evaluation entry point
├── requirements.txt
└── README.md
````

---

## Problem Formulation

Given a **depth image**, the model predicts dense grasp maps:

* **quality map**: probability that a grasp at each pixel is feasible
* **angle map**: grasp orientation
* **width map**: gripper opening width

The robust version augments the input with an **occlusion-related cue**, allowing the model to reason more explicitly about degraded local geometry.

---

## Training Targets

The Cornell grasp rectangles are converted into dense supervision maps for:

* **grasp quality**
* **grasp angle**
* **grasp width**

Typical losses include:

* classification loss for quality
* regression loss for width
* angle regression using a stable representation such as sine/cosine or an equivalent encoded form

---

## Data Split Protocol

This repository uses explicit split files stored under:

```text
data/splits/
```

The split protocol should be fixed and documented before reporting results. Depending on the experiment, this may be:

* **image-wise split**, or
* **object-wise split**

This is important because Cornell results are sensitive to the chosen evaluation protocol.

---

## Quick Start

### 1. Create environment

```bash
pip install -r requirements.txt
```

### 2. Download the Cornell dataset

```bash
bash scripts/download_cornell.sh
```

### 3. Verify the pipeline

```bash
python tests/test_pipeline.py
```

### 4. Train the baseline model

```bash
python train.py --config configs/baseline.yaml
```

### 5. Train the robust model

```bash
python train.py --config configs/robust.yaml
```

### 6. Evaluate both models

```bash
python evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/best.pth
python evaluate.py --config configs/robust.yaml --checkpoint checkpoints/robust/best.pth
```

---

## Cornell Dataset

The original Cornell hosting links are no longer consistently reliable, so this repository uses the provided helper script:

```bash
bash scripts/download_cornell.sh
```

After extraction, the dataset should be placed under:

```text
data/raw/cornell/
```

A typical sample includes:

* `pcd####r.png` — RGB image
* `pcd####d.tiff` — depth image
* `pcd####cpos.txt` — positive grasp rectangles
* `pcd####cneg.txt` — negative grasp rectangles

The Cornell dataset is a small RGB-D grasping benchmark with **885 annotated images** across **240 household objects**.

---

## Evaluation Protocol

A predicted grasp is considered correct if it satisfies both:

* **angle error < 30°**
* **IoU (Jaccard overlap) > 0.25** with a ground-truth grasp rectangle

Depending on the evaluation setting, the repository may report:

* overall grasp accuracy
* accuracy under synthetic occlusion
* occlusion-stratified performance curves
* calibration or uncertainty diagnostics for the robust model

---

## Results

| Model         | Accuracy (all) | Accuracy (occluded 30%) | Accuracy (occluded 60%) |
| ------------- | -------------: | ----------------------: | ----------------------: |
| Baseline      |              — |                       — |                       — |
| Robust (ours) |              — |                       — |                       — |

Replace these entries after training and evaluation.

---

## Reproducibility Notes

For stable experiments, it is recommended to keep:

* configuration files under `configs/`
* split definitions under `data/splits/`
* cached outputs under `data/processed/`
* saved checkpoints under `checkpoints/`

This helps keep training, evaluation, and report generation consistent.

---

## Team

| Member | Role                                                   |
| ------ | ------------------------------------------------------ |
| Marwah | Data pipeline, evaluation, occlusion metric design     |
| Rawda  | Model training, augmentation experiments               |
| Ayah   | Robust model extension, report writing, visualisations |

---

## References

1. Jiang et al., *Efficient Grasping from RGBD Images* — Cornell grasping benchmark
2. Redmon and Angelova, *Real-Time Grasp Detection Using Convolutional Neural Networks*
3. Morrison et al., *GG-CNN: Generative Grasping Convolutional Neural Network*

```
```
