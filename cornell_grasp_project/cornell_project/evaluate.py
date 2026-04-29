"""
evaluate.py
-----------
Full evaluation script.

Computes:
    1. Standard Cornell grasp accuracy (IoU > 0.25, angle < 30°)
    2. Accuracy vs. occlusion level sweep (0% → 60% of pixels masked)
    3. Per-sample results saved to JSON
    4. Figures: accuracy curve, sample predictions

Usage
-----
    python evaluate.py \
        --config configs/baseline.yaml \
        --checkpoint checkpoints/baseline/best.pth

    python evaluate.py \
        --config configs/robust.yaml \
        --checkpoint checkpoints/robust/best.pth \
        --out_dir results/robust

    # Compare two models
    python evaluate.py \
        --config configs/baseline.yaml \
        --checkpoint checkpoints/baseline/best.pth \
        --compare_config configs/robust.yaml \
        --compare_checkpoint checkpoints/robust/best.pth
"""

import argparse
import os
import json
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.cornell_loader import CornellDataset
from torch.utils.data import DataLoader
from utils.grasp_utils import (
    prediction_to_grasp, is_grasp_correct, load_grasp_rectangles
)
from utils.occlusion_utils import apply_occlusion_fraction
from models.baseline_model import BaselineGraspCNN, GraspLoss
from models.robust_model import RobustGraspCNN


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg, ckpt_path, device):
    name = cfg['model']['name']
    if name == 'baseline':
        model = BaselineGraspCNN.from_config(cfg)
    else:
        model = RobustGraspCNN.from_config(cfg)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print(f"  Loaded: {ckpt_path}  (epoch {ckpt.get('epoch','?')}  "
          f"best_acc={ckpt.get('metric',0)*100:.1f}%)")
    return model


def build_eval_dataset(cfg):
    """Build validation dataset — same split as training."""
    d = cfg['data']
    return CornellDataset(
        root            = d['root'],
        split           = 'val',
        split_ratio     = d.get('split_ratio', [0.9, 0.1])[0],
        image_size      = tuple(d.get('image_size', [480, 640])),
        use_depth_only  = d.get('use_depth_only', True),
        normalize_depth = d.get('normalize_depth', True),
        augment         = False,
        occlusion_aug   = False,
    )


# ─────────────────────────────────────────────────────────────
# Standard evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_standard(model, dataset, cfg, device):
    """
    Evaluate grasp accuracy on the full validation set
    at zero synthetic occlusion.

    Returns:
        accuracy (float): fraction of correct grasps
        results  (list of bool)
    """
    iou_thr = cfg['evaluation']['iou_threshold']
    ang_thr = cfg['evaluation']['angle_threshold']

    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=2, pin_memory=True)
    correct_list = []

    for batch in tqdm(loader, desc="Standard eval"):
        x   = batch['input'].to(device)
        idx = batch['idx'].item()

        pred_q, pred_a, pred_w = model(x)
        q_np = pred_q[0, 0].cpu().numpy()
        a_np = pred_a[0, 0].cpu().numpy()
        w_np = pred_w[0, 0].cpu().numpy()

        sample   = dataset.samples[idx]
        gt_rects = load_grasp_rectangles(sample['label_path'])
        preds    = prediction_to_grasp(q_np, a_np, w_np,
                                       threshold=0.3, num_peaks=1)

        correct = any(is_grasp_correct(p, gt_rects, iou_thr, ang_thr)
                      for p in preds)
        correct_list.append(correct)

    accuracy = float(np.mean(correct_list)) if correct_list else 0.0
    return accuracy, correct_list


# ─────────────────────────────────────────────────────────────
# Occlusion sweep evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_occlusion_sweep(model, dataset, cfg, device,
                              occlusion_levels=None):
    """
    Evaluate accuracy at increasing levels of synthetic occlusion.

    For each occlusion fraction f:
        - Mask ~f fraction of each depth image with zeros
        - Recompute occlusion map
        - Run model and evaluate

    Args:
        occlusion_levels (list): fractions to test, e.g. [0, 0.1, ..., 0.6]

    Returns:
        dict: {fraction: accuracy}
    """
    if occlusion_levels is None:
        occlusion_levels = cfg['evaluation'].get(
            'occlusion_levels', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    iou_thr      = cfg['evaluation']['iou_threshold']
    ang_thr      = cfg['evaluation']['angle_threshold']
    use_occ_map  = not cfg['data'].get('use_depth_only', True)
    occ_map_type = cfg['data'].get('occlusion_map_type', 'variance')
    occ_window   = cfg['data'].get('variance_window', 15)
    norm_depth   = cfg['data'].get('normalize_depth', True)

    from utils.occlusion_utils import compute_occlusion_map

    results = {}

    for frac in occlusion_levels:
        correct_list = []

        for idx, sample in enumerate(tqdm(
                dataset.samples,
                desc=f"Occ={frac:.0%}", leave=False)):

            # Load raw depth
            depth = dataset._load_depth(sample['depth_path'])
            if dataset.image_size:
                depth = dataset._resize(depth, dataset.image_size)

            # Apply synthetic occlusion
            if frac > 0:
                depth = apply_occlusion_fraction(depth, fraction=frac)

            # Normalise
            if norm_depth:
                depth_in = dataset._normalize(depth)
            else:
                depth_in = depth.copy()

            # Build input tensor
            if use_occ_map:
                occ = compute_occlusion_map(depth, mode=occ_map_type,
                                             window=occ_window)
                x = torch.FloatTensor(
                    np.stack([depth_in, occ], axis=0)
                ).unsqueeze(0).to(device)    # (1, 2, H, W)
            else:
                x = torch.FloatTensor(depth_in[None, None]).to(device)

            pred_q, pred_a, pred_w = model(x)
            q_np = pred_q[0, 0].cpu().numpy()
            a_np = pred_a[0, 0].cpu().numpy()
            w_np = pred_w[0, 0].cpu().numpy()

            gt_rects = load_grasp_rectangles(sample['label_path'])
            preds    = prediction_to_grasp(q_np, a_np, w_np,
                                           threshold=0.3, num_peaks=1)

            correct = any(is_grasp_correct(p, gt_rects, iou_thr, ang_thr)
                          for p in preds)
            correct_list.append(correct)

        acc = float(np.mean(correct_list)) if correct_list else 0.0
        results[round(frac, 2)] = acc
        print(f"  Occlusion {frac:.0%} → accuracy {acc*100:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_occlusion_curve(sweep_results_dict, out_path):
    """
    Plot accuracy vs. occlusion fraction for one or more models.

    Args:
        sweep_results_dict (dict): {model_name: {fraction: accuracy}}
        out_path (str): where to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

    for i, (name, results) in enumerate(sweep_results_dict.items()):
        x = sorted(results.keys())
        y = [results[k] * 100 for k in x]
        ax.plot(x, y, marker='o', linewidth=2, markersize=7,
                color=colours[i % len(colours)], label=name)

    ax.set_xlabel('Occlusion fraction (proportion of pixels masked)')
    ax.set_ylabel('Grasp accuracy (%)')
    ax.set_title('Grasp Accuracy vs. Synthetic Occlusion Level\n'
                 '(IoU > 0.25, angle error < 30°)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-0.02, max(max(r.keys()) for r in sweep_results_dict.values()) + 0.02)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_degradation_bar(sweep_results_dict, out_path):
    """Bar chart: accuracy at 0% vs 50% occlusion per model."""
    models = list(sweep_results_dict.keys())
    acc_0  = [sweep_results_dict[m].get(0.0, 0) * 100 for m in models]
    acc_50 = [sweep_results_dict[m].get(0.5, 0) * 100 for m in models]

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, acc_0,  w, label='No occlusion',   color='#2ecc71')
    ax.bar(x + w/2, acc_50, w, label='50% occlusion',  color='#e74c3c')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Degradation Under Occlusion')
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 105)
    for i, (a, b) in enumerate(zip(acc_0, acc_50)):
        ax.text(i - w/2, a + 1, f'{a:.1f}%', ha='center', fontsize=9)
        ax.text(i + w/2, b + 1, f'{b:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',              required=True)
    parser.add_argument('--checkpoint',          required=True)
    parser.add_argument('--compare_config',      default=None)
    parser.add_argument('--compare_checkpoint',  default=None)
    parser.add_argument('--out_dir',             default='results/eval')
    parser.add_argument('--gpu',                 type=int, default=0)
    parser.add_argument('--no_sweep',            action='store_true',
                        help='Skip occlusion sweep (faster)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    cfg   = load_config(args.config)
    model = load_model(cfg, args.checkpoint, device)

    # Build dataset
    dataset = build_eval_dataset(cfg)
    print(f"Eval samples: {len(dataset)}\n")

    # ── Standard accuracy ─────────────────────────────────────
    print("Running standard evaluation...")
    acc, per_sample = evaluate_standard(model, dataset, cfg, device)
    print(f"\n  Standard accuracy: {acc*100:.1f}%  "
          f"({sum(per_sample)}/{len(per_sample)} correct)\n")

    all_results = {cfg['experiment']['name']: {}}

    # ── Occlusion sweep ───────────────────────────────────────
    if not args.no_sweep:
        print("Running occlusion sweep...")
        sweep = evaluate_occlusion_sweep(model, dataset, cfg, device)
        all_results[cfg['experiment']['name']] = sweep
        print()

    # ── Compare second model (optional) ───────────────────────
    if args.compare_config and args.compare_checkpoint:
        cfg2   = load_config(args.compare_config)
        model2 = load_model(cfg2, args.compare_checkpoint, device)
        ds2    = build_eval_dataset(cfg2)

        print("Standard eval — comparison model...")
        acc2, ps2 = evaluate_standard(model2, ds2, cfg2, device)
        print(f"\n  Comparison accuracy: {acc2*100:.1f}%\n")

        if not args.no_sweep:
            print("Occlusion sweep — comparison model...")
            sweep2 = evaluate_occlusion_sweep(model2, ds2, cfg2, device)
            all_results[cfg2['experiment']['name']] = sweep2
            print()

    # ── Save metrics ──────────────────────────────────────────
    metrics = {
        cfg['experiment']['name']: {
            'standard_accuracy': acc,
            'n_correct': int(sum(per_sample)),
            'n_total':   len(per_sample),
            'occlusion_sweep': all_results.get(cfg['experiment']['name'], {}),
        }
    }
    out_json = os.path.join(args.out_dir, 'metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {out_json}")

    # ── Plots ─────────────────────────────────────────────────
    if not args.no_sweep and all_results:
        # Convert fraction keys back to float
        float_results = {}
        for name, res in all_results.items():
            float_results[name] = {float(k): v for k, v in res.items()}

        plot_occlusion_curve(
            float_results,
            os.path.join(args.out_dir, 'occlusion_curve.png'))

        if len(float_results) > 1:
            plot_degradation_bar(
                float_results,
                os.path.join(args.out_dir, 'degradation_bar.png'))

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Model      : {cfg['experiment']['name']}")
    print(f"  Accuracy   : {acc*100:.1f}%")
    if not args.no_sweep:
        sweep_r = all_results.get(cfg['experiment']['name'], {})
        for frac, a in sorted(sweep_r.items()):
            print(f"  Occ {frac:.0%}     : {a*100:.1f}%")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
