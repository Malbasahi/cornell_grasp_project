"""
utils/visualization.py
-----------------------
Visualisation utilities for the Cornell grasp detection project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm


def plot_sample(depth, quality, angle, width, gt_rects=None,
                pred_rects=None, title='Sample', figsize=(14, 5)):
    """
    Display a Cornell sample with its pixel maps and optional grasp overlays.

    Args:
        depth    (np.ndarray): H×W depth image
        quality  (np.ndarray): H×W quality map
        angle    (np.ndarray): H×W angle map
        width    (np.ndarray): H×W width map
        gt_rects   (list): ground-truth GraspRectangle objects
        pred_rects (list): predicted GraspRectangle objects
        title    (str)
        figsize  (tuple)
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    titles = ['Depth', 'Quality (GT)', 'Angle (GT)', 'Width (GT)']
    maps   = [depth, quality, angle, width]
    cmaps  = ['gray', 'RdYlGn', 'hsv', 'viridis']

    for ax, img, t, cmap in zip(axes, maps, titles, cmaps):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(t, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if gt_rects:
        for rect in gt_rects:
            pts = np.vstack([rect.points, rect.points[0]])
            axes[1].plot(pts[:, 1], pts[:, 0],
                         'g-', linewidth=1.5, alpha=0.8)

    if pred_rects:
        for rect in pred_rects:
            pts = np.vstack([rect.points, rect.points[0]])
            axes[1].plot(pts[:, 1], pts[:, 0],
                         'r--', linewidth=2.0, alpha=0.9)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_occlusion_maps(depth, binary_map, variance_map, figsize=(12, 4)):
    """
    Show the three occlusion representations side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(depth, cmap='gray')
    axes[0].set_title('Depth Image')
    axes[0].axis('off')

    axes[1].imshow(binary_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Binary Occlusion Map\n(1 = missing depth)')
    axes[1].axis('off')

    im = axes[2].imshow(variance_map, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Local Variance Map\n(high = uncertain boundary)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_occlusion_augmentation(original, augmented_list, labels,
                                figsize=(18, 4)):
    """
    Show original depth + several augmented versions side by side.
    """
    n = len(augmented_list) + 1
    fig, axes = plt.subplots(1, n, figsize=figsize)

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original\n(no occlusion)', fontsize=9)
    axes[0].axis('off')

    for ax, img, label in zip(axes[1:], augmented_list, labels):
        frac = (img == 0).mean()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{label}\n({frac:.0%} missing)', fontsize=9)
        ax.axis('off')

    plt.suptitle('Synthetic Occlusion Augmentation', fontsize=11)
    plt.tight_layout()
    return fig


def plot_prediction(depth, pred_q, pred_a, pred_w,
                    pred_rects=None, gt_rects=None,
                    correct=None, figsize=(14, 5)):
    """
    Show model prediction maps and reconstructed grasp rectangles.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Depth
    axes[0].imshow(depth, cmap='gray')
    axes[0].set_title('Input Depth')
    axes[0].axis('off')

    # Predicted quality
    axes[1].imshow(pred_q, cmap='RdYlGn', vmin=0, vmax=1)
    if gt_rects:
        for r in gt_rects:
            pts = np.vstack([r.points, r.points[0]])
            axes[1].plot(pts[:, 1], pts[:, 0], 'b-', lw=1.5, alpha=0.7,
                         label='GT')
    if pred_rects:
        for r in pred_rects:
            pts = np.vstack([r.points, r.points[0]])
            color = 'lime' if correct else 'red'
            axes[1].plot(pts[:, 1], pts[:, 0], '--', color=color,
                         lw=2, label='Pred')
    title = 'Quality + Grasps'
    if correct is not None:
        title += ' [✓ CORRECT]' if correct else ' [✗ WRONG]'
    axes[1].set_title(title, fontsize=9)
    axes[1].axis('off')

    # Predicted angle
    axes[2].imshow(pred_a, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
    axes[2].set_title('Predicted Angle')
    axes[2].axis('off')

    # Predicted width
    axes[3].imshow(pred_w, cmap='viridis', vmin=0, vmax=1)
    axes[3].set_title('Predicted Width')
    axes[3].axis('off')

    plt.tight_layout()
    return fig


def plot_training_curves(log_dir, out_path=None):
    """
    Load TensorBoard logs and plot training/validation curves.
    Requires tensorboard package.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
    except ImportError:
        print("tensorboard not installed — skipping curve plot")
        return

    ea = EventAccumulator(log_dir)
    ea.Reload()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    if 'train/loss' in ea.Tags()['scalars']:
        train_loss = ea.Scalars('train/loss')
        steps = [s.step for s in train_loss]
        vals  = [s.value for s in train_loss]
        axes[0].plot(steps, vals, alpha=0.4, color='blue', label='train (per step)')

    if 'val/loss' in ea.Tags()['scalars']:
        val_loss = ea.Scalars('val/loss')
        epochs = [s.step for s in val_loss]
        vals   = [s.value for s in val_loss]
        axes[0].plot(epochs, vals, 'b-o', markersize=4, label='val (per epoch)')

    axes[0].set_xlabel('Step / Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    if 'val/accuracy' in ea.Tags()['scalars']:
        val_acc = ea.Scalars('val/accuracy')
        epochs  = [s.step for s in val_acc]
        vals    = [s.value * 100 for s in val_acc]
        axes[1].plot(epochs, vals, 'g-o', markersize=4)
        axes[1].axhline(max(vals), color='g', linestyle='--', alpha=0.5,
                        label=f'Best: {max(vals):.1f}%')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Grasp Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    return fig


def save_results_table(metrics_dict, out_path):
    """
    Save a formatted results table as a text file.

    Args:
        metrics_dict: {model_name: {'standard_accuracy': float,
                                    'occlusion_sweep': {frac: acc}}}
        out_path (str)
    """
    lines = []
    lines.append("=" * 65)
    lines.append(f"  RESULTS TABLE")
    lines.append("=" * 65)

    for name, m in metrics_dict.items():
        lines.append(f"\nModel: {name}")
        lines.append(f"  Standard accuracy: {m['standard_accuracy']*100:.1f}%")
        lines.append(f"  ({m['n_correct']}/{m['n_total']} correct)")
        if m.get('occlusion_sweep'):
            lines.append("  Occlusion sweep:")
            for frac, acc in sorted(m['occlusion_sweep'].items()):
                bar = '█' * int(acc * 30) + '░' * (30 - int(acc * 30))
                lines.append(f"    {float(frac):.0%}  {acc*100:5.1f}%  {bar}")

    lines.append("\n" + "=" * 65)
    text = '\n'.join(lines)
    print(text)

    with open(out_path, 'w') as f:
        f.write(text)
    print(f"\nSaved: {out_path}")
