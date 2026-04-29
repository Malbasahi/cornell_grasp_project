"""
train.py
---------
Main training script for Cornell grasp detection.

Trains on the FULL Cornell dataset (no artificial subsetting).
Saves the best checkpoint based on validation accuracy.
Logs to TensorBoard.

Usage
-----
    # Train baseline (depth only)
    python train.py --config configs/baseline.yaml

    # Train occlusion-robust model
    python train.py --config configs/robust.yaml

    # Resume from checkpoint
    python train.py --config configs/robust.yaml --resume checkpoints/robust/best.pth

    # Override epochs
    python train.py --config configs/baseline.yaml --epochs 30
"""

import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.cornell_loader import build_dataloaders
from utils.grasp_utils import (
    prediction_to_grasp, evaluate_predictions,
    load_grasp_rectangles
)
from models.baseline_model import BaselineGraspCNN, GraspLoss
from models.robust_model import RobustGraspCNN


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    name = cfg['model']['name']
    if name == 'baseline':
        return BaselineGraspCNN.from_config(cfg)
    elif name == 'robust':
        return RobustGraspCNN.from_config(cfg)
    else:
        raise ValueError(f"Unknown model: {name}")


def build_optimizer(model, cfg):
    lr  = cfg['training']['learning_rate']
    wd  = cfg['training'].get('weight_decay', 1e-4)
    opt = cfg['training'].get('optimizer', 'adam').lower()
    if opt == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == 'sgd':
        return optim.SGD(model.parameters(), lr=lr,
                         weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {opt}")


def build_scheduler(optimizer, cfg, n_epochs):
    sched = cfg['training'].get('lr_scheduler', 'cosine')
    if sched == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif sched == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


def save_checkpoint(model, optimizer, scheduler, epoch, metric, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'metric':          metric,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler and ckpt.get('scheduler_state'):
        scheduler.load_state_dict(ckpt['scheduler_state'])
    return ckpt.get('epoch', 0) + 1


# ─────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loss_fn, loader, optimizer, device,
                    writer, epoch, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    loss_components = {'quality_loss': 0, 'angle_loss': 0,
                       'width_loss': 0, 'total_loss': 0}
    n = len(loader)

    pbar = tqdm(enumerate(loader), total=n,
                desc=f"Epoch {epoch:03d} [train]", leave=False)

    for step, batch in pbar:
        x    = batch['input'].to(device)     # (B, C, H, W)
        gt_q = batch['quality'].to(device)   # (B, 1, H, W)
        gt_a = batch['angle'].to(device)
        gt_w = batch['width'].to(device)

        pred_q, pred_a, pred_w = model(x)
        loss, ld = loss_fn(pred_q, pred_a, pred_w, gt_q, gt_a, gt_w)

        optimizer.zero_grad()
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += ld.get(k, 0)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        global_step = (epoch - 1) * n + step
        writer.add_scalar('train/loss', loss.item(), global_step)

    avg = {k: v / n for k, v in loss_components.items()}
    return avg


# ─────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg):
    """
    Run validation: compute loss AND grasp accuracy
    using the standard Cornell evaluation criterion.
    """
    model.eval()
    iou_thr   = cfg['evaluation']['iou_threshold']
    ang_thr   = cfg['evaluation']['angle_threshold']
    total_loss = 0.0
    all_correct = []

    pbar = tqdm(loader, desc="Validation", leave=False)

    for batch in pbar:
        x    = batch['input'].to(device)
        gt_q = batch['quality'].to(device)
        gt_a = batch['angle'].to(device)
        gt_w = batch['width'].to(device)

        pred_q, pred_a, pred_w = model(x)
        loss, _ = loss_fn(pred_q, pred_a, pred_w, gt_q, gt_a, gt_w)
        total_loss += loss.item()

        # Convert predictions → grasp rectangles and evaluate
        for i in range(x.shape[0]):
            q_np = pred_q[i, 0].cpu().numpy()
            a_np = pred_a[i, 0].cpu().numpy()
            w_np = pred_w[i, 0].cpu().numpy()

            # Get ground-truth rectangles from label file
            sample = loader.dataset.samples[batch['idx'][i].item()]
            gt_rects = load_grasp_rectangles(sample['label_path'])

            # Predict top grasp
            pred_rects = prediction_to_grasp(q_np, a_np, w_np,
                                              threshold=0.3, num_peaks=1)

            if len(pred_rects) == 0:
                all_correct.append(False)
                continue

            # Check if any prediction is correct
            correct = False
            for pr in pred_rects:
                from utils.grasp_utils import is_grasp_correct
                if is_grasp_correct(pr, gt_rects, iou_thr, ang_thr):
                    correct = True
                    break
            all_correct.append(correct)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = float(np.mean(all_correct)) if all_correct else 0.0
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  required=True)
    parser.add_argument('--resume',  default=None)
    parser.add_argument('--epochs',  type=int, default=None)
    parser.add_argument('--gpu',     type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg['training']['epochs'] = args.epochs

    torch.manual_seed(cfg['experiment']['seed'])
    np.random.seed(cfg['experiment']['seed'])

    # ── Device ───────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU. Training will be slow.")

    # ── Data ─────────────────────────────────────────────────
    print("Loading Cornell dataset (full)...")
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val   batches: {len(val_loader)}")

    # ── Model & Loss ─────────────────────────────────────────
    model   = build_model(cfg).to(device)
    loss_fn = GraspLoss(
        quality_weight = cfg['loss']['quality_weight'],
        angle_weight   = cfg['loss']['angle_weight'],
        width_weight   = cfg['loss']['width_weight'],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']}  |  Parameters: {n_params:,}")

    # ── Optimiser & Scheduler ─────────────────────────────────
    n_epochs  = cfg['training']['epochs']
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, n_epochs)

    # ── Resume ───────────────────────────────────────────────
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch - 1}")

    # ── Logging ──────────────────────────────────────────────
    writer  = SummaryWriter(log_dir=cfg['experiment']['log_dir'])
    out_dir = cfg['experiment']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    best_accuracy = 0.0
    clip_norm     = cfg['training'].get('clip_grad_norm', 1.0)

    print(f"\n{'='*60}")
    print(f"  Experiment : {cfg['experiment']['name']}")
    print(f"  Epochs     : {start_epoch} → {n_epochs}")
    print(f"  Batch size : {cfg['training']['batch_size']}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────
        train_losses = train_one_epoch(
            model, loss_fn, train_loader, optimizer,
            device, writer, epoch, clip_norm)

        # ── Validate ───────────────────────────────────────
        val_loss, val_acc = validate(model, val_loader, loss_fn,
                                      device, cfg)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0

        # ── Log ────────────────────────────────────────────
        writer.add_scalar('val/loss',     val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc,  epoch)
        writer.add_scalar('train/loss_epoch',
                          train_losses['total_loss'], epoch)
        writer.add_scalar('lr',
                          optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:03d}/{n_epochs:03d} | "
              f"train_loss={train_losses['total_loss']:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc*100:.1f}% | "
              f"time={elapsed:.0f}s")

        # ── Save checkpoints ───────────────────────────────
        # Latest
        save_checkpoint(model, optimizer, scheduler, epoch, val_acc,
                        os.path.join(out_dir, 'latest.pth'))
        # Best
        if val_acc >= best_accuracy:
            best_accuracy = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc,
                            os.path.join(out_dir, 'best.pth'))
            print(f"  ★ New best accuracy: {best_accuracy*100:.1f}%")

    writer.close()
    print(f"\nTraining complete.")
    print(f"Best validation accuracy : {best_accuracy*100:.1f}%")
    print(f"Checkpoints saved to     : {out_dir}/")


if __name__ == '__main__':
    main()
