"""Training script for fall detection model.

Usage:
    python -m fall_detection.train --data_root ./dataset --epochs 60

See --help for all options.
"""
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

from .utils import set_seed, save_json, FEATURE_DIM, DEFAULT_FPS
from .dataset import FallDataset
from .model import FallDetectionTCN
from .losses import CombinedLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train fall detection model')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='work_dirs/fall_detection')
    p.add_argument('--fps', type=float, default=None,
                   help='Override fps (default: read from JSON)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--window_sec', type=float, default=1.5)
    p.add_argument('--stride_sec', type=float, default=0.2)
    p.add_argument('--horizon_sec', type=float, default=1.0)
    # Loss weights
    p.add_argument('--state_weight', type=float, default=1.0)
    p.add_argument('--impact_weight', type=float, default=2.0)
    p.add_argument('--pred_weight', type=float, default=1.0)
    # Model
    p.add_argument('--channels', type=str, default='64,64,128,128',
                   help='TCN channel sizes, comma-separated')
    p.add_argument('--dropout', type=float, default=0.15)
    p.add_argument('--no_conf_gate', action='store_true')
    return p.parse_args()


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: CombinedLoss, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Dict[str, float]:
    model.train()
    total_losses = {'loss_total': 0, 'loss_state': 0,
                    'loss_impact': 0, 'loss_pred': 0}
    n_batches = 0
    correct_state = 0
    total_samples = 0

    for features, state, impact, pred in loader:
        features = features.to(device)
        state = state.to(device)
        impact = impact.to(device)
        pred = pred.to(device)

        outputs = model(features)
        losses = criterion(
            outputs['state'], outputs['impact'], outputs['predict'],
            state, impact, pred
        )

        optimizer.zero_grad()
        losses['loss_total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = features.size(0)
        for k, v in losses.items():
            total_losses[k] += v.item() * bs
        total_samples += bs
        n_batches += 1

        # State accuracy
        pred_state = outputs['state'].argmax(dim=1)
        correct_state += (pred_state == state).sum().item()

    avg = {k: v / total_samples for k, v in total_losses.items()}
    avg['state_acc'] = correct_state / total_samples
    return avg


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             criterion: CombinedLoss, device: torch.device
             ) -> Dict[str, float]:
    model.eval()
    total_losses = {'loss_total': 0, 'loss_state': 0,
                    'loss_impact': 0, 'loss_pred': 0}
    total_samples = 0
    correct_state = 0

    # For AP/F1
    all_impact_logits = []
    all_impact_labels = []
    all_pred_logits = []
    all_pred_labels = []

    for features, state, impact, pred in loader:
        features = features.to(device)
        state = state.to(device)
        impact = impact.to(device)
        pred = pred.to(device)

        outputs = model(features)
        losses = criterion(
            outputs['state'], outputs['impact'], outputs['predict'],
            state, impact, pred
        )

        bs = features.size(0)
        for k, v in losses.items():
            total_losses[k] += v.item() * bs
        total_samples += bs

        pred_state = outputs['state'].argmax(dim=1)
        correct_state += (pred_state == state).sum().item()

        all_impact_logits.append(outputs['impact'].squeeze(-1).cpu())
        all_impact_labels.append(impact.cpu())
        all_pred_logits.append(outputs['predict'].squeeze(-1).cpu())
        all_pred_labels.append(pred.cpu())

    avg = {k: v / total_samples for k, v in total_losses.items()}
    avg['state_acc'] = correct_state / total_samples

    # Compute F1 for impact and prediction
    impact_logits_all = torch.cat(all_impact_logits)
    impact_labels_all = torch.cat(all_impact_labels)
    pred_logits_all = torch.cat(all_pred_logits)
    pred_labels_all = torch.cat(all_pred_labels)

    avg['impact_f1'] = _binary_f1(impact_logits_all, impact_labels_all)
    avg['pred_f1'] = _binary_f1(pred_logits_all, pred_labels_all)

    return avg


def _binary_f1(logits: torch.Tensor, labels: torch.Tensor,
               threshold: float = 0.5) -> float:
    """Compute binary F1 score."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build datasets
    print("Loading datasets...")
    ds_kwargs = dict(
        fps=args.fps,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        horizon_sec=args.horizon_sec,
    )
    train_ds, val_ds = FallDataset.build_splits(
        args.data_root, val_ratio=args.val_ratio, seed=args.seed, **ds_kwargs
    )

    if len(train_ds) == 0:
        print("ERROR: No training samples. Check data_root and files.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Build model
    channels = [int(c) for c in args.channels.split(',')]
    model = FallDetectionTCN(
        in_features=FEATURE_DIM,
        channels=channels,
        dropout=args.dropout,
        use_confidence_gate=not args.no_conf_gate,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(
        state_weight=args.state_weight,
        impact_weight=args.impact_weight,
        pred_weight=args.pred_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    best_val_loss = float('inf')
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) lr={lr:.6f}")
        print(f"  Train: loss={train_metrics['loss_total']:.4f} "
              f"state_acc={train_metrics['state_acc']:.3f}")
        print(f"  Val:   loss={val_metrics['loss_total']:.4f} "
              f"state_acc={val_metrics['state_acc']:.3f} "
              f"impact_f1={val_metrics['impact_f1']:.3f} "
              f"pred_f1={val_metrics['pred_f1']:.3f}")

        record = {'epoch': epoch, 'lr': lr, **{f'train_{k}': v for k, v in train_metrics.items()},
                  **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(record)

        # Save checkpoint
        is_best = val_metrics['loss_total'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss_total']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
            }, output_dir / 'best.pth')
            print(f"  ** Best model saved (val_loss={best_val_loss:.4f})")

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'args': vars(args),
        }, output_dir / 'latest.pth')

    print(f"\nTraining complete. Best epoch: {best_epoch} "
          f"(val_loss={best_val_loss:.4f})")

    # Save history
    save_json(history, str(output_dir / 'history.json'))
    print(f"History saved to {output_dir / 'history.json'}")


if __name__ == '__main__':
    main()
