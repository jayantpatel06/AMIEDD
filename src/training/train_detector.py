"""Training loop for Model 1: Degradation Estimator."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.degradation_detector import DegradationDetector
from .losses import MultiTaskDetectorLoss


def train_detector(
    model: DegradationDetector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints/detector",
    save_best: bool = True,
) -> Dict[str, list]:
    """
    Train Model 1 (Degradation Detector).
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = MultiTaskDetectorLoss(severity_weight=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_presence_acc": [],
        "val_severity_mae": [],
    }

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Detector Epoch {epoch}/{epochs} [Train]", leave=False)
        for degraded, clean, vector, _ in pbar:
            degraded = degraded.to(device)
            vector = vector.to(device)

            optimizer.zero_grad()
            presence_logits, severity_preds = model(degraded)
            loss, _, _ = criterion(presence_logits, severity_preds, vector)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = degraded.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = running_loss / max(total_samples, 1)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_presence = 0
        total_presence_checks = 0
        severity_diffs = 0.0
        val_samples = 0

        with torch.no_grad():
            for degraded, clean, vector, _ in val_loader:
                degraded = degraded.to(device)
                vector = vector.to(device)

                presence_logits, severity_preds = model(degraded)
                loss, _, _ = criterion(presence_logits, severity_preds, vector)

                batch_size = degraded.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size

                # Compute Presence Accuracy (Threshold at logit 0.0 / prob 0.5)
                pred_presence = (presence_logits > 0.0).float()
                gt_presence = (vector > 0.05).float()
                correct_presence += (pred_presence == gt_presence).sum().item()
                total_presence_checks += vector.numel()

                # Compute Severity MAE
                severity_diffs += torch.abs(severity_preds - vector).sum().item()

        val_loss = val_loss / max(val_samples, 1)
        val_acc = correct_presence / max(total_presence_checks, 1)
        val_mae = severity_diffs / max(total_presence_checks, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_presence_acc"].append(val_acc)
        history["val_severity_mae"].append(val_mae)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Presence Acc: {val_acc:.3f} | Severity MAE: {val_mae:.3f}")

        # Save Checkpoint
        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path / "best_detector.pth")

        torch.save(model.state_dict(), ckpt_path / "last_detector.pth")

    return history
