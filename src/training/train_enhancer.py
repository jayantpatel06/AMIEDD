"""Training loop for Model 2: Adaptive U-Net with FiLM Conditioning."""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.adaptive_unet import AdaptiveUNet
from ..models.degradation_detector import DegradationDetector
from .losses import CompositeEnhancerLoss


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * torch.log10(1.0 / mse).item())


def train_enhancer(
    model: AdaptiveUNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    detector: Optional[DegradationDetector] = None,
    epochs: int = 25,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    checkpoint_dir: str = "checkpoints/enhancer",
    save_best: bool = True,
) -> Dict[str, list]:
    """
    Train Model 2 (Adaptive U-Net).
    Optionally utilizes Model 1 during training to bridge realistic estimation error.
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    if detector is not None:
        detector = detector.to(device)
        detector.eval()

    criterion = CompositeEnhancerLoss(w_l1=1.0, w_ssim=0.2, w_edge=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    best_val_psnr = -float("inf")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Enhancer Epoch {epoch}/{epochs} [Train]", leave=False)
        for degraded, clean, vector, _ in pbar:
            degraded = degraded.to(device)
            clean = clean.to(device)
            vector = vector.to(device)

            # Conditioning vector: use GT or Model 1 prediction with 30% probability
            if detector is not None and torch.rand(1).item() < 0.30:
                with torch.no_grad():
                    condition = detector.predict_vector(degraded)
            else:
                condition = vector

            optimizer.zero_grad()
            restored = model(degraded, condition)
            loss, details = criterion(restored, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = degraded.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "l1": f"{details['l1']:.4f}"})

        scheduler.step()
        train_loss = running_loss / max(total_samples, 1)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for degraded, clean, vector, _ in val_loader:
                degraded = degraded.to(device)
                clean = clean.to(device)
                vector = vector.to(device)

                # At test/validation, use detector if available, otherwise vector
                if detector is not None:
                    condition = detector.predict_vector(degraded)
                else:
                    condition = vector

                restored = model(degraded, condition)
                loss, details = criterion(restored, clean)

                batch_size = degraded.size(0)
                val_loss += loss.item() * batch_size
                val_psnr_sum += compute_psnr(restored, clean) * batch_size
                val_ssim_sum += (1.0 - details["ssim"]) * batch_size
                val_samples += batch_size

        val_loss = val_loss / max(val_samples, 1)
        val_psnr = val_psnr_sum / max(val_samples, 1)
        val_ssim = val_ssim_sum / max(val_samples, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.3f}")

        # Save Checkpoint based on PSNR
        if save_best and val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            }, ckpt_path / "best_enhancer.pth")

        torch.save(model.state_dict(), ckpt_path / "last_enhancer.pth")

    return history
