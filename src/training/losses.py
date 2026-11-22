"""Loss functions for degradation detection and adaptive image restoration."""
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskDetectorLoss(nn.Module):
    """
    Multi-task loss for Model 1 (Degradation Estimator):
      - Binary Cross Entropy for presence of each degradation
      - Smooth L1 Loss for severity estimation (masked where degradation is present)
    """

    def __init__(self, severity_weight: float = 2.0, threshold: float = 0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.severity_weight = severity_weight
        self.threshold = threshold

    def forward(
        self,
        presence_logits: torch.Tensor,
        severity_preds: torch.Tensor,
        target_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            presence_logits: (B, 4)
            severity_preds: (B, 4) in [0, 1]
            target_vector: (B, 4) in [0, 1]
        """
        # Ground-truth binary presence
        presence_targets = (target_vector > self.threshold).float()
        loss_presence = self.bce(presence_logits, presence_targets)

        # Severity loss masked to active degradations
        raw_sev_loss = self.smooth_l1(severity_preds, target_vector)
        # Apply mask
        active_count = torch.sum(presence_targets)
        if active_count > 0:
            loss_severity = torch.sum(raw_sev_loss * presence_targets) / active_count
        else:
            loss_severity = torch.tensor(0.0, device=target_vector.device)

        total_loss = loss_presence + self.severity_weight * loss_severity
        return total_loss, loss_presence, loss_severity


class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity Index (SSIM) loss for PyTorch."""

    def __init__(self, window_size: int = 11, in_channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.register_buffer("window", self._create_window(window_size, in_channels))

    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        window = self.window.to(img1.device, dtype=img1.dtype)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.in_channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.in_channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.in_channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.in_channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.in_channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()


class SobelEdgeLoss(nn.Module):
    """Sobel edge difference loss to preserve subtle radiographic boundaries."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sx = self.sobel_x.to(pred.device, dtype=pred.dtype)
        sy = self.sobel_y.to(pred.device, dtype=pred.dtype)

        gx_pred = F.conv2d(pred, sx, padding=1, groups=self.in_channels)
        gy_pred = F.conv2d(pred, sy, padding=1, groups=self.in_channels)
        gx_target = F.conv2d(target, sx, padding=1, groups=self.in_channels)
        gy_target = F.conv2d(target, sy, padding=1, groups=self.in_channels)

        loss_x = F.l1_loss(gx_pred, gx_target)
        loss_y = F.l1_loss(gy_pred, gy_target)
        return loss_x + loss_y


class CompositeEnhancerLoss(nn.Module):
    """
    Total restoration loss:
      L_total = w_l1 * L1 + w_ssim * SSIM_loss + w_edge * Edge_loss
    """

    def __init__(self, w_l1: float = 1.0, w_ssim: float = 0.2, w_edge: float = 0.1):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_edge = w_edge

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = SobelEdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_edge = self.edge(pred, target)

        total = self.w_l1 * loss_l1 + self.w_ssim * loss_ssim + self.w_edge * loss_edge
        details = {
            "l1": float(loss_l1.item()),
            "ssim": float(loss_ssim.item()),
            "edge": float(loss_edge.item()),
            "total": float(total.item()),
        }
        return total, details
