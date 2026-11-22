"""Unit tests for loss functions."""
import torch
import pytest

from src.training.losses import MultiTaskDetectorLoss, SSIMLoss, SobelEdgeLoss, CompositeEnhancerLoss


def test_detector_loss():
    b = 4
    logits = torch.randn(b, 4, requires_grad=True)
    severity = torch.rand(b, 4, requires_grad=True)
    target = torch.tensor([
        [0.7, 0.0, 0.0, 0.4],
        [0.0, 0.5, 0.0, 0.0],
        [0.8, 0.6, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    loss_fn = MultiTaskDetectorLoss()
    total, bce, sev = loss_fn(logits, severity, target)

    assert total.item() > 0
    total.backward()
    assert logits.grad is not None
    assert severity.grad is not None


def test_composite_enhancer_loss():
    b = 2
    pred = torch.rand(b, 1, 64, 64, requires_grad=True)
    target = torch.rand(b, 1, 64, 64)

    criterion = CompositeEnhancerLoss()
    total, details = criterion(pred, target)

    assert total.item() > 0
    assert "l1" in details and "ssim" in details and "edge" in details
    total.backward()
    assert pred.grad is not None
