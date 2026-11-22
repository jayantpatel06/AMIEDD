"""Unit tests for neural network architectures."""
import torch
import pytest

from src.models.film_layer import FiLMGenerator, FiLMBlock
from src.models.degradation_detector import DegradationDetector
from src.models.adaptive_unet import AdaptiveUNet
from src.models.disease_classifier import DiseaseClassifier


def test_film_layer():
    b, c, h, w = 2, 32, 28, 28
    cond_dim = 4
    x = torch.randn(b, c, h, w)
    condition = torch.rand(b, cond_dim)

    film_block = FiLMBlock(in_channels=c, out_channels=c, cond_dim=cond_dim)
    out = film_block(x, condition)

    assert out.shape == (b, c, h, w)
    assert not torch.isnan(out).any()


def test_degradation_detector():
    b = 2
    x = torch.rand(b, 1, 224, 224)
    model = DegradationDetector(in_channels=1, num_degradations=4, pretrained=False)

    presence_logits, severity_preds = model(x)
    assert presence_logits.shape == (b, 4)
    assert severity_preds.shape == (b, 4)
    assert (severity_preds >= 0.0).all() and (severity_preds <= 1.0).all()

    vec = model.predict_vector(x)
    assert vec.shape == (b, 4)


def test_adaptive_unet():
    b = 2
    x = torch.rand(b, 1, 128, 128)
    condition = torch.rand(b, 4)
    model = AdaptiveUNet(in_channels=1, out_channels=1, base_channels=16, cond_dim=4, residual=True)

    out = model(x, condition)
    assert out.shape == (b, 1, 128, 128)
    assert (out >= 0.0).all() and (out <= 1.0).all()

    # Test backward pass
    loss = out.sum()
    loss.backward()
    assert model.enc1.block[0].weight.grad is not None


def test_disease_classifier():
    b = 2
    x = torch.rand(b, 1, 224, 224)
    model = DiseaseClassifier(architecture="densenet121", num_classes=2, pretrained=False)

    logits = model(x)
    assert logits.shape == (b, 2)

    probs = model.predict_proba(x)
    assert probs.shape == (b, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(b), atol=1e-5)
