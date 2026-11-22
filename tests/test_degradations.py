"""Unit tests for degradation operators and pipeline."""
import numpy as np
import pytest

from src.degradations.noise import GaussianNoise, PoissonNoise
from src.degradations.blur import GaussianBlur, MotionBlur
from src.degradations.contrast import LowContrast, GammaDistortion
from src.degradations.compression import JPEGCompression
from src.degradations.pipeline import DegradationPipeline


@pytest.fixture
def sample_image():
    """Create a synthetic 224x224 gradient test pattern in [0, 1]."""
    x = np.linspace(0, 1, 224)
    y = np.linspace(0, 1, 224)
    xx, yy = np.meshgrid(x, y)
    return (0.5 * xx + 0.5 * yy).astype(np.float32)


def test_gaussian_noise(sample_image):
    op = GaussianNoise()
    out, sev = op.apply(sample_image, severity=0.6)
    assert out.shape == sample_image.shape
    assert out.dtype == np.float32
    assert np.min(out) >= 0.0 and np.max(out) <= 1.0
    assert not np.allclose(out, sample_image)


def test_gaussian_blur(sample_image):
    op = GaussianBlur()
    out, sev = op.apply(sample_image, severity=0.7)
    assert out.shape == sample_image.shape
    assert np.min(out) >= 0.0 and np.max(out) <= 1.0


def test_low_contrast(sample_image):
    op = LowContrast()
    out, sev = op.apply(sample_image, severity=0.8)
    assert out.shape == sample_image.shape
    assert np.std(out) < np.std(sample_image)


def test_jpeg_compression(sample_image):
    op = JPEGCompression()
    out, sev = op.apply(sample_image, severity=0.9)
    assert out.shape == sample_image.shape
    assert np.min(out) >= 0.0 and np.max(out) <= 1.0


def test_pipeline_vector(sample_image):
    pipeline = DegradationPipeline()
    vector = np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float32)
    out, v = pipeline.apply_vector(sample_image, vector)
    assert out.shape == sample_image.shape
    assert np.allclose(v, vector)
    assert np.min(out) >= 0.0 and np.max(out) <= 1.0


def test_pipeline_random_modes(sample_image):
    pipeline = DegradationPipeline()
    for mode in ["single", "pair", "triple", "quad", "clean"]:
        out, v = pipeline.apply_random(sample_image, mode=mode)
        assert out.shape == sample_image.shape
        active = np.sum(v > 0.0)
        if mode == "single":
            assert active == 1
        elif mode == "pair":
            assert active == 2
        elif mode == "triple":
            assert active == 3
        elif mode == "quad":
            assert active == 4
        elif mode == "clean":
            assert active == 0
