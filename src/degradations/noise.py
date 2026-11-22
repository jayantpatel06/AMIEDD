"""Noise degradation operators for medical radiography simulation."""
from typing import Tuple
import numpy as np
from .base import BaseDegradation


class GaussianNoise(BaseDegradation):
    """Additive Gaussian noise simulating electronic detector noise."""

    def __init__(self, min_sigma: float = 0.02, max_sigma: float = 0.25):
        super().__init__("noise")
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def apply(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, float]:
        if severity <= 0.0:
            return image.copy(), 0.0

        sigma = self.min_sigma + severity * (self.max_sigma - self.min_sigma)
        noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
        degraded = np.clip(image + noise, 0.0, 1.0)
        return degraded, severity


class PoissonNoise(BaseDegradation):
    """Poisson (shot) noise simulating quantum mottle from low-dose X-rays."""

    def __init__(self, min_photons: float = 20.0, max_photons: float = 250.0):
        super().__init__("noise")
        self.min_photons = min_photons
        self.max_photons = max_photons

    def apply(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, float]:
        if severity <= 0.0:
            return image.copy(), 0.0

        # Severity 1.0 means few photons (maximum noise), 0.0 means high photon count
        photons = self.max_photons - severity * (self.max_photons - self.min_photons)
        scaled = np.clip(image, 0.0, 1.0) * photons
        noisy = np.random.poisson(scaled).astype(np.float32) / max(photons, 1.0)
        degraded = np.clip(noisy, 0.0, 1.0)
        return degraded, severity
