"""Contrast and dynamic range degradation operators for medical radiography."""
from typing import Tuple
import numpy as np
from .base import BaseDegradation


class LowContrast(BaseDegradation):
    """Reduces dynamic range, simulating underexposed scans or poor detector calibration."""

    def __init__(self, min_factor: float = 0.20, max_factor: float = 0.75):
        super().__init__("contrast")
        self.min_factor = min_factor  # Severity 1.0 reduces contrast to 20%
        self.max_factor = max_factor  # Severity 0.0 reduces contrast to 75%

    def apply(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, float]:
        if severity <= 0.0:
            return image.copy(), 0.0

        # Contrast reduction factor: lower means less contrast / flatter gray image
        contrast_factor = self.max_factor - severity * (self.max_factor - self.min_factor)
        mean_val = np.mean(image)

        degraded = mean_val + contrast_factor * (image - mean_val)
        return np.clip(degraded, 0.0, 1.0).astype(np.float32), severity


class GammaDistortion(BaseDegradation):
    """Applies non-linear gamma distortion simulating faulty look-up tables or display calibration."""

    def __init__(self, min_gamma: float = 0.4, max_gamma: float = 2.2):
        super().__init__("contrast")
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def apply(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, float]:
        if severity <= 0.0:
            return image.copy(), 0.0

        # Gamma can deviate either lower (<1.0 wash out) or higher (>1.0 crush blacks)
        if np.random.rand() > 0.5:
            gamma = 1.0 + severity * (self.max_gamma - 1.0)
        else:
            gamma = 1.0 - severity * (1.0 - self.min_gamma)

        degraded = np.power(np.clip(image, 1e-6, 1.0), gamma)
        return np.clip(degraded, 0.0, 1.0).astype(np.float32), severity
