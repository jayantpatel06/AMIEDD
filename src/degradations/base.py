"""Base degradation interface."""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseDegradation(ABC):
    """Abstract base class for image degradation operators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, float]:
        """
        Apply degradation to input image.

        Args:
            image: Float32 numpy array with values in [0, 1], shape (H, W) or (H, W, C).
            severity: Continuous severity value in [0.0, 1.0].

        Returns:
            Tuple of (degraded_image, actual_severity_applied)
        """
        pass

    def __call__(self, image: np.ndarray, severity: float = 0.5) -> Tuple[np.ndarray, float]:
        severity = float(np.clip(severity, 0.0, 1.0))
        return self.apply(image, severity)
