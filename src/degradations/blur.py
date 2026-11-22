"""Blur degradation operators simulating patient motion and focal blur."""
from typing import Tuple
import cv2
import numpy as np
from .base import BaseDegradation


class GaussianBlur(BaseDegradation):
    """Gaussian blur simulating optical defocus and scanner modulation transfer function loss."""

    def __init__(self, min_kernel: int = 3, max_kernel: int = 15, min_sigma: float = 0.5, max_sigma: float = 3.5):
        super().__init__("blur")
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def apply(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, float]:
        if severity <= 0.0:
            return image.copy(), 0.0

        # Map severity to odd kernel size
        raw_k = int(self.min_kernel + severity * (self.max_kernel - self.min_kernel))
        k = raw_k if raw_k % 2 == 1 else raw_k + 1
        sigma = self.min_sigma + severity * (self.max_sigma - self.min_sigma)

        degraded = cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)
        return np.clip(degraded, 0.0, 1.0).astype(np.float32), severity


class MotionBlur(BaseDegradation):
    """Directional motion blur simulating patient breathing or movement during scanning."""

    def __init__(self, min_length: int = 3, max_length: int = 15):
        super().__init__("blur")
        self.min_length = min_length
        self.max_length = max_length

    def apply(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, float]:
        if severity <= 0.0:
            return image.copy(), 0.0

        length = int(self.min_length + severity * (self.max_length - self.min_length))
        length = max(length, 1)
        angle = np.random.uniform(0, 180)

        # Build motion kernel
        kernel = np.zeros((length, length), dtype=np.float32)
        center = length // 2
        kernel[center, :] = 1.0

        # Rotate kernel
        rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (length, length))
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel[center, center] = 1.0

        degraded = cv2.filter2D(image, -1, kernel)
        return np.clip(degraded, 0.0, 1.0).astype(np.float32), severity
