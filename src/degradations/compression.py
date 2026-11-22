"""JPEG compression and quantization artifact simulation."""
from typing import Tuple
import cv2
import numpy as np
from .base import BaseDegradation


class JPEGCompression(BaseDegradation):
    """Simulates lossy DCT block quantization and high-frequency ringing artifacts from PACS transmission."""

    def __init__(self, min_quality: int = 10, max_quality: int = 75):
        super().__init__("compression")
        self.min_quality = min_quality  # Maximum degradation (severity 1.0)
        self.max_quality = max_quality  # Mild degradation (severity 0.0)

    def apply(self, image: np.ndarray, severity: float) -> Tuple[np.ndarray, float]:
        if severity <= 0.0:
            return image.copy(), 0.0

        # High severity -> low JPEG quality factor
        quality = int(round(self.max_quality - severity * (self.max_quality - self.min_quality)))
        quality = int(np.clip(quality, 5, 95))

        uint8_img = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        success, encimg = cv2.imencode(".jpg", uint8_img, encode_param)
        if not success:
            return image.copy(), 0.0

        decoded = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
        degraded = (decoded.astype(np.float32) / 255.0).reshape(image.shape)
        return np.clip(degraded, 0.0, 1.0), severity
