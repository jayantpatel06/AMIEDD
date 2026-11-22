"""Degradation composition pipeline for single and mixed medical image degradations."""
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .base import BaseDegradation
from .noise import GaussianNoise, PoissonNoise
from .blur import GaussianBlur, MotionBlur
from .contrast import LowContrast, GammaDistortion
from .compression import JPEGCompression


class DegradationPipeline:
    """
    Physical radiography acquisition & storage degradation pipeline.
    Vector indices:
      0: noise
      1: blur
      2: contrast
      3: compression
    """

    DEGRADATION_NAMES = ["noise", "blur", "contrast", "compression"]

    def __init__(
        self,
        noise_op: Optional[BaseDegradation] = None,
        blur_op: Optional[BaseDegradation] = None,
        contrast_op: Optional[BaseDegradation] = None,
        compression_op: Optional[BaseDegradation] = None,
    ):
        self.noise = noise_op or GaussianNoise()
        self.blur = blur_op or GaussianBlur()
        self.contrast = contrast_op or LowContrast()
        self.compression = compression_op or JPEGCompression()

    def apply_vector(
        self,
        image: np.ndarray,
        vector: Union[np.ndarray, List[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply degradations according to a specified severity vector [noise, blur, contrast, compression].
        Physical ordering applied:
          1. Contrast reduction (sensor/attenuation loss)
          2. Blur (optics/motion)
          3. Noise (quantum/detector electronics)
          4. Compression (PACS lossy archive/network)
        """
        v = np.clip(np.array(vector, dtype=np.float32), 0.0, 1.0)
        assert len(v) == 4, f"Vector must have 4 elements, got {len(v)}"

        out = image.copy()
        s_noise, s_blur, s_contrast, s_compression = float(v[0]), float(v[1]), float(v[2]), float(v[3])

        # Step 1: Contrast
        if s_contrast > 0.0:
            out, _ = self.contrast.apply(out, s_contrast)

        # Step 2: Blur
        if s_blur > 0.0:
            out, _ = self.blur.apply(out, s_blur)

        # Step 3: Noise
        if s_noise > 0.0:
            out, _ = self.noise.apply(out, s_noise)

        # Step 4: JPEG Compression
        if s_compression > 0.0:
            out, _ = self.compression.apply(out, s_compression)

        return np.clip(out, 0.0, 1.0).astype(np.float32), v

    def apply_random(
        self,
        image: np.ndarray,
        mode: str = "any",
        min_severity: float = 0.20,
        max_severity: float = 0.90,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random single or combined degradation.

        Args:
            image: Input image [0, 1] float32.
            mode: 'single', 'pair', 'triple', 'quad', 'clean', or 'any'.
            min_severity: Minimum active degradation severity.
            max_severity: Maximum active degradation severity.
        """
        vector = np.zeros(4, dtype=np.float32)

        if mode == "clean":
            return image.copy(), vector

        if mode == "single":
            num_active = 1
        elif mode == "pair":
            num_active = 2
        elif mode == "triple":
            num_active = 3
        elif mode == "quad":
            num_active = 4
        elif mode == "any":
            # Probability distribution: 5% clean, 45% single, 35% pair, 15% triple/quad
            rand = random.random()
            if rand < 0.05:
                num_active = 0
            elif rand < 0.50:
                num_active = 1
            elif rand < 0.85:
                num_active = 2
            else:
                num_active = random.choice([3, 4])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if num_active > 0:
            active_indices = random.sample(range(4), num_active)
            for idx in active_indices:
                vector[idx] = float(np.random.uniform(min_severity, max_severity))

        return self.apply_vector(image, vector)

    @classmethod
    def vector_to_dict(cls, vector: np.ndarray) -> Dict[str, float]:
        """Convert 4D vector to readable dictionary."""
        return {name: float(vector[i]) for i, name in enumerate(cls.DEGRADATION_NAMES)}

    @classmethod
    def dict_to_vector(cls, d: Dict[str, float]) -> np.ndarray:
        """Convert degradation dictionary to 4D vector."""
        return np.array([float(d.get(name, 0.0)) for name in cls.DEGRADATION_NAMES], dtype=np.float32)
