"""Chest X-Ray Dataset supporting on-the-fly degradation simulation."""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..degradations.pipeline import DegradationPipeline


class CXRDataset(Dataset):
    """
    PyTorch Dataset for Chest X-Ray images.
    Returns:
      degraded: (1, H, W) float32 tensor in [0, 1]
      clean: (1, H, W) float32 tensor in [0, 1]
      degradation_vector: (4,) float32 tensor [noise, blur, contrast, compression]
      label: int (0: Normal, 1: Pneumonia)
    """

    CLASS_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        pipeline: Optional[DegradationPipeline] = None,
        deterministic: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = Path(root_dir) / split
        self.image_size = image_size
        self.pipeline = pipeline or DegradationPipeline()
        self.deterministic = deterministic
        self.seed = seed

        # Collect image filepaths and labels
        self.samples: List[Tuple[Path, int]] = []
        for class_name, label in self.CLASS_MAP.items():
            class_folder = self.root_dir / class_name
            if class_folder.exists():
                for ext in ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"):
                    for file_path in class_folder.glob(ext):
                        self.samples.append((file_path, label))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No CXR images found in {self.root_dir} for split '{split}'")

        self.samples.sort(key=lambda x: str(x[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load grayscale image resized to target dimensions, normalized to [0, 1]."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read image: {path}")

        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        return (img.astype(np.float32) / 255.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        file_path, label = self.samples[idx]
        clean_img = self._load_image(file_path)

        if self.deterministic:
            # Deterministic pseudo-random generation per index for reproducible validation
            np.random.seed(self.seed + idx)
            degraded_img, vector = self.pipeline.apply_random(clean_img, mode="any")
        else:
            degraded_img, vector = self.pipeline.apply_random(clean_img, mode="any")

        # Convert to PyTorch tensors (1, H, W)
        clean_t = torch.from_numpy(clean_img).unsqueeze(0).float()
        degraded_t = torch.from_numpy(degraded_img).unsqueeze(0).float()
        vector_t = torch.from_numpy(vector).float()

        return degraded_t, clean_t, vector_t, label
