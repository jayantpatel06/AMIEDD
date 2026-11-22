"""Helper utilities for configuration, device management, and reproducibility."""
import os
import random
from pathlib import Path
from typing import Dict, Any, Union

# Configure temporary writable matplotlib directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import yaml
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto", num_threads: int = 8) -> torch.device:
    """Resolve and configure execution device."""
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif preferred == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if num_threads > 0:
            torch.set_num_threads(num_threads)
    return device


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration dictionary to YAML."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
