"""Model 3: Downstream clinical disease classifiers (DenseNet121, ResNet50, EfficientNet-B0)."""
import torch
import torch.nn as nn
from torchvision.models import (
    densenet121, DenseNet121_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)
from typing import Dict, Any, Optional


class DiseaseClassifier(nn.Module):
    """
    Downstream disease classifier wrapper supporting DenseNet121, ResNet50, and EfficientNet-B0.
    Standardized interface for Chest X-ray pneumonia detection.
    """

    SUPPORTED_ARCHITECTURES = ["densenet121", "resnet50", "efficientnet_b0"]

    def __init__(
        self,
        architecture: str = "densenet121",
        num_classes: int = 2,
        pretrained: bool = True,
        in_channels: int = 1,
    ):
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        self.in_channels = in_channels

        if self.architecture == "densenet121":
            try:
                weights = DenseNet121_Weights.DEFAULT if pretrained else None
                self.model = densenet121(weights=weights)
            except Exception:
                self.model = densenet121(weights=None)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)

        elif self.architecture == "resnet50":
            try:
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                self.model = resnet50(weights=weights)
            except Exception:
                self.model = resnet50(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

        elif self.architecture == "efficientnet_b0":
            try:
                weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
                self.model = efficientnet_b0(weights=weights)
            except Exception:
                self.model = efficientnet_b0(weights=None)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose from {self.SUPPORTED_ARCHITECTURES}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. If single-channel input is passed, replicate to 3 channels for ImageNet compatibility.
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Normalize with ImageNet standard mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x - mean) / std

        return self.model(x_norm)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def fit_classifier_head(
        self,
        clean_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epochs: int = 4,
        learning_rate: float = 1e-3,
        class_weights: Optional[torch.Tensor] = None,
        fine_tune_backbone: bool = False,
        fine_tune_lr: float = 5e-5,
    ) -> None:
        """
        Calibrate and optionally fine-tune classification head on clean CXR images.
        Supports class weighting for handling imbalanced pneumonia vs normal cohorts.
        """
        self.to(device)

        # Stage 1: Freeze backbone and train classification head
        for name, param in self.model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        head_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=learning_rate, weight_decay=1e-3)

        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(epochs):
            for _, clean, _, labels in clean_loader:
                clean = clean.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self.forward(clean)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        # Stage 2: Optional end-to-end fine-tuning at lower learning rate
        if fine_tune_backbone:
            for param in self.parameters():
                param.requires_grad = True
            ft_optimizer = torch.optim.AdamW(self.parameters(), lr=fine_tune_lr, weight_decay=1e-4)
            for epoch in range(max(epochs // 2, 2)):
                for _, clean, _, labels in clean_loader:
                    clean = clean.to(device)
                    labels = labels.to(device)
                    ft_optimizer.zero_grad()
                    logits = self.forward(clean)
                    loss = criterion(logits, labels)
                    loss.backward()
                    ft_optimizer.step()

        # Finalize
        for param in self.parameters():
            param.requires_grad = True
        self.eval()
