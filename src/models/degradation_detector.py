"""Model 1: Degradation Estimator ('Problem Detective') using ResNet18."""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Tuple, Optional


class DegradationDetector(nn.Module):
    """
    Multi-task ResNet18 that diagnoses the presence and continuous severity of
    image defects: [noise, blur, contrast, compression].
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_degradations: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_degradations = num_degradations

        # Load ResNet18 backbone (with graceful fallback if offline)
        try:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except Exception:
            backbone = resnet18(weights=None)
            pretrained = False

        # Modify first layer if grayscale
        if in_channels == 1:
            orig_conv = backbone.conv1
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                # Average ImageNet RGB weights for grayscale
                with torch.no_grad():
                    self.conv1.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
        else:
            self.conv1 = backbone.conv1

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

        in_features = backbone.fc.in_features  # 512

        # Multi-task heads:
        # Head A: Defect presence (logits for BCE)
        self.presence_head = nn.Linear(in_features, num_degradations)
        # Head B: Defect continuous severity in [0, 1]
        self.severity_head = nn.Sequential(
            nn.Linear(in_features, num_degradations),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            presence_logits: (B, num_degradations)
            severity_preds: (B, num_degradations) in [0, 1]
        """
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)

        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        h = self.avgpool(h)
        feat = torch.flatten(h, 1)
        feat = self.dropout(feat)

        presence_logits = self.presence_head(feat)
        severity_preds = self.severity_head(feat)

        return presence_logits, severity_preds

    def predict_vector(
        self,
        x: torch.Tensor,
        threshold: float = 0.35,
    ) -> torch.Tensor:
        """
        Predict gated continuous degradation vector in [0, 1]^4.
        """
        self.eval()
        with torch.no_grad():
            presence_logits, severity_preds = self.forward(x)
            presence_probs = torch.sigmoid(presence_logits)
            # Gate severity by presence probability
            mask = (presence_probs >= threshold).float()
            vector = presence_probs * severity_preds * mask
        return vector
