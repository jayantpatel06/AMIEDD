"""Model 2: Adaptive U-Net with Feature-wise Linear Modulation (FiLM)."""
import torch
import torch.nn as nn
from typing import Optional
from .film_layer import FiLMBlock


class ConvBlock(nn.Module):
    """Standard double convolution block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip-connection concatenation, and FiLM modulation."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, cond_dim: int, film_hidden_dim: int = 128):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        total_in = (in_channels // 2) + skip_channels
        self.conv1 = nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.film = FiLMBlock(out_channels, out_channels, cond_dim=cond_dim, hidden_dim=film_hidden_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if shapes differ slightly due to odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.film(x, condition)
        return x


class AdaptiveUNet(nn.Module):
    """
    Adaptive U-Net with FiLM conditioning.
    Takes degraded image and degradation severity vector [s_noise, s_blur, s_contrast, s_compression].
    Outputs clean restored image via residual learning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        cond_dim: int = 4,
        residual: bool = True,
        film_hidden_dim: int = 128,
    ):
        super().__init__()
        self.residual = residual
        self.cond_dim = cond_dim

        # Encoder
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256
        c_bottleneck = base_channels * 16  # 512

        self.enc1 = ConvBlock(in_channels, c1)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c1, c2)
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c2, c3)
        self.down3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(c3, c4)
        self.down4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(c4, c_bottleneck)

        # Decoder with FiLM
        self.dec4 = DecoderBlock(c_bottleneck, c4, c4, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
        self.dec3 = DecoderBlock(c4, c3, c3, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
        self.dec2 = DecoderBlock(c3, c2, c2, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)
        self.dec1 = DecoderBlock(c2, c1, c1, cond_dim=cond_dim, film_hidden_dim=film_hidden_dim)

        # Output projection
        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Degraded image tensor (B, C, H, W) in [0, 1].
            condition: Degradation condition vector (B, cond_dim) in [0, 1].
                       If None, default to zeros (generic unconditioned mode).
        Returns:
            Restored image tensor (B, C, H, W) in [0, 1].
        """
        if condition is None:
            condition = torch.zeros((x.size(0), self.cond_dim), device=x.device, dtype=x.dtype)

        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Bottleneck
        b = self.bottleneck(self.down4(e4))

        # Decoder path with FiLM
        d4 = self.dec4(b, e4, condition)
        d3 = self.dec3(d4, e3, condition)
        d2 = self.dec2(d3, e2, condition)
        d1 = self.dec1(d2, e1, condition)

        delta = self.final_conv(d1)

        if self.residual:
            # Predict residual correction bounded in [-1, 1]
            out = torch.clamp(x + self.tanh(delta), 0.0, 1.0)
        else:
            out = torch.clamp(delta, 0.0, 1.0)

        return out
