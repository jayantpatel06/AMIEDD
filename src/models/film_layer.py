"""Feature-wise Linear Modulation (FiLM) modules."""
import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """
    Multilayer perceptron that predicts affine modulation parameters (gamma, beta)
    from a degradation condition vector.
    """

    def __init__(self, cond_dim: int, num_features: int, hidden_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, num_features * 2),
        )
        # Initialize final layer near zero so FiLM starts as near-identity mapping
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: Tensor of shape (B, cond_dim)
        Returns:
            gamma: (B, num_features, 1, 1)
            beta: (B, num_features, 1, 1)
        """
        params = self.mlp(condition)  # (B, 2 * num_features)
        gamma, beta = torch.split(params, self.num_features, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


class FiLMBlock(nn.Module):
    """
    Convolutional block modulated by FiLM:
    Output = (1 + gamma) * Norm(Conv(x)) + beta
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.film_gen = FiLMGenerator(cond_dim=cond_dim, num_features=out_channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        gamma, beta = self.film_gen(condition)
        h = (1.0 + gamma) * h + beta
        return self.act(h)
