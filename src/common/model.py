"""Model definitions for the federated learning task.

Provides the small multilayer perceptron used for binary classification
on tabular features, with configurable hidden widths and dropout.
"""
import torch
from torch import nn


class TabularMLP(nn.Module):
    """Simple MLP for binary classification on tabular features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        hidden_dims = hidden_dims or [64, 32]
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return binary logits for a batch of tabular features."""
        return self.network(features)
