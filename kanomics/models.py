"""
Contains custom neural network architectures used across the KANomics pipeline.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Multi-Layer Perceptron baseline model for protein expression regression."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
