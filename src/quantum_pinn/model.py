import torch
from torch import nn

class HarmonicPINN(nn.Module):

    def __init__(self, hidden_layer: int = 10):
        super().__init__()

        # Three layer neural network
        self.network = nn.Sequential(nn.Linear(1, hidden_layer), nn.Tanh(), nn.Linear(hidden_layer, hidden_layer),
                                     nn.Tanh(), nn.Linear(hidden_layer, 1),)

        # Parameter to be trained - Energy of the state
        self.energy = nn.Parameter(torch.tensor(1.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)