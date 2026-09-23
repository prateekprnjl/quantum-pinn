import torch

from quantum_pinn.potentials import HarmonicPotential

def test_harmonic_potential():
    potential = HarmonicPotential()
    x = torch.tensor([[0.0], [1.0], [2.0]])
    expected = torch.tensor([[0.0], [0.5], [2.0]])
    result = potential(x)

    assert torch.allclose(result, expected)
