import torch

class HarmonicPotential:
    # Harmonic Oscillator potential with analytical solutions
    # V(x) = 1/2 * x^2 : E0 = 0.5 a.u.

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x**2