import torch

def derivatives(psi: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Calculates the first and second derivative of the first variable (psi) wrt. second variable (x)

    dpsi_dx = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    d2psi_dx = torch.autograd.grad(dpsi_dx, x, grad_outputs=torch.ones_like(dpsi_dx), create_graph=True)[0]

    return dpsi_dx, d2psi_dx
