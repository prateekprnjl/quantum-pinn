import torch

def derivatives(psi: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Calculates the first and second derivative of the first variable (psi) wrt. second variable (x)

    dpsi_dx = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    d2psi_dx = torch.autograd.grad(dpsi_dx, x, grad_outputs=torch.ones_like(dpsi_dx), create_graph=True)[0]

    return dpsi_dx, d2psi_dx

def se_residual(psi: torch.Tensor, x: torch.Tensor, energy: torch.Tensor, potential) -> torch.Tensor:
    # Calculates the residual of expected energy vs exact energy
    # enforce: [-0.5 * d2/dx2 + V(x)] Psi = E*Psi

    _, d2psi_dx = derivatives(psi, x)

    kinetic = -0.5 * d2psi_dx
    potential_energy = potential * psi

    residual = kinetic + potential_energy - energy * psi

    return torch.mean(residual**2)

def norm_residual(psi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Calculate the residual from expected normalization of wavefunction
    # enforce: || Psi ||^2 = 1

    integral = torch.trapezoid(psi.squeeze()**2, x.squeeze())

    return (integral - 1.0) ** 2

def boundary_residual(model, x_min: float, x_max: float) -> torch.Tensor:
    # Soft boundary condition
    # enforce: Psi(x_min) = Psi(x_max) = 0

    x_boundary = torch.tensor([[x_min], [x_max]], dtype=torch.float32)
    psi_boundary = model(x_boundary)

    return torch.mean(psi_boundary**2)

def loss(model, x: torch.Tensor, potential, norm_weight: float = 5.0, boundary_weight = 5.0) -> tuple[torch.Tensor, dict[str, float]]:
    # Return total loss from the simulation
    # Hyperparameters: norm_weight, boundary_weight

    psi = model(x)

    se = se_residual(psi, x, model.energy, potential)
    norm = norm_residual(psi, x)
    boundary = boundary_residual(model, float(torch.min(x)), float(torch.max(x)))

    total_loss = se + (norm_weight * norm) + (boundary_weight * boundary)
    losses = {
        "Total loss": float(total_loss.detach()), 
        "SE loss": float(se.detach()),
        "Norm loss": float(norm.detach()),
        "Boundary loss": float(boundary.detach()),
    }

    return total_loss, losses
