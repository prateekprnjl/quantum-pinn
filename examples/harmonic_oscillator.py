import torch
import matplotlib.pyplot as plt

from quantum_pinn.potentials import HarmonicPotential
from quantum_pinn.model import HarmonicPINN
from quantum_pinn.numerics import loss

def main():
    torch.manual_seed(3)

    # grid
    x = torch.linspace(-5.0, 5.0, 500).reshape(-1, 1)
    x.requires_grad_(True)

    potential = HarmonicPotential()
    model = HarmonicPINN(hidden_layer=30)

    # Hyperparameter: learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training
    for epoch in range(5_000):
        optimizer.zero_grad()

        loss_val, losses = loss(model, x, potential)
        loss_val.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss {losses['Total loss']:.4e} | Energy {model.energy.item():.4f}")

    print("\n")
    print(f"Predicted energy: {model.energy.item():.4f}")
    print("Exact energy : 0.5000")

#    plot_solution(model, x)

# Plot generated vs exact
def plot_solution(model, x):
    model.eval()

    with torch.no_grad():
        psi_pinn = model(x).squeeze()

    psi_exact = (torch.pi ** (-0.25) * torch.exp(-0.5 * x.squeeze()) ** 2)

    plt.plot(x.detach().numpy(), psi_pinn.numpy(), color='blue', label="generated")
    plt.plot(x.detach().numpy(), psi_exact.detach().numpy(), color='red', label="exact")

    plt.legend()
    plt.title("Harmonic Oscillator Ground State")

    plt.show()

if __name__ == "__main__":
    main()