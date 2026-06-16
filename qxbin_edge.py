import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class QxBinEdge:
    """QxBin Logic on Edge: Personal Cubit Simulator
    
    Room-temperature simulation of quantum-like superposition
    using Binary Probability Matrices, fractions, and exponents.
    Runs on any laptop. Perfect for education, prototyping, edge AI.
    """
    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        # Initialize as normalized probability matrix (analog nuance)
        self.state = np.random.rand(grid_size, grid_size).astype(np.float64)
        self.state /= self.state.sum()

    def apply_superposition(self, bias: float = 0.75, power_n: int = 2, power_m: int = 1):
        """Simulate spinning coin superposition with fractional leaning states."""
        # Fractional contributions: bias**n for heads lean, (1-bias)**m for tails
        frac_heads = bias ** power_n
        frac_tails = (1 - bias) ** power_m
        
        # Build probability matrix (punch-card / grid style)
        x = np.linspace(frac_heads, frac_tails, self.grid_size)
        prob_matrix = np.outer(x, x)
        
        # Blend with existing state (multi-dimensional chain evolution)
        self.state = (self.state + prob_matrix) / 2.0
        self.state /= self.state.sum()  # Renormalize
        return self.state

    def measure(self) -> np.ndarray:
        """Probabilistic collapse to classical outcome (like measurement)."""
        flat = self.state.flatten()
        outcome_idx = np.random.choice(len(flat), p=flat)
        collapsed = np.zeros_like(flat)
        collapsed[outcome_idx] = 1.0
        return collapsed.reshape(self.state.shape)

    def visualize(self, title: str = "QxBin Probability Grid (Edge)"):
        """Visualize as colorful probability matrix with node clusters."""
        plt.figure(figsize=(8, 6))
        plt.imshow(self.state, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Probability Amplitude')
        plt.title(title)
        
        # Overlay nodes for superposition clusters (matching demo visuals)
        y, x = np.indices(self.state.shape)
        sizes = self.state.flatten() * 800 + 20
        plt.scatter(x.flatten(), y.flatten(), s=sizes, 
                    c='white', alpha=0.7, edgecolors='black', linewidths=0.5)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=== QxBin Edge Simulator ===")
    qx = QxBinEdge(grid_size=6)
    print("Initial normalized probability matrix shape:", qx.state.shape)
    
    qx.apply_superposition(bias=0.78, power_n=3, power_m=1)  # Strong heads lean
    print("After superposition (fractional exponents applied):")
    print(qx.state.round(3))
    
    qx.visualize()
    
    measured = qx.measure()
    print("\nMeasured (collapsed) state:")
    print(measured)
