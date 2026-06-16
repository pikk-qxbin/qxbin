import numpy as np
import matplotlib.pyplot as plt


class QxBinEdge:
    """
    QxBin Edge Simulator - Personal Cubit on Classical Hardware
    
    Implements core QxBin Logic:
    - Binary Probability Matrix (2D grid instead of linear bits)
    - Fractional probabilities via bias and exponents (n, m)
    - Superposition-like blending of states
    - Probabilistic measurement
    
    Designed for laptop use, education, edge AI, and rapid prototyping.
    Room temperature. No special hardware required.
    """
    def __init__(self, grid_size: int = 6):
        self.grid_size = grid_size
        # Start with uniform-ish random probability matrix (analog starting point)
        self.state = np.random.rand(grid_size, grid_size).astype(np.float64)
        self._normalize()

    def _normalize(self):
        s = self.state.sum()
        if s > 0:
            self.state /= s

    def apply_superposition(self, bias: float = 0.75, n: int = 2, m: int = 1):
        """
        Apply QxBin-style superposition.
        
        Uses fractional 'leaning coin' : bias**n for one direction,
        (1-bias)**m for the other.
        Builds a new probability matrix via outer product (grid coordinate system).
        Blends with current state.
        
        n and m act as the exponent coordinates for directed contributions.
        """
        frac = bias ** n
        tail = (1.0 - bias) ** m
        
        # Create coordinate vectors using the powered fractions
        vec = np.linspace(frac, tail, self.grid_size)
        new_matrix = np.outer(vec, vec)
        
        # Blend (superposition-like update)
        self.state = (self.state + new_matrix) / 2.0
        self._normalize()
        return self.state

    def measure(self):
        """Probabilistic collapse to a classical outcome (weighted by current probabilities)."""
        flat = self.state.flatten()
        idx = np.random.choice(len(flat), p=flat)
        result = np.zeros_like(flat)
        result[idx] = 1.0
        return result.reshape(self.state.shape)

    def visualize(self, title="QxBin Probability Grid"):
        """Visualize the Binary Probability Matrix with node emphasis."""
        plt.figure(figsize=(8, 6))
        plt.imshow(self.state, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Probability Amplitude")
        plt.title(title)
        
        # Nodes for visual clusters (inspired by QxBin demo grids)
        y, x = np.indices(self.state.shape)
        sizes = np.clip(self.state.flatten() * 1200, 10, 800)
        plt.scatter(x.flatten(), y.flatten(), s=sizes, c="white", alpha=0.65,
                    edgecolors="black", linewidths=0.4)
        plt.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("QxBin Edge v2 - Personal Cubit Simulator")
    qx = QxBinEdge(grid_size=5)
    print("Initial state shape:", qx.state.shape)
    
    qx.apply_superposition(bias=0.8, n=3, m=1)  # Strong directional lean
    print("State after superposition (fractional exponents n=3, m=1):")
    print(np.round(qx.state, 3))
    
    qx.visualize("Edge QxBin - Probability Matrix")
    
    collapsed = qx.measure()
    print("\nMeasured (collapsed) outcome:")
    print(collapsed)
