import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


@njit(parallel=True, fastmath=True)
def _evolve_batch(states, biases, ns, ms):
    """Numba-accelerated parallel evolution of QxBin probability matrices."""
    n_cubits = states.shape[0]
    for i in prange(n_cubits):
        b = biases[i]
        nn = ns[i]
        mm = ms[i]
        
        frac = b ** nn
        tail = (1.0 - b) ** mm
        
        # Element-wise blend using fractional exponents
        blended = (states[i] * frac + (1.0 - states[i]) * tail) * 0.5
        
        # Stable normalize
        total = blended.sum()
        if total > 1e-12:
            states[i] = blended / total
        else:
            states[i] = np.ones_like(blended) / blended.size
    return states


class QxBinCloud:
    """
    QxBin Cloud / Server Simulator - Scalable Ensemble of Cubit Chains
    
    Runs many Binary Probability Matrices in parallel.
    Uses the same fractional exponent (n, m) logic as the edge version.
    Ideal for batch simulation, optimization, Monte Carlo-style workloads.
    """
    def __init__(self, num_cubits: int = 24, grid_size: int = 7):
        self.num_cubits = num_cubits
        self.grid_size = grid_size
        self.states = np.random.rand(num_cubits, grid_size, grid_size).astype(np.float64)
        # Normalize all
        for i in range(num_cubits):
            s = self.states[i].sum()
            if s > 0:
                self.states[i] /= s

    def evolve_chains(self, biases=None):
        """Evolve all cubit chains in parallel using QxBin fractional logic."""
        if biases is None:
            biases = np.random.uniform(0.5, 0.85, self.num_cubits)
        ns = np.random.randint(1, 5, self.num_cubits)
        ms = np.random.randint(1, 5, self.num_cubits)
        
        self.states = _evolve_batch(self.states, biases, ns, ms)
        return self.states.mean(0)  # Return aggregate for convenience

    def optimize_to_target(self, target_mean: float = 0.7, max_steps: int = 80):
        """Simple probabilistic optimization loop (tune ensemble toward target)."""
        for step in range(max_steps):
            agg = self.evolve_chains()
            current = agg.mean()
            if abs(current - target_mean) < 0.003:
                print(f"Converged after {step} steps | mean prob ≈ {current:.4f}")
                break
        return self.states

    def visualize(self, title="QxBin Cloud - Aggregate Probability Landscape"):
        agg = self.states.mean(0)
        plt.figure(figsize=(8, 6))
        plt.imshow(agg, cmap="plasma", interpolation="nearest")
        plt.colorbar(label="Average Amplitude")
        plt.title(title)
        plt.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("QxBin Cloud v2 - Scalable Cubit Ensemble")
    cloud = QxBinCloud(num_cubits=30, grid_size=6)
    print(f"Evolving {cloud.num_cubits} parallel cubit chains...")
    
    cloud.optimize_to_target(target_mean=0.68)
    cloud.visualize()
    
    print("\nDone. Ensemble ready.")
