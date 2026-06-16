import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


@njit(parallel=True, fastmath=True)
def evolve_qxbin_chains(states: np.ndarray, biases: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Parallel evolution of multiple QxBin probability chains (server/cloud scale)."""
    n = states.shape[0]
    for i in prange(n):
        b = biases[i]
        pn, pm = powers[i, 0], powers[i, 1]
        frac = b ** pn
        tail = (1.0 - b) ** pm
        
        # Simple matrix evolution for each cubit chain
        new_state = (states[i] * frac + (1.0 - states[i]) * tail) * 0.5
        new_state = new_state / new_state.sum()
        states[i] = new_state
    return states


class QxBinCloud:
    """QxBin Logic on Cloud/Server: Scalable Ensemble Simulator
    
    Batch processing of many cubit chains for optimization,
    Monte Carlo, or large-scale probabilistic workloads.
    Numba-accelerated for speed. Easy to extend to GPU.
    """
    def __init__(self, num_cubits: int = 20, grid_size: int = 8):
        self.num_cubits = num_cubits
        self.grid_size = grid_size
        self.states = np.random.rand(num_cubits, grid_size, grid_size).astype(np.float64)
        # Normalize each matrix
        for i in range(num_cubits):
            self.states[i] /= self.states[i].sum()

    def evolve(self, biases: np.ndarray = None):
        """Evolve all chains in parallel."""
        if biases is None:
            biases = np.random.uniform(0.45, 0.85, self.num_cubits)
        powers = np.random.randint(1, 5, size=(self.num_cubits, 2)).astype(np.int32)
        
        self.states = evolve_qxbin_chains(self.states, biases, powers)
        return self.states.mean(axis=0)  # Aggregate view

    def probabilistic_optimize(self, target: float = 0.75, max_iter: int = 100):
        """Simple feedback loop to tune ensemble toward target probability."""
        for it in range(max_iter):
            agg = self.evolve()
            current_mean = agg.mean()
            if abs(current_mean - target) < 0.005:
                print(f"Converged at iteration {it} to mean prob ~{current_mean:.3f}")
                break
        return self.states

    def visualize_aggregate(self, title: str = "QxBin Cloud Aggregate (Many Cubits)"):
        """Show mean probability landscape across the ensemble."""
        agg = self.states.mean(axis=0)
        plt.figure(figsize=(8, 6))
        plt.imshow(agg, cmap='plasma', interpolation='nearest')
        plt.colorbar(label='Avg Probability Amplitude')
        plt.title(title)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=== QxBin Cloud / Server Simulator ===")
    qx = QxBinCloud(num_cubits=25, grid_size=7)
    print(f"Evolving {qx.num_cubits} cubit chains...")
    
    qx.probabilistic_optimize(target=0.72)
    qx.visualize_aggregate()
    
    print("\nDone. Ensemble mean probability matrix ready for downstream use.")
