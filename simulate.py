import numpy as np

# simulate.py
def generate_covariance_matrix(r: float) -> np.ndarray:
    """Returns 4x4 V(r) for the two-mode squeezed vacuum."""
    

def reduce_to_accessible(V: np.ndarray) -> np.ndarray:
    """Top-left 2x2 block — the physically observable state."""

def sample_measurements(V_A, n_samples, noise_level, measurement_type, rng):
    """
    Draws x ~ N(0, V_A) via Cholesky (fast, seedable),
    adds detector noise eps ~ N(0, sigma^2 I),
    applies measurement_type logic (see below), returns (n_samples, 2) array.
    """

def main():
    # Example usage
    r = 0.5  # Squeezing parameter
    V = generate_covariance_matrix(r)
    V_A = reduce_to_accessible(V)
    
    n_samples = 1000
    noise_level = 0.1
    measurement_type = 'homodyne'  # or 'heterodyne'
    rng = np.random.default_rng(seed=42)
    
    samples = sample_measurements(V_A, n_samples, noise_level, measurement_type, rng)
    print(samples)

if __name__ == "__main__":
    main()