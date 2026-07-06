import numpy as np

# simulate.py
def generate_covariance_matrix(r: float) -> np.ndarray:
    """Returns 4x4 V(r) for the two-mode squeezed vacuum."""
    cov_matrix = np.array([[np.cosh(2*r), 0, np.sinh(2*r), 0],
                           [0, np.cosh(2*r), 0, -np.sinh(2*r)],
                           [np.sinh(2*r), 0, np.cosh(2*r), 0],
                           [0, -np.sinh(2*r), 0, np.cosh(2*r)]])
    return cov_matrix

def reduce_to_accessible(V: np.ndarray) -> np.ndarray:
    """Top-left 2x2 block — the physically observable state."""
    top_left_block = V[:2, :2]
    return top_left_block

def sample_measurements(V_A, n_samples, noise_level, measurement_type, rng):
    """
    Draws x ~ N(0, V_A) via Cholesky (fast, seedable),
    adds detector noise eps ~ N(0, sigma^2 I),
    applies measurement_type logic (see below), returns (n_samples, 2) array.
    """
    # Cholesky decomposition of V_A
    L = np.linalg.cholesky(V_A)
    
    # Sample from standard normal distribution
    z = rng.standard_normal(size=(n_samples, 2))
    
    # Transform to have covariance V_A
    samples = z @ L.T
    
    # Add detector noise
    noise = rng.normal(loc=0.0, scale=noise_level, size=(n_samples, 2))
    samples += noise
    
    if measurement_type == 'homodyne':
        # For homodyne, we can simulate measuring one quadrature (e.g., x)
        samples[:, 1] = 0  # Set p quadrature to zero
    elif measurement_type == 'heterodyne':
        # For heterodyne, we can simulate measuring both quadratures with added noise
        samples += rng.normal(loc=0.0, scale=noise_level, size=(n_samples, 2))
    
    return samples

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