import numpy as np
import matplotlib.pyplot as plt

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
    L = np.linalg.cholesky(V_A) # lower triangular matrix such that V_A = L @ L.T
    
    # Sample from standard normal distribution
    z = rng.standard_normal(size=(n_samples, 2))
    
    # Transform to have covariance V_A
    samples = z @ L.T
    
    # Add detector noise
    noise = rng.normal(loc=0.0, scale=noise_level, size=(n_samples, 2))
    samples += noise
    
    # p quadrature is not directly accessible in homodyne detection, 
    # so we can simulate that by setting it to zero or adding noise depending on the measurement type.

    # q quadrature is accessible in both homodyne and heterodyne detection, 
    # but the p quadrature is only accessible in heterodyne detection. 
    # Therefore, we can simulate the measurement process by modifying the samples based on the measurement type.

    if measurement_type == 'homodyne':
        # each shot, randomly choose to measure x or p (LO phase choice), mask the other
        mask = rng.integers(0, 2, size=n_samples)  # 0 -> measure x, 1 -> measure p
        samples[mask == 0, 1] = np.nan
        samples[mask == 1, 0] = np.nan
    elif measurement_type == 'heterodyne':
        # For heterodyne, we can simulate measuring both quadratures with added noise
        samples += rng.normal(loc=0.0, scale=noise_level, size=(n_samples, 2))
    
    return samples

def plot_samples(samples):
    """Plots the samples in phase space."""
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
    plt.title("Phase Space Samples")
    plt.xlabel("X Quadrature")
    plt.ylabel("P Quadrature")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def main():
    # Example usage
    r = 0.5  # Squeezing parameter
    V = generate_covariance_matrix(r)
    V_A = reduce_to_accessible(V)
    
    n_samples = 1000
    noise_level = 0.1
    measurement_type = 'heterodyne'  # or 'homodyne'
    rng = np.random.default_rng(seed=42)
    
    samples = sample_measurements(V_A, n_samples, noise_level, measurement_type, rng)
    plot_samples(samples)
    print(samples)

if __name__ == "__main__":
    main()