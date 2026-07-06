import numpy as np
from simulate import generate_covariance_matrix, reduce_to_accessible, sample_measurements
from estimate import estimate_covariance
from metrics import compute_particle_number, compute_entropy, estimate_squeezing, matrix_error
import matplotlib.pyplot as plt

def run_single_trial(r, n_samples, noise_level, measurement_type, seed):
    rng = np.random.default_rng(seed)

    V_true = generate_covariance_matrix(r)
    V_A_true = reduce_to_accessible(V_true)

    samples = sample_measurements(V_A_true, n_samples, noise_level, measurement_type, rng)
    V_A_hat = estimate_covariance(samples)

    return {
        "r_true": r,
        "n_samples": n_samples,
        "noise_level": noise_level,
        "measurement_type": measurement_type,
        "seed": seed,
        "r_hat": estimate_squeezing(V_A_hat),
        "N_true": compute_particle_number(V_A_true),
        "N_hat": compute_particle_number(V_A_hat),
        "S_true": compute_entropy(V_A_true),
        "S_hat": compute_entropy(V_A_hat),
        "V_error": matrix_error(V_A_true, V_A_hat),
    }

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

def plot_results(results):
    """Plots the results of multiple trials."""
    plt.figure(figsize=(10, 6))
    plt.scatter([res["r_true"] for res in results], [res["r_hat"] for res in results], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label="Ideal")
    plt.xlabel("True Squeezing Parameter (r)")
    plt.ylabel("Estimated Squeezing Parameter (r_hat)")
    plt.title("Squeezing Parameter Estimation")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    result = run_single_trial(r=0.5, n_samples=1000, noise_level=0.1,
                               measurement_type="heterodyne", seed=42)
    
    samples = sample_measurements(reduce_to_accessible(generate_covariance_matrix(0.5)), 1000, 0.1, "heterodyne", np.random.default_rng(seed=42))
    plot_samples(samples)

    print(result)
    plot_results([result])
