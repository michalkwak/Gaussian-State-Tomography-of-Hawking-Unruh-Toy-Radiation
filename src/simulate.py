import numpy as np
import matplotlib.pyplot as plt

# simulate.py

def generate_covariance_matrix(r: float) -> np.ndarray:
    """Returns 4x4 V(r) for the two-mode squeezed vacuum (joint state,
    both wedges). Used only for the idealized joint-access comparison —
    the physically realistic pipeline uses reduce_to_accessible(V) below."""
    cov_matrix = np.array([[np.cosh(2*r), 0, np.sinh(2*r), 0],
                           [0, np.cosh(2*r), 0, -np.sinh(2*r)],
                           [np.sinh(2*r), 0, np.cosh(2*r), 0],
                           [0, -np.sinh(2*r), 0, np.cosh(2*r)]])
    return cov_matrix


def reduce_to_accessible(V: np.ndarray) -> np.ndarray:
    """Top-left 2x2 block — the one-wedge (right-wedge) marginal state
    V_R(r) = cosh(2r) I_2, i.e. what an outside observer actually has
    access to. This is the physically realistic starting point for the
    main experiment; recovering the full joint state from this alone is
    not possible, since many joint states share the same marginal."""
    top_left_block = V[:2, :2]
    return top_left_block


def sample_measurements(V_A, n_samples, noise_level, measurement_type, rng):
    """
    Simulates detector outcomes for the one-wedge accessible mode V_A (2x2).

    This is a two-step model, kept explicitly separate:
      1. State sampling: x ~ N(0, V_A) via Cholesky — samples the Gaussian
         state's Wigner function. This alone is NOT a measurement outcome.
      2. Measurement protocol: converts state samples into what a given
         detector scheme would actually report.
           - joint:     both quadratures observed exactly (idealized limit)
           - homodyne:  only one quadrature observed per shot, chosen at
                        random; the other is missing (NaN), not zero
           - heterodyne: both quadratures observed every shot, but with an
                        unavoidable +1 unit of vacuum noise added
                        (Sigma_het = V_A + I_2), per Weedbrook et al. Sec II.5
    Independent Gaussian detector noise (sigma_noise) is added on top of
    the ideal measurement outcome in all cases, to emulate real apparatus
    imperfection.

    Returns (n_samples, 2) array.
    """
    # --- Step 1: state sampling (Wigner-function sampling, not a measurement yet) ---
    L = np.linalg.cholesky(V_A)  # lower triangular matrix such that V_A = L @ L.T
    z = rng.standard_normal(size=(n_samples, 2))
    samples = z @ L.T  # samples ~ N(0, V_A)

    # --- Step 2: measurement protocol ---
    if measurement_type == 'joint':
        pass  # idealized: both quadratures observed exactly, nothing to add here
    elif measurement_type == 'homodyne':
        # each shot, randomly choose to measure x or p (local-oscillator phase
        # choice); mask the unmeasured quadrature as missing, not zero
        mask = rng.integers(0, 2, size=n_samples)  # 0 -> measure x, 1 -> measure p
        samples[mask == 0, 1] = np.nan
        samples[mask == 1, 0] = np.nan
    elif measurement_type == 'heterodyne':
        # fixed +1 unit of vacuum noise (Sigma_het = V_A + I_2), NOT scaled
        # by noise_level -- this is a fundamental measurement penalty, not
        # a technical/detector imperfection
        samples += rng.standard_normal(size=(n_samples, 2))
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")

    # --- detector noise, on top of the measurement outcome, all schemes ---
    detector_noise = rng.normal(loc=0.0, scale=noise_level, size=(n_samples, 2))
    samples += detector_noise

    return samples


def sample_idealized_joint_access(V, n_samples, noise_level, rng):
    """Idealized COMPARISON case only: samples both modes (right + left
    wedge) as if an observer had access to the full joint state. This is
    not physically realistic (the left-wedge/partner mode is inaccessible
    behind the horizon) -- it exists purely to quantify, by contrast, how
    much reconstruction accuracy is lost when restricted to one wedge.
    V must be the full 4x4 joint covariance matrix. Returns (n_samples, 4).
    """
    L = np.linalg.cholesky(V)
    z = rng.standard_normal(size=(n_samples, 4))
    samples = z @ L.T
    noise = rng.normal(loc=0.0, scale=noise_level, size=(n_samples, 4))
    samples += noise
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
    # Example usage: physically realistic one-wedge pipeline
    r = 0.5  # Squeezing parameter
    V = generate_covariance_matrix(r)
    V_A = reduce_to_accessible(V)  # one-wedge marginal, the physical case

    n_samples = 1000
    noise_level = 0.1
    rng = np.random.default_rng(seed=42)

    for measurement_type in ('joint', 'homodyne', 'heterodyne'):
        samples = sample_measurements(V_A, n_samples, noise_level, measurement_type, rng)
        print(measurement_type, samples[:5])

    # Idealized joint-access comparison (both wedges) -- NOT the physical case
    joint_samples = sample_idealized_joint_access(V, n_samples, noise_level, rng)
    print("idealized joint access:", joint_samples[:5])


if __name__ == "__main__":
    main()