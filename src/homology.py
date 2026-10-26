import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import ripser
import teaspoon.TDA.Draw as Draw

import simulate

# homology.py
#
# Persistent-homology analysis of the synthetic measurement point cloud
# produced by simulate.sample_measurements(). Consumes the same (n_samples, 2)
# arrays (or (n_samples, 4) for the idealized joint-access comparison,
# see simulate.sample_idealized_joint_access) as the rest of the pipeline --
# no separate data generation logic lives here.


def compute_persistence_diagrams(samples: jax.Array, maxdim: int = 1) -> list:
    """
    Computes Vietoris-Rips persistence diagrams directly from a point cloud.

    ripser computes pairwise Euclidean distances internally, so no explicit
    distance matrix needs to be constructed here -- passing a covariance
    matrix (as in the original script) is NOT a valid substitute: a
    covariance matrix has none of the properties required of a distance
    matrix (symmetric zero diagonal, non-negativity, triangle inequality).

    maxdim=1 (H0, H1) is fast even for n_samples in the thousands. Going to
    higher maxdim (per the H0-H2 study described in the paper)
    is combinatorially more expensive -- test on a small n_samples (50-100)
    before scaling up.

    Returns the list of persistence diagrams, diagrams[0] = H0, diagrams[1] = H1, etc.
    """
    return ripser.ripser(jax.device_get(samples), maxdim=maxdim)["dgms"]


def plot_persistence(samples: jax.Array, diagrams: list, R: float = 2.0):
    """Plots the point cloud alongside its H0 and H1 persistence diagrams."""
    samples = jax.device_get(samples)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    plt.sca(axes[0])
    plt.title("Point Cloud")
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)

    plt.sca(axes[1])
    plt.title("0-dim Diagram")
    Draw.drawDgm(diagrams[0])

    plt.sca(axes[2])
    plt.title("1-dim Diagram")
    Draw.drawDgm(diagrams[1])
    plt.axis([0, R, 0, R])

    plt.show()


def main():
    # Example usage, mirroring simulate.py's main() so this can be run
    # standalone or dropped straight into the shared pipeline.
    r = 0.5
    V = simulate.generate_covariance_matrix(r)
    V_A = simulate.reduce_to_accessible(V)

    n_samples = 1000
    noise_level = 0.1
    measurement_type = 'heterodyne'  # 'joint' (ideal benchmark) | 'heterodyne'
    key = random.key(42)

    samples = simulate.sample_measurements(V_A, n_samples, noise_level, measurement_type, key)
    diagrams = compute_persistence_diagrams(samples, maxdim=1)
    plot_persistence(samples, diagrams)


if __name__ == "__main__":
    main()