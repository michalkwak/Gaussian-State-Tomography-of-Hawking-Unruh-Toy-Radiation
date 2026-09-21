import numpy as np
import matplotlib.pyplot as plt
import ripser
import teaspoon.TDA.Draw as Draw

import simulate

# topology.py
#
# Persistent-homology analysis of the synthetic measurement point cloud
# produced by simulate.sample_measurements(). Consumes the same (n_samples, 2)
# arrays (or (n_samples, 4) for the idealized joint-access comparison,
# see simulate.sample_idealized_joint_access) as the rest of the pipeline --
# no separate data generation logic lives here.


def compute_persistence_diagrams(samples: np.ndarray, maxdim: int = 1) -> list:
    """
    Computes Vietoris-Rips persistence diagrams directly from a point cloud.

    ripser computes pairwise Euclidean distances internally, so no explicit
    distance matrix needs to be constructed here -- passing a covariance
    matrix (as in the original script) is NOT a valid substitute: a
    covariance matrix has none of the properties required of a distance
    matrix (symmetric zero diagonal, non-negativity, triangle inequality).

    homodyne data contains NaN entries for the unmeasured quadrature each
    shot (see simulate.sample_measurements); rows with any NaN are dropped
    here, since ripser cannot handle missing coordinates.

    NOTE: for pure homodyne data, every shot has exactly one missing
    quadrature (never both), so dropping rows with any NaN discards ALL
    rows, not "roughly half" -- there is no row where both coordinates are
    jointly observed. This function will raise rather than silently return
    an empty/broken result. Running persistent homology on homodyne data
    therefore requires an actual imputation or reconstruction step first
    (e.g. feeding it estimate.py's reconstructed V_hat and resampling, or
    an imputation method the DS/ML person settles on) -- this function
    does not attempt that, it only handles the case where NaNs are sparse
    (e.g. a mix of measurement types, or occasional dropped shots).

    maxdim=1 (H0, H1) is fast even for n_samples in the thousands. Going to
    higher maxdim (per the H0-H4 sweep mentioned in the project abstract)
    is combinatorially more expensive -- test on a small n_samples (50-100)
    before scaling up.

    Returns the list of persistence diagrams, diagrams[0] = H0, diagrams[1] = H1, etc.
    """
    if np.isnan(samples).any():
        mask = ~np.isnan(samples).any(axis=1)
        n_dropped = (~mask).sum()
        samples = samples[mask]
        if len(samples) == 0:
            raise ValueError(
                "compute_persistence_diagrams: every row has a missing "
                "quadrature -- this looks like pure homodyne data, where no "
                "shot has both quadratures jointly observed. Row-dropping "
                "cannot recover a usable point cloud here; homodyne data "
                "needs imputation or reconstruction (e.g. via estimate.py) "
                "before persistent homology can be run on it."
            )
        print(f"compute_persistence_diagrams: dropping {n_dropped}/{n_dropped + len(samples)} "
              f"rows with missing quadratures")

    return ripser.ripser(samples, maxdim=maxdim)["dgms"]


def plot_persistence(samples: np.ndarray, diagrams: list, R: float = 2.0):
    """Plots the point cloud alongside its H0 and H1 persistence diagrams."""
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
    measurement_type = 'heterodyne'  # 'joint' | 'homodyne' | 'heterodyne'
    rng = np.random.default_rng(seed=42)

    samples = simulate.sample_measurements(V_A, n_samples, noise_level, measurement_type, rng)
    diagrams = compute_persistence_diagrams(samples, maxdim=1)
    plot_persistence(samples, diagrams)


if __name__ == "__main__":
    main()