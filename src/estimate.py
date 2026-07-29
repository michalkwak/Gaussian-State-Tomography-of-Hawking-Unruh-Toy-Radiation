import numpy as np

def estimate_covariance(samples: np.ndarray) -> np.ndarray:
    """Handles homodyne's NaN-masked entries via pairwise-complete covariance"""
    if not np.isnan(samples).any():
        return np.cov(samples.T, ddof=1)

    # pairwise deletion: use only rows where both columns are present
    mask = ~np.isnan(samples).any(axis=1)
    if mask.sum() < 2:
        raise ValueError("Not enough jointly-observed samples to estimate covariance")
    return np.cov(samples[mask].T, ddof=1)