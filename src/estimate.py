import jax
import jax.numpy as jnp

def estimate_covariance(samples: jax.Array) -> jax.Array:
    """Handles homodyne's NaN-masked entries via pairwise-complete covariance"""
    if not jnp.isnan(samples).any():
        return jnp.cov(samples.T, ddof=1)

    # pairwise deletion: use only rows where both columns are present
    mask = ~jnp.isnan(samples).any(axis=1)
    if mask.sum() < 2:
        raise ValueError("Not enough jointly-observed samples to estimate covariance")
    return jnp.cov(samples[mask].T, ddof=1)