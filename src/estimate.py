import jax
import jax.numpy as jnp


def estimate_covariance(samples: jax.Array, measurement_type: str = "joint",
                        noise_level: float = 0.0) -> jax.Array:
    """
    Estimates the accessible-mode covariance V_A from measured samples
    produced by simulate.sample_measurements.

    - heterodyne: the known +I_2 vacuum penalty (Sigma_het = V_A + I_2) is
      subtracted.
    - the known detector-noise variance noise_level^2 * I_2 is subtracted in
      all schemes; pass noise_level=0 to get the raw estimate.
    """
    if measurement_type not in ("joint", "heterodyne"):
        raise ValueError(f"Unknown measurement type: {measurement_type}")
    V = jnp.cov(samples.T, ddof=1)

    if measurement_type == "heterodyne":
        V = V - jnp.eye(2)
    return V - noise_level ** 2 * jnp.eye(2)
