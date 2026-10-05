import jax
import jax.numpy as jnp


def estimate_covariance(samples: jax.Array, measurement_type: str = "joint",
                        noise_level: float = 0.0) -> jax.Array:
    """
    Estimates the accessible-mode covariance V_A from measured samples
    produced by simulate.sample_measurements.

    - homodyne: x and p are never observed in the same shot (the other is
      NaN), so Cov(x, p) is unmeasurable. Var(x) and Var(p) are estimated
      from their own sub-samples and the cross-covariance is ASSUMED zero
      (true for the thermal state family V_A = cosh(2r) I_2).
    - heterodyne: the known +I_2 vacuum penalty (Sigma_het = V_A + I_2) is
      subtracted.
    - the known detector-noise variance noise_level^2 * I_2 is subtracted in
      all schemes; pass noise_level=0 to get the raw estimate.
    """
    if measurement_type == "homodyne":
        n_x = int((~jnp.isnan(samples[:, 0])).sum())
        n_p = int((~jnp.isnan(samples[:, 1])).sum())
        if min(n_x, n_p) < 2:
            raise ValueError("Need at least 2 shots of each quadrature to estimate homodyne variances")
        vx = jnp.nanvar(samples[:, 0], ddof=1)
        vp = jnp.nanvar(samples[:, 1], ddof=1)
        V = jnp.diag(jnp.array([vx, vp]))
    elif measurement_type in ("joint", "heterodyne"):
        if jnp.isnan(samples).any():
            raise ValueError(f"{measurement_type} samples should not contain missing quadratures")
        V = jnp.cov(samples.T, ddof=1)
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")

    if measurement_type == "heterodyne":
        V = V - jnp.eye(2)
    return V - noise_level ** 2 * jnp.eye(2)
