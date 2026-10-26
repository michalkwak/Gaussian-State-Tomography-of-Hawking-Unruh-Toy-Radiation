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


def estimate_squeezing_joint_access(samples: jax.Array, noise_level: float = 0.0,
                                    r_max: float = 4.0, n_grid: int = 4001) -> jax.Array:
    """
    Maximum-likelihood r from the idealized two-wedge samples (n, 4) of
    simulate.sample_idealized_joint_access, under the model
    N(0, V(r) + noise_level^2 I_4). The likelihood depends on the data only
    through S = X^T X / n, and is maximized on a grid over [0, r_max].
    """
    from simulate import generate_covariance_matrix
    S = samples.T @ samples / samples.shape[0]
    rs = jnp.linspace(0.0, r_max, n_grid)
    covs = jax.vmap(generate_covariance_matrix)(rs) + noise_level ** 2 * jnp.eye(4)
    # per-sample negative log-likelihood up to constants: log det Sigma + Tr(Sigma^-1 S)
    nll = jnp.linalg.slogdet(covs)[1] + jnp.trace(jnp.linalg.solve(covs, S), axis1=1, axis2=2)
    return rs[jnp.argmin(nll)]
