import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy


def symplectic_eigenvalue(V: jax.Array) -> jax.Array:
    # clamp: noise subtraction can push a reconstructed det slightly below 0
    return jnp.sqrt(jnp.maximum(jnp.linalg.det(V), 0.0))

def compute_particle_number(V: jax.Array) -> jax.Array:
    return (jnp.trace(V) - 2) / 4

def compute_entropy(V: jax.Array) -> jax.Array:
    # clamp at vacuum: reconstruction noise can push nu below 1
    nu = jnp.maximum(symplectic_eigenvalue(V), 1.0)
    a, b = (nu+1)/2, (nu-1)/2
    # xlogy(0, 0) = 0, so nu = 1 gives S = 0 without branching (jit/vmap safe)
    return (xlogy(a, a) - xlogy(b, b)) / jnp.log(2.0)

def estimate_squeezing(V: jax.Array) -> jax.Array:
    nu = symplectic_eigenvalue(V)
    return 0.5 * jnp.arccosh(jnp.maximum(nu, 1.0))

def squeezing_crb(r, n_samples, noise_level, measurement_type):
    """
    Cramer-Rao lower bound on std(r_hat) from n_samples shots of the
    accessible mode. Each observed quadrature has variance
    s(r) = cosh(2r) + c, with c = noise_level^2 (+1 for heterodyne), and
    each shot observes both quadratures, giving Fisher information per shot
    4 sinh^2(2r) / s^2.
    """
    c = noise_level ** 2 + (1.0 if measurement_type == "heterodyne" else 0.0)
    s = jnp.cosh(2 * r) + c
    return s / (2 * jnp.sinh(2 * r) * jnp.sqrt(n_samples))

def gaussian_fisher_information(cov_fn, r):
    """
    Fisher information about r carried by one draw from N(0, cov_fn(r)):
    I(r) = 1/2 Tr[(Sigma^-1 dSigma/dr)^2]. Works for any dimension, so it
    covers both the one-wedge (2x2) and the two-wedge (4x4) cases.
    """
    A = jnp.linalg.solve(cov_fn(r), jax.jacfwd(cov_fn)(r))
    return 0.5 * jnp.trace(A @ A)

def squeezing_crb_joint_access(r, n_samples, noise_level):
    """
    Cramer-Rao lower bound on std(r_hat) for the idealized two-wedge
    comparison: n_samples draws from V(r) + noise_level^2 I_4. Unlike the
    one-wedge bound, the Fisher information stays finite as r -> 0, because
    the A-B correlations (sinh 2r) are linear in r there.
    """
    from simulate import generate_covariance_matrix
    cov_fn = lambda r: generate_covariance_matrix(r) + noise_level ** 2 * jnp.eye(4)
    return 1.0 / jnp.sqrt(n_samples * gaussian_fisher_information(cov_fn, r))

def matrix_error(V_true, V_hat) -> float:
    return jnp.linalg.norm(V_true - V_hat, ord='fro')
