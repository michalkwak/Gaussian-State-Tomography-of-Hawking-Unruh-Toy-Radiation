import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import simulate


def compute_particle_number(V: jax.Array) -> jax.Array:
    return (jnp.trace(V) - 2) / 4

def compute_entropy(V: jax.Array) -> jax.Array:
    nu = jnp.sqrt(jnp.linalg.det(V))
    if nu <= 1:  # guard against reconstruction noise pushing below vacuum
        return 0.0
    a, b = (nu+1)/2, (nu-1)/2
    return a*jnp.log2(a) - b*jnp.log2(b)

def estimate_squeezing(V: jax.Array) -> jax.Array:
    nu = jnp.sqrt(jnp.linalg.det(V))
    return 0.5 * jnp.arccosh(jnp.maximum(nu, 1.0))

def matrix_error(V_true, V_hat) -> float:
    return jnp.linalg.norm(V_true - V_hat, ord='fro')