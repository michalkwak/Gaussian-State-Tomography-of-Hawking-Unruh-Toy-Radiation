import numpy as np
import matplotlib.pyplot as plt
import simulate


def compute_particle_number(V: np.ndarray) -> float:
    return (np.trace(V) - 2) / 4

def compute_entropy(V: np.ndarray) -> float:
    nu = np.sqrt(np.linalg.det(V))
    if nu <= 1:  # guard against reconstruction noise pushing below vacuum
        return 0.0
    a, b = (nu+1)/2, (nu-1)/2
    return a*np.log2(a) - b*np.log2(b)

def estimate_squeezing(V: np.ndarray) -> float:
    nu = np.sqrt(np.linalg.det(V))
    return 0.5 * np.arccosh(max(nu, 1.0))

def matrix_error(V_true, V_hat) -> float:
    return np.linalg.norm(V_true - V_hat, ord='fro')