import numpy as np

ESTIMATORS = {}
def register_estimator(name):
    def deco(fn):
        ESTIMATORS[name] = fn
        return fn
    return deco

@register_estimator("sample_cov")
def estimate_covariance(samples: np.ndarray) -> np.ndarray:
    return np.cov(samples.T, ddof=1)