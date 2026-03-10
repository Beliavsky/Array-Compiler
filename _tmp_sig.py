import numpy as np

def fit_arma_initial_numpy(x, p, q, k,
                           mode="both",
                           pacf_weight=0.5,
                           npsi=None,
                           seed=None):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    if npsi is None:
        npsi = max(10, k + 1)
    return x
