import numpy as np
from typing import Final
from array_compiler.annotations import Array1D, Array2D, Array3D

def fit_arma_initial_numpy(x: Array1D[float], p, q, k, mode: str = "both", pacf_weight: float = 0.5, npsi: int | None = None, seed: int | None = None) -> Array1D[float]:
    x: Final[Array1D[float]] = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    if npsi is None:
        npsi: Final[int] = max(10, k + 1)
    return x
