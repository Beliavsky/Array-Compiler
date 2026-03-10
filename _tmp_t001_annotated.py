import numpy as np
from typing import Final
from array_compiler.annotations import Array1D, Array2D, Array3D

np.set_printoptions(precision=6, suppress=True, linewidth=120)

x: Final[Array1D[int]] = np.array([1, 2, 3], dtype=np.int32)
y: Final[Array1D[float]] = x.astype(np.float64) * 0.5
print("x", x, x.dtype)
print("y", y, y.dtype)
