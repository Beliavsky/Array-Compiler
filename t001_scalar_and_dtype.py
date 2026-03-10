import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=120)

x = np.array([1, 2, 3], dtype=np.int32)
y = x.astype(np.float64) * 0.5
print("x", x, x.dtype)
print("y", y, y.dtype)
