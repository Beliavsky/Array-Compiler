"""Fortran helper module registry.

The initial registry is intentionally metadata-only. The actual helper source
files can be copied in from Pure-Fortran as the backend starts using them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HelperModule:
    key: str
    filename: str
    description: str


class HelperRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2] / "runtime" / "fortran"
        self._helpers: dict[str, HelperModule] = {
            "lapack_d": HelperModule("lapack_d", "lapack_d.f90", "LAPACK and dense linear algebra helpers"),
            "python": HelperModule("python", "python.f90", "Python/NumPy compatibility helpers"),
            "octave_funcs": HelperModule("octave_funcs", "octave_funcs.f90", "Octave/Matlab compatibility helpers"),
            "r": HelperModule("r", "r.f90", "R compatibility helpers"),
        }

    def get(self, key: str) -> HelperModule:
        return self._helpers[key]

    def path_for(self, key: str) -> Path:
        helper = self.get(key)
        return self.root / helper.filename
