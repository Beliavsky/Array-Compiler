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
    external_path: Path | None = None


class HelperRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2] / "runtime" / "fortran"
        external_root = Path(r"c:\python\fortran")
        self._helpers: dict[str, HelperModule] = {
            "kind_mod": HelperModule("kind_mod", "kind.f90", "Shared kind parameters for generated Fortran"),
            "ac_constants": HelperModule("ac_constants", "ac_constants.f90", "Named mathematical constants for generated Fortran"),
            "ac_stats": HelperModule("ac_stats", "ac_stats.f90", "Statistical runtime helpers for generated Fortran"),
            "ac_string": HelperModule("ac_string", "ac_string.f90", "Small string helper runtime for generated Fortran"),
            "ac_random": HelperModule("ac_random", "ac_random.f90", "Small RNG helper runtime for translated scalar Monte Carlo code"),
            "ac_numpy": HelperModule("ac_numpy", "ac_numpy.f90", "Small NumPy-style runtime helpers for generated Fortran"),
            "ac_lapack": HelperModule("ac_lapack", "ac_lapack.f90", "LAPACK-backed linear algebra helpers for generated Fortran"),
            "lapack_d": HelperModule(
                "lapack_d",
                "lapack_d.f90",
                "LAPACK and dense linear algebra helpers",
                external_path=external_root / "lapack_d.f90",
            ),
            "python": HelperModule(
                "python",
                "python.f90",
                "Python/NumPy compatibility helpers",
                external_path=external_root / "python.f90",
            ),
            "octave_funcs": HelperModule(
                "octave_funcs",
                "octave_funcs.f90",
                "Octave/Matlab compatibility helpers",
                external_path=external_root / "octave_funcs.f90",
            ),
            "r": HelperModule(
                "r",
                "r.f90",
                "R compatibility helpers",
                external_path=external_root / "r.f90",
            ),
        }

    def get(self, key: str) -> HelperModule:
        return self._helpers[key]

    def path_for(self, key: str) -> Path:
        helper = self.get(key)
        local_path = self.root / helper.filename
        if local_path.exists():
            return local_path
        if helper.external_path is not None and helper.external_path.exists():
            return helper.external_path
        return local_path
