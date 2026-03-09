"""Top-level compiler orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .backends.fortran.emitter import FortranBackend
from .ir.module import Module


@dataclass
class Compiler:
    """Compiler entrypoint around a shared semantic IR."""

    fortran_backend: FortranBackend | None = None

    def emit_fortran(self, module: Module) -> str:
        backend = self.fortran_backend or FortranBackend()
        return backend.emit(module)
