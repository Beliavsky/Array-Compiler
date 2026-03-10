"""Top-level compiler orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .backends.fortran.bindc import BindCArtifact, FortranBindCGenerator
from .backends.fortran.emitter import FortranBackend
from .fortran_style import apply_fortran_style
from .ir.module import Module


@dataclass
class Compiler:
    """Compiler entrypoint around a shared semantic IR."""

    fortran_backend: FortranBackend | None = None
    fortran_bindc_generator: FortranBindCGenerator | None = None

    def emit_fortran(self, module: Module, *, style_level: str = "full") -> str:
        backend = self.fortran_backend or FortranBackend()
        return apply_fortran_style(backend.emit_raw(module), style_level=style_level)

    def emit_fortran_library(self, module: Module, exports: list[str] | None = None, *, style_level: str = "full") -> str:
        backend = self.fortran_backend or FortranBackend()
        library_module = Module(
            name=module.name,
            records=list(module.records),
            functions=list(module.functions),
            program=module.program,
            exports=list(exports) if exports is not None else list(module.exports),
            library_mode=True,
            diagnostics=list(module.diagnostics),
        )
        return apply_fortran_style(backend.emit_raw(library_module), style_level=style_level)

    def emit_fortran_bindc(self, module: Module) -> BindCArtifact:
        generator = self.fortran_bindc_generator or FortranBindCGenerator()
        artifact = generator.emit(module)
        if artifact.diagnostics:
            module.diagnostics.extend(artifact.diagnostics)
        return artifact
