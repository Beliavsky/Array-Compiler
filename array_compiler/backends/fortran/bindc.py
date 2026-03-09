"""Generate small bind(C) wrappers for exported scalar procedures."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...ir.module import Module
from ...ir.nodes import Function, RecordTypeRef, ScalarType
from .formatter import wrap_fortran_source


@dataclass
class BindCArtifact:
    source: str
    exported_wrappers: list[str] = field(default_factory=list)
    skipped_exports: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class FortranBindCGenerator:
    def emit(self, module: Module) -> BindCArtifact:
        functions_by_name = {fn.name: fn for fn in module.functions}
        wrapper_lines: list[str] = [
            f"module {module.name}_bindc",
            "use, intrinsic :: iso_c_binding, only: c_int, c_double",
            f"use {module.name}, only: {', '.join(self._export_names(module))}" if self._export_names(module) else f"use {module.name}",
            "implicit none",
            "private",
        ]
        wrapper_blocks: list[list[str]] = []
        exported_wrappers: list[str] = []
        skipped_exports: list[str] = []
        diagnostics: list[str] = []

        for export_name in self._export_names(module):
            fn = functions_by_name.get(export_name)
            if fn is None:
                diagnostics.append(f"bind(C): export '{export_name}' not found in module functions")
                skipped_exports.append(export_name)
                continue
            wrapper = self._emit_wrapper(fn)
            if wrapper is None:
                diagnostics.append(f"bind(C): export '{export_name}' is not yet C-interoperable")
                skipped_exports.append(export_name)
                continue
            exported_wrappers.append(f"{export_name}_c")
            wrapper_blocks.append(wrapper)

        if exported_wrappers:
            wrapper_lines.append("public :: " + ", ".join(exported_wrappers))
        wrapper_lines.extend(["", "contains", ""])
        for block in wrapper_blocks:
            wrapper_lines.extend(block)
            wrapper_lines.append("")
        wrapper_lines.append(f"end module {module.name}_bindc")
        wrapper_lines.append("")

        return BindCArtifact(
            source=wrap_fortran_source("\n".join(wrapper_lines), max_len=132),
            exported_wrappers=exported_wrappers,
            skipped_exports=skipped_exports,
            diagnostics=diagnostics,
        )

    def _emit_wrapper(self, fn: Function) -> list[str] | None:
        if not all(self._is_c_scalar(arg_type) for _, arg_type in fn.args):
            return None
        if fn.result_type is not None and not self._is_c_scalar(fn.result_type):
            return None

        wrapper_name = f"{fn.name}_c"
        arg_text = ", ".join(name for name, _ in fn.args)
        call_text = f"{fn.name}({arg_text})"

        if fn.result_type is None:
            lines = [f"subroutine {wrapper_name}({arg_text}) bind(C, name='{wrapper_name}')", "implicit none"]
            for name, arg_type in fn.args:
                lines.append(f"{self._c_type_name(arg_type)}, value :: {name}")
            lines.append(f"call {call_text}")
            lines.append(f"end subroutine {wrapper_name}")
            return lines

        lines = [f"function {wrapper_name}({arg_text}) result(result_value) bind(C, name='{wrapper_name}')", "implicit none"]
        for name, arg_type in fn.args:
            lines.append(f"{self._c_type_name(arg_type)}, value :: {name}")
        lines.append(f"{self._c_type_name(fn.result_type)} :: result_value")
        lines.append(f"result_value = {call_text}")
        lines.append(f"end function {wrapper_name}")
        return lines

    def _is_c_scalar(self, value_type: object) -> bool:
        if isinstance(value_type, RecordTypeRef):
            return False
        return value_type in {ScalarType.INTEGER, ScalarType.REAL64}

    def _c_type_name(self, value_type: object) -> str:
        mapping = {
            ScalarType.INTEGER: "integer(c_int)",
            ScalarType.REAL64: "real(c_double)",
        }
        return mapping[value_type]

    def _export_names(self, module: Module) -> list[str]:
        if module.exports:
            return list(dict.fromkeys(module.exports))
        return [fn.name for fn in module.functions]
