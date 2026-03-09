"""IR module container."""

from __future__ import annotations

from dataclasses import dataclass, field

from .nodes import Function, Program, RecordDef


@dataclass
class Module:
    name: str
    records: list[RecordDef] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    program: Program | None = None
    exports: list[str] = field(default_factory=list)
    library_mode: bool = True
    diagnostics: list[str] = field(default_factory=list)
