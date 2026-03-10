from __future__ import annotations

from typing import Literal

from ..backends.fortran.formatter import wrap_fortran_source

StyleLevel = Literal["none", "safe", "full"]
STYLE_LEVELS: tuple[StyleLevel, ...] = ("none", "safe", "full")


def apply_fortran_style(source: str, *, style_level: StyleLevel = "full") -> str:
    if style_level == "none":
        return source
    if style_level in {"safe", "full"}:
        # Initial separation step: line wrapping is the first pass moved out of
        # the backend. Additional structural style passes can be layered onto
        # "full" later without changing the CLI contract.
        return wrap_fortran_source(source, max_len=80)
    raise ValueError(f"unsupported Fortran style level: {style_level}")
