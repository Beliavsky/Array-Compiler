from array_compiler.backends.fortran import FortranBackend
from array_compiler.compiler import Compiler
from array_compiler.ir.module import Module
from array_compiler.ir.nodes import Call, Constant, Function, Return, ScalarType, ValueRef
from test_utils import assert_max_fortran_line_length


def test_simple_function_emission() -> None:
    module = Module(
        name="demo_mod",
        functions=[
            Function(
                name="f",
                args=[("x", ScalarType.REAL64)],
                result_type=ScalarType.REAL64,
                body=[Return(Call("sin", (ValueRef("x"),)))],
            )
        ],
    )

    source = FortranBackend().emit(module)
    assert "module demo_mod" in source
    assert "use kind_mod, only: dp" in source
    assert "private" in source
    assert "public :: f" in source
    assert "function f(x) result(result_value)" in source
    assert "result_value = sin(x)" in source


def test_library_exports_can_be_set_explicitly() -> None:
    module = Module(
        name="lib_mod",
        functions=[
            Function(name="f", args=[("x", ScalarType.REAL64)], result_type=ScalarType.REAL64, body=[Return(ValueRef("x"))]),
            Function(name="g", args=[("x", ScalarType.REAL64)], result_type=ScalarType.REAL64, body=[Return(ValueRef("x"))]),
        ],
        exports=["f"],
        library_mode=True,
    )

    source = Compiler().emit_fortran_library(module)
    assert "public :: f" in source
    assert "public :: g" not in source
