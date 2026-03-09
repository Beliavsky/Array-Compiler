from array_compiler.backends.fortran import FortranBackend
from array_compiler.ir.module import Module
from array_compiler.ir.nodes import Call, Constant, Function, Return, ScalarType, ValueRef


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
    assert "function f(x) result(result_value)" in source
    assert "result_value = sin(x)" in source
