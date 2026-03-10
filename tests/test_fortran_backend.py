from array_compiler.backends.fortran import FortranBackend
from array_compiler.compiler import Compiler
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from array_compiler.ir.module import Module
from array_compiler.ir.nodes import Call, Compare, CompareOperator, Function, If, Raise, Return, ScalarType, ValueRef
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


def test_emitter_uses_module_level_implicit_none_one_line_if_and_pure_attrs() -> None:
    module = Module(
        name="style_mod",
        functions=[
            Function(
                name="normal_cdf",
                args=[("x", ScalarType.REAL64)],
                result_type=ScalarType.REAL64,
                body=[Return(Call("sin", (ValueRef("x"),)))],
            ),
            Function(
                name="guard",
                args=[("x", ScalarType.REAL64)],
                result_type=None,
                body=[If(Compare(ValueRef("x"), CompareOperator.LE, ValueRef("x")), (Raise("bad"),), ())],
            ),
        ],
    )
    source = FortranBackend().emit(module)
    assert source.count("implicit none") == 1
    assert "pure elemental function normal_cdf(x) result(result_value)" in source
    assert 'if (x <= x) error stop "bad"' in source


def test_python_comments_are_carried_into_emitted_fortran() -> None:
    source = """
def f(x: float) -> float:
    # comment after signature
    # another one
    # before assignment
    y = x
    # before return
    return y
""".strip()

    module = PythonNumpyFrontend().lower_source(source, module_name="comment_mod")
    emitted = FortranBackend().emit(module)

    block = emitted[emitted.index("function f"):emitted.index("end function f")]
    assert "! comment after signature" in block
    assert "! another one" in block
    assert block.index("! comment after signature") < block.index("real(dp), intent(in) :: x")
    assert "! before assignment" in block
    assert "! before return" in block
