from pathlib import Path
import subprocess
import tempfile
import textwrap

from array_compiler.backends.fortran import FortranBackend
from array_compiler.backends.fortran.helpers import HelperRegistry
from array_compiler.compiler import Compiler
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from array_compiler.ir.module import Module
from array_compiler.ir.nodes import Call, Function, Return, ScalarType, ValueRef


XBS_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xbs.py")


def test_bindc_wrapper_generation_for_scalar_function() -> None:
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
        exports=["f"],
        library_mode=True,
    )

    artifact = Compiler().emit_fortran_bindc(module)
    assert artifact.exported_wrappers == ["f_c"]
    assert artifact.skipped_exports == []
    assert artifact.diagnostics == []
    assert "module demo_mod_bindc" in artifact.source
    assert "use, intrinsic :: iso_c_binding, only: c_int, c_double" in artifact.source
    assert "function f_c(x) result(result_value) bind(C, name='f_c')" in artifact.source
    assert "real(c_double), value :: x" in artifact.source


def test_bindc_reports_partial_translation_for_xbs() -> None:
    module = PythonNumpyFrontend().lower_source(XBS_PATH.read_text(), module_name="xbs_mod")
    artifact = Compiler().emit_fortran_bindc(module)

    assert "normal_cdf_c" in artifact.exported_wrappers
    assert "run_example_c" in artifact.exported_wrappers
    assert "black_scholes_price" in artifact.skipped_exports
    assert any("black_scholes_price" in diagnostic for diagnostic in artifact.diagnostics)
    assert any("black_scholes_price" in diagnostic for diagnostic in module.diagnostics)


def test_bindc_scalar_wrapper_compiles_and_runs() -> None:
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
        exports=["f"],
        library_mode=True,
    )

    compiler = Compiler()
    source = FortranBackend().emit(module)
    bindc = compiler.emit_fortran_bindc(module).source
    kind_mod_path = HelperRegistry().path_for("kind_mod")
    driver = textwrap.dedent(
        """
        program test_driver
        use demo_mod_bindc, only: f_c
        implicit none
        print *, f_c(1.0d0)
        end program test_driver
        """
    ).strip() + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        kind_mod_file = tmp / kind_mod_path.name
        kind_mod_file.write_text(kind_mod_path.read_text())
        module_file = tmp / "demo_mod.f90"
        module_file.write_text(source)
        bindc_file = tmp / "demo_mod_bindc.f90"
        bindc_file.write_text(bindc)
        driver_file = tmp / "test_driver.f90"
        driver_file.write_text(driver)
        exe_file = tmp / "test_driver.exe"

        compile_proc = subprocess.run(
            ["gfortran", "-std=f2018", str(kind_mod_file), str(module_file), str(bindc_file), str(driver_file), "-o", str(exe_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compile_proc.returncode == 0, compile_proc.stderr

        run_proc = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
        assert run_proc.returncode == 0, run_proc.stderr
        assert run_proc.stdout.strip() != ""
