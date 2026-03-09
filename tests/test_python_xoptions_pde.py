from pathlib import Path
import subprocess
import tempfile

from array_compiler.backends.fortran import FortranBackend
from array_compiler.backends.fortran.helpers import HelperRegistry
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from test_utils import assert_max_fortran_line_length


XPDE_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xoptions_pde.py")


def test_xoptions_pde_lowers_and_emits_fortran() -> None:
    module = PythonNumpyFrontend().lower_source(XPDE_PATH.read_text(), module_name="xpde_mod")
    source = FortranBackend().emit(module)

    assert module.exports == ["solve_tridiagonal", "finite_difference_option_price", "print_example", "run_example"]
    assert "function solve_tridiagonal(lower, diag, upper, rhs) result(result_value)" in source
    assert "real(dp), intent(in) :: diag(:)" in source
    assert "real(dp), allocatable :: result_value(:)" in source
    assert "spots = [( (i * d_spot), i = 0, ((num_asset_steps + 1) - 1), 1 )]" in source
    assert "values = [( max((spots(s_index + 1) - strike), 0.0d0), s_index = 0, size(spots) - 1 )]" in source
    assert "values = [[[left_boundary_current], interior_values], [right_boundary_current]]" in source
    assert "values(i + 1) = max(values(i + 1), intrinsic_value)" in source
    assert "call print_example(0.0d0)" in source
    assert "call print_example(0.08d0)" in source


def test_xoptions_pde_emitted_fortran_compiles() -> None:
    module = PythonNumpyFrontend().lower_source(XPDE_PATH.read_text(), module_name="xpde_mod")
    source = FortranBackend().emit(module)
    kind_mod_path = HelperRegistry().path_for("kind_mod")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        kind_mod_copy = tmp / kind_mod_path.name
        kind_mod_copy.write_text(kind_mod_path.read_text())
        module_file = tmp / "xpde_mod.f90"
        module_file.write_text(source)

        compile_proc = subprocess.run(
            ["gfortran", "-std=f2018", str(kind_mod_copy), str(module_file), "-c"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compile_proc.returncode == 0, compile_proc.stderr


def test_xoptions_pde_emitted_fortran_stays_within_132_columns() -> None:
    module = PythonNumpyFrontend().lower_source(XPDE_PATH.read_text(), module_name="xpde_mod")
    source = FortranBackend().emit(module)
    assert_max_fortran_line_length(source, max_len=132)
