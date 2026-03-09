from pathlib import Path
import subprocess
import tempfile

from array_compiler.backends.fortran import FortranBackend
from array_compiler.backends.fortran.helpers import HelperRegistry
from array_compiler.frontends.python_numpy import PythonNumpyFrontend


XAMERICAN_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xamerican_options.py")


def test_xamerican_options_lowers_and_emits_fortran() -> None:
    module = PythonNumpyFrontend().lower_source(XAMERICAN_PATH.read_text(), module_name="xamerican_mod")
    source = FortranBackend().emit(module)

    assert module.exports == ["binomial_tree_option_price", "print_example", "run_example"]
    assert "real(dp) , allocatable :: values(:)" in source
    assert "values = [real(dp) :: ]" in source
    assert "values = [values, max((terminal_price - strike), 0.0d0)]" in source
    assert "do step = (num_steps - 1), (-1), (-1)" in source
    assert "continuation_value = (discount * ((risk_neutral_prob * values(node + 1)) + ((1.0d0 - risk_neutral_prob) * values((node + 1) + 1))))" in source
    assert "call print_example(0.0d0)" in source
    assert "print *" in source
    assert "call print_example(0.08d0)" in source


def test_xamerican_options_emitted_fortran_compiles() -> None:
    module = PythonNumpyFrontend().lower_source(XAMERICAN_PATH.read_text(), module_name="xamerican_mod")
    source = FortranBackend().emit(module)
    kind_mod_path = HelperRegistry().path_for("kind_mod")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        kind_mod_copy = tmp / kind_mod_path.name
        kind_mod_copy.write_text(kind_mod_path.read_text())
        module_file = tmp / "xamerican_mod.f90"
        module_file.write_text(source)

        compile_proc = subprocess.run(
            ["gfortran", str(kind_mod_copy), str(module_file), "-c"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compile_proc.returncode == 0, compile_proc.stderr
