from pathlib import Path
import os
import subprocess
import tempfile

from array_compiler.backends.fortran import FortranBackend
from array_compiler.backends.fortran.helpers import HelperRegistry
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from test_utils import assert_max_fortran_line_length, compiled_fortran_module_with_driver, run_command


XAMERICAN_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xamerican_options.py")


def test_xamerican_options_lowers_and_emits_fortran() -> None:
    module = PythonNumpyFrontend().lower_source(XAMERICAN_PATH.read_text(), module_name="xamerican_mod")
    source = FortranBackend().emit(module)

    assert module.exports == ["binomial_tree_option_price", "print_example", "run_example"]
    assert "real(dp), allocatable :: values(:)" in source
    assert "values = [real(dp) :: ]" in source
    assert "values = [values, max(terminal_price - strike, 0.0d0)]" in source
    assert "do step = num_steps - 1, merge((-1 - 1), (-1 + 1), (-1 > 0)), -1" in source
    assert "continuation_value = discount * (risk_neutral_prob * values(node + 1) &" in source
    assert "& + (1.0d0 - risk_neutral_prob) * values(node + 2))" in source
    assert "call print_example(0.0d0)" in source
    assert "print *" in source
    assert "call print_example(0.08d0)" in source


def test_xamerican_options_emitted_fortran_compiles() -> None:
    module = PythonNumpyFrontend().lower_source(XAMERICAN_PATH.read_text(), module_name="xamerican_mod")
    source = FortranBackend().emit(module)
    kind_mod_path = HelperRegistry().path_for("kind_mod")
    ac_string_path = HelperRegistry().path_for("ac_string")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        kind_mod_copy = tmp / kind_mod_path.name
        kind_mod_copy.write_text(kind_mod_path.read_text())
        ac_string_copy = tmp / ac_string_path.name
        ac_string_copy.write_text(ac_string_path.read_text())
        module_file = tmp / "xamerican_mod.f90"
        module_file.write_text(source)

        compile_proc = subprocess.run(
            ["gfortran", "-std=f2018", str(kind_mod_copy), str(ac_string_copy), str(module_file), "-c"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compile_proc.returncode == 0, compile_proc.stderr


def test_xamerican_options_emitted_fortran_stays_within_80_columns() -> None:
    module = PythonNumpyFrontend().lower_source(XAMERICAN_PATH.read_text(), module_name="xamerican_mod")
    source = FortranBackend().emit(module)
    assert_max_fortran_line_length(source, max_len=80)


def test_xamerican_options_python_and_generated_fortran_outputs_agree() -> None:
    source_text = XAMERICAN_PATH.read_text().replace("num_steps = 10000", "num_steps = 1000", 1)
    module = PythonNumpyFrontend().lower_source(source_text, module_name="xamerican_mod")
    source = FortranBackend().emit(module)
    with compiled_fortran_module_with_driver(
        source,
        module_name="xamerican_mod",
        driver_source="\n".join(
            [
                "program xamerican_driver",
                "use xamerican_mod, only: run_example",
                "implicit none",
                "call run_example()",
                "end program xamerican_driver",
                "",
            ]
        ),
    ) as exe_file:
        python_file = _write_temp_python_script(source_text, prefix="xamerican_options_")
        try:
            python_blocks = _parse_labeled_blocks(run_command(["python", str(python_file)]))
            fortran_blocks = _parse_labeled_blocks(run_command([str(exe_file)]))
        finally:
            python_file.unlink(missing_ok=True)

        assert len(python_blocks) == len(fortran_blocks) == 2
        for python_block, fortran_block in zip(python_blocks, fortran_blocks, strict=True):
            assert python_block["status_1"] == fortran_block["status_1"] == "American prices are at least as large as European prices"
            assert python_block["status_2"] == fortran_block["status_2"] == "European parity equality and American parity bounds passed"
            for key in (
                "dividend_yield",
                "num_steps",
                "european call",
                "american call",
                "call premium",
                "european put",
                "american put",
                "put premium",
                "eu parity lhs",
                "eu parity rhs",
                "eu abs error",
                "am parity val",
                "am parity low",
                "am parity high",
            ):
                assert abs(python_block[key] - fortran_block[key]) <= 5.0e-8, key


def _parse_labeled_blocks(stdout: str) -> list[dict[str, float | str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    parsed_blocks: list[dict[str, float | str]] = []
    for block in blocks:
        parsed: dict[str, float | str] = {}
        status_index = 1
        for line in block:
            if ":" in line:
                label, value = line.split(":", 1)
                try:
                    parsed[label.strip()] = float(value.strip())
                    continue
                except ValueError:
                    pass
            parsed[f"status_{status_index}"] = line
            status_index += 1
        parsed_blocks.append(parsed)
    return parsed_blocks


def _write_temp_python_script(source_text: str, prefix: str) -> Path:
    fd, script_name = tempfile.mkstemp(suffix=".py", prefix=prefix)
    os.close(fd)
    script_path = Path(script_name)
    script_path.write_text(source_text)
    return script_path
