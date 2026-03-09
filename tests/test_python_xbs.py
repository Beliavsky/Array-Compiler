from pathlib import Path

from array_compiler.backends.fortran import FortranBackend
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from test_utils import compiled_fortran_module_with_driver, run_command


XBS_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xbs.py")


def test_xbs_lowers_and_emits_fortran() -> None:
    module = PythonNumpyFrontend().lower_source(XBS_PATH.read_text(), module_name="xbs_mod")
    source = FortranBackend().emit(module)

    assert module.library_mode is True
    assert module.exports == ["normal_cdf", "black_scholes_price", "run_example"]
    assert "module xbs_mod" in source
    assert "use kind_mod, only: dp" in source
    assert "private" in source
    assert "public :: normal_cdf, black_scholes_price, run_example" in source
    assert "function normal_cdf(x) result(result_value)" in source
    assert "function black_scholes_price(spot, strike, rate, volatility, time_to_maturity, option_type) result(result_value)" in source
    assert "subroutine run_example()" in source
    assert "subroutine run_main()" in source
    assert "option_kind = ac_lower(trim(adjustl(option_type)))" in source
    assert "if (((option_kind /= 'call') .and. (option_kind /= 'put'))) then" in source
    assert "error stop 'option_type must be ''call'' or ''put'''" in source
    assert "call run_example()" in source


def test_python_frontend_accepts_utf8_bom_prefixed_source() -> None:
    source = "\ufeffdef f(x: float) -> float:\n    return x\n"
    module = PythonNumpyFrontend().lower_source(source, module_name="bom_mod")
    assert [fn.name for fn in module.functions] == ["f"]


def test_xbs_python_and_generated_fortran_outputs_agree() -> None:
    module = PythonNumpyFrontend().lower_source(XBS_PATH.read_text(), module_name="xbs_mod")
    source = FortranBackend().emit(module)
    with compiled_fortran_module_with_driver(
        source,
        module_name="xbs_mod",
        driver_source="\n".join(
            [
                "program xbs_driver",
                "use xbs_mod, only: run_example",
                "implicit none",
                "call run_example()",
                "end program xbs_driver",
                "",
            ]
        ),
    ) as exe_file:
        python_result = _parse_xbs_output(run_command(["python", str(XBS_PATH)]))
        fortran_result = _parse_xbs_output(run_command([str(exe_file)]))

        assert python_result["status"] == fortran_result["status"] == "put-call parity check passed"
        for key in ("call price", "put price", "parity lhs", "parity rhs", "abs error"):
            assert abs(python_result[key] - fortran_result[key]) <= 5.0e-10, key


def _parse_xbs_output(stdout: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            label, value = line.split(":", 1)
            try:
                result[label.strip()] = float(value.strip())
                continue
            except ValueError:
                pass
        result["status"] = line
    return result
