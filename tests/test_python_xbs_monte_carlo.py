from pathlib import Path
import os
import subprocess
import tempfile
import textwrap

from array_compiler.backends.fortran import FortranBackend
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from array_compiler.backends.fortran.helpers import HelperRegistry
from test_utils import compiled_fortran_module_with_driver, run_command


XBS_MC_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xbs_monte_carlo.py")


def test_xbs_monte_carlo_lowers_and_emits_fortran() -> None:
    module = PythonNumpyFrontend().lower_source(XBS_MC_PATH.read_text(), module_name="xbs_mc_mod")
    source = FortranBackend().emit(module)

    record_names = {record.name for record in module.records}
    assert "mean_standard_error_and_ci_result" in record_names
    assert "monte_carlo_option_prices_result" in record_names
    assert "ac_random_state" not in record_names

    assert "use kind_mod, only: dp" in source
    assert "use ac_random_support, only: ac_random_state, ac_random_init, ac_gauss" in source
    assert "type :: monte_carlo_option_prices_result" in source
    assert "function mean_standard_error_and_ci(payoff_sum, payoff_sum_sq, num_paths, &" in source
    assert "& discount_factor) result(result_value)" in source
    assert "function monte_carlo_option_prices(spot, strike, rate, volatility, &" in source
    assert "& time_to_maturity, num_paths, seed) result(result_value)" in source
    assert "rng = ac_random_init(seed)" in source
    assert "z = ac_gauss(rng, 0.0d0, 1.0d0)" in source
    assert "do ac_loop_index = 0, (num_paths - 1), 1" in source
    assert "call_price = mean_standard_error_and_ci_value%item1" in source
    assert "result_value = monte_carlo_option_prices_result(call_price=call_price" in source
    assert "call_price = results%call_price" in source
    assert 'write(*, "(a)") "call se:    " // ac_format_fixed(results%call_standard_error,' in source


def test_xbs_monte_carlo_emitted_fortran_compiles_and_runs() -> None:
    module = PythonNumpyFrontend().lower_source(XBS_MC_PATH.read_text(), module_name="xbs_mc_mod")
    source = FortranBackend().emit(module)
    kind_mod_path = HelperRegistry().path_for("kind_mod")
    ac_string_path = HelperRegistry().path_for("ac_string")
    helper_path = HelperRegistry().path_for("ac_random")

    driver = textwrap.dedent(
        """
        program test_driver
        use xbs_mc_mod, only: monte_carlo_option_prices, monte_carlo_option_prices_result
        implicit none
        type(monte_carlo_option_prices_result) :: results
        results = monte_carlo_option_prices(100.0d0, 100.0d0, 0.05d0, 0.2d0, 1.0d0, 16, 12345)
        print *, results%call_price
        end program test_driver
        """
    ).strip() + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        kind_mod_copy = tmp / kind_mod_path.name
        kind_mod_copy.write_text(kind_mod_path.read_text())
        ac_string_copy = tmp / ac_string_path.name
        ac_string_copy.write_text(ac_string_path.read_text())
        helper_copy = tmp / helper_path.name
        helper_copy.write_text(helper_path.read_text())
        module_file = tmp / "xbs_mc_mod.f90"
        module_file.write_text(source)
        driver_file = tmp / "test_driver.f90"
        driver_file.write_text(driver)
        exe_file = tmp / "test_driver.exe"

        compile_proc = subprocess.run(
            ["gfortran", "-std=f2018", str(kind_mod_copy), str(ac_string_copy), str(helper_copy), str(module_file), str(driver_file), "-o", str(exe_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compile_proc.returncode == 0, compile_proc.stderr

        run_proc = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
        assert run_proc.returncode == 0, run_proc.stderr
        assert run_proc.stdout.strip() != ""


def test_xbs_monte_carlo_python_and_generated_fortran_outputs_are_statistically_consistent() -> None:
    source_text = XBS_MC_PATH.read_text().replace("num_paths = 10_000_000", "num_paths = 200_000", 1)
    module = PythonNumpyFrontend().lower_source(source_text, module_name="xbs_mc_mod")
    source = FortranBackend().emit(module)
    helper_path = HelperRegistry().path_for("ac_random")
    with compiled_fortran_module_with_driver(
        source,
        module_name="xbs_mc_mod",
        driver_source="\n".join(
            [
                "program xbs_mc_driver",
                "use xbs_mc_mod, only: run_example",
                "implicit none",
                "call run_example()",
                "end program xbs_mc_driver",
                "",
            ]
        ),
        extra_sources=[helper_path],
    ) as exe_file:
        python_file = _write_temp_python_script(source_text, prefix="xbs_monte_carlo_")
        try:
            python_result = _parse_xbs_mc_output(run_command(["python", str(python_file)]))
            fortran_result = _parse_xbs_mc_output(run_command([str(exe_file)]))
        finally:
            python_file.unlink(missing_ok=True)

        assert python_result["status"] == fortran_result["status"] == "put-call parity check passed"
        assert python_result["num_paths"] == fortran_result["num_paths"] == 200000.0
        assert python_result["seed"] == fortran_result["seed"] == 12345.0
        assert abs(python_result["parity rhs"] - fortran_result["parity rhs"]) <= 5.0e-10
        assert python_result["abs error"] <= 5.0e-2
        assert fortran_result["abs error"] <= 5.0e-2

        for price_key, se_key in (("call price", "call se"), ("put price", "put se")):
            diff = abs(python_result[price_key] - fortran_result[price_key])
            tolerance = 5.0 * (python_result[se_key] + fortran_result[se_key])
            assert diff <= tolerance, price_key

        for lo_key, hi_key in (("call_ci_low", "call_ci_high"), ("put_ci_low", "put_ci_high")):
            assert python_result[lo_key] <= python_result[hi_key]
            assert fortran_result[lo_key] <= fortran_result[hi_key]


def _parse_xbs_mc_output(stdout: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("call 95% ci:["):
            left, right = _parse_ci(line.removeprefix("call 95% ci:["))
            result["call_ci_low"] = left
            result["call_ci_high"] = right
            continue
        if line.startswith("put 95% ci: ["):
            left, right = _parse_ci(line.removeprefix("put 95% ci: ["))
            result["put_ci_low"] = left
            result["put_ci_high"] = right
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


def _parse_ci(text: str) -> tuple[float, float]:
    trimmed = text.strip().rstrip("]")
    left_text, right_text = trimmed.split(",", 1)
    return float(left_text.strip()), float(right_text.strip())


def _write_temp_python_script(source_text: str, prefix: str) -> Path:
    fd, script_name = tempfile.mkstemp(suffix=".py", prefix=prefix)
    os.close(fd)
    script_path = Path(script_name)
    script_path.write_text(source_text)
    return script_path
