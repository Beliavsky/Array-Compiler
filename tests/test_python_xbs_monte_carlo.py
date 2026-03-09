from pathlib import Path
import subprocess
import tempfile
import textwrap

from array_compiler.backends.fortran import FortranBackend
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from array_compiler.backends.fortran.helpers import HelperRegistry


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
    assert "function mean_standard_error_and_ci(payoff_sum, payoff_sum_sq, num_paths, discount_factor) result(result_value)" in source
    assert "function monte_carlo_option_prices(spot, strike, rate, volatility, time_to_maturity, num_paths, seed) result(result_value)" in source
    assert "rng = ac_random_init(seed)" in source
    assert "z = ac_gauss(rng, 0.0d0, 1.0d0)" in source
    assert "do ac_loop_index = 0, (num_paths - 1), 1" in source
    assert "call_price = mean_standard_error_and_ci_value%item1" in source
    assert "result_value = monte_carlo_option_prices_result(call_price=call_price" in source
    assert "call_price = results%call_price" in source
    assert "print *, 'call se:    ', results%call_standard_error" in source


def test_xbs_monte_carlo_emitted_fortran_compiles_and_runs() -> None:
    module = PythonNumpyFrontend().lower_source(XBS_MC_PATH.read_text(), module_name="xbs_mc_mod")
    source = FortranBackend().emit(module)
    kind_mod_path = HelperRegistry().path_for("kind_mod")
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
        helper_copy = tmp / helper_path.name
        helper_copy.write_text(helper_path.read_text())
        module_file = tmp / "xbs_mc_mod.f90"
        module_file.write_text(source)
        driver_file = tmp / "test_driver.f90"
        driver_file.write_text(driver)
        exe_file = tmp / "test_driver.exe"

        compile_proc = subprocess.run(
            ["gfortran", "-std=f2018", str(kind_mod_copy), str(helper_copy), str(module_file), str(driver_file), "-o", str(exe_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compile_proc.returncode == 0, compile_proc.stderr

        run_proc = subprocess.run([str(exe_file)], capture_output=True, text=True, check=False)
        assert run_proc.returncode == 0, run_proc.stderr
        assert run_proc.stdout.strip() != ""
