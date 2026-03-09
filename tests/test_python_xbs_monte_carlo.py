from pathlib import Path

from array_compiler.backends.fortran import FortranBackend
from array_compiler.frontends.python_numpy import PythonNumpyFrontend


XBS_MC_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xbs_monte_carlo.py")


def test_xbs_monte_carlo_lowers_and_emits_fortran() -> None:
    module = PythonNumpyFrontend().lower_source(XBS_MC_PATH.read_text(), module_name="xbs_mc_mod")
    source = FortranBackend().emit(module)

    record_names = {record.name for record in module.records}
    assert "mean_standard_error_and_ci_result" in record_names
    assert "monte_carlo_option_prices_result" in record_names
    assert "ac_random_state" in record_names

    assert "type :: monte_carlo_option_prices_result" in source
    assert "function mean_standard_error_and_ci(payoff_sum, payoff_sum_sq, num_paths, discount_factor) result(result_value)" in source
    assert "function monte_carlo_option_prices(spot, strike, rate, volatility, time_to_maturity, num_paths, seed) result(result_value)" in source
    assert "rng = ac_random_init(seed)" in source
    assert "z = ac_gauss(rng, 0.0, 1.0)" in source
    assert "do _ = 1, num_paths" in source
    assert "call_price = mean_standard_error_and_ci_value%item1" in source
    assert "result_value = monte_carlo_option_prices_result(call_price=call_price" in source
    assert "call_price = results%call_price" in source
    assert "print *, 'call se:    ', results%call_standard_error" in source
