from pathlib import Path

from array_compiler.backends.fortran import FortranBackend
from array_compiler.frontends.python_numpy import PythonNumpyFrontend


XBS_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xbs.py")


def test_xbs_lowers_and_emits_fortran() -> None:
    module = PythonNumpyFrontend().lower_source(XBS_PATH.read_text(), module_name="xbs_mod")
    source = FortranBackend().emit(module)

    assert "module xbs_mod" in source
    assert "function normal_cdf(x) result(result_value)" in source
    assert "function black_scholes_price(spot, strike, rate, volatility, time_to_maturity, option_type) result(result_value)" in source
    assert "subroutine run_example()" in source
    assert "subroutine run_main()" in source
    assert "option_kind = ac_lower(trim(adjustl(option_type)))" in source
    assert "if (((option_kind /= 'call') .and. (option_kind /= 'put'))) then" in source
    assert "error stop 'option_type must be ''call'' or ''put'''" in source
    assert "call run_example()" in source
