from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from x2f import (
    build_command,
    compare_outputs,
    compiler_kind,
    default_ifx_compiler,
    helper_cache_dir,
    helper_cache_root,
    helper_compile_command,
    helper_object_suffix,
    resolve_compiler_command,
    transpile_python_to_fortran,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
X2F_PATH = REPO_ROOT / "x2f.py"
XBS_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xbs.py")
T025_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\python_numpy_examples_1\t025_linalg_eig_svd.py")
XAR_PATH = REPO_ROOT / "xar_sim.py"
XAR_ACF_PACF_PATH = REPO_ROOT / "xar_sim_acf_pacf.py"
XARMA_ACF_PACF_PATH = REPO_ROOT / "xarma_sim_acf_pacf.py"
XGARCH_FIT_GRID_PATH = REPO_ROOT / "xgarch_fit_grid.py"
XCCC_GARCH_SIM_PATH = REPO_ROOT / "xccc_garch_sim.py"


def test_x2f_time_both_runs_end_to_end(tmp_path: Path) -> None:
    local_input = tmp_path / "xbs.py"
    local_input.write_text(XBS_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--time-both"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    stdout = proc.stdout
    assert "Run (python):" in stdout
    assert "Run (python): PASS" in stdout
    assert "wrote xbs_p.f90" in stdout
    assert "Build: PASS" in stdout
    assert "Run: PASS" in stdout
    assert "Run diff:" in stdout
    assert "Timing summary (seconds):" in stdout
    assert (tmp_path / "xbs_p.f90").exists()
    assert (tmp_path / "xbs_p.exe").exists()


def test_x2f_run_both_runs_end_to_end_without_diff_or_timings(tmp_path: Path) -> None:
    local_input = tmp_path / "xbs.py"
    local_input.write_text(XBS_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--run-both"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    stdout = proc.stdout
    assert "Run (python):" in stdout
    assert "Run (python): PASS" in stdout
    assert "wrote xbs_p.f90" in stdout
    assert "Build: PASS" in stdout
    assert "Run: PASS" in stdout
    assert "Run diff:" not in stdout
    assert "Timing summary (seconds):" not in stdout
    assert (tmp_path / "xbs_p.f90").exists()
    assert (tmp_path / "xbs_p.exe").exists()


def test_x2f_print_main_shows_generated_driver(tmp_path: Path) -> None:
    local_input = tmp_path / "xbs.py"
    local_input.write_text(XBS_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile", "--print-main"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    stdout = proc.stdout
    assert "program xbs_p_driver" in stdout
    assert "use xbs_p, only:" in stdout
    assert "call " in stdout
    assert "end program xbs_p_driver" in stdout


def test_x2f_does_not_print_timing_summary_when_transpile_fails(tmp_path: Path) -> None:
    bad_input = tmp_path / "bad.py"
    bad_input.write_text("def f():\n    class C:\n        pass\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(bad_input), "--time-both"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "Transpile failed:" in proc.stdout
    assert "Timing summary (seconds):" not in proc.stdout


def test_x2f_rejects_run_both_with_time_both(tmp_path: Path) -> None:
    local_input = tmp_path / "xbs.py"
    local_input.write_text(XBS_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--run-both", "--time-both"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "--run-both and --time-both cannot be used together" in proc.stderr


def test_x2f_compile_builds_executable_for_module_with_run_main(tmp_path: Path) -> None:
    local_input = tmp_path / "xar_sim.py"
    local_input.write_text(XAR_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Build: PASS" in proc.stdout
    assert (tmp_path / "xar_sim_p.exe").exists()


def test_x2f_compile_builds_executable_for_xar_sim_acf_pacf(tmp_path: Path) -> None:
    local_input = tmp_path / "xar_sim_acf_pacf.py"
    local_input.write_text(XAR_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Build: PASS" in proc.stdout
    assert (tmp_path / "xar_sim_acf_pacf_p.exe").exists()


def test_x2f_replaces_simple_ac_array_with_native_constructor(tmp_path: Path) -> None:
    local_input = tmp_path / "xar_sim.py"
    local_input.write_text(XAR_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = (tmp_path / "xar_sim_p.f90").read_text(encoding="utf-8")
    assert "real(dp), parameter :: phi(3) = [0.7d0, -0.2d0, 0.1d0]" in generated
    assert "ac_array(" not in generated


def test_x2f_compile_builds_executable_for_linalg_eig_svd_example(tmp_path: Path) -> None:
    local_input = tmp_path / "t025_linalg_eig_svd.py"
    local_input.write_text(T025_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Auto helper files:" in proc.stdout
    assert "lapack_d.f90" in proc.stdout
    assert "Build: PASS" in proc.stdout
    assert (tmp_path / "t025_linalg_eig_svd_p.exe").exists()


def test_x2f_compile_builds_executable_for_ccc_garch_cholesky_example(tmp_path: Path) -> None:
    local_input = tmp_path / "xccc_garch_sim.py"
    local_input.write_text(XCCC_GARCH_SIM_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Build: PASS" in proc.stdout
    assert (tmp_path / "xccc_garch_sim_p.exe").exists()


def test_x2f_uses_meaningful_terminal_local_as_function_result(tmp_path: Path) -> None:
    local_input = tmp_path / "xgarch_fit_grid.py"
    local_input.write_text(XGARCH_FIT_GRID_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = (tmp_path / "xgarch_fit_grid_p.f90").read_text(encoding="utf-8")
    assert "function read_returns(filename) result(x)" in generated
    assert "real(dp), allocatable :: x(:)" in generated
    assert "result_value = x" not in generated
    assert "ac_copy(" not in generated


def test_x2f_replaces_first_ac_empty_assignment_with_allocate(tmp_path: Path) -> None:
    local_input = tmp_path / "xgarch_fit_grid.py"
    local_input.write_text(XGARCH_FIT_GRID_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = (tmp_path / "xgarch_fit_grid_p.f90").read_text(encoding="utf-8")
    assert "allocate(h(n))" in generated
    assert "h = ac_empty(n)" not in generated
    assert "best_params = params" in generated
    assert "ac_copy(" not in generated


def test_x2f_shifts_simple_zero_based_loops_to_one_based_fortran(tmp_path: Path) -> None:
    local_input = tmp_path / "xgarch_fit_grid.py"
    local_input.write_text(XGARCH_FIT_GRID_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = (tmp_path / "xgarch_fit_grid_p.f90").read_text(encoding="utf-8")
    assert "do omega_index = 1, size(omega_grid)" in generated
    assert "do alpha_index = 1, size(alpha_grid)" in generated
    assert "do beta_index = 1, size(beta_grid)" in generated
    assert "omega = omega_grid(omega_index)" in generated
    assert "alpha = alpha_grid(alpha_index)" in generated
    assert "beta = beta_grid(beta_index)" in generated
    assert "omega_grid(omega_index + 1)" not in generated


def test_x2f_collapses_copy_reshape_center_sequence(tmp_path: Path) -> None:
    local_input = tmp_path / "xgarch_fit_grid.py"
    local_input.write_text(XGARCH_FIT_GRID_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = (tmp_path / "xgarch_fit_grid_p.f90").read_text(encoding="utf-8")
    assert "x = x_arg - ac_mean(x_arg)" in generated
    assert "x = ac_reshape(ac_asarray(x), -1)" not in generated
    assert "x = x - ac_mean(x)" not in generated


def test_x2f_replaces_first_ac_empty2_assignment_with_allocate(tmp_path: Path) -> None:
    source_path = Path(
        r"c:\python\public_domain\github\Pure-Fortran-Examples\mixtures\xsim_mix_mv.py"
    )
    local_input = tmp_path / "xsim_mix_mv.py"
    local_input.write_text(source_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--style-level", "full"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = (tmp_path / "xsim_mix_mv_p.f90").read_text(encoding="utf-8")
    assert "allocate(x(n, d))" in generated
    assert "x = ac_empty2(n, d)" not in generated


def test_x2f_does_not_print_timing_summary_when_build_fails(tmp_path: Path) -> None:
    local_input = tmp_path / "xbs.py"
    local_input.write_text(XBS_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(X2F_PATH),
            str(local_input),
            "--time-both",
            "--compiler",
            'python -c "import sys; sys.exit(1)"',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "Build: FAIL" in proc.stdout
    assert "Timing summary (seconds):" not in proc.stdout


def test_compare_outputs_accepts_close_numeric_lines(capsys) -> None:
    assert compare_outputs("value: 1.0000000000\n", "value: 1.0\n")
    captured = capsys.readouterr()
    assert "Run diff: CLOSE" in captured.out


def test_compare_outputs_uses_approx_mode_for_stochastic_output(capsys) -> None:
    assert compare_outputs("x = [1.0 2.0]\n", "x = 3.0 4.0\n", stochastic=True)
    captured = capsys.readouterr()
    assert "Run diff: APPROX" in captured.out


def test_resolve_compiler_command_uses_ifx_preset() -> None:
    assert resolve_compiler_command(compiler=None, ifx=True) == default_ifx_compiler()


def test_resolve_compiler_command_prefers_explicit_compiler_over_ifx() -> None:
    assert resolve_compiler_command(compiler="customfc -O2", ifx=True) == "customfc -O2"


def test_ifx_compiler_kind_and_object_suffix() -> None:
    assert compiler_kind(["ifx", "/O3"]) == "ifx"
    expected_suffix = ".obj" if sys.platform.startswith("win") else ".o"
    assert helper_object_suffix("ifx") == expected_suffix


def test_ifx_helper_compile_command_uses_intel_flags_on_windows() -> None:
    if not sys.platform.startswith("win"):
        return
    cmd = helper_compile_command(
        compiler_parts=["ifx", "/O3"],
        source_path=Path(r"c:\tmp\kind.f90"),
        cache_dir=Path(r"c:\tmp\cache"),
        object_path=Path(r"c:\tmp\cache\kind.obj"),
    )
    assert "/c" in cmd
    assert "/module:c:\\tmp\\cache" in [part.lower() for part in cmd]
    assert "/object:c:\\tmp\\cache\\kind.obj" in [part.lower() for part in cmd]


def test_ifx_build_command_uses_intel_exe_flag_on_windows() -> None:
    if not sys.platform.startswith("win"):
        return
    cmd = build_command(
        compiler_parts=["ifx", "/O3"],
        cache_dir=Path(r"c:\tmp\cache"),
        helper_objects=[Path(r"c:\tmp\cache\kind.obj")],
        output_path=Path(r"c:\tmp\foo_p.f90"),
        driver_path=Path(r"c:\tmp\foo_p_driver.f90"),
        exe_path=Path(r"c:\tmp\foo_p.exe"),
        compile_only=False,
    )
    lower_cmd = [part.lower() for part in cmd]
    assert "/i" in lower_cmd[2][:2]
    assert "/exe:c:\\tmp\\foo_p.exe" in lower_cmd


def test_helper_cache_dir_changes_with_cache_format_version_inputs() -> None:
    cache_dir = helper_cache_dir(REPO_ROOT, ["ifx", "/O3"])
    assert cache_dir.name


def test_x2f_clean_cache_removes_cache_tree(tmp_path: Path) -> None:
    cache_root = helper_cache_root(REPO_ROOT)
    cached_file = cache_root / "fortran" / "dummy" / "stale.txt"
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_text("stale", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), "--clean-cache"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Removed cache:" in proc.stdout
    assert not cache_root.exists()


def test_x2f_clean_cache_succeeds_when_cache_missing(tmp_path: Path) -> None:
    cache_root = helper_cache_root(REPO_ROOT)
    if cache_root.exists():
        shutil.rmtree(cache_root)

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), "--clean-cache"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Cache not present:" in proc.stdout


def test_xarma_simple_python_slices_emit_as_fortran_sections(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)

    assert "result_value = x(burn + r + 1:size(x))" in fortran_source
    assert "ac_dot(" not in fortran_source
    assert "denom = sum(xc**2)" in fortran_source
    assert "sum(xc(1:size(xc) - lag) * xc(lag + 1:size(xc)))" in fortran_source
    assert "result_value = pacf_from_acf([1.0d0, theoretical_acf_arma(ar, ma, sigma, k)])" in fortran_source
    assert "gamma(1:m + 1) = gamma0_to_m(1:min(m, k) + 1)" in fortran_source
    assert "gamma(h + 1) = sum(ar * gamma(h:h - p + 1:-1))" in fortran_source
    assert "a(h + 1, h + 1) = 1.0d0" in fortran_source
    assert "if (q > 0) theta(2:size(theta)) = ma" in fortran_source
    assert "rho(0 + 1) = 1.0d0" not in fortran_source
    assert "rho((1) + 1:size(rho)) = acf" not in fortran_source
    assert "gamma(h + 1) = ac_dot(ar, gamma(h:h - (p + 1) + 2:-1))" not in fortran_source
    assert "call ac_set_item2(a, h, h, 1.0d0)" not in fortran_source
    assert "call ac_set_slice(theta, 1, size(theta), ma)" not in fortran_source
    assert "rho = ac_empty(k + 1)" not in fortran_source


def test_adjacent_string_literals_in_print_are_combined(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)

    assert 'write(*, "(a, i0)") "n: ", n' in fortran_source
    assert 'write(*, "(a)") "n: " // ac_format_int(n, 0)' not in fortran_source
    assert 'write(*, "(a)") "n:" // " " // ac_format_int(n, 0)' not in fortran_source


def test_xarma_sim_fit_acf_pacf_compiles(tmp_path: Path) -> None:
    source_path = Path("c:/python/Array-Compiler/xarma_sim_fit_acf_pacf.py")
    proc = subprocess.run(
        [sys.executable, "x2f.py", str(source_path), "--compile"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Build: PASS" in proc.stdout


def test_additive_index_expression_omits_unneeded_parentheses(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)

    assert "s = s + theta(j + 1) * psi(j - h + 1)" in fortran_source
    assert "s = s + theta(j + 1) * psi((j - h) + 1)" not in fortran_source


def test_mutated_dummy_arguments_emit_shadow_copies_with_intent_in(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)

    assert "real(dp), intent(in) :: ar_arg(:), ma_arg(:)" in fortran_source
    assert "real(dp), allocatable :: ar(:), ma(:)" in fortran_source
    assert "\nar = ar_arg\n" in fortran_source
    assert "\nma = ma_arg\n" in fortran_source


def test_nonconsecutive_like_local_declarations_are_grouped(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)
    block = fortran_source[
        fortran_source.index("function theoretical_acovf_arma"):
        fortran_source.index("end function theoretical_acovf_arma")
    ]
    local_decl_line = next(
        line for line in block.splitlines()
        if "real(dp), allocatable :: ar(:), ma(:), theta(:), psi(:), a(:, :), b(:), &" in line
    )

    assert "function theoretical_acovf_arma(ar_arg, ma_arg, sigma, k) result(gamma)" in block
    assert "real(dp), allocatable :: ar(:), ma(:), theta(:), psi(:), a(:, :), b(:), &" in block
    assert "& gamma0_to_m(:)" in block
    assert "gamma(:)" not in local_decl_line
    assert "integer :: p, q, m, h, i, j" in block


def test_main_promotes_scalar_and_literal_array_setup_values_to_parameters(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)
    block = fortran_source[
        fortran_source.index("subroutine main()"):
        fortran_source.index("end subroutine main")
    ]

    assert "integer, parameter :: n = 5000" in block
    assert "integer, parameter :: k = 10, seed = 123" in block
    assert "real(dp), parameter ::" in block
    assert "ar(2) = [0.6d0, -0.3d0]" in block
    assert "ma(1) = [0.1d0]" in block
    assert "sigma = 1.0d0" in block
    assert "real(dp) :: sigma" not in block
    assert "real(dp), allocatable :: ar(:), ma(:)" not in block


def test_single_use_temps_are_inlined_in_theoretical_pacf_arma(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)
    block = fortran_source[
        fortran_source.index("function theoretical_pacf_arma"):
        fortran_source.index("end function theoretical_pacf_arma")
    ]

    assert "real(dp), allocatable :: acf(:), rho(:)" not in block
    assert "real(dp), allocatable :: acf(:)" not in block
    assert "rho = [1.0d0, acf]" not in block
    assert "result_value = pacf_from_acf([1.0d0, theoretical_acf_arma(ar, ma, sigma, k)])" in block


def test_function_docstring_is_emitted_after_signature(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)
    block = fortran_source[
        fortran_source.index("function theoretical_acf_arma"):
        fortran_source.index("end function theoretical_acf_arma")
    ]

    assert "! return theoretical autocorrelations rho(1),...,rho(k)" in block
    assert block.index("! return theoretical autocorrelations rho(1),...,rho(k)") < block.index("real(dp), intent(in) :: ar(:), ma(:), sigma")


def test_style_level_none_preserves_unwrapped_output(tmp_path: Path) -> None:
    local_input = tmp_path / "xbs.py"
    local_input.write_text(XBS_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_none = tmp_path / "xbs_none.f90"
    output_full = tmp_path / "xbs_full.f90"

    _, source_none = transpile_python_to_fortran(local_input, output_none, style_level="none")
    _, source_full = transpile_python_to_fortran(local_input, output_full, style_level="full")

    assert max(len(line) for line in source_none.splitlines()) > 80
    assert max(len(line) for line in source_full.splitlines()) <= 80


def test_simple_np_r_concat_emits_native_array_constructor(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)

    assert "all(abs(ac_roots([-ar(size(ar):1:-1), 1.0d0])) > 1.0d0)" in fortran_source
    assert "all(abs(ac_roots([ma(size(ma):1:-1), 1.0d0])) > 1.0d0)" in fortran_source
    assert "ac_r_concat(" not in fortran_source
    assert "ac_array(" not in fortran_source


def test_default_do_step_is_omitted_for_positive_unit_stride(tmp_path: Path) -> None:
    local_input = tmp_path / "xarma_sim_acf_pacf.py"
    local_input.write_text(XARMA_ACF_PACF_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    output_path = tmp_path / "xarma_sim_acf_pacf_p.f90"

    _, fortran_source = transpile_python_to_fortran(local_input, output_path)

    assert "do h = m + 1, k" in fortran_source
    assert "do h = m + 1, k, 1" not in fortran_source
