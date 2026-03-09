from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from x2f import compare_outputs


REPO_ROOT = Path(__file__).resolve().parents[1]
X2F_PATH = REPO_ROOT / "x2f.py"
XBS_PATH = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\option_pricing\xbs.py")
XAR_PATH = REPO_ROOT / "xar_sim.py"
XAR_ACF_PACF_PATH = REPO_ROOT / "xar_sim_acf_pacf.py"


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
