from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from array_compiler.frontends.r import RSubsetFrontend
from x2p import error_source_line


REPO_ROOT = Path(__file__).resolve().parents[1]
X2P_PATH = REPO_ROOT / "x2p.py"
X2F_PATH = REPO_ROOT / "x2f.py"
R_EXAMPLES_DIR = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\r_examples_1")
XBASE_GARCH_SIM = REPO_ROOT / "xbase_garch_sim.r"
XBASE_GARCH_MOMENT_FIT = REPO_ROOT / "xbase_garch_moment_fit.r"
XTIMER = REPO_ROOT / "xtimer.r"


def test_r_subset_standalone_translation_is_self_contained() -> None:
    source = "\n".join(
        [
            "x = 1:5",
            'cat(x, "\\n")',
            "",
        ]
    )

    translated = RSubsetFrontend().translate_source(source, standalone=True)

    assert "import numpy as np" in translated
    assert "from array_compiler.annotations" not in translated
    assert "def r_cat(" in translated


def test_r_subset_standalone_translation_runs_full_example_corpus(tmp_path: Path) -> None:
    frontend = RSubsetFrontend()

    for source_path in sorted(R_EXAMPLES_DIR.glob("*.r")):
        translated = frontend.translate_source(source_path.read_text(encoding="utf-8-sig"), standalone=True)
        output_path = tmp_path / f"{source_path.stem}_translated.py"
        output_path.write_text(translated, encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(output_path)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode == 0, f"{source_path.name}: {proc.stdout}\n{proc.stderr}"


def test_x2p_run_both_for_lm_example(tmp_path: Path) -> None:
    local_input = tmp_path / "test45_lm.r"
    local_input.write_text((R_EXAMPLES_DIR / "test45_lm.r").read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2P_PATH), str(local_input), "--run-both"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Run (r): PASS" in proc.stdout
    assert "Run (python): PASS" in proc.stdout
    assert "wrote test45_lm_p.py" in proc.stdout
    assert "Run diff:" in proc.stdout
    assert (tmp_path / "test45_lm_p.py").exists()


def test_x2p_x2f_mode_emits_typed_python_subset(tmp_path: Path) -> None:
    local_input = tmp_path / "test02_as_integer_trunc.r"
    local_input.write_text((R_EXAMPLES_DIR / "test02_as_integer_trunc.r").read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2P_PATH), str(local_input), "--mode", "x2f"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = (tmp_path / "test02_as_integer_trunc_p.py").read_text(encoding="utf-8")
    assert "from array_compiler.annotations import Array1D, Array2D" in generated
    assert "print(int(x))" in generated


def test_x2p_x2f_mode_output_compiles_through_x2f(tmp_path: Path) -> None:
    local_input = tmp_path / "test02_as_integer_trunc.r"
    local_input.write_text((R_EXAMPLES_DIR / "test02_as_integer_trunc.r").read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2P_PATH), str(local_input), "--mode", "x2f"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    generated_py = tmp_path / "test02_as_integer_trunc_p.py"
    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(generated_py), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Build: PASS" in proc.stdout


def test_x2p_x2f_mode_rep_example_compiles_through_x2f(tmp_path: Path) -> None:
    local_input = tmp_path / "test04_rep.r"
    local_input.write_text((R_EXAMPLES_DIR / "test04_rep.r").read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2P_PATH), str(local_input), "--mode", "x2f"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    generated_py = tmp_path / "test04_rep_p.py"
    generated = generated_py.read_text(encoding="utf-8")
    assert "np.asarray(3).reshape(-1)" in generated

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(generated_py), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Build: PASS" in proc.stdout


def test_x2p_translates_and_runs_xbase_garch_sim(tmp_path: Path) -> None:
    local_input = tmp_path / "xbase_garch_sim.r"
    local_input.write_text(XBASE_GARCH_SIM.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2P_PATH), str(local_input)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wrote xbase_garch_sim_p.py" in proc.stdout

    generated_py = tmp_path / "xbase_garch_sim_p.py"
    proc = subprocess.run(
        [sys.executable, str(generated_py), "100", "tmp_garch_returns.txt", "0.01", "0.08", "0.9", "0"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wrote 100 returns to tmp_garch_returns.txt" in proc.stdout
    assert "['" not in proc.stdout
    assert "seed = NA" not in proc.stdout
    assert (tmp_path / "tmp_garch_returns.txt").exists()


def test_x2p_translates_xbase_garch_moment_fit(tmp_path: Path) -> None:
    local_input = tmp_path / "xbase_garch_moment_fit.r"
    local_input.write_text(XBASE_GARCH_MOMENT_FIT.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2P_PATH), str(local_input)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wrote xbase_garch_moment_fit_p.py" in proc.stdout

    generated_py = tmp_path / "xbase_garch_moment_fit_p.py"
    proc = subprocess.run(
        [sys.executable, str(generated_py), "200", "tmp_moment_fit_returns.txt", "0.01", "0.08", "0.9", "0"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "estimated garch(1,1) parameters" in proc.stdout
    assert "elapsed times in seconds" in proc.stdout
    assert (tmp_path / "tmp_moment_fit_returns.txt").exists()


def test_x2f_compiles_xbase_garch_moment_fit_r(tmp_path: Path) -> None:
    local_input = tmp_path / "xbase_garch_moment_fit.r"
    local_input.write_text(XBASE_GARCH_MOMENT_FIT.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Build: PASS" in proc.stdout


def test_x2f_compiles_xtimer_r(tmp_path: Path) -> None:
    local_input = tmp_path / "xtimer.r"
    local_input.write_text(XTIMER.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Build: PASS" in proc.stdout


def test_x2f_mode_translation_formats_joined_tables_for_xbase_garch_moment_fit(tmp_path: Path) -> None:
    local_input = tmp_path / "xbase_garch_moment_fit.r"
    local_input.write_text(XBASE_GARCH_MOMENT_FIT.read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2P_PATH), str(local_input), "--mode", "x2f"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr

    generated_py = tmp_path / "xbase_garch_moment_fit_p.py"
    generated = generated_py.read_text(encoding="utf-8")

    assert "piece = x[i].rstrip()" in generated
    assert "ac_format_fixed(values[(i - 1)], digits, width)" in generated


def test_x2p_error_source_line_returns_full_continued_statement() -> None:
    source = "\n".join(
        [
            "if (is.finite(acf_sq[i]) &&",
            "    is.finite(acf_sq_theory[i])) {",
            "  x <- 1",
            "}",
            "",
        ]
    )

    snippet = error_source_line(source, "unsupported R expression near line 2")

    assert snippet == "\n".join(
        [
            "if (is.finite(acf_sq[i]) &&",
            "    is.finite(acf_sq_theory[i])) {",
        ]
    )


def test_x2p_reports_full_multiline_statement_on_translate_failure(tmp_path: Path) -> None:
    local_input = tmp_path / "broken.r"
    local_input.write_text(
        "\n".join(
            [
                "if (is.finite(acf_sq[i]) &&",
                "    is.finite(acf_sq_theory[i])) ;",
                "",
            ]
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(X2P_PATH), str(local_input)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "Translate failed:" in proc.stdout
    assert "if (is.finite(acf_sq[i]) &&" in proc.stdout
    assert "    is.finite(acf_sq_theory[i])) ;" in proc.stdout
