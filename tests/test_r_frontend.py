from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from array_compiler.frontends.r import RSubsetFrontend
from x2f import detect_source_language, source_run_command


REPO_ROOT = Path(__file__).resolve().parents[1]
X2F_PATH = REPO_ROOT / "x2f.py"
R_EXAMPLES_DIR = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\r_examples_1")


def test_detect_source_language_supports_r_suffixes() -> None:
    assert detect_source_language(Path("example.py")) == "python"
    assert detect_source_language(Path("example.r")) == "r"
    assert detect_source_language(Path("example.R")) == "r"


def test_source_run_command_uses_rscript_for_r_sources() -> None:
    assert source_run_command(Path("example.r")) == ["Rscript", "example.r"]


def test_r_subset_translation_promotes_scalar_calls_and_transpose() -> None:
    source = "\n".join(
        [
            "f = function(x) x*x + 2*x + 1",
            'cat(f(3), "\\n")',
            "a = matrix(1:6, nrow=2, ncol=3)",
            "b = t(a)",
            "",
        ]
    )

    translated = RSubsetFrontend().translate_source(source)

    assert "def f(x: float):" in translated
    assert "print(f(3.0))" in translated
    assert "b = (a).T" in translated


@pytest.mark.parametrize(
    "example_name",
    [
        "test03_seq_colon.r",
        "test20_matrix_basic.r",
        "test21_transpose.r",
        "test26_for_accum.r",
        "test31_fun_scalar.r",
        "test33_fun_defaults.r",
        "test43_mod_intdiv.r",
        "test44_cat_format.r",
    ],
)
def test_x2f_compiles_supported_r_examples(tmp_path: Path, example_name: str) -> None:
    local_input = tmp_path / example_name
    local_input.write_text((R_EXAMPLES_DIR / example_name).read_text(encoding="utf-8-sig"), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(X2F_PATH), str(local_input), "--compile"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Build: PASS" in proc.stdout
    assert (tmp_path / f"{local_input.stem}_p.exe").exists()
