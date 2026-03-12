from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from array_compiler.rchecker import DiagnosticInfo
from array_compiler.rchecker import RChecker


REPO_ROOT = Path(__file__).resolve().parents[1]
XRCHECK_PATH = REPO_ROOT / "xrcheck.py"


def test_r_checker_warns_and_fixes_direct_integer_literal() -> None:
    source = "y <- runif(5)\n"

    checker = RChecker()
    diagnostics = checker.check_source(source)
    fixed, _ = checker.fix_source(source)

    assert diagnostics == [
        DiagnosticInfo(
            level="warning",
            line=1,
            message="double literal 5 is used as an integer in runif() count; prefer 5L",
        )
    ]
    assert fixed == "y <- runif(5L)\n"


def test_r_checker_warns_and_fixes_integer_like_binding() -> None:
    source = "n <- 5\ny <- runif(n)\n"

    checker = RChecker()
    diagnostics = checker.check_source(source)
    fixed, _ = checker.fix_source(source)

    assert diagnostics == [
        DiagnosticInfo(
            level="warning",
            line=1,
            message="n is assigned double literal 5 but used as an integer in runif() count; prefer 5L",
        )
    ]
    assert fixed == "n <- 5L\ny <- runif(n)\n"


def test_r_checker_leaves_explicit_integer_literal_alone() -> None:
    source = "n <- 5L\ny <- runif(n)\n"

    checker = RChecker()
    diagnostics = checker.check_source(source)
    fixed, _ = checker.fix_source(source)

    assert diagnostics == []
    assert fixed == source


def test_r_checker_warns_and_fixes_t_and_f() -> None:
    source = "if (T) x <- 1\nif (F) x <- 2\n"

    checker = RChecker()
    diagnostics = checker.check_source(source)
    fixed, _ = checker.fix_source(source)

    assert diagnostics == [
        DiagnosticInfo(level="warning", line=1, message="prefer TRUE over T"),
        DiagnosticInfo(level="warning", line=2, message="prefer FALSE over F"),
    ]
    assert fixed == "if (TRUE) x <- 1\nif (FALSE) x <- 2\n"


def test_r_checker_warns_on_bare_na() -> None:
    source = "x <- NA\n"

    diagnostics = RChecker().check_source(source)

    assert diagnostics == [
        DiagnosticInfo(
            level="warning",
            line=1,
            message="bare NA is untyped; prefer NA_integer_, NA_real_, or NA_character_ when the type matters",
        )
    ]


def test_r_checker_warns_on_numeric_used_as_logical() -> None:
    source = "if (1) x <- 2\n"

    diagnostics = RChecker().check_source(source)

    assert diagnostics == [
        DiagnosticInfo(
            level="warning",
            line=1,
            message="numeric literal 1 is used as a logical in if condition; prefer TRUE/FALSE or an explicit comparison",
        )
    ]


def test_r_checker_warns_on_partial_argument_matching() -> None:
    source = "x <- matrix(0, nr=2, nc=3)\n"

    diagnostics = RChecker().check_source(source)

    assert diagnostics == [
        DiagnosticInfo(level="warning", line=1, message="partial argument name 'nr' in matrix(); prefer 'nrow'"),
        DiagnosticInfo(level="warning", line=1, message="partial argument name 'nc' in matrix(); prefer 'ncol'"),
    ]


def test_r_checker_warns_on_incompatible_recycling() -> None:
    source = "x <- c(1, 2) + c(1, 2, 3)\n"

    diagnostics = RChecker().check_source(source)

    assert diagnostics == [
        DiagnosticInfo(
            level="warning",
            line=1,
            message="vector lengths 2 and 3 are incompatible for recycling with '+'",
        )
    ]


def test_r_checker_warns_on_sequential_type_change() -> None:
    source = "x <- 1\nx <- \"a\"\n"

    diagnostics = RChecker().check_source(source)

    assert diagnostics == [
        DiagnosticInfo(
            level="warning",
            line=2,
            message="x changes type: scalar_double -> scalar_character",
        )
    ]


def test_r_checker_warns_on_branch_type_change() -> None:
    source = "x <- 1\nif (TRUE) x <- \"a\" else x <- 2\n"

    diagnostics = RChecker().check_source(source)

    assert diagnostics == [
        DiagnosticInfo(
            level="warning",
            line=2,
            message="x changes type: scalar_double -> scalar_character",
        )
    ]


def test_r_checker_can_fix_simple_sequential_type_change() -> None:
    source = "x <- 1\nx <- as.character(x)\nprint(x)\n"

    fixed, diagnostics = RChecker().fix_source(source, fix_type_changes=True)

    assert diagnostics == [
        DiagnosticInfo(
            level="warning",
            line=2,
            message="x changes type: scalar_double -> vector_character",
        )
    ]
    assert fixed == "x <- 1\nx_character <- as.character(x)\nprint(x_character)\n"


def test_r_checker_does_not_fix_type_change_through_control_flow() -> None:
    source = "x <- 1\nx <- as.character(x)\nif (TRUE) print(x)\n"

    fixed, diagnostics = RChecker().fix_source(source, fix_type_changes=True)

    assert diagnostics == [
        DiagnosticInfo(
            level="warning",
            line=2,
            message="x changes type: scalar_double -> vector_character",
        )
    ]
    assert fixed == source


def test_xrcheck_cli_can_fix_file_in_place(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.r"
    input_path.write_text("n <- 5\nif (T) y <- runif(n)\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(XRCHECK_PATH), str(input_path), "--fix"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wrote" in proc.stdout
    assert "warning: line 1:" in proc.stdout
    assert "warning: line 2:" in proc.stdout
    assert input_path.read_text(encoding="utf-8") == "n <- 5L\nif (TRUE) y <- runif(n)\n"


def test_xrcheck_cli_can_fix_simple_type_change(tmp_path: Path) -> None:
    input_path = tmp_path / "rename_demo.r"
    input_path.write_text("x <- 1\nx <- as.character(x)\nprint(x)\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(XRCHECK_PATH), str(input_path), "--fix-type-changes"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wrote" in proc.stdout
    assert input_path.read_text(encoding="utf-8") == "x <- 1\nx_character <- as.character(x)\nprint(x_character)\n"
