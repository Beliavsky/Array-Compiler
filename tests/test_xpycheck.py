from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from array_compiler.pychecker import DiagnosticInfo
from array_compiler.pychecker import PythonChecker


REPO_ROOT = Path(__file__).resolve().parents[1]
XPYCHECK_PATH = REPO_ROOT / "xpycheck.py"


def test_python_checker_reports_variable_annotation_mismatch() -> None:
    source = "\n".join(
        [
            "import numpy as np",
            "from array_compiler.annotations import Array1D",
            "",
            "a: Array1D[float] = np.zeros((2, 3))",
            "",
        ]
    )
    diagnostics = PythonChecker().check_source(source)
    messages = [diag.message for diag in diagnostics]
    assert "a is annotated Array1D[float] but assigned Array2D[float]" in messages


def test_python_checker_reports_final_rebinding() -> None:
    source = "\n".join(
        [
            "from typing import Final",
            "",
            "x: Final[int] = 1",
            "x = 2",
            "",
        ]
    )
    diagnostics = PythonChecker().check_source(source)
    messages = [diag.message for diag in diagnostics]
    assert "x is annotated Final but rebound" in messages


def test_python_checker_reports_return_mismatch() -> None:
    source = "\n".join(
        [
            "import numpy as np",
            "from array_compiler.annotations import Array1D",
            "",
            "def f() -> float:",
            "    return np.arange(5)",
            "",
        ]
    )
    diagnostics = PythonChecker().check_source(source)
    messages = [diag.message for diag in diagnostics]
    assert "f is annotated to return float but returns Array1D[float]" in messages


def test_python_checker_can_report_intent_notes() -> None:
    source = "\n".join(
        [
            "def f(x, y):",
            "    y[0] = 1.0",
            "    return x",
            "",
        ]
    )
    diagnostics = PythonChecker().check_source(source, check_intent=True)
    notes = [diag.message for diag in diagnostics if diag.level == "note"]
    assert "parameter x intent=in" in notes
    assert "parameter y intent=mutated" in notes


def test_xpycheck_cli_json_and_strict(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.py"
    input_path.write_text("from typing import Final\nx: Final[int] = 1\nx = 2\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(XPYCHECK_PATH), str(input_path), "--json", "--strict"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    diagnostics = json.loads(proc.stdout)
    assert any(diag["message"] == "x is annotated Final but rebound" for diag in diagnostics)


def test_python_checker_can_fix_simple_type_change() -> None:
    source = "\n".join(
        [
            "x = 1",
            "x = str(x)",
            "print(x)",
            "",
        ]
    )

    fixed, diagnostics = PythonChecker().fix_source(source, fix_type_changes=True)

    assert diagnostics == [
        DiagnosticInfo(level="warning", line=2, message="x has unstable type: int -> str"),
    ]
    assert fixed == "\n".join(
        [
            "x = 1",
            "x_str = str(x)",
            "print(x_str)",
            "",
        ]
    )


def test_python_checker_does_not_fix_type_change_through_control_flow() -> None:
    source = "\n".join(
        [
            "x = 1",
            "x = str(x)",
            "if True:",
            "    print(x)",
            "",
        ]
    )

    fixed, diagnostics = PythonChecker().fix_source(source, fix_type_changes=True)

    assert diagnostics == [
        DiagnosticInfo(level="warning", line=2, message="x has unstable type: int -> str"),
    ]
    assert fixed == source


def test_xpycheck_cli_can_fix_simple_type_change(tmp_path: Path) -> None:
    input_path = tmp_path / "demo_fix.py"
    input_path.write_text("x = 1\nx = str(x)\nprint(x)\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(XPYCHECK_PATH), str(input_path), "--fix-type-changes"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wrote" in proc.stdout
    assert "warning: line 2: x has unstable type: int -> str" in proc.stdout
    assert input_path.read_text(encoding="utf-8") == "x = 1\nx_str = str(x)\nprint(x_str)\n"
