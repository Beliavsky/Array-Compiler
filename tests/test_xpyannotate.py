from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from array_compiler.annotator import PythonAnnotator


REPO_ROOT = Path(__file__).resolve().parents[1]
XPYANNOTATE_PATH = REPO_ROOT / "xpyannotate.py"


def test_python_annotator_adds_final_and_array_annotations() -> None:
    source = "\n".join(
        [
            "import numpy as np",
            "",
            "n = 5",
            "sigma = 1.0",
            "x = np.arange(10)",
            "a = np.zeros((2, 3))",
            "",
        ]
    )
    annotated, warnings = PythonAnnotator().annotate_source(source)
    assert "from typing import Final" in annotated
    assert "from array_compiler.annotations import Array1D, Array2D, Array3D" in annotated
    assert "n: Final[int] = 5" in annotated
    assert "sigma: Final[float] = 1.0" in annotated
    assert "x: Final[Array1D[float]] = np.arange(10)" in annotated
    assert "a: Final[Array2D[float]] = np.zeros((2, 3))" in annotated
    assert warnings == []


def test_python_annotator_warns_on_unstable_types() -> None:
    source = "\n".join(
        [
            "x = 1",
            "x = 1.5",
            "",
        ]
    )
    annotated, warnings = PythonAnnotator().annotate_source(source)
    assert "x: int = 1" in annotated
    assert "x: float = 1.5" in annotated
    assert len(warnings) == 1
    assert warnings[0].message == "x has unstable type: int -> float"


def test_python_annotator_adds_function_return_annotations() -> None:
    source = "\n".join(
        [
            "import numpy as np",
            "",
            "def f():",
            "    return 1.0",
            "",
            "def g():",
            "    return np.arange(5)",
            "",
        ]
    )
    annotated, warnings = PythonAnnotator().annotate_source(source)
    assert "def f() -> float:" in annotated
    assert "def g() -> Array1D[float]:" in annotated
    assert warnings == []


def test_python_annotator_adds_function_argument_annotations() -> None:
    source = "\n".join(
        [
            "def f(n, x, a=1.0):",
            "    total = 0.0",
            "    for i in range(n):",
            "        total = total + x[i]",
            "    return total + a",
            "",
        ]
    )
    annotated, warnings = PythonAnnotator().annotate_source(source)
    assert "def f(n: int, x: Array1D[float], a: float = 1.0) -> float:" in annotated
    messages = [warning.message for warning in warnings]
    assert "parameter n intent=in" in messages
    assert "parameter x intent=in" in messages
    assert "parameter a intent=in" in messages


def test_python_annotator_adds_final_locals_inside_function() -> None:
    source = "\n".join(
        [
            "import numpy as np",
            "",
            "def empirical_acf(x: Array1D[float], k: int) -> Array1D[float]:",
            "    x: Array1D[float] = np.asarray(x, dtype=float)",
            "    n = len(x)",
            "    xc = x - x.mean()",
            "    denom = np.dot(xc, xc)",
            "    acf = np.empty(k, dtype=float)",
            "    return acf",
            "",
        ]
    )
    annotated, warnings = PythonAnnotator().annotate_source(source)
    assert "n: Final[int] = len(x)" in annotated
    assert "xc: Final[Array1D[float]] = x - x.mean()" in annotated
    assert "denom: Final[float] = np.dot(xc, xc)" in annotated
    assert "acf: Final[Array1D[float]] = np.empty(k, dtype=float)" in annotated
    messages = [warning.message for warning in warnings]
    assert "parameter x intent=rebound" in messages
    assert "parameter k intent=in" in messages


def test_python_annotator_upgrades_existing_annotation_to_final() -> None:
    source = "\n".join(
        [
            "def f(x: Array1D[float]):",
            "    n: int = len(x)",
            "    return n",
            "",
        ]
    )
    annotated, warnings = PythonAnnotator().annotate_source(source)
    assert "n: Final[int] = len(x)" in annotated
    assert [warning.message for warning in warnings] == ["parameter x intent=in"]


def test_python_annotator_warns_on_unstable_return_types() -> None:
    source = "\n".join(
        [
            "def f(flag):",
            "    if flag:",
            "        return 1",
            "    return 1.0",
            "",
        ]
    )
    annotated, warnings = PythonAnnotator().annotate_source(source)
    assert "def f(flag):" in annotated
    assert "->" not in annotated.splitlines()[0]
    assert len(warnings) == 1
    assert warnings[0].message == "f has unstable return type"


def test_python_annotator_inferrs_int_from_annotated_callee_and_shape_expr() -> None:
    source = "\n".join(
        [
            "import numpy as np",
            "",
            "def empirical_acf(x: Array1D[float], k: int) -> Array1D[float]:",
            "    return np.empty(k, dtype=float)",
            "",
            "def empirical_pacf(x, k):",
            "    acf = empirical_acf(x, k=k)",
            "    rho: Array1D[float] = np.empty(k + 1, dtype=float)",
            "    rho[0] = 1.0",
            "    rho[1:] = acf",
            "    return rho",
            "",
        ]
    )
    annotated, warnings = PythonAnnotator().annotate_source(source)
    assert "def empirical_pacf(x: Array1D[float], k: int) -> Array1D[float]:" in annotated
    messages = [warning.message for warning in warnings]
    assert "parameter x intent=in" in messages
    assert "parameter k intent=in" in messages


def test_python_annotator_reports_parameter_intents() -> None:
    source = "\n".join(
        [
            "def f(x, y, z):",
            "    y = y + 1",
            "    z[0] = 1.0",
            "    return x",
            "",
        ]
    )
    _annotated, warnings = PythonAnnotator().annotate_source(source)
    messages = [warning.message for warning in warnings]
    assert "parameter x intent=in" in messages
    assert "parameter y intent=rebound" in messages
    assert "parameter z intent=mutated" in messages


def test_python_annotator_can_disable_parameter_intents() -> None:
    source = "\n".join(
        [
            "def f(x):",
            "    return x",
            "",
        ]
    )
    _annotated, warnings = PythonAnnotator().annotate_source(source, infer_intent=False)
    assert warnings == []


def test_python_annotator_rewrites_multiline_signature_and_optional_defaults() -> None:
    source = "\n".join(
        [
            "import numpy as np",
            "",
            "def fit_arma_initial_numpy(x, p, q, k,",
            "                           mode=\"both\",",
            "                           pacf_weight=0.5,",
            "                           npsi=None,",
            "                           seed=None):",
            "    x = np.asarray(x, dtype=float)",
            "    rng = np.random.default_rng(seed)",
            "    if npsi is None:",
            "        npsi = max(10, k + 1)",
            "    return x",
            "",
        ]
    )
    annotated, _warnings = PythonAnnotator().annotate_source(source)
    assert (
        "def fit_arma_initial_numpy(x: Array1D[float], p, q, k, "
        "mode: str = \"both\", pacf_weight: float = 0.5, npsi: int | None = None, "
        "seed: int | None = None) -> Array1D[float]:"
    ) in annotated


def test_xpyannotate_cli_writes_output_and_warnings(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.py"
    input_path.write_text("x = 1\nx = 2.0\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(XPYANNOTATE_PATH), str(input_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    output_path = tmp_path / "demo_annotated.py"
    warnings_path = tmp_path / "demo_annotated.py.warnings.txt"
    assert output_path.exists()
    assert warnings_path.exists()
    assert "wrote" in proc.stdout
    assert "warning: line 2: x has unstable type: int -> float" in proc.stdout


def test_xpyannotate_cli_can_disable_intent_warnings(tmp_path: Path) -> None:
    input_path = tmp_path / "demo.py"
    input_path.write_text("def f(x):\n    return x\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(XPYANNOTATE_PATH), str(input_path), "--no-infer-intent"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    warnings_path = tmp_path / "demo_annotated.py.warnings.txt"
    assert not warnings_path.exists()
    assert "intent=" not in proc.stdout
