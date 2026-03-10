from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
X2F_BATCH_PATH = REPO_ROOT / "x2f_batch.py"


def test_x2f_batch_stops_on_first_failure_and_writes_csv(tmp_path: Path) -> None:
    good_path = tmp_path / "good.py"
    bad_path = tmp_path / "bad.py"
    csv_path = tmp_path / "results.csv"

    good_path.write_text("print('ok')\n", encoding="utf-8")
    bad_path.write_text("class C:\n    pass\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(X2F_BATCH_PATH),
            str(good_path),
            str(bad_path),
            "--compile",
            "--stop-on-first-failure",
            "--csv",
            str(csv_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "[1/2]" in proc.stdout
    assert "Stopped on first failure:" in proc.stdout
    assert "Totals: 1 files, 0 pass, 1 fail" in proc.stdout
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "bad.py" in csv_text


def test_x2f_batch_compile_examples_limit_one(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(X2F_BATCH_PATH),
            r"c:\python\public_domain\github\Pure-Fortran-Examples\python_numpy_examples_1\*.py",
            "--compile",
            "--limit",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert "[1/1]" in proc.stdout
    assert "Summary:" in proc.stdout


def test_x2f_batch_prints_grouped_failures(tmp_path: Path) -> None:
    bad1_path = tmp_path / "bad1.py"
    bad2_path = tmp_path / "bad2.py"

    bad1_path.write_text("import numpy as np\nprint(np.sign(np.array([1.0, -2.0])))\n", encoding="utf-8")
    bad2_path.write_text("import numpy as np\nx = np.sign(np.array([3.0]))\nprint(x)\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(X2F_BATCH_PATH),
            str(bad1_path),
            str(bad2_path),
            "--compile",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "Failure groups:" in proc.stdout
    assert "unsupported call: np.sign" in proc.stdout
