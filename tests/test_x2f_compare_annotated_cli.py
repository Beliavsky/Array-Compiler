from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
X2F_COMPARE_PATH = REPO_ROOT / "x2f_compare_annotated.py"


def test_x2f_compare_annotated_writes_summary_and_csv(tmp_path: Path) -> None:
    good_path = tmp_path / "good.py"
    bad_path = tmp_path / "bad.py"
    csv_path = tmp_path / "compare.csv"

    good_path.write_text("print('ok')\n", encoding="utf-8")
    bad_path.write_text("import numpy as np\nprint(np.meshgrid([1.0], [2.0]))\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(X2F_COMPARE_PATH),
            str(good_path),
            str(bad_path),
            "--compile",
            "--csv",
            str(csv_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert "[1/2]" in proc.stdout
    assert "Summary:" in proc.stdout
    assert "baseline pass :" in proc.stdout
    assert "annotated pass:" in proc.stdout
    assert "regressed" in proc.stdout
    assert "changed" in proc.stdout
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "comparison" in csv_text
    assert "good.py" in csv_text
