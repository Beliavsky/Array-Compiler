from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
X2P_BATCH_PATH = REPO_ROOT / "x2p_batch.py"
R_EXAMPLES_DIR = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\r_examples_1")


def test_x2p_batch_generates_x2f_mode_python_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "translated"

    proc = subprocess.run(
        [
            sys.executable,
            str(X2P_BATCH_PATH),
            str(R_EXAMPLES_DIR),
            "--mode",
            "x2f",
            "--out-dir",
            str(out_dir),
            "--limit",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[1/2]" in proc.stdout
    assert "Totals: 2 files, 2 pass, 0 fail" in proc.stdout
    generated = sorted(out_dir.glob("*.py"))
    assert len(generated) == 2
    assert any(path.name.endswith("_from_r.py") for path in generated)
