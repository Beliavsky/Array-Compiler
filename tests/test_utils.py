import os
from pathlib import Path
import subprocess
import tempfile
from contextlib import contextmanager

from array_compiler.backends.fortran.helpers import HelperRegistry


def assert_max_fortran_line_length(source: str, max_len: int = 132) -> None:
    overlong = [
        (line_no, len(line), line)
        for line_no, line in enumerate(source.splitlines(), start=1)
        if len(line) > max_len
    ]
    assert not overlong, (
        f"found {len(overlong)} line(s) longer than {max_len}; "
        f"first is line {overlong[0][0]} length {overlong[0][1]}: {overlong[0][2]!r}"
    )


@contextmanager
def compiled_fortran_module_with_driver(
    module_source: str,
    module_name: str,
    driver_source: str,
    extra_sources: list[Path] | None = None,
) -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        kind_mod_path = HelperRegistry().path_for("kind_mod")
        copied_sources = [tmp / kind_mod_path.name]
        copied_sources[0].write_text(kind_mod_path.read_text())

        for source_path in extra_sources or []:
            copied = tmp / source_path.name
            copied.write_text(source_path.read_text())
            copied_sources.append(copied)

        module_file = tmp / f"{module_name}.f90"
        module_file.write_text(module_source)
        driver_file = tmp / f"{module_name}_driver.f90"
        driver_file.write_text(driver_source)
        exe_file = tmp / f"{module_name}_driver.exe"

        compile_proc = subprocess.run(
            ["gfortran", "-std=f2018", *(str(path) for path in copied_sources), str(module_file), str(driver_file), "-o", str(exe_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compile_proc.returncode == 0, compile_proc.stderr

        fd, saved_exe_name = tempfile.mkstemp(suffix=".exe", prefix=f"{module_name}_")
        os.close(fd)
        saved_exe = Path(saved_exe_name)
        try:
            saved_exe.write_bytes(exe_file.read_bytes())
            yield saved_exe
        finally:
            saved_exe.unlink(missing_ok=True)


def run_command(command: list[str]) -> str:
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout
