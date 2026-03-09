import os
from pathlib import Path
import re
import subprocess
import tempfile
from contextlib import contextmanager

from array_compiler.backends.fortran.helpers import HelperRegistry


HELPER_MODULE_KEYS = {
    "kind_mod": "kind_mod",
    "ac_string_mod": "ac_string",
    "ac_random_support": "ac_random",
    "ac_numpy_mod": "ac_numpy",
}

HELPER_ORDER = ["kind_mod", "ac_string", "ac_random", "ac_numpy", "lapack_d", "python", "octave_funcs", "r"]
HELPER_DEPENDENCIES = {
    "ac_numpy": {"ac_random"},
}


def helper_keys_for_source(fortran_source: str) -> list[str]:
    used_modules = {
        match.group(1).lower()
        for match in re.finditer(r"^\s*use\s+([a-z][a-z0-9_]*)\b", fortran_source, flags=re.IGNORECASE | re.MULTILINE)
    }
    helper_keys = {HELPER_MODULE_KEYS[module] for module in used_modules if module in HELPER_MODULE_KEYS}
    changed = True
    while changed:
        changed = False
        for key in list(helper_keys):
            for dependency in HELPER_DEPENDENCIES.get(key, set()):
                if dependency not in helper_keys:
                    helper_keys.add(dependency)
                    changed = True
    return [key for key in HELPER_ORDER if key in helper_keys]


def assert_max_fortran_line_length(source: str, max_len: int = 80) -> None:
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
        registry = HelperRegistry()
        copied_sources: list[Path] = []
        copied_names: set[str] = set()
        for key in helper_keys_for_source(module_source):
            source_path = registry.path_for(key)
            if source_path.name in copied_names:
                continue
            copied = tmp / source_path.name
            copied.write_text(source_path.read_text())
            copied_sources.append(copied)
            copied_names.add(source_path.name)

        for source_path in extra_sources or []:
            if source_path.name in copied_names:
                continue
            copied = tmp / source_path.name
            copied.write_text(source_path.read_text())
            copied_sources.append(copied)
            copied_names.add(source_path.name)

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
