from __future__ import annotations

import argparse
import difflib
import hashlib
import math
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from array_compiler.compiler import Compiler
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from array_compiler.backends.fortran.helpers import HelperRegistry
from array_compiler.ir.nodes import Function


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
HELPER_MOD_FILES = {
    "kind_mod": "kind_mod.mod",
    "ac_string": "ac_string_mod.mod",
    "ac_random": "ac_random_support.mod",
    "ac_numpy": "ac_numpy_mod.mod",
}


def quote_cmd_arg(value: str) -> str:
    return subprocess.list2cmdline([value])


def format_command(parts: list[str]) -> str:
    return " ".join(quote_cmd_arg(part) for part in parts)


def run_capture(cmd: list[str], *, cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


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


def choose_run_entry(module_name: str, module) -> str:
    if module.program is not None:
        return module.program.name

    exported = set(module.exports)
    subroutines: list[Function] = [
        fn for fn in module.functions if fn.result_type is None and len(fn.args) == 0 and fn.name in exported
    ]
    for preferred in ("main", "run_example"):
        for fn in subroutines:
            if fn.name == preferred:
                return fn.name
    if subroutines:
        return subroutines[0].name
    raise RuntimeError(f"no runnable entry point found in translated module {module_name!r}")


def build_driver_source(module_name: str, entry_name: str) -> str:
    return "\n".join(
        [
            f"program {module_name}_driver",
            f"use {module_name}, only: {entry_name}",
            "implicit none",
            f"call {entry_name}()",
            f"end program {module_name}_driver",
            "",
        ]
    )


def emit_timing_summary(timings: dict[str, float]) -> None:
    if not timings:
        return
    print("Timing summary (seconds):")
    base = timings.get("python run")
    rows: list[tuple[str, float, float | None]] = []
    order = ["python run", "transpile", "compile", "fortran run", "total"]
    for name in order:
        if name in timings:
            value = timings[name]
            ratio = (value / base) if (base is not None and base > 0.0) else None
            rows.append((name, value, ratio))
    print("  stage         seconds    ratio(vs python run)")
    for name, value, ratio in rows:
        ratio_text = f"{ratio:>22.6f}" if ratio is not None else " " * 22
        print(f"  {name:<12} {value:>9.6f}{ratio_text}")


NUMERIC_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def _normalize_output_line(line: str) -> str:
    normalized = line.replace("True", "T").replace("False", "F")
    normalized = NUMERIC_TOKEN_RE.sub("#", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _numeric_tokens(line: str) -> list[float]:
    return [float(token) for token in NUMERIC_TOKEN_RE.findall(line)]


def _outputs_numerically_close(python_stdout: str, fortran_stdout: str, *, rel_tol: float = 1.0e-9, abs_tol: float = 1.0e-9) -> bool:
    py_lines = python_stdout.rstrip().splitlines()
    ft_lines = fortran_stdout.rstrip().splitlines()
    if len(py_lines) != len(ft_lines):
        return False
    for py_line, ft_line in zip(py_lines, ft_lines, strict=True):
        if _normalize_output_line(py_line) != _normalize_output_line(ft_line):
            return False
        py_numbers = _numeric_tokens(py_line)
        ft_numbers = _numeric_tokens(ft_line)
        if len(py_numbers) != len(ft_numbers):
            return False
        if any(not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(py_numbers, ft_numbers, strict=True)):
            return False
    return True


def _outputs_have_same_shape(python_stdout: str, fortran_stdout: str) -> bool:
    py_lines = [line for line in python_stdout.rstrip().splitlines() if line.strip()]
    ft_lines = [line for line in fortran_stdout.rstrip().splitlines() if line.strip()]
    if len(py_lines) != len(ft_lines):
        return False
    return all(_normalize_output_line(a) == _normalize_output_line(b) for a, b in zip(py_lines, ft_lines, strict=True))


def compare_outputs(python_stdout: str, fortran_stdout: str, *, stochastic: bool = False) -> bool:
    py_lines = python_stdout.rstrip().splitlines()
    ft_lines = fortran_stdout.rstrip().splitlines()
    if py_lines == ft_lines:
        print("Run diff: MATCH")
        return True
    if _outputs_numerically_close(python_stdout, fortran_stdout):
        print("Run diff: CLOSE")
        return True
    if stochastic and _outputs_have_same_shape(python_stdout, fortran_stdout):
        print("Run diff: APPROX")
        print("  numeric output differs, but line structure and labels match")
        return True
    if stochastic:
        print("Run diff: APPROX")
        print("  stochastic output differs; exact text comparison skipped")
        return True
    print("Run diff: DIFF")
    mismatch_index = next((i for i, (a, b) in enumerate(zip(py_lines, ft_lines), start=1) if a != b), None)
    if mismatch_index is None:
        mismatch_index = min(len(py_lines), len(ft_lines)) + 1
    py_line = py_lines[mismatch_index - 1] if mismatch_index - 1 < len(py_lines) else ""
    ft_line = ft_lines[mismatch_index - 1] if mismatch_index - 1 < len(ft_lines) else ""
    print(f"  first mismatch line: {mismatch_index}")
    print(f"  python : {py_line}")
    print(f"  fortran: {ft_line}")
    diff = list(difflib.unified_diff(py_lines, ft_lines, fromfile="python", tofile="fortran", lineterm=""))
    for line in diff[:12]:
        print(line)
    return False


def transpile_python_to_fortran(input_path: Path, output_path: Path):
    source = input_path.read_text(encoding="utf-8-sig")
    module_name = output_path.stem
    frontend = PythonNumpyFrontend()
    module = frontend.lower_source(source, module_name=module_name)
    fortran_source = Compiler().emit_fortran(module)
    output_path.write_text(fortran_source, encoding="utf-8")
    return module, fortran_source


def helper_cache_dir(repo_root: Path, compiler_parts: list[str]) -> Path:
    compiler_key = hashlib.sha256("\0".join(compiler_parts).encode("utf-8")).hexdigest()[:16]
    return repo_root / ".array_compiler_cache" / "fortran" / compiler_key


def ensure_cached_helpers(
    helper_keys: list[str],
    helper_registry: HelperRegistry,
    compiler_parts: list[str],
    cache_dir: Path,
) -> tuple[list[Path], list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_objects: list[Path] = []
    rebuilt: list[str] = []
    for key in helper_keys:
        source_path = helper_registry.path_for(key)
        object_path = cache_dir / f"{source_path.stem}.o"
        module_path = cache_dir / HELPER_MOD_FILES.get(key, f"{source_path.stem}.mod")
        needs_rebuild = (
            not object_path.exists()
            or (source_path.stat().st_mtime > object_path.stat().st_mtime)
            or not module_path.exists()
        )
        if needs_rebuild:
            compile_cmd = compiler_parts + ["-c", str(source_path), "-J", str(cache_dir), "-o", str(object_path)]
            compile_proc = subprocess.run(
                compile_cmd,
                cwd=cache_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if compile_proc.returncode != 0:
                message = compile_proc.stderr.strip() or compile_proc.stdout.strip() or "helper build failed"
                raise RuntimeError(f"failed to compile helper {source_path.name}: {message}")
            rebuilt.append(source_path.name)
        cached_objects.append(object_path)
    return cached_objects, rebuilt


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate Python to Fortran with optional compile/run timing")
    parser.add_argument("input_py", help="Python source file")
    parser.add_argument("--out", help="Output Fortran file")
    parser.add_argument("--compile", action="store_true", help="Compile generated Fortran")
    parser.add_argument("--run", action="store_true", help="Compile and run generated Fortran")
    parser.add_argument("--time-both", action="store_true", help="Run Python, transpile, compile, run Fortran, and print timings")
    parser.add_argument("--tee", action="store_true", help="Print the generated Fortran source after transpilation")
    parser.add_argument("--compiler", default="gfortran -O3 -march=native -flto -Wfatal-errors", help="Compiler command")
    args = parser.parse_args()

    if args.time_both:
        args.run = True
        args.compile = True
    elif args.run:
        args.compile = True

    input_path = Path(args.input_py).resolve()
    output_path = Path(args.out).resolve() if args.out else input_path.with_name(f"{input_path.stem}_p.f90")
    timings: dict[str, float] = {}
    python_stdout = ""
    python_rc = 0

    if args.time_both:
        py_cmd = [sys.executable, str(input_path)]
        print("Run (python):", format_command(py_cmd))
        t0 = time.perf_counter()
        python_rc, python_stdout, python_stderr = run_capture(py_cmd, cwd=input_path.parent)
        timings["python run"] = time.perf_counter() - t0
        if python_rc != 0:
            print(f"Run (python): FAIL (exit {python_rc})")
            if python_stdout.strip():
                print(python_stdout.rstrip())
            if python_stderr.strip():
                print(python_stderr.rstrip())
            emit_timing_summary(timings)
            return python_rc
        print("Run (python): PASS")
        if python_stdout.strip():
            print(python_stdout.rstrip())
        if python_stderr.strip():
            print(python_stderr.rstrip())

    t0 = time.perf_counter()
    try:
        module, fortran_source = transpile_python_to_fortran(input_path, output_path)
    except Exception as exc:  # pragmatic CLI surface for now
        timings["transpile"] = time.perf_counter() - t0
        print(f"Transpile failed: {exc}")
        return 1
    timings["transpile"] = time.perf_counter() - t0
    print(f"wrote {output_path.name}")
    if args.tee:
        print(fortran_source.rstrip())

    if not args.compile:
        return 0

    helper_registry = HelperRegistry()
    helper_keys = helper_keys_for_source(fortran_source)
    auto_added = [helper_registry.get(key).filename for key in helper_keys if key != "kind_mod"]
    compiler_parts = shlex.split(args.compiler)
    if len(compiler_parts) > 1:
        print("Compile options:", " ".join(compiler_parts[1:]))
    else:
        print("Compile options: <none>")
    if auto_added:
        print("Auto helper files:", " ".join(auto_added))

    entry_name = ""
    try:
        entry_name = choose_run_entry(output_path.stem, module)
    except RuntimeError:
        entry_name = ""

    repo_root = Path(__file__).resolve().parent
    cache_dir = helper_cache_dir(repo_root, compiler_parts)
    t0 = time.perf_counter()
    try:
        helper_objects, rebuilt_helpers = ensure_cached_helpers(helper_keys, helper_registry, compiler_parts, cache_dir)
    except RuntimeError as exc:
        timings["compile"] = time.perf_counter() - t0
        print(f"Build: FAIL ({exc})")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        build_dir = Path(tmpdir)
        driver_path = build_dir / f"{output_path.stem}_driver.f90"
        exe_path = output_path.with_suffix(".exe")
        if entry_name:
            driver_path.write_text(build_driver_source(output_path.stem, entry_name), encoding="utf-8")

        build_cmd = compiler_parts + ["-I", str(cache_dir)] + [*(str(path) for path in helper_objects), str(output_path)]
        if entry_name:
            build_cmd += [str(driver_path), "-o", str(exe_path)]
        else:
            build_cmd.insert(len(compiler_parts), "-c")
        t0 = time.perf_counter()
        compile_proc = subprocess.run(build_cmd, cwd=build_dir, capture_output=True, text=True, check=False)
        timings["compile"] = time.perf_counter() - t0
        print("Build:", format_command(build_cmd))
        if rebuilt_helpers:
            print("Rebuilt helpers:", " ".join(rebuilt_helpers))
        if compile_proc.returncode != 0:
            print(f"Build: FAIL (exit {compile_proc.returncode})")
            if compile_proc.stdout.strip():
                print(compile_proc.stdout.rstrip())
            if compile_proc.stderr.strip():
                print(compile_proc.stderr.rstrip())
            return compile_proc.returncode
        print("Build: PASS")

        if args.run:
            t0 = time.perf_counter()
            run_rc, fortran_stdout, fortran_stderr = run_capture([str(exe_path)], cwd=input_path.parent)
            timings["fortran run"] = time.perf_counter() - t0
            if run_rc != 0:
                print(f"Run: FAIL (exit {run_rc})")
                if fortran_stdout.strip():
                    print(fortran_stdout.rstrip())
                if fortran_stderr.strip():
                    print(fortran_stderr.rstrip())
                emit_timing_summary(timings)
                return run_rc
            print("Run: PASS")
            if fortran_stdout.strip():
                print(fortran_stdout.rstrip())
            if fortran_stderr.strip():
                print(fortran_stderr.rstrip())
            if args.time_both:
                compare_outputs(python_stdout, fortran_stdout, stochastic=("ac_random" in helper_keys))

    if args.time_both:
        timings["total"] = timings.get("transpile", 0.0) + timings.get("compile", 0.0) + timings.get("fortran run", 0.0)
        emit_timing_summary(timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
