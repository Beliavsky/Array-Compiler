from __future__ import annotations

import argparse
import difflib
import hashlib
import locale
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from array_compiler.compiler import Compiler
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from array_compiler.frontends.r import RSubsetFrontend
from array_compiler.backends.fortran.helpers import HelperRegistry
from array_compiler.fortran_style import STYLE_LEVELS
from array_compiler.ir.nodes import Function


HELPER_MODULE_KEYS = {
    "kind_mod": "kind_mod",
    "ac_constants_mod": "ac_constants",
    "ac_stats_mod": "ac_stats",
    "ac_string_mod": "ac_string",
    "ac_random_support": "ac_random",
    "ac_numpy_mod": "ac_numpy",
    "ac_lapack_mod": "ac_lapack",
}

HELPER_ORDER = ["kind_mod", "ac_constants", "ac_stats", "ac_string", "ac_random", "ac_numpy", "ac_lapack", "lapack_d", "python", "octave_funcs", "r"]
HELPER_DEPENDENCIES = {
    "ac_stats": {"ac_constants"},
    "ac_numpy": {"ac_random"},
    "ac_lapack": {"lapack_d"},
}
HELPER_MOD_FILES = {
    "kind_mod": "kind_mod.mod",
    "ac_constants": "ac_constants_mod.mod",
    "ac_stats": "ac_stats_mod.mod",
    "ac_string": "ac_string_mod.mod",
    "ac_random": "ac_random_support.mod",
    "ac_numpy": "ac_numpy_mod.mod",
}
HELPER_CACHE_FORMAT_VERSION = "v4"

DEFAULT_GFORTRAN_COMPILER = "gfortran -O3 -march=native -flto -Wfatal-errors -Werror"
DEFAULT_IFX_COMPILER_WINDOWS = "ifx /O3"
DEFAULT_IFX_COMPILER_POSIX = "ifx -O3"


def quote_cmd_arg(value: str) -> str:
    return subprocess.list2cmdline([value])


def format_command(parts: list[str]) -> str:
    return " ".join(quote_cmd_arg(part) for part in parts)


def _decode_output(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    for encoding in ("utf-8", locale.getpreferredencoding(False) or "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def emit_text(text: str) -> None:
    if not text:
        return
    encoding = getattr(sys.stdout, "encoding", None) or locale.getpreferredencoding(False) or "utf-8"
    try:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode(encoding, errors="replace"))
        if not text.endswith("\n"):
            sys.stdout.buffer.write(b"\n")


def run_capture(cmd: list[str], *, cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=False, check=False)
    return proc.returncode, _decode_output(proc.stdout), _decode_output(proc.stderr)


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
            f"use {module_name}, only: ac_init_argv, {entry_name}",
            "implicit none",
            "call ac_init_argv()",
            f"call {entry_name}()",
            f"end program {module_name}_driver",
            "",
        ]
    )


def emit_timing_summary(timings: dict[str, float], *, source_run_label: str = "python run") -> None:
    if not timings:
        return
    print("Timing summary (seconds):")
    base = timings.get(source_run_label)
    rows: list[tuple[str, float, float | None]] = []
    order = [source_run_label, "transpile", "compile", "fortran run", "total"]
    for name in order:
        if name in timings:
            value = timings[name]
            ratio = (value / base) if (base is not None and base > 0.0) else None
            rows.append((name, value, ratio))
    stage_header = "stage"
    seconds_header = "seconds"
    ratio_header = f"ratio(vs {source_run_label})"
    stage_width = max(len(stage_header), *(len(name) for name, _, _ in rows))
    seconds_width = max(len(seconds_header), 9)
    ratio_width = max(len(ratio_header), 22)
    print(f"  {stage_header:<{stage_width}}  {seconds_header:>{seconds_width}}  {ratio_header:>{ratio_width}}")
    for name, value, ratio in rows:
        ratio_text = f"{ratio:>{ratio_width}.6f}" if ratio is not None else " " * ratio_width
        print(f"  {name:<{stage_width}}  {value:>{seconds_width}.6f}  {ratio_text}")


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


def detect_source_language(input_path: Path) -> str:
    suffix = input_path.suffix
    if suffix == ".py":
        return "python"
    if suffix in {".r", ".R"}:
        return "r"
    raise ValueError(f"unsupported source suffix {suffix!r}; expected .py, .r, or .R")


def default_fortran_output_tag(input_path: Path) -> str:
    suffix = input_path.suffix.lower().lstrip(".")
    if suffix == "py":
        return "p"
    return suffix or "src"


def frontend_for_language(language: str):
    if language == "python":
        return PythonNumpyFrontend()
    if language == "r":
        return RSubsetFrontend()
    raise ValueError(f"unsupported source language {language!r}")


def transpile_source_to_fortran(input_path: Path, output_path: Path, *, style_level: str = "full"):
    source = input_path.read_text(encoding="utf-8-sig")
    module_name = output_path.stem
    frontend = frontend_for_language(detect_source_language(input_path))
    module = frontend.lower_source(source, module_name=module_name)
    fortran_source = Compiler().emit_fortran(module, style_level=style_level)
    output_path.write_text(fortran_source, encoding="utf-8")
    return module, fortran_source


def transpile_python_to_fortran(input_path: Path, output_path: Path, *, style_level: str = "full"):
    return transpile_source_to_fortran(input_path, output_path, style_level=style_level)


def source_run_command(input_path: Path) -> list[str]:
    language = detect_source_language(input_path)
    if language == "python":
        return [sys.executable, str(input_path)]
    if language == "r":
        return ["Rscript", str(input_path)]
    raise ValueError(f"unsupported source language {language!r}")


def default_ifx_compiler() -> str:
    return DEFAULT_IFX_COMPILER_WINDOWS if os.name == "nt" else DEFAULT_IFX_COMPILER_POSIX


def resolve_compiler_command(*, compiler: str | None, ifx: bool) -> str:
    if compiler:
        return compiler
    if ifx:
        return default_ifx_compiler()
    return DEFAULT_GFORTRAN_COMPILER


def compiler_kind(compiler_parts: list[str]) -> str:
    executable = Path(compiler_parts[0]).name.lower() if compiler_parts else ""
    if executable.startswith("ifx"):
        return "ifx"
    return "gfortran"


def helper_cache_dir(repo_root: Path, compiler_parts: list[str]) -> Path:
    compiler_key = hashlib.sha256(
        "\0".join([HELPER_CACHE_FORMAT_VERSION, *compiler_parts]).encode("utf-8")
    ).hexdigest()[:16]
    return repo_root / ".array_compiler_cache" / "fortran" / compiler_key


def helper_cache_root(repo_root: Path) -> Path:
    return repo_root / ".array_compiler_cache"


def helper_object_suffix(kind: str) -> str:
    if kind == "ifx" and os.name == "nt":
        return ".obj"
    return ".o"


def helper_compile_command(
    *,
    compiler_parts: list[str],
    source_path: Path,
    cache_dir: Path,
    object_path: Path,
) -> list[str]:
    kind = compiler_kind(compiler_parts)
    if kind == "ifx" and os.name == "nt":
        return [
            *compiler_parts,
            "/c",
            f"/module:{cache_dir}",
            f"/object:{object_path}",
            f"/I{cache_dir}",
            str(source_path),
        ]
    return [*compiler_parts, "-c", str(source_path), "-J", str(cache_dir), "-o", str(object_path)]


def build_command(
    *,
    compiler_parts: list[str],
    cache_dir: Path,
    helper_objects: list[Path],
    output_path: Path,
    driver_path: Path | None,
    exe_path: Path,
    compile_only: bool,
) -> list[str]:
    kind = compiler_kind(compiler_parts)
    if kind == "ifx" and os.name == "nt":
        cmd = [*compiler_parts, f"/I{cache_dir}", *(str(path) for path in helper_objects), str(output_path)]
        if driver_path is not None:
            cmd.extend([str(driver_path), f"/exe:{exe_path}"])
        else:
            cmd.append("/c")
        return cmd
    cmd = [*compiler_parts, "-I", str(cache_dir), *(str(path) for path in helper_objects), str(output_path)]
    if driver_path is not None:
        cmd += [str(driver_path), "-o", str(exe_path)]
    else:
        cmd.insert(len(compiler_parts), "-c")
    return cmd


def ensure_cached_helpers(
    helper_keys: list[str],
    helper_registry: HelperRegistry,
    compiler_parts: list[str],
    cache_dir: Path,
) -> tuple[list[Path], list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_objects: list[Path] = []
    rebuilt: list[str] = []
    object_suffix = helper_object_suffix(compiler_kind(compiler_parts))
    for key in helper_keys:
        source_path = helper_registry.path_for(key)
        object_path = cache_dir / f"{source_path.stem}{object_suffix}"
        module_path = cache_dir / HELPER_MOD_FILES.get(key, f"{source_path.stem}.mod")
        needs_rebuild = (
            not object_path.exists()
            or (source_path.stat().st_mtime > object_path.stat().st_mtime)
            or not module_path.exists()
        )
        if needs_rebuild:
            compile_cmd = helper_compile_command(
                compiler_parts=compiler_parts,
                source_path=source_path,
                cache_dir=cache_dir,
                object_path=object_path,
            )
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
    parser = argparse.ArgumentParser(description="Translate Python or restricted R to Fortran with optional compile/run timing")
    parser.add_argument("input_py", nargs="?", help="Source file (.py, .r, .R)")
    parser.add_argument("--out", help="Output Fortran file")
    parser.add_argument("--compile", action="store_true", help="Compile generated Fortran")
    parser.add_argument("--run", action="store_true", help="Compile and run generated Fortran")
    parser.add_argument("--run-both", action="store_true", help="Run Python, transpile, compile, and run Fortran without timings or diff")
    parser.add_argument("--time-both", action="store_true", help="Run Python, transpile, compile, run Fortran, and print timings")
    parser.add_argument("--print-main", action="store_true", help="Print the generated temporary driver program")
    parser.add_argument("--tee", action="store_true", help="Print the generated Fortran source after transpilation")
    parser.add_argument("--compiler", help="Compiler command")
    parser.add_argument("--ifx", action="store_true", help="Use the Intel Fortran compiler preset")
    parser.add_argument("--clean-cache", action="store_true", help="Remove cached Fortran helper build artifacts and exit")
    parser.add_argument("--style-level", choices=STYLE_LEVELS, default="full", help="Fortran post-processing level")
    parser.add_argument("--no-style", action="store_true", help="Disable Fortran style post-processing")
    args = parser.parse_args()

    if args.no_style:
        args.style_level = "none"

    repo_root = Path(__file__).resolve().parent

    if args.clean_cache:
        cache_root = helper_cache_root(repo_root)
        if cache_root.exists():
            shutil.rmtree(cache_root)
            print(f"Removed cache: {cache_root}")
        else:
            print(f"Cache not present: {cache_root}")
        return 0

    if not args.input_py:
        parser.error("the following arguments are required: input_py")

    if args.run_both and args.time_both:
        parser.error("--run-both and --time-both cannot be used together")

    if args.time_both or args.run_both:
        args.run = True
        args.compile = True
    elif args.run:
        args.compile = True

    input_path = Path(args.input_py).resolve()
    source_language = detect_source_language(input_path)
    output_tag = default_fortran_output_tag(input_path)
    output_path = Path(args.out).resolve() if args.out else input_path.with_name(f"{input_path.stem}_{output_tag}.f90")
    source_run_label = "python run" if source_language == "python" else f"{source_language.upper()} run"
    timings: dict[str, float] = {}
    python_stdout = ""
    python_rc = 0

    if args.time_both or args.run_both:
        py_cmd = source_run_command(input_path)
        print(f"Run ({source_language}):", format_command(py_cmd))
        t0 = time.perf_counter()
        python_rc, python_stdout, python_stderr = run_capture(py_cmd, cwd=input_path.parent)
        timings[source_run_label] = time.perf_counter() - t0
        if python_rc != 0:
            print(f"Run ({source_language}): FAIL (exit {python_rc})")
            if python_stdout.strip():
                emit_text(python_stdout.rstrip())
            if python_stderr.strip():
                emit_text(python_stderr.rstrip())
            if args.time_both:
                emit_timing_summary(timings, source_run_label=source_run_label)
            return python_rc
        print(f"Run ({source_language}): PASS")
        if python_stdout.strip():
            emit_text(python_stdout.rstrip())
        if python_stderr.strip():
            emit_text(python_stderr.rstrip())

    t0 = time.perf_counter()
    try:
        module, fortran_source = transpile_source_to_fortran(input_path, output_path, style_level=args.style_level)
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
    compiler_command = resolve_compiler_command(compiler=args.compiler, ifx=args.ifx)
    compiler_parts = shlex.split(compiler_command)
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
        local_output_path = build_dir / output_path.name
        exe_path = output_path.with_suffix(".exe")
        local_output_path.write_text(fortran_source, encoding="utf-8")
        driver_source = ""
        if entry_name:
            driver_source = build_driver_source(output_path.stem, entry_name)
            driver_path.write_text(driver_source, encoding="utf-8")
            if args.print_main:
                print(driver_source.rstrip())
        elif args.print_main:
            print("No generated main program.")

        build_cmd = build_command(
            compiler_parts=compiler_parts,
            cache_dir=cache_dir,
            helper_objects=helper_objects,
            output_path=local_output_path,
            driver_path=(driver_path if entry_name else None),
            exe_path=exe_path,
            compile_only=not bool(entry_name),
        )
        t0 = time.perf_counter()
        compile_proc = subprocess.run(build_cmd, cwd=build_dir, capture_output=True, text=True, check=False)
        timings["compile"] = time.perf_counter() - t0
        print("Build:", format_command(build_cmd))
        if rebuilt_helpers:
            print("Rebuilt helpers:", " ".join(rebuilt_helpers))
        if compile_proc.returncode != 0:
            print(f"Build: FAIL (exit {compile_proc.returncode})")
            if compile_proc.stdout.strip():
                emit_text(compile_proc.stdout.rstrip())
            if compile_proc.stderr.strip():
                emit_text(compile_proc.stderr.rstrip())
            return compile_proc.returncode
        print("Build: PASS")

        if args.run:
            t0 = time.perf_counter()
            run_rc, fortran_stdout, fortran_stderr = run_capture([str(exe_path)], cwd=input_path.parent)
            timings["fortran run"] = time.perf_counter() - t0
            if run_rc != 0:
                print(f"Run: FAIL (exit {run_rc})")
                if fortran_stdout.strip():
                    emit_text(fortran_stdout.rstrip())
                if fortran_stderr.strip():
                    emit_text(fortran_stderr.rstrip())
                if args.time_both:
                    emit_timing_summary(timings, source_run_label=source_run_label)
                return run_rc
            print("Run: PASS")
            if fortran_stdout.strip():
                emit_text(fortran_stdout.rstrip())
            if fortran_stderr.strip():
                emit_text(fortran_stderr.rstrip())
            if args.time_both:
                compare_outputs(python_stdout, fortran_stdout, stochastic=("ac_random" in helper_keys))

    if args.time_both:
        timings["total"] = timings.get("transpile", 0.0) + timings.get("compile", 0.0) + timings.get("fortran run", 0.0)
        emit_timing_summary(timings, source_run_label=source_run_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
