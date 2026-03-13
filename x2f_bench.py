from __future__ import annotations

import argparse
import csv
import glob
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from array_compiler.backends.fortran.helpers import HelperRegistry
from array_compiler.benchmarking import (
    DEFAULT_BENCHMARK_CONFIG_PATH,
    BenchmarkCase,
    BenchmarkEvaluation,
    configured_benchmarks,
    evaluate_benchmark_case,
    median_seconds,
)
from x2f import (
    build_command,
    build_driver_source,
    choose_run_entry,
    compiler_kind,
    default_fortran_output_tag,
    ensure_cached_helpers,
    format_command,
    helper_cache_dir,
    helper_keys_for_source,
    resolve_compiler_command,
    run_capture,
    source_run_command,
    transpile_source_to_fortran,
)


@dataclass(frozen=True)
class BenchmarkRunResult:
    case: BenchmarkCase
    source_median: float | None
    fortran_median: float | None
    repeats_used: int
    warmups_used: int
    transpile_seconds: float
    compile_seconds: float
    source_run_seconds: float
    fortran_run_seconds: float
    speedup: float | None
    source_ok: bool
    executable_built: bool
    executable_run_ok: bool
    evaluation: BenchmarkEvaluation
    error: str = ""


def _has_glob_meta(text: str) -> bool:
    return any(ch in text for ch in "*?[]")


def _expand_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for item in items:
        matches = glob.glob(item, recursive=True) if _has_glob_meta(item) else [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                for pattern in ("*.py", "*.r", "*.R"):
                    for file_path in sorted(path.glob(pattern)):
                        key = str(file_path.resolve()).lower()
                        if key not in seen:
                            seen.add(key)
                            out.append(file_path.resolve())
                continue
            if path.suffix.lower() not in {".py", ".r"} or not path.exists():
                continue
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(path.resolve())
    return sorted(out, key=lambda p: str(p).lower())


def _ad_hoc_case(path: Path) -> BenchmarkCase:
    return BenchmarkCase(name=path.stem, path=path.resolve())


def _time_command(cmd: list[str], *, cwd: Path, repeats: int, warmups: int) -> tuple[list[float], float]:
    elapsed_total = 0.0
    for _ in range(max(warmups, 0)):
        t0 = time.perf_counter()
        rc, stdout, stderr = run_capture(cmd, cwd=cwd)
        elapsed_total += time.perf_counter() - t0
        if rc != 0:
            message = stderr.strip() or stdout.strip() or f"command failed with exit code {rc}"
            raise RuntimeError(message)
    samples: list[float] = []
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        rc, stdout, stderr = run_capture(cmd, cwd=cwd)
        elapsed = time.perf_counter() - t0
        if rc != 0:
            message = stderr.strip() or stdout.strip() or f"command failed with exit code {rc}"
            raise RuntimeError(message)
        samples.append(elapsed)
        elapsed_total += elapsed
    return samples, elapsed_total


def _compile_fortran_executable(
    case: BenchmarkCase,
    *,
    compiler: str | None,
    ifx: bool,
    style_level: str,
    repo_root: Path,
    work_dir: Path,
) -> tuple[Path, float, float]:
    input_path = case.path.resolve()
    output_tag = default_fortran_output_tag(input_path)
    output_path = work_dir / f"{input_path.stem}_{output_tag}.f90"

    t0 = time.perf_counter()
    module, fortran_source = transpile_source_to_fortran(input_path, output_path, style_level=style_level)
    transpile_seconds = time.perf_counter() - t0

    helper_registry = HelperRegistry()
    helper_keys = helper_keys_for_source(fortran_source)
    compiler_command = resolve_compiler_command(compiler=compiler, ifx=ifx)
    compiler_parts = shlex.split(compiler_command)
    cache_dir = helper_cache_dir(repo_root, compiler_parts)
    helper_objects, _ = ensure_cached_helpers(helper_keys, helper_registry, compiler_parts, cache_dir)

    entry_name = choose_run_entry(output_path.stem, module)
    local_output_path = work_dir / output_path.name
    local_output_path.write_text(fortran_source, encoding="utf-8")
    driver_path = work_dir / f"{output_path.stem}_driver.f90"
    driver_path.write_text(build_driver_source(output_path.stem, entry_name), encoding="utf-8")
    exe_suffix = ".exe" if compiler_kind(compiler_parts) in {"gfortran", "ifx"} else ""
    exe_path = work_dir / f"{output_path.stem}{exe_suffix}"

    build_cmd = build_command(
        compiler_parts=compiler_parts,
        cache_dir=cache_dir,
        helper_objects=helper_objects,
        output_path=local_output_path,
        driver_path=driver_path,
        exe_path=exe_path,
        compile_only=False,
    )
    t0 = time.perf_counter()
    proc = subprocess.run(build_cmd, cwd=work_dir, capture_output=True, text=True, check=False)
    compile_seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"build failed with exit code {proc.returncode}"
        raise RuntimeError(f"{format_command(build_cmd)}\n{message}")
    return exe_path, transpile_seconds, compile_seconds


def run_benchmark_case(
    case: BenchmarkCase,
    *,
    compiler: str | None,
    ifx: bool,
    style_level: str,
    repeats_override: int | None,
    warmups_override: int | None,
    repo_root: Path,
) -> BenchmarkRunResult:
    repeats = repeats_override if repeats_override is not None else case.repeats
    warmups = warmups_override if warmups_override is not None else case.warmups

    transpile_seconds = 0.0
    compile_seconds = 0.0
    source_run_seconds = 0.0
    fortran_run_seconds = 0.0
    source_ok = False
    executable_built = False
    executable_run_ok = False
    source_median: float | None = None
    fortran_median: float | None = None
    speedup: float | None = None
    evaluation = BenchmarkEvaluation(status="fail", issues=())
    error = ""

    with tempfile.TemporaryDirectory(prefix=f"x2f_bench_{case.name}_") as tmpdir:
        work_dir = Path(tmpdir)
        try:
            exe_path, transpile_seconds, compile_seconds = _compile_fortran_executable(
                case,
                compiler=compiler,
                ifx=ifx,
                style_level=style_level,
                repo_root=repo_root,
                work_dir=work_dir,
            )
            executable_built = True
            source_samples, source_run_seconds = _time_command(
                source_run_command(case.path), cwd=case.path.parent, repeats=repeats, warmups=warmups
            )
            source_ok = True
            source_median = median_seconds(source_samples)
            fortran_samples, fortran_run_seconds = _time_command(
                [str(exe_path)], cwd=case.path.parent, repeats=repeats, warmups=warmups
            )
            executable_run_ok = True
            fortran_median = median_seconds(fortran_samples)
            speedup = (source_median / fortran_median) if fortran_median > 0.0 else float("inf")
            evaluation = evaluate_benchmark_case(case, source_seconds=source_median, fortran_seconds=fortran_median)
        except Exception as exc:
            error = str(exc)
    return BenchmarkRunResult(
        case=case,
        source_median=source_median,
        fortran_median=fortran_median,
        repeats_used=max(repeats, 1),
        warmups_used=max(warmups, 0),
        transpile_seconds=transpile_seconds,
        compile_seconds=compile_seconds,
        source_run_seconds=source_run_seconds,
        fortran_run_seconds=fortran_run_seconds,
        speedup=speedup,
        source_ok=source_ok,
        executable_built=executable_built,
        executable_run_ok=executable_run_ok,
        evaluation=evaluation,
        error=error,
    )


def _write_csv(path: Path, results: list[BenchmarkRunResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "path",
                "source_median_s",
                "fortran_median_s",
                "speedup",
                "transpile_s",
                "compile_s",
                "status",
                "issues",
            ]
        )
        for result in results:
            issues = " | ".join(f"{issue.scope}:{issue.status}:{issue.message}" for issue in result.evaluation.issues)
            writer.writerow(
                [
                    result.case.name,
                    str(result.case.path),
                    "" if result.source_median is None else f"{result.source_median:.6f}",
                    "" if result.fortran_median is None else f"{result.fortran_median:.6f}",
                    "" if result.speedup is None else f"{result.speedup:.6f}",
                    f"{result.transpile_seconds:.6f}",
                    f"{result.compile_seconds:.6f}",
                    result.evaluation.status,
                    result.error or issues,
                ]
            )


def _print_aggregate_summary(
    results: list[BenchmarkRunResult],
    *,
    total_elapsed: float,
    repeats_override: int | None,
    warmups_override: int | None,
) -> None:
    sources_tested = len(results)
    source_ok = sum(1 for result in results if result.source_ok)
    transpile_attempted = len(results)
    compile_attempted = sum(1 for result in results if result.transpile_seconds > 0.0 or result.compile_seconds > 0.0 or result.executable_built)
    executables_built = sum(1 for result in results if result.executable_built)
    executables_run_ok = sum(1 for result in results if result.executable_run_ok)
    total_transpile = sum(result.transpile_seconds for result in results)
    total_compile = sum(result.compile_seconds for result in results)
    total_source_run = sum(result.source_run_seconds for result in results)
    matched_source_run = sum(result.source_run_seconds for result in results if result.executable_run_ok)
    total_fortran_run = sum(result.fortran_run_seconds for result in results)
    source_run_count = sum(1 for result in results if result.source_ok)
    matched_source_run_count = sum(1 for result in results if result.executable_run_ok)
    executable_run_count = sum(
        (result.repeats_used + result.warmups_used) for result in results if result.executable_run_ok
    )

    def _fmt_seconds(value: float) -> str:
        return f"{value:.3f}s"

    def _avg(total: float, count: int) -> str:
        return _fmt_seconds(total / count) if count > 0 else "n/a"

    print("Aggregate timing:")
    print(f"  repeats setting:              {repeats_override if repeats_override is not None else 'per-case config'}")
    print(f"  warmups setting:              {warmups_override if warmups_override is not None else 'per-case config'}")
    print(f"  sources tested:               {sources_tested}")
    print(f"  source runs without error:    {source_ok}")
    print(f"  executables built:            {executables_built}")
    print(f"  executable runs without error:{executables_run_ok:>5}")
    print(f"  {'metric':<28} {'count':>8} {'total':>14} {'avg/count':>14}")
    print(f"  {'total elapsed':<28} {sources_tested:>8} {_fmt_seconds(total_elapsed):>14} {_avg(total_elapsed, sources_tested):>14}")
    print(f"  {'transpile time':<28} {transpile_attempted:>8} {_fmt_seconds(total_transpile):>14} {_avg(total_transpile, transpile_attempted):>14}")
    print(f"  {'compile time':<28} {compile_attempted:>8} {_fmt_seconds(total_compile):>14} {_avg(total_compile, compile_attempted):>14}")
    print(f"  {'source run time':<28} {source_run_count:>8} {_fmt_seconds(total_source_run):>14} {_avg(total_source_run, source_run_count):>14}")
    print(f"  {'matched source run time':<28} {matched_source_run_count:>8} {_fmt_seconds(matched_source_run):>14} {_avg(matched_source_run, matched_source_run_count):>14}")
    print(f"  {'executable run time':<28} {executable_run_count:>8} {_fmt_seconds(total_fortran_run):>14} {_avg(total_fortran_run, executable_run_count):>14}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark source runtimes against transpiled Fortran runtimes.")
    parser.add_argument("inputs", nargs="*", help="Optional source files, directories, or glob patterns (.py, .r, .R)")
    parser.add_argument("--config", default=str(DEFAULT_BENCHMARK_CONFIG_PATH), help="Benchmark config TOML")
    parser.add_argument("--case", action="append", dest="cases", help="Benchmark case name to run (repeatable)")
    parser.add_argument("--compiler", help="Compiler command")
    parser.add_argument("--ifx", action="store_true", help="Use the Intel Fortran compiler preset")
    parser.add_argument("--repeats", type=int, help="Override repeats per case")
    parser.add_argument("--warmups", type=int, help="Override warmups per case")
    parser.add_argument("--style-level", choices=["none", "safe", "full"], default="full")
    parser.add_argument("--csv", help="Write CSV report to this path")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    selected_names = set(args.cases or [])
    cases: list[BenchmarkCase] = []
    seen_paths: set[str] = set()
    if args.inputs:
        for path in _expand_inputs(args.inputs):
            key = str(path.resolve()).lower()
            if key not in seen_paths:
                seen_paths.add(key)
                cases.append(_ad_hoc_case(path))
    all_cases = configured_benchmarks(config_path)
    include_config_cases = (not args.inputs) or bool(selected_names)
    if include_config_cases:
        for case in all_cases:
            if selected_names and case.name not in selected_names:
                continue
            key = str(case.path.resolve()).lower()
            if key not in seen_paths:
                seen_paths.add(key)
                cases.append(case)
    if not cases:
        print("No benchmark cases selected.")
        return 1

    repo_root = Path(__file__).resolve().parent
    results: list[BenchmarkRunResult] = []
    bench_start = time.perf_counter()
    for case in cases:
        print(f"[bench] {case.name}: {case.path}")
        result = run_benchmark_case(
            case,
            compiler=args.compiler,
            ifx=args.ifx,
            style_level=args.style_level,
            repeats_override=args.repeats,
            warmups_override=args.warmups,
            repo_root=repo_root,
        )
        results.append(result)
        print(
            f"  source median:  {('n/a' if result.source_median is None else f'{result.source_median:.3f}s')}\n"
            f"  fortran median: {('n/a' if result.fortran_median is None else f'{result.fortran_median:.3f}s')}\n"
            f"  speedup:        {('n/a' if result.speedup is None else f'{result.speedup:.3f}x')}\n"
            f"  transpile:      {result.transpile_seconds:.3f}s\n"
            f"  compile:        {result.compile_seconds:.3f}s\n"
            f"  status:         {result.evaluation.status}"
        )
        if result.error:
            print(f"    error: {result.error}")
        for issue in result.evaluation.issues:
            print(f"    {issue.scope} {issue.status}: {issue.message}")
        print("")

    if args.csv:
        csv_path = Path(args.csv).resolve()
        _write_csv(csv_path, results)
        print(f"Wrote CSV: {csv_path}")

    fail_count = sum(1 for result in results if result.evaluation.status == "fail")
    warn_count = sum(1 for result in results if result.evaluation.status == "warn")
    print(f"Totals: {len(results)} cases, {fail_count} fail, {warn_count} warn")
    _print_aggregate_summary(
        results,
        total_elapsed=time.perf_counter() - bench_start,
        repeats_override=args.repeats,
        warmups_override=args.warmups,
    )
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
