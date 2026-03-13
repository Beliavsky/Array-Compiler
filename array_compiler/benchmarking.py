from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
import tomllib


DEFAULT_BENCHMARK_CONFIG_PATH = Path(__file__).resolve().parents[1] / "benchmarks.toml"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    path: Path
    repeats: int = 5
    warmups: int = 1
    baseline_fortran_seconds: float | None = None
    min_source_seconds_for_source_ratio: float = 1.0
    warn_if_fortran_slower_than_source: float = 1.5
    fail_if_fortran_slower_than_source: float = 2.5
    warn_if_fortran_slower_by_seconds: float = 5.0
    fail_if_fortran_slower_by_seconds: float = 15.0
    warn_if_fortran_slower_than_baseline: float = 1.10
    fail_if_fortran_slower_than_baseline: float = 1.35
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkIssue:
    scope: str
    status: str
    message: str


@dataclass(frozen=True)
class BenchmarkEvaluation:
    status: str
    issues: tuple[BenchmarkIssue, ...]


def load_benchmark_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_BENCHMARK_CONFIG_PATH
    with path.open("rb") as handle:
        return tomllib.load(handle)


def configured_benchmarks(config_path: Path | None = None) -> list[BenchmarkCase]:
    path = config_path or DEFAULT_BENCHMARK_CONFIG_PATH
    config = load_benchmark_config(path)
    base_dir = path.resolve().parent
    cases: list[BenchmarkCase] = []
    for entry in config.get("benchmark", []):
        case_path = Path(entry["path"])
        if not case_path.is_absolute():
            case_path = (base_dir / case_path).resolve()
        cases.append(
            BenchmarkCase(
                name=entry["name"],
                path=case_path,
                repeats=int(entry.get("repeats", 5)),
                warmups=int(entry.get("warmups", 1)),
                baseline_fortran_seconds=float(entry["baseline_fortran_seconds"])
                if "baseline_fortran_seconds" in entry
                else None,
                min_source_seconds_for_source_ratio=float(entry.get("min_source_seconds_for_source_ratio", 1.0)),
                warn_if_fortran_slower_than_source=float(entry.get("warn_if_fortran_slower_than_source", 1.5)),
                fail_if_fortran_slower_than_source=float(entry.get("fail_if_fortran_slower_than_source", 2.5)),
                warn_if_fortran_slower_by_seconds=float(entry.get("warn_if_fortran_slower_by_seconds", 5.0)),
                fail_if_fortran_slower_by_seconds=float(entry.get("fail_if_fortran_slower_by_seconds", 15.0)),
                warn_if_fortran_slower_than_baseline=float(entry.get("warn_if_fortran_slower_than_baseline", 1.10)),
                fail_if_fortran_slower_than_baseline=float(entry.get("fail_if_fortran_slower_than_baseline", 1.35)),
                tags=tuple(entry.get("tags", [])),
            )
        )
    return cases


def median_seconds(samples: list[float]) -> float:
    if not samples:
        raise ValueError("median_seconds requires at least one sample")
    return float(statistics.median(samples))


def evaluate_benchmark_case(case: BenchmarkCase, *, source_seconds: float, fortran_seconds: float) -> BenchmarkEvaluation:
    issues: list[BenchmarkIssue] = []

    if case.baseline_fortran_seconds is not None and case.baseline_fortran_seconds > 0.0:
        baseline_ratio = fortran_seconds / case.baseline_fortran_seconds
        if baseline_ratio > case.fail_if_fortran_slower_than_baseline:
            issues.append(
                BenchmarkIssue(
                    scope="baseline",
                    status="fail",
                    message=(
                        f"Fortran median {fortran_seconds:.6f}s is {baseline_ratio:.3f}x slower "
                        f"than baseline {case.baseline_fortran_seconds:.6f}s"
                    ),
                )
            )
        elif baseline_ratio > case.warn_if_fortran_slower_than_baseline:
            issues.append(
                BenchmarkIssue(
                    scope="baseline",
                    status="warn",
                    message=(
                        f"Fortran median {fortran_seconds:.6f}s is {baseline_ratio:.3f}x slower "
                        f"than baseline {case.baseline_fortran_seconds:.6f}s"
                    ),
                )
            )

    if source_seconds >= case.min_source_seconds_for_source_ratio and source_seconds > 0.0 and fortran_seconds > source_seconds:
        source_ratio = fortran_seconds / source_seconds
        source_gap = fortran_seconds - source_seconds
        if source_ratio > case.fail_if_fortran_slower_than_source or source_gap > case.fail_if_fortran_slower_by_seconds:
            issues.append(
                BenchmarkIssue(
                    scope="source",
                    status="fail",
                    message=(
                        f"Fortran median {fortran_seconds:.6f}s is {source_ratio:.3f}x slower than source "
                        f"{source_seconds:.6f}s (gap {source_gap:.6f}s)"
                    ),
                )
            )
        elif source_ratio > case.warn_if_fortran_slower_than_source or source_gap > case.warn_if_fortran_slower_by_seconds:
            issues.append(
                BenchmarkIssue(
                    scope="source",
                    status="warn",
                    message=(
                        f"Fortran median {fortran_seconds:.6f}s is {source_ratio:.3f}x slower than source "
                        f"{source_seconds:.6f}s (gap {source_gap:.6f}s)"
                    ),
                )
            )

    status = "ok"
    if any(issue.status == "fail" for issue in issues):
        status = "fail"
    elif issues:
        status = "warn"
    return BenchmarkEvaluation(status=status, issues=tuple(issues))
