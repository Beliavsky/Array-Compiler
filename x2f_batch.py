from __future__ import annotations

import argparse
import csv
import glob
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
X2F_PATH = REPO_ROOT / "x2f.py"


@dataclass
class CaseResult:
    source: str
    rc: int
    status: str
    outcome: str
    elapsed_s: float
    first_error: str


UNSUPPORTED_CALL_RE = re.compile(r"unsupported call: .*?attr='([A-Za-z_][A-Za-z0-9_]*)'")
UNSUPPORTED_EXPR_ATTR_RE = re.compile(r"unsupported expression: Attribute\(value=.*?attr='([A-Za-z_][A-Za-z0-9_]*)'")
COMPILE_ERROR_RE = re.compile(r"^\s*Error: (.+)$", flags=re.MULTILINE)


def has_glob_meta(text: str) -> bool:
    return any(ch in text for ch in "*?[]")


def expand_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for item in items:
        matches = glob.glob(item, recursive=True) if has_glob_meta(item) else [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                for file_path in sorted(path.glob("*.py")):
                    key = str(file_path.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        out.append(file_path)
                continue
            if path.suffix.lower() != ".py" or not path.exists():
                continue
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(path)
    return sorted(out, key=lambda p: str(p).lower())


def classify_outcome(stdout: str, stderr: str, rc: int) -> tuple[str, str]:
    text = f"{stdout}\n{stderr}"
    for line in text.splitlines():
        if line.startswith("Transpile failed:"):
            return "transpile_fail", line.strip()
        if line.startswith("Build: FAIL"):
            match = COMPILE_ERROR_RE.search(text)
            if match:
                return "compile_fail", f"Build: FAIL - {match.group(1).strip()}"
            return "compile_fail", line.strip()
        if line.startswith("Run: FAIL"):
            return "run_fail", line.strip()
    if rc == 0:
        return "pass", ""
    last_nonempty = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    return "other_fail", last_nonempty


def normalize_error(first_error: str) -> str:
    if not first_error:
        return ""
    if first_error.startswith("Transpile failed: unsupported call:"):
        match = UNSUPPORTED_CALL_RE.search(first_error)
        if match:
            return f"unsupported call: np.{match.group(1)}"
        return "unsupported call"
    if first_error.startswith("Transpile failed: unsupported expression:"):
        match = UNSUPPORTED_EXPR_ATTR_RE.search(first_error)
        if match:
            return f"unsupported expression: .{match.group(1)}"
        return "unsupported expression"
    if first_error.startswith("Transpile failed:"):
        return first_error.removeprefix("Transpile failed:").strip()
    if first_error.startswith("Build: FAIL"):
        return first_error
    if first_error.startswith("Run: FAIL"):
        return "Run: FAIL"
    return first_error


def print_grouped_failures(results: list[CaseResult]) -> None:
    groups: dict[str, list[str]] = {}
    for result in results:
        if result.rc == 0:
            continue
        key = normalize_error(result.first_error)
        groups.setdefault(key, []).append(result.source)
    if not groups:
        return
    print("")
    print("Failure groups:")
    grouped = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for key, files in grouped:
        print(f"{len(files):>3}  {key}")


def build_x2f_command(args, source: Path) -> list[str]:
    cmd = [sys.executable, str(X2F_PATH), str(source)]
    if args.compile:
        cmd.append("--compile")
    if args.run:
        cmd.append("--run")
    if args.run_both:
        cmd.append("--run-both")
    if args.time_both:
        cmd.append("--time-both")
    if args.tee:
        cmd.append("--tee")
    if args.ifx:
        cmd.append("--ifx")
    if args.compiler:
        cmd.extend(["--compiler", args.compiler])
    if args.no_style:
        cmd.append("--no-style")
    else:
        cmd.extend(["--style-level", args.style_level])
    return cmd


def write_csv(path: Path, results: list[CaseResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "status", "outcome", "rc", "elapsed_s", "first_error"])
        for result in results:
            writer.writerow(
                [
                    result.source,
                    result.status,
                    result.outcome,
                    result.rc,
                    f"{result.elapsed_s:.6f}",
                    result.first_error,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run x2f.py over many Python files.")
    parser.add_argument("inputs", nargs="+", help="Python files, directories, or glob patterns")
    parser.add_argument("--compile", action="store_true", help="Pass --compile to x2f.py")
    parser.add_argument("--run", action="store_true", help="Pass --run to x2f.py")
    parser.add_argument("--run-both", action="store_true", help="Pass --run-both to x2f.py")
    parser.add_argument("--time-both", action="store_true", help="Pass --time-both to x2f.py")
    parser.add_argument("--tee", action="store_true", help="Pass --tee to x2f.py")
    parser.add_argument("--compiler", help="Pass --compiler to x2f.py")
    parser.add_argument("--ifx", action="store_true", help="Pass --ifx to x2f.py")
    parser.add_argument("--style-level", choices=["none", "safe", "full"], default="full")
    parser.add_argument("--no-style", action="store_true", help="Pass --no-style to x2f.py")
    parser.add_argument("--stop-on-first-failure", action="store_true", help="Stop after the first failing file")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many files (0 = no limit)")
    parser.add_argument("--csv", help="Write a CSV report to this path")
    args = parser.parse_args()

    if args.run_both and args.time_both:
        parser.error("--run-both and --time-both cannot be used together")

    py_files = expand_inputs(args.inputs)
    if not py_files:
        print("No Python files matched the provided inputs.")
        return 1
    if args.limit > 0:
        py_files = py_files[: args.limit]

    total = len(py_files)
    results: list[CaseResult] = []

    for index, py_file in enumerate(py_files, start=1):
        print(f"[{index}/{total}] {py_file}")
        cmd = build_x2f_command(args, py_file)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        elapsed_s = time.perf_counter() - t0
        if proc.stdout.strip():
            print(proc.stdout.rstrip())
        if proc.stderr.strip():
            print(proc.stderr.rstrip())
        outcome, first_error = classify_outcome(proc.stdout or "", proc.stderr or "", proc.returncode)
        status = "PASS" if proc.returncode == 0 else "FAIL"
        results.append(
            CaseResult(
                source=str(py_file),
                rc=proc.returncode,
                status=status,
                outcome=outcome,
                elapsed_s=elapsed_s,
                first_error=first_error,
            )
        )
        if proc.returncode != 0 and args.stop_on_first_failure:
            print(f"Stopped on first failure: {py_file}")
            break
        if index < total:
            print("")

    print("")
    print("Summary:")
    for result in results:
        print(
            f"{result.status:<4}  {result.outcome:<15}  {result.elapsed_s:>8.3f} s  {result.source}"
        )
    n_pass = sum(1 for result in results if result.rc == 0)
    n_fail = len(results) - n_pass
    print(f"Totals: {len(results)} files, {n_pass} pass, {n_fail} fail")
    print_grouped_failures(results)

    if args.csv:
        csv_path = Path(args.csv)
        write_csv(csv_path, results)
        print(f"Wrote CSV: {csv_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
