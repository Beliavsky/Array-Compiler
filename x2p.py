from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from array_compiler.frontends.r import RSubsetFrontend
from x2f import compare_outputs, emit_timing_summary, format_command, run_capture


def detect_source_language(input_path: Path) -> str:
    suffix = input_path.suffix
    if suffix in {".r", ".R"}:
        return "r"
    raise ValueError(f"unsupported source suffix {suffix!r}; expected .r or .R")


def translate_source_to_python(input_path: Path, output_path: Path, *, standalone: bool = True) -> str:
    source = input_path.read_text(encoding="utf-8-sig")
    language = detect_source_language(input_path)
    if language != "r":
        raise ValueError(f"unsupported source language {language!r}")
    python_source = RSubsetFrontend().translate_source(source, standalone=standalone)
    output_path.write_text(python_source, encoding="utf-8")
    return python_source


def source_run_command(input_path: Path) -> list[str]:
    return ["Rscript", str(input_path)]


def error_source_line(source: str, message: str) -> str | None:
    match = re.search(r"\bnear line (\d+)\b", message)
    if match is None:
        return None
    line_number = int(match.group(1))
    lines = source.splitlines()
    if not (1 <= line_number <= len(lines)):
        return None
    for start, end in statement_ranges(lines):
        if start <= line_number - 1 <= end:
            return "\n".join(lines[start : end + 1])
    return lines[line_number - 1]


def statement_ranges(lines: list[str]) -> list[tuple[int, int]]:
    if not lines:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    balance = 0
    for index, line in enumerate(lines):
        balance += paren_balance_delta(line)
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if balance <= 0 and not line_ends_with_continuation(line) and not line_starts_with_continuation(next_line):
            ranges.append((start, index))
            start = index + 1
            balance = 0
    if start < len(lines):
        ranges.append((start, len(lines) - 1))
    return ranges


def line_ends_with_continuation(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped:
        return False
    continuations = ("&&", "||", "<-", "=", "+", "-", "*", "/", "^", ",", "(", "[", "%*%", "%/%", "%%", "&", "|")
    return any(stripped.endswith(token) for token in continuations)


def line_starts_with_continuation(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    continuations = ("&&", "||", "+", "-", "*", "/", "^", ",", "%*%", "%/%", "%%", "&", "|")
    return any(stripped.startswith(token) for token in continuations)


def paren_balance_delta(line: str) -> int:
    delta = 0
    in_single = False
    in_double = False
    escape = False
    for ch in line:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch in "([":
            delta += 1
        elif ch in ")]":
            delta -= 1
    return delta


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate restricted R to standalone Python/NumPy")
    parser.add_argument("input", help="R source file (.r, .R)")
    parser.add_argument("--out", help="Output Python file")
    parser.add_argument("--mode", choices=["standalone", "x2f"], default="standalone", help="Emit standalone Python/NumPy or the typed x2f-oriented Python subset")
    parser.add_argument("--run", action="store_true", help="Run translated Python after writing it")
    parser.add_argument("--run-both", action="store_true", help="Run R source, translate, run Python, and compare outputs")
    parser.add_argument("--time-both", action="store_true", help="Run R source, translate, run Python, compare outputs, and print timings")
    parser.add_argument("--tee", action="store_true", help="Print the generated Python source after translation")
    args = parser.parse_args()

    if args.run_both and args.time_both:
        parser.error("--run-both and --time-both cannot be used together")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")

    output_path = Path(args.out).resolve() if args.out else input_path.with_name(f"{input_path.stem}_p.py")
    timings: dict[str, float] = {}
    source_stdout = ""

    if args.run_both or args.time_both:
        cmd = source_run_command(input_path)
        print("Run (r):", format_command(cmd))
        t0 = time.perf_counter()
        rc, stdout, stderr = run_capture(cmd, cwd=input_path.parent)
        timings["r run"] = time.perf_counter() - t0
        if rc != 0:
            print(f"Run (r): FAIL (exit {rc})")
            if stdout.strip():
                print(stdout.rstrip())
            if stderr.strip():
                print(stderr.rstrip())
            return rc
        print("Run (r): PASS")
        if stdout.strip():
            print(stdout.rstrip())
        if stderr.strip():
            print(stderr.rstrip())
        source_stdout = stdout

    source = input_path.read_text(encoding="utf-8-sig")
    t0 = time.perf_counter()
    try:
        language = detect_source_language(input_path)
        if language != "r":
            raise ValueError(f"unsupported source language {language!r}")
        python_source = RSubsetFrontend().translate_source(source, standalone=args.mode == "standalone")
        output_path.write_text(python_source, encoding="utf-8")
    except Exception as exc:
        timings["translate"] = time.perf_counter() - t0
        message = str(exc)
        print(f"Translate failed: {message}")
        source_line = error_source_line(source, message)
        if source_line is not None:
            print(source_line)
        if args.time_both:
            emit_timing_summary(timings)
        return 1
    timings["translate"] = time.perf_counter() - t0
    print(f"wrote {output_path.name}")

    if args.tee:
        print(python_source.rstrip())

    if not (args.run or args.run_both or args.time_both):
        if args.time_both:
            emit_timing_summary(timings)
        return 0

    cmd = [sys.executable, str(output_path)]
    print("Run (python):", format_command(cmd))
    t0 = time.perf_counter()
    rc, stdout, stderr = run_capture(cmd, cwd=output_path.parent)
    timings["python run"] = time.perf_counter() - t0
    if rc != 0:
        print(f"Run (python): FAIL (exit {rc})")
        if stdout.strip():
            print(stdout.rstrip())
        if stderr.strip():
            print(stderr.rstrip())
        if args.time_both:
            emit_timing_summary(timings)
        return rc
    print("Run (python): PASS")
    if stdout.strip():
        print(stdout.rstrip())
    if stderr.strip():
        print(stderr.rstrip())

    if args.run_both or args.time_both:
        compare_outputs(source_stdout, stdout)
    if args.time_both:
        timings["total"] = sum(timings.values())
        emit_timing_summary(timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
