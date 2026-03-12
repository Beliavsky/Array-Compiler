from __future__ import annotations

import argparse
from pathlib import Path

from array_compiler.pychecker import PythonChecker


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Python type/rank/Final annotations for consistency.")
    parser.add_argument("input_py", help="Python source file to check")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="Emit diagnostics as JSON")
    parser.add_argument("--check-intent", action="store_true", help="Report conservative parameter intent notes")
    parser.add_argument("--fortran-preflight", action="store_true", help="Warn about parameters that remain untyped for translation")
    parser.add_argument(
        "--fix-type-changes",
        action="store_true",
        help="Conservatively rewrite simple sequential type changes to a fresh variable name",
    )
    parser.add_argument("--output", help="Write fixed output to a separate file instead of modifying the input")
    args = parser.parse_args()

    input_path = Path(args.input_py)
    source = input_path.read_text(encoding="utf-8-sig")
    checker = PythonChecker()
    if args.fix_type_changes:
        fixed_source, diagnostics = checker.fix_source(source, fix_type_changes=True)
        target = Path(args.output) if args.output else input_path
        if fixed_source != source:
            target.write_text(fixed_source, encoding="utf-8")
            print(f"wrote {target}")
    else:
        diagnostics = checker.check_source(
            source,
            check_intent=args.check_intent,
            fortran_preflight=args.fortran_preflight,
        )

    if args.json:
        print(checker.format_json(diagnostics), end="")
    else:
        print(checker.format_human(diagnostics), end="")

    has_error = any(diag.level == "error" for diag in diagnostics)
    has_warning = any(diag.level == "warning" for diag in diagnostics)
    return 1 if has_error or (args.strict and has_warning) else 0


if __name__ == "__main__":
    raise SystemExit(main())
