from __future__ import annotations

import argparse
from pathlib import Path

from array_compiler.rchecker import RChecker


def main() -> int:
    parser = argparse.ArgumentParser(description="Warn about R double literals used where integer values are clearer.")
    parser.add_argument("input_r", help="R source file to check")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--json", action="store_true", help="Emit diagnostics as JSON")
    parser.add_argument("--fix", action="store_true", help="Rewrite safe cases such as 5 -> 5L in place")
    parser.add_argument(
        "--fix-type-changes",
        action="store_true",
        help="Conservatively rewrite simple sequential type changes to a fresh variable name",
    )
    parser.add_argument("--output", help="Write fixed output to a separate file instead of modifying the input")
    args = parser.parse_args()

    path = Path(args.input_r)
    source = path.read_text(encoding="utf-8-sig")
    checker = RChecker()

    if args.fix or args.fix_type_changes:
        fixed_source, diagnostics = checker.fix_source(source, fix_type_changes=args.fix_type_changes)
        target = Path(args.output) if args.output else path
        if fixed_source != source:
            target.write_text(fixed_source, encoding="utf-8")
            print(f"wrote {target}")
    else:
        diagnostics = checker.check_source(source)

    if args.json:
        print(checker.format_json(diagnostics), end="")
    else:
        print(checker.format_human(diagnostics), end="")

    has_warning = any(diag.level == "warning" for diag in diagnostics)
    return 1 if args.strict and has_warning else 0


if __name__ == "__main__":
    raise SystemExit(main())
