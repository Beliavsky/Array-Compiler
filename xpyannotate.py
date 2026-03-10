from __future__ import annotations

import argparse
from pathlib import Path

from array_compiler.annotator import PythonAnnotator


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_annotated{input_path.suffix}")


def default_warnings_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".warnings.txt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate Python source with conservative type/rank/Final hints.")
    parser.add_argument("input_py", help="Python source file to annotate")
    parser.add_argument("--out", help="Output annotated Python path")
    parser.add_argument("--warnings-out", help="Optional warnings output path")
    parser.add_argument("--no-infer-intent", action="store_true", help="Do not report conservative parameter intent analysis")
    args = parser.parse_args()

    input_path = Path(args.input_py)
    output_path = Path(args.out) if args.out else default_output_path(input_path)
    warnings_path = Path(args.warnings_out) if args.warnings_out else default_warnings_path(output_path)

    source = input_path.read_text(encoding="utf-8-sig")
    annotated, warnings = PythonAnnotator().annotate_source(source, infer_intent=not args.no_infer_intent)
    output_path.write_text(annotated, encoding="utf-8")
    print(f"wrote {output_path}")

    if warnings:
        warnings_text = "\n".join(f"line {warning.line}: {warning.message}" for warning in warnings) + "\n"
        warnings_path.write_text(warnings_text, encoding="utf-8")
        print(f"wrote {warnings_path}")
        for warning in warnings:
            print(f"warning: line {warning.line}: {warning.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
