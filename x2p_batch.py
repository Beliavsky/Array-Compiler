from __future__ import annotations

import argparse
import glob
from pathlib import Path

from x2p import translate_source_to_python


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
                for pattern in ("*.r", "*.R"):
                    for file_path in sorted(path.glob(pattern)):
                        key = str(file_path.resolve()).lower()
                        if key not in seen:
                            seen.add(key)
                            out.append(file_path)
                continue
            if path.suffix not in {".r", ".R"} or not path.exists():
                continue
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(path)
    return sorted(out, key=lambda p: str(p).lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate many R source files to Python.")
    parser.add_argument("inputs", nargs="+", help="R files, directories, or glob patterns")
    parser.add_argument("--out-dir", required=True, help="Directory to write translated Python files into")
    parser.add_argument("--mode", choices=["standalone", "x2f"], default="x2f", help="Emit standalone Python/NumPy or typed x2f-oriented Python")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many files (0 = no limit)")
    parser.add_argument("--stop-on-first-failure", action="store_true", help="Stop after the first failing file")
    args = parser.parse_args()

    r_files = expand_inputs(args.inputs)
    if not r_files:
        print("No R source files matched the provided inputs.")
        return 1
    if args.limit > 0:
        r_files = r_files[: args.limit]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_fail = 0
    total = len(r_files)
    for index, r_file in enumerate(r_files, start=1):
        print(f"[{index}/{total}] {r_file}")
        output_path = out_dir / f"{r_file.stem}_from_r.py"
        try:
            translate_source_to_python(r_file, output_path, standalone=args.mode == "standalone")
        except Exception as exc:
            n_fail += 1
            print(f"FAIL  {r_file}  {exc}")
            if args.stop_on_first_failure:
                break
            continue
        print(f"PASS  {output_path}")

    print("")
    print(f"Totals: {total if not args.stop_on_first_failure else index} files, {total - n_fail if not args.stop_on_first_failure else index - n_fail} pass, {n_fail} fail")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
