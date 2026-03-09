"""CLI entrypoint scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from .compiler import Compiler
from .frontends.python_numpy import PythonNumpyFrontend


def main() -> int:
    parser = argparse.ArgumentParser(description="Array Compiler")
    parser.add_argument("input", help="source file")
    parser.add_argument("--from-lang", choices=["python"], default="python")
    parser.add_argument("--to-lang", choices=["fortran"], default="fortran")
    parser.add_argument("--out", help="output file")
    args = parser.parse_args()

    input_path = Path(args.input)
    source = input_path.read_text(encoding="utf-8-sig")

    if args.from_lang != "python" or args.to_lang != "fortran":
        raise NotImplementedError("Only the Python -> Fortran path is scaffolded so far")

    frontend = PythonNumpyFrontend()
    module = frontend.lower_source(source, module_name=input_path.stem)
    output = Compiler().emit_fortran(module)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
