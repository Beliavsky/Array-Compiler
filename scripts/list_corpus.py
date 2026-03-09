from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from array_compiler.corpus import configured_cases, configured_roots


def main() -> int:
    print("Configured corpora:")
    for root in configured_roots().values():
        print(f"- {root.name}: {root.path}")
    print()
    print("Selected cases:")
    for case in configured_cases():
        print(f"- [{case.language}] {case.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
