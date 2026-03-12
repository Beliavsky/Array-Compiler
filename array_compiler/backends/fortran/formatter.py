"""Formatting helpers for emitted free-form Fortran."""

from __future__ import annotations


def wrap_fortran_source(source: str, max_len: int = 80) -> str:
    """Wrap overlong free-form Fortran lines conservatively."""
    wrapped_lines: list[str] = []
    for line in source.splitlines():
        wrapped_lines.extend(_wrap_line(line, max_len=max_len))
    return "\n".join(wrapped_lines) + ("\n" if source.endswith("\n") else "")


def _wrap_line(line: str, max_len: int) -> list[str]:
    if len(line) <= max_len:
        return [line]

    stripped = line.lstrip()
    if not stripped or stripped.startswith("!") or stripped.startswith("#"):
        return [line]

    pieces: list[str] = []
    current = line
    continuation_prefix = "    & "
    available = max_len - len(continuation_prefix)

    while len(current) > max_len:
        previous = current
        split = _find_split_point(current, max_len - 3)
        if split is None:
            return [line]
        split_at, include_left = split
        if include_left:
            left = current[: split_at + 1].rstrip()
            right = current[split_at + 1 :].lstrip()
        else:
            left = current[:split_at].rstrip()
            right = current[split_at:].lstrip()
        pieces.append(f"{left} &")
        current = continuation_prefix + right
        if len(current) >= len(previous):
            return [line]
        if len(current) <= max_len:
            pieces.append(current)
            return pieces
        if available <= 20:
            return [line]

    pieces.append(current)
    return pieces


def _find_split_point(line: str, limit: int) -> tuple[int, bool] | None:
    in_single = False
    in_double = False
    best: tuple[int, int, bool] | None = None
    for idx, ch in enumerate(line):
        if idx > limit:
            break
        if ch == "'" and not in_double:
            if in_single and idx + 1 < len(line) and line[idx + 1] == "'":
                continue
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            if in_double and idx + 1 < len(line) and line[idx + 1] == '"':
                continue
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch.isspace():
            left_text = line[:idx].rstrip().lower()
            if left_text.endswith(".or.") or left_text.endswith(".and."):
                candidate = (3, idx, False)
            else:
                candidate = (1, idx, False)
            if best is None or candidate >= best:
                best = candidate
            continue
        if ch in ",)]":
            candidate = (2, idx, True)
            if best is None or candidate >= best:
                best = candidate
            continue
        if ch in "+-*":
            candidate = (0, idx, False)
            if best is None or candidate >= best:
                best = candidate
            continue
        if ch == "/" and not (
            (idx > 0 and line[idx - 1] == "/") or (idx + 1 < len(line) and line[idx + 1] == "/")
        ):
            candidate = (0, idx, False)
            if best is None or candidate >= best:
                best = candidate
    if best is None:
        return None
    return best[1], best[2]
