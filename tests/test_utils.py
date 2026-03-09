def assert_max_fortran_line_length(source: str, max_len: int = 132) -> None:
    overlong = [
        (line_no, len(line), line)
        for line_no, line in enumerate(source.splitlines(), start=1)
        if len(line) > max_len
    ]
    assert not overlong, (
        f"found {len(overlong)} line(s) longer than {max_len}; "
        f"first is line {overlong[0][0]} length {overlong[0][1]}: {overlong[0][2]!r}"
    )
