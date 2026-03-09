from array_compiler.corpus import configured_cases, configured_roots, discover_language_files


def test_corpus_root_exists() -> None:
    roots = configured_roots()
    root = roots["pure_fortran_examples"]
    assert root.path.exists()


def test_selected_cases_exist() -> None:
    cases = configured_cases()
    assert cases
    for case in cases:
        assert case.path.exists(), f"missing corpus case: {case.path}"


def test_can_discover_python_examples_from_python_numpy_examples_1() -> None:
    roots = configured_roots()
    root = roots["pure_fortran_examples"].path / "python_numpy_examples_1"
    found = discover_language_files(root, (".py",))
    assert found
    assert any(path.name == "t004_reshape_ravel_flatten.py" for path in found)
