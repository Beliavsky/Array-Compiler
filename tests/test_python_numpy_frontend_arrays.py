from pathlib import Path
import re
import tempfile

import pytest

from array_compiler.backends.fortran.emitter import FortranBackend
from array_compiler.backends.fortran.helpers import HelperRegistry
from array_compiler.frontends.python_numpy import PythonNumpyFrontend
from array_compiler.ir.nodes import ArrayTypeRef, Assignment, Call, FieldAccess, If, RecordTypeRef, ScalarType
from test_utils import compiled_fortran_module_with_driver, run_command


MIXTURES_DIR = Path(r"c:\python\public_domain\github\Pure-Fortran-Examples\mixtures")


def test_frontend_lowers_numpy_array_constructors_and_metadata() -> None:
    source = """import numpy as np

def f(x):
    rng = np.random.default_rng(123)
    a = np.asarray(x, dtype=float)
    n, d = a.shape
    return a.size
"""
    module = PythonNumpyFrontend().lower_source(source, module_name="numpy_slice_mod")
    fn = module.functions[0]
    local_types = dict(fn.locals)

    assert local_types["rng"] == RecordTypeRef("ac_random_state")
    assert local_types["a"] == ArrayTypeRef(ScalarType.REAL64, rank=2)
    assert local_types["n"] == ScalarType.INTEGER
    assert local_types["d"] == ScalarType.INTEGER
    assert any(record.name == "ac_shape_2d" for record in module.records)

    assert isinstance(fn.body[0], Assignment)
    assert isinstance(fn.body[0].value, Call)
    assert fn.body[0].value.func == "ac_random_init"
    assert isinstance(fn.body[1], Assignment)
    assert isinstance(fn.body[1].value, Call)
    assert fn.body[1].value.func == "ac_asarray"
    assert isinstance(fn.body[2], If)
    assert isinstance(fn.body[2].body[1].value, Call)
    assert fn.body[2].body[1].value.func == "ac_shape_dim"


def test_xsim_mix_mv_now_lowers_through_the_frontend() -> None:
    source = (MIXTURES_DIR / "xsim_mix_mv.py").read_text(encoding="utf-8-sig")
    module = PythonNumpyFrontend().lower_source(source, module_name="xsim_mix_mv_mod")
    assert [fn.name for fn in module.functions] == ["simulate_mixture_mvnorm", "main"]
    assert module.program is not None
    emitted = FortranBackend().emit(module)
    assert "use ac_numpy_mod" in emitted


def test_xfit_mix_mv_now_lowers_through_the_frontend() -> None:
    source = (MIXTURES_DIR / "xfit_mix_mv.py").read_text(encoding="utf-8-sig")
    module = PythonNumpyFrontend().lower_source(source, module_name="xfit_mix_mv_mod")
    assert [fn.name for fn in module.functions] == ["fit_gmm_em", "main"]
    assert module.program is not None


def test_frontend_lowers_rng_choice_variants_used_by_mixture_examples() -> None:
    source = """import numpy as np

def f(weights, n, k):
    rng = np.random.default_rng(123)
    a = rng.choice(k, size=n, p=weights)
    b = rng.choice(n, size=k, replace=False)
    return 0
"""
    module = PythonNumpyFrontend().lower_source(source, module_name="choice_mod")
    fn = module.functions[0]
    local_types = dict(fn.locals)

    assert local_types["a"] == ArrayTypeRef(ScalarType.INTEGER)
    assert local_types["b"] == ArrayTypeRef(ScalarType.INTEGER)
    assert isinstance(fn.body[1], Assignment)
    assert isinstance(fn.body[1].value, Call)
    assert fn.body[1].value.func == "ac_choice_weighted"
    assert isinstance(fn.body[2], Assignment)
    assert isinstance(fn.body[2].value, Call)
    assert fn.body[2].value.func == "ac_choice_no_replace"


def test_ac_numpy_runtime_module_compiles() -> None:
    import subprocess
    import tempfile

    helper_registry = HelperRegistry()
    kind_path = helper_registry.path_for("kind_mod")
    random_path = helper_registry.path_for("ac_random")
    numpy_path = helper_registry.path_for("ac_numpy")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        kind_copy = tmp / kind_path.name
        random_copy = tmp / random_path.name
        numpy_copy = tmp / numpy_path.name
        kind_copy.write_text(kind_path.read_text())
        random_copy.write_text(random_path.read_text())
        numpy_copy.write_text(numpy_path.read_text())
        for source_file in (kind_copy, random_copy, numpy_copy):
            proc = subprocess.run(
                ["gfortran", "-std=f2018", str(source_file), "-c"],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 0, proc.stderr


def test_xsim_mix_mv_generated_fortran_compiles_and_runs() -> None:
    source = (MIXTURES_DIR / "xsim_mix_mv.py").read_text(encoding="utf-8-sig").replace("n = 10000", "n = 500", 1)
    module = PythonNumpyFrontend().lower_source(source, module_name="xsim_mix_mv_mod")
    emitted = FortranBackend().emit(module)
    helper_registry = HelperRegistry()
    extra_sources = [
        helper_registry.path_for("ac_random"),
        helper_registry.path_for("ac_numpy"),
    ]
    driver_source = "\n".join(
        [
            "program xsim_mix_mv_driver",
            "use xsim_mix_mv_mod, only: main",
            "implicit none",
            "call main('temp_xsim.txt')",
            "end program xsim_mix_mv_driver",
            "",
        ]
    )

    with compiled_fortran_module_with_driver(
        emitted,
        module_name="xsim_mix_mv_mod",
        driver_source=driver_source,
        extra_sources=extra_sources,
    ) as exe_file:
        stdout = run_command([str(exe_file)])

    assert "wrote data file:temp_xsim.txt" in stdout


def test_xfit_mix_mv_emitted_fortran_compiles() -> None:
    import subprocess
    import tempfile

    source = (MIXTURES_DIR / "xfit_mix_mv.py").read_text(encoding="utf-8-sig")
    module = PythonNumpyFrontend().lower_source(source, module_name="xfit_mix_mv_mod")
    emitted = FortranBackend().emit(module)
    helper_registry = HelperRegistry()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        copied_sources = []
        for key in ("kind_mod", "ac_random", "ac_numpy"):
            source_path = helper_registry.path_for(key)
            copied = tmp / source_path.name
            copied.write_text(source_path.read_text())
            copied_sources.append(copied)
        module_file = tmp / "xfit_mix_mv_mod.f90"
        module_file.write_text(emitted)
        proc = subprocess.run(
            ["gfortran", "-std=f2018", *(str(path) for path in copied_sources), str(module_file), "-c"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


def _parse_vector_payload(text: str, label: str) -> list[float]:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(label):
            payload = stripped[len(label) :].strip()
            return [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?|[-+]?\.\d+(?:[Ee][-+]?\d+)?", payload)]
    raise AssertionError(f"missing label {label!r}")


def _parse_scalar_payload(text: str, label: str) -> float:
    values = _parse_vector_payload(text, label)
    assert values, f"no numeric payload for {label!r}"
    return values[0]


def _parse_means_payload(text: str) -> list[float]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "means") + 1
    end = next(i for i, line in enumerate(lines[start:], start) if line.strip().startswith("covariances"))
    payload = "\n".join(lines[start:end])
    values = [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?|[-+]?\.\d+(?:[Ee][-+]?\d+)?", payload)]
    assert values, "missing means payload"
    return values


def _fortran_column_major_to_row_major(values: list[float], nrow: int, ncol: int) -> list[float]:
    assert len(values) == nrow * ncol
    row_major: list[float] = []
    for i in range(nrow):
        for j in range(ncol):
            row_major.append(values[j * nrow + i])
    return row_major


def test_xfit_mix_mv_generated_fortran_runs_close_to_python() -> None:
    xsim_source = (MIXTURES_DIR / "xsim_mix_mv.py").read_text(encoding="utf-8-sig").replace("n = 10000", "n = 3000", 1)
    xfit_source = (MIXTURES_DIR / "xfit_mix_mv.py").read_text(encoding="utf-8-sig")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        data_file = tmp / "xsim_mix_mv_data.txt"
        xsim_script = tmp / "xsim_mix_mv_small.py"
        xsim_script.write_text(xsim_source)
        run_command(["python", str(xsim_script), str(data_file)])

        python_stdout = run_command(["python", str(MIXTURES_DIR / "xfit_mix_mv.py"), str(data_file)])

        patched_xfit_source = xfit_source.replace(
            'data_file = "xsim_mix_mv_data.txt"',
            f'data_file = r"{str(data_file).replace("\\\\", "/")}"',
            1,
        )
        module = PythonNumpyFrontend().lower_source(patched_xfit_source, module_name="xfit_mix_mv_mod")
        emitted = FortranBackend().emit(module)
        helper_registry = HelperRegistry()
        extra_sources = [
            helper_registry.path_for("ac_random"),
            helper_registry.path_for("ac_numpy"),
        ]
        driver_source = "\n".join(
            [
                "program xfit_mix_mv_driver",
                "use xfit_mix_mv_mod, only: main",
                "implicit none",
                "call main()",
                "end program xfit_mix_mv_driver",
                "",
            ]
        )

        with compiled_fortran_module_with_driver(
            emitted,
            module_name="xfit_mix_mv_mod",
            driver_source=driver_source,
            extra_sources=extra_sources,
        ) as exe_file:
            fortran_stdout = run_command([str(exe_file)])

    python_iter = _parse_scalar_payload(python_stdout, "iterations")
    fortran_iter = _parse_scalar_payload(fortran_stdout, "iterations")
    python_ll = _parse_scalar_payload(python_stdout, "log_likelihood")
    fortran_ll = _parse_scalar_payload(fortran_stdout, "log_likelihood")
    python_weights = _parse_vector_payload(python_stdout, "weights")
    fortran_weights = _parse_vector_payload(fortran_stdout, "weights")
    python_means = _parse_means_payload(python_stdout)
    fortran_means = _fortran_column_major_to_row_major(_parse_means_payload(fortran_stdout), 3, 2)

    assert python_iter > 0
    assert 1 <= fortran_iter <= 200
    assert abs(fortran_ll - python_ll) < 5.0
    assert len(fortran_weights) == len(python_weights) == 3
    assert max(abs(a - b) for a, b in zip(fortran_weights, python_weights, strict=True)) < 2.0e-2
    assert len(fortran_means) == len(python_means) == 6
    assert max(abs(a - b) for a, b in zip(fortran_means, python_means, strict=True)) < 1.0e-1
