# Array Compiler

Compiler/transpiler framework for translating array-oriented scientific code into modern Fortran first, with C++ planned later as a second backend.

## Status

The repository now has three practical command-line paths:

- `x2f.py`: translate Python/NumPy and a restricted R subset to modern Fortran, with optional compile/run/timing support
- `x2f_bench.py`: benchmark source runtimes against translated Fortran runtimes using repeated runs, medians, configurable slowdown thresholds, and aggregate timing summaries
- `x2p.py`: translate a restricted R subset to Python/NumPy, either as standalone runnable Python or as the narrower typed subset used by `x2f.py`

The current architecture is still Fortran-first, but the repository now also includes:

- a reusable normalized array IR
- a Fortran runtime/helper registry
- batch drivers for corpus testing
- conservative source annotators and checkers for Python and R
- an R reducer for shrinking translation/runtime mismatches
- early support for comparing source-language output against translated output

The initial migration source is the [`Pure-Fortran`](https://github.com/beliavsky/pure-fortran) project.

## Current Capabilities

### `x2f.py`

`x2f.py` now accepts:

- `.py` for Python/NumPy input
- `.r` and `.R` for restricted R input

It can:

- emit Fortran source
- compile the generated Fortran with `gfortran`
- run the compiled executable
- run source and translated code side by side with `--run-both`
- time both sides with `--time-both`

Recent `x2f.py` work includes:

- initial `R -> Python/NumPy -> Python frontend -> Fortran` support
- stricter `gfortran` builds with `-Werror`
- source-suffix-based Fortran outputs such as `_p.f90` for Python and `_r.f90` for R
- argument passing from the generated Fortran driver into translated programs
- better formatted Fortran console output for translated R scripts
- working elapsed-time support for translated `proc.time()`-based R timer examples
- broader shared runtime support for translated R statistical and numeric helpers

### `x2p.py`

`x2p.py` is a standalone R-to-Python/NumPy translator.

It supports two modes:

- `--mode standalone`: emit runnable standalone Python/NumPy
- `--mode x2f`: emit the narrower typed Python subset intended to feed `x2f.py`

It can also:

- run the translated Python with `--run`
- run R and translated Python side by side with `--run-both`
- compare timings with `--time-both`
- print the relevant source line, including continued R statements, on translation failures

The current R frontend covers a useful numeric subset rather than full R. It includes enough support to run and batch-translate the `Pure-Fortran` `r_examples_1` corpus through `x2p.py`, and many of those scripts now also compile through `x2f.py`.

## Command-Line Tools

### Translation

- `x2f.py`: translate Python or restricted R to Fortran
- `x2f_batch.py`: batch-run `x2f.py` over many `.py`, `.r`, or `.R` files, with optional CSV output
- `x2f_bench.py`: benchmark selected cases from `benchmarks.toml` or ad hoc file/directory/glob inputs, report medians and speedups, flag baseline regressions, and flag suspicious cases where Fortran is much slower than the source program
- `x2p.py`: translate restricted R to Python/NumPy
- `x2p_batch.py`: batch-run `x2p.py` over many `.r` and `.R` files

### Annotation and Checking

- `xpyannotate.py`: add conservative Python type/rank/`Final` hints to runnable Python source
- `xpycheck.py`: check typed/transpiler-friendly Python for consistency issues
- `xrcheck.py`: check R source for patterns that are valid R but poor style for translation

### Reduction

- `xrreduce.py`: reduce an R script while preserving a differential failure such as "R runs, translation succeeds, generated Python fails"

### Repository Workflow

- `upload_to_github.bat`: stage curated project files, commit, and push without sweeping in generated `.exe`, `_p.py`, `_p.f90`, temp files, caches, or ad hoc local test artifacts
- `upload_core_to_github.bat`: stage a curated core-project set
- `upload_strict_source_to_github.bat`: stage only the strict compiler/runtime source whitelist, excluding tests, examples, docs, and generated files

## Annotation and Checker Notes

### `xpyannotate.py`

The annotator now does more than basic type hints. It can emit:

- conservative type and rank annotations
- `Final[...]` where the variable binding is not rebound
- generated parameter-intent comments such as `# xpyannotate: intent x=in, y=local-rebind`
- generated readonly-array comments such as `# xpyannotate: readonly-array`

Generated comments use the reserved `xpyannotate:` prefix so they can be distinguished from ordinary comments.

### `xpycheck.py`

The Python checker can report:

- annotation/type inconsistencies
- conservative parameter-intent diagnostics
- simple unstable variable-type changes

It also has a conservative `--fix-type-changes` mode for simple sequential cases where a new variable name can be introduced safely.

### `xrcheck.py`

The R checker warns about translation-hostile patterns such as:

- double literals used where integer values are clearer
- `T` and `F` instead of `TRUE` and `FALSE`
- bare `NA`
- numeric values used as logicals
- partial argument matching
- obviously incompatible recycling
- simple variable type changes

It can safely rewrite a subset of these cases with `--fix`, and it also has a conservative `--fix-type-changes` mode for simple sequential rewrites.

### `xrreduce.py`

The R reducer is aimed at debugging translation/runtime mismatches. It can shrink an R script while preserving a differential failure such as:

- `Rscript file.r` succeeds
- `x2p.py file.r` succeeds
- generated `file_p.py` fails

The current reducer starts with conservative structural passes and can optionally use slower line-based reduction.

## Layout

- `array_compiler/ir/`: normalized array IR and core compiler model
- `array_compiler/frontends/`: source-language frontends
- `array_compiler/backends/fortran/`: Fortran lowering and emission
- `array_compiler/runtime/fortran/`: runtime/helper modules used by generated Fortran
- `array_compiler/annotator.py`: Python annotation engine
- `array_compiler/pychecker.py`: Python checker logic
- `array_compiler/rchecker.py`: R checker logic
- `array_compiler/benchmarking.py`: benchmark config loading and slowdown evaluation
- `array_compiler/annotations.py`: runnable Python annotation aliases such as `Array1D[T]`
- `docs/`: architecture and migration notes
- `tests/`: regression coverage for frontends, backend, CLI tools, and helpers

## Example Commands

Translate Python to Fortran:

```bat
python x2f.py xccc_garch_fit.py --compile
```

Translate R to Fortran and run both source and translated code:

```bat
python x2f.py xbase_garch_moment_fit.r --run-both
```

Translate R to standalone Python:

```bat
python x2p.py xbase_garch_sim.r
```

Generate x2f-oriented Python from R:

```bat
python x2p.py xbase_garch_sim.r --mode x2f
```

Batch-translate an R corpus to x2f-oriented Python:

```bat
python x2p_batch.py c:\python\public_domain\github\Pure-Fortran-Examples\r_examples_1 --mode x2f --out-dir c:\python\Array-Compiler\tmp_r_x2f_py
```

Batch-run `x2f.py` over a directory of Python or R sources:

```bat
python x2f_batch.py c:\python\public_domain\github\Pure-Fortran-Examples\r_examples_1 --compile --csv r_compile_results.csv
```

Benchmark selected translation cases with repeated runs, median timings, and aggregate stage totals:

```bat
python x2f_bench.py --case xoptions_pde --repeats 3 --warmups 1
```

Benchmark ad hoc Python or R sources directly, including Windows-friendly glob patterns:

```bat
python x2f_bench.py x*.py --repeats 1 --warmups 0
```

Write a benchmark CSV report:

```bat
python x2f_bench.py --case xoptions_pde --csv bench_report.csv
```

Annotate Python source:

```bat
python xpyannotate.py xccc_garch_fit.py
```

Check Python source:

```bat
python xpycheck.py xccc_garch_fit_annotated.py --check-intent
```

Check and optionally rewrite R source:

```bat
python xrcheck.py xtimer.r --fix
```

Reduce an R differential failure:

```bat
python xrreduce.py xccc_garch_sim.r --output xccc_garch_sim_reduced.r --verbose
```

Curated commit/push:

```bat
upload_to_github.bat "update r and fortran translation docs"
```

Strict source-only commit/push:

```bat
upload_strict_source_to_github.bat "update core compiler sources"
```

## Near-Term Plan

1. Keep widening the Python subset accepted by the Python-to-Fortran frontend, especially where generated R-to-Python code depends on it.
2. Continue expanding the R subset with tests driven by real example corpora.
3. Move more language semantics out of ad hoc text rewrites and toward cleaner lowering into the shared IR.
4. Grow the shared Fortran runtime/helper modules for commonly used R statistical and numeric functions.
5. Preserve full-program translation, but keep library/module generation, benchmarking, and helper/runtime management as first-class features.
