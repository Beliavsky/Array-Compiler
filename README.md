# Array Compiler

Compiler/transpiler framework for translating array-oriented scientific code into modern Fortran first, with C++ planned as a second backend.

## Status

This repository is the start of a shared architecture that replaces one-off pairwise transpilers with:

- language-specific frontends
- a shared normalized array IR
- a Fortran-first backend
- a helper/runtime registry
- explicit library/module output for partial translation

The initial migration source is the [`Pure-Fortran`](https://github.com/beliavsky/pure-fortran) project.

The first backend target is modern Fortran. The architecture is being designed so a later C++ backend can reuse the same semantic core.

The translator is intended to support both:

- complete program generation when the input is translatable end to end
- library/module generation when only individual functions or a partial translation is practical

That second mode is essential for packaging translated numerical kernels behind foreign-function interfaces.

The current Fortran path now includes the beginning of that packaging layer:

- exported library procedures in generated Fortran modules
- a small `bind(C)` wrapper generator for scalar numeric procedures
- diagnostics for exported procedures that are not yet C-interoperable

## Initial Goals

- extract shared compiler concepts from `xp2f.py`, `xoct2f.py`, `xr2f.py`, and `xc2f.py`
- define a normalized IR for array-heavy scientific code
- build a reusable Fortran backend instead of emitting Fortran directly from each frontend
- make function-level and package-oriented translation first-class, not just full-program translation
- register helper modules such as `lapack_d.f90`, `python.f90`, `octave_funcs.f90`, and `r.f90`
- migrate the Python/NumPy path first because `xp2f.py` is the most mature

## Layout

- `array_compiler/ir/`: normalized array IR and core compiler model
- `array_compiler/frontends/`: source-language frontends
- `array_compiler/backends/fortran/`: Fortran-specific lowering and emission
- `array_compiler/runtime/fortran/`: helper registry and copied runtime helpers
- `docs/`: architecture and migration notes

## Near-Term Plan

1. Stabilize the shared IR and Fortran backend skeleton.
2. Migrate reusable pieces from `Pure-Fortran`:
   - Fortran build helpers
   - Fortran post-processing
   - helper module registry
3. Add a Python/NumPy frontend that lowers into the shared IR.
4. Use the same IR for Octave, R, C, and later restricted C++.
