# Architecture

## Goal

Build a universal compiler/transpiler framework for array-oriented scientific
code with:

- multiple source languages
- a shared normalized semantic IR
- multiple backends

The first backend is modern Fortran. A later C++ backend should reuse the same
IR and analysis passes.

## Core Design

### Frontends

Each source language gets its own frontend:

- Python/NumPy
- Matlab/Octave
- R
- C
- restricted C++

Frontend responsibilities:

- parse source language
- resolve language-specific syntax
- normalize source semantics into a shared IR

Frontends should not emit Fortran or C++ directly once migrated.

### Shared IR

The shared IR should capture:

- scalar and array types
- rank and shape
- slices and array sections
- elementwise operations
- reductions
- matrix operations
- calls to recognized intrinsics or helper operations
- structured control flow

The current repository contains only a small initial skeleton. The next stage is
to expand this into a real normalized array IR.

### Backends

Backends are target-specific:

- Fortran backend
- later C++ backend

The Fortran backend should take advantage of native array semantics. The later
C++ backend will likely require a runtime/view layer that the Fortran backend
does not need.

## Fortran-First Migration

Initial migration source:

`c:\python\public_domain\github\Pure-Fortran`

Priority order:

1. Migrate reusable Fortran build and post-processing helpers.
2. Migrate the Python/NumPy path from `xp2f.py` into:
   - Python frontend
   - shared IR lowering
   - Fortran backend
3. Migrate Octave, R, and C frontends onto the same IR.

## Shared Components To Extract First

These should move out of monolithic transpilers early:

- type mapping
- rank and shape inference
- helper module selection
- intrinsic recognition
- declaration emission support
- post-processing and formatting hooks

## Helper Strategy

Fortran helper modules such as:

- `lapack_d.f90`
- `python.f90`
- `octave_funcs.f90`
- `r.f90`

should be managed through a shared helper registry instead of frontend-specific
ad hoc imports.
