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
- exported procedures and library/package boundaries
- optional program entrypoints
- partial-translation diagnostics

The current repository contains only a small initial skeleton. The next stage is
to expand this into a real normalized array IR.

### Backends

Backends are target-specific:

- Fortran backend
- later C++ backend

The Fortran backend should take advantage of native array semantics. The later
C++ backend will likely require a runtime/view layer that the Fortran backend
does not need.

Backends should support at least two output modes:

- program mode: emit a complete executable unit when the source is sufficiently complete
- library mode: emit callable modules/packages when only translated functions are available

Library mode is a core requirement because many useful translations will be
function-level kernels intended for reuse from Python, R, or Matlab.

For the Fortran backend, that implies a layered output strategy:

- native Fortran module emission
- optional `bind(C)` wrapper generation for C-interoperable exported procedures
- later package generators that can target Python, R, or Matlab on top of the stable C ABI

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

## Partial Translation

The system should not assume all-or-nothing translation.

The normalized IR and backend-facing module model should be able to represent:

- translated exported procedures
- optional main-program wrappers
- untranslated or deferred pieces recorded as diagnostics

This allows the compiler to remain useful even when only part of the original
program can be lowered cleanly.
