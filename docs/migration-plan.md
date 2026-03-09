# Migration Plan

## Source Project

`c:\python\public_domain\github\Pure-Fortran`

Most relevant existing files:

- `xp2f.py`
- `xoct2f.py`
- `xr2f.py`
- `xc2f.py`
- `fortran_build.py`
- `fortran_pipeline.py`
- `fortran_post.py`

## Start Point

`xp2f.py` is the first migration target because it is the most mature frontend.

## Milestones

### Milestone 1

Create project skeleton:

- package layout
- IR skeleton
- Fortran backend skeleton
- helper registry
- architecture docs

### Milestone 2

Extract reusable Fortran backend utilities from `Pure-Fortran`:

- build helpers
- helper source management
- post-processing

### Milestone 3

Add a small Python/NumPy frontend slice that lowers a narrow subset into the
shared IR and emits Fortran through the shared backend.

Candidate first subset:

- typed scalar functions
- simple assignments
- returns
- a few intrinsic calls
- small array constructors

### Milestone 4

Expand the normalized IR to cover:

- array sections
- broadcasting
- reductions
- matrix operations

Then migrate more of `xp2f.py`.
