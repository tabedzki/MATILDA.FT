# MATILDA.FT test suite

Unit tests for the core C++/CUDA components, built on GoogleTest 1.14
(vendored in `external/googletest`, no download needed). The tests link
against the object files produced by `src/makefile` and run real device code,
so an NVIDIA GPU + CUDA toolkit are required (same requirements as the main
build).

## Running

```bash
make -C tests test           # builds src objects if needed, then builds + runs
make -C tests ARCH=sm_86     # override GPU arch, same convention as src/
./matilda_tests --gtest_filter='Gsd.*'   # run a subset (from tests/)
```

The binary resolves the `fixtures/` directory relative to the directory it is
launched from; when running it from somewhere other than `tests/`, set
`MATILDA_FIXTURES_DIR=/path/to/tests/fixtures`. Tests are hermetic: all
generated files go to `mkdtemp` directories under /tmp.

## Layout

- `makefile` — builds the single `matilda_tests` binary. Auto-globs
  `unit/*.cu`; new test files need no makefile changes. Reuses
  `src/objects/*.o` (except `main.cu.o`).
- `support/test_main.cu` — gtest `main()` plus the globals that `src/main.cu`
  normally provides via the `#define MAIN` pattern (`idum`, timing counters,
  `giveQuote()`).
- `unit/test_random.cu` — host RNG (`ran2`): determinism per seed, range,
  uniform moments.
- `unit/test_gsd.cu` — GSD file format library (`src/gsd.c`): create/write/
  read roundtrips, multi-frame indexing, type sizes, error paths.
- `unit/test_device_utils.cu` — standalone CUDA kernels from
  `src/device_utils.cu`: element-wise float/complex ops and the tree-reduction
  sum (power-of-two block sizes only — that is the kernel's contract).
- `unit/test_ps_box.cu` — integration-level tests on a `PS_Box` built from
  `fixtures/ps2d.input` (200 particles, 2D, two species): data-file parsing,
  type-group membership, cell-list neighbor list vs. an O(N²) brute-force
  reference, VV integration sanity (finite, in-box), and single-precision FFT
  wrapper roundtrip/convolution identities.
- `unit/test_box_utils.cu` — host-side `Box` index/geometry math via the
  ps2d fixture: `unstack`/`stack` roundtrip, `get_r` vs. manual grid
  arithmetic, `get_kD` FFT wraparound convention, `pbc_dr2` minimum-image
  (float and double).
- `unit/test_bonds.cu` — analytic harmonic bond/angle checks on a 7-particle
  bonded fixture (`fixtures/psbond.*`): parsed topology, energies and forces
  vs. closed-form values, equal/opposite pair forces. Documents a suspected
  bond/angle double-count on the data-file init path (see the comment in
  `ParsesBondedFixture`).
- `unit/test_fts_box.cu` — `FTS_Box` built from `fixtures/fts2d.input`:
  double-precision FFT roundtrip/DC normalization, convolution identity, and
  `integTComplexD` volume integration.
- `unit/test_gsd_traj.cu` — `PS_Box` GSD trajectory writing/appending and
  position roundtrip (avoids asserting `configuration/step`, which is garbage
  until issue #1 finding 9 is fixed).
- `fixtures/` — input + data files for the PS_Box/FTS_Box tests, written in the
  *current* input-file format (`box ps` … `endBox`). Note the shipped
  `examples/` inputs (except `examples/ft`) predate the parser rewrite and do
  not run unmodified.

## Conventions

- Anything accumulated on the GPU is compared with `EXPECT_NEAR`, never exact
  equality: force/density kernels use `atomicAdd`, so floating-point ordering
  is not reproducible.
- Deterministic host RNG: seed by assigning a negative value to `idum`;
  input files seed both CPU and GPU RNGs with `randSeed <n>`.
- Tests must not modify anything under `src/`; suspected bugs found while
  testing get documented (test comment or report), not silently fixed.

## Regression harness

`make -C tests regression` (or `tests/regression/run_regression.sh`) runs
short, deterministic full-system simulations with the `matilda.ft` binary and
compares every `.dat` output token-by-token against golden files
(`compare_numeric.py`, tight default tolerances — SCFT output is
bit-reproducible on a given GPU; the tolerance absorbs FFT rounding
differences across GPU generations). Each case is a `regression/cases/<name>/`
directory holding `case.input` and `golden/`; the harness discovers cases
automatically. `run_regression.sh --update` regenerates goldens after an
intentional behavior change — eyeball the diff before committing it.

Current cases:

- `fts-scft-2d` — 200 SCFT steps of a symmetric A/B homopolymer blend
  (Helfand + Flory potentials, 48² grid), adapted from `examples/ft/input`
  to the current parser syntax.

Note the case input file is named `case.input` because the repo-root
`.gitignore` ignores files named `input` everywhere.

## Planned

- A particle-simulation (`box ps`) regression case: needs a tolerance-based
  trajectory comparison (GPU atomicAdd ordering makes runs non-bit-identical)
  and a fix for RNG seeding (issue #1 finding 15) to be meaningful.
