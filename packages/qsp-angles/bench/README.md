# Benchmarks & validation

Everything here is built so a third party can reproduce the claims with one
command and independently verify correctness against a reference solver. Nothing
is "trust the screenshot."

## One command

```bash
bash bench/run_all.sh
```

This prints a **provenance header** (CPU, compiler + flags, library versions,
thread pinning), runs the C++ `sym_qsp` benchmark, the pyqsp timing baseline, and
the independent cross-validation, writing all outputs to `bench/results/`
(git-ignored). A captured reference run is in [SAMPLE_RUN.md](SAMPLE_RUN.md).

## What's here

| file | purpose |
|------|---------|
| `bench_symqsp_core.cpp` | C++ `sym_qsp` timing/accuracy across degree 11…1001 |
| `bench_core.cpp` | the legacy homotopy solver (for the historical comparison) |
| `bench_pyqsp.py` | pyqsp `sym_qsp` timing baseline |
| `validate_vs_pyqsp.py` | **independent cross-validation** — generates generic targets, solves with pyqsp (oracle), drives the C++ solver, asserts both reach tolerance |
| `validate.cpp` | C++ driver the cross-validation calls on the shared manifest |
| `BENCHMARK.md` | results + honest verdict |

## Methodology (so the numbers are interpretable)

- **Same algorithm both sides.** The C++ `sym_qsp` and pyqsp's `newton_solver`
  are the *same* Dong–Lin–Ni–Wang Newton method (arXiv:2307.12468). The speed
  difference is therefore a **compiled-vs-interpreted constant factor plus
  embeddability**, not a better algorithm — and we say so plainly. We are not
  claiming a novel solver; we claim the state-of-the-art solver at C++ speed,
  callable from a compiled stack.
- **Single-threaded, both sides.** `run_all.sh` pins `OMP/OpenBLAS/MKL` to one
  thread so the C++ (single-threaded) vs numpy/scipy comparison is apples-to-apples.
- **Target.** Timing uses scaled Chebyshev `0.8·T_d` (isolates angle-finding from
  polynomial approximation). Validation uses *random generic* multi-coefficient
  targets (every allowed Chebyshev mode populated), both parities, degrees up to
  101 — the realistic QSVT case.
- **Grading is convention-neutral.** Each solver is graded in its own convention
  against the same target polynomial (the C++ Re-convention residual; pyqsp's
  Im-convention residual). They agree to machine precision.

## Caveats (read before quoting numbers)

- **Absolute timings are hardware-dependent.** Quote them only with the
  provenance header. The *ratio* (≈25–70× here) and the trend are the portable claim.
- **Correctness is hardware-independent.** The validation residuals (~1e-14) are
  reproducible anywhere; that is the part to lean on.
- **`-march=native` + FFT** means the last bits can differ across CPUs. The
  validation uses tolerances (`1e-10`), not exact equality, for this reason.
- This benchmarks against **pyqsp** (the established open-source solver). MATLAB
  **QSPPACK** implements the same method and is more optimized than pyqsp; a
  QSPPACK comparison would strengthen the claim further and is a good next step.
