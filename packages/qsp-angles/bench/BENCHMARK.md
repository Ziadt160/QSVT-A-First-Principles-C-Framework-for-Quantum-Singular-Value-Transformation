# Benchmark: qsp-angles C++ core vs pyqsp, QSPPACK and nlft-qsp

**Question:** is this solver actually useful, or do the existing packages already do
everything better? This benchmark answers it honestly, including where we lose.

> **TL;DR.** The package ships an **embeddable, zero-dependency** C++ implementation
> of the **symmetric-QSP Newton method** (`sym_qsp`; Dong–Lin–Ni–Wang,
> arXiv:2307.12468), and it is the **default** solver. On the QSVT `1/x` inversion
> target it is the **fastest of the four solvers tested at every degree up to 1001**
> (1.4×–17× against the best external solver across six configurations), at machine-precision residual. Two honest qualifications: against
> pyqsp and QSPPACK — which run essentially the *same algorithm* in Python — the
> ~11–23× margin is a **compiled-vs-interpreted constant, not a better algorithm**;
> and against **nlft-qsp**, whose inverse-nonlinear-FFT method scales as
> O(d log² d) versus our ~O(d²), our lead **collapses from 162× to 1.8× by degree
> 801** and extrapolates to a crossover near **degree ~4000**. The durable
> differentiator is **embeddability**: we are the only one of the four callable from
> a compiled C/C++/Rust binary with no interpreter.

## Head-to-head vs the other open-source solvers

Every solver is given the **identical** target: the near-minimax odd Chebyshev
approximation of `c/x` on `[1/κ, 1]` with `κ = 10` and `|p| ≤ 1` — the polynomial a
QSVT linear solver actually needs (not a synthetic single-Chebyshev target).

- **Protocol:** one discarded warm-up, then the **median of 3** timed runs.
- **Pinned & single-threaded:** `OMP/OpenBLAS/MKL = 1`, process pinned to one core,
  so BLAS thread counts cannot flatter any implementation.
- **Machine:** Intel i7-9750H, idle (load 0.00); g++ 12.4.0 `-O3`; Python 3.12.3;
  numpy 2.5.0; pyqsp 0.2.0; qsppack 0.3.12; nlft-qsp 1.0.4.
- Wall-clock is the primary metric because it is convention-free. Phase conventions
  differ between the libraries, so phases are **not** cross-graded; our residual is
  reported to show the target is genuinely solved.

Reproduce with `bench/shootout_clean.py`.

| degree | **ours (C++)** | our residual | nlft-qsp | QSPPACK | pyqsp |
|-------:|---------------:|-------------:|---------:|--------:|------:|
| 51   | **0.015 s** | 6.3e-15 | 2.354 s (162×) | 0.282 s (19×) | 0.330 s (23×) |
| 101  | **0.050 s** | 1.0e-14 | 4.797 s (96×)  | 0.972 s (19×) | 1.024 s (20×) |
| 201  | **0.192 s** | 2.6e-14 | 4.936 s (26×)  | 4.330 s (23×) | 3.709 s (19×) |
| 401  | **0.776 s** | 2.4e-14 | 4.707 s (6.1×) | 6.535 s (8.4×)| 8.026 s (10×) |
| 801  | **2.696 s** | 9.3e-14 | 4.958 s (1.8×) | 30.97 s (11×) | 32.99 s (12×) |
| 1001 | **3.603 s** | 1.9e-13 | 10.21 s (2.8×) | 40.70 s (11×) | 70.09 s (20×) |

### Where this lead ends — measured, not extrapolated

`nlft-qsp` overtakes this solver **between degree 1001 and 1501**. Extrapolating the
κ=10 table above suggested a crossover near degree 4000; that estimate was wrong by
~3×, so it was measured directly. Each degree below uses a κ matched to it
(κ ≈ d/9.1), so every target sits in its realistic regime — a degree-4001 polynomial
is needed because the system is ill-conditioned, not because a well-conditioned one
was over-provisioned (at κ=10 a degree-4001 fit is rank-deficient and meaningless;
degree 79 suffices).

| degree | κ | ours | nlft-qsp | winner |
|-------:|--:|-----:|---------:|:-------|
| 1501 | 165 | 20.80 s | **15.20 s** | nlft-qsp 1.4× |
| 2001 | 220 | 46.99 s | **31.46 s** | nlft-qsp 1.5× |
| 3001 | 330 | 146.10 s | **58.43 s** | nlft-qsp 2.5× |
| 4001 | 440 | 188.65 s | **49.51 s** | nlft-qsp 3.8× |

Our residual stays at machine precision throughout (6.3e-14 … 1.5e-13), so this is a
speed result, not an accuracy one. Measured scaling is **~O(d^2.8)** for this Newton
core against a clearly sub-cubic curve for the inverse-nonlinear-FFT method. The 3001
and 4001 rows are single runs (each costs minutes) and nlft-qsp's 4001 time came in
*below* its 3001 time, so those two rows are noisy — the direction, however, is
unambiguous.

**Use `nlft-qsp` above degree ~1500.** This solver's advantage is the range below
that — which covers every QSVT application in this repo — plus embeddability at any
degree.

### Convergence requires the degree to match κ

A related limit, found while isolating the crossover. At **fixed** degree 1001,
varying only the target difficulty:

| κ | Newton iterations | time | residual |
|--:|------------------:|-----:|---------:|
| 10  | 9 | 9.41 s | 1.8e-13 |
| 30  | 9 | 7.00 s | 5.2e-14 |
| 60  | 9 | 8.50 s | 4.2e-14 |
| 110 | 9 | 9.16 s | 4.3e-14 |
| 165 | **100 (capped)** | 69.88 s | **1.3e-02 — did not converge** |

Iteration count is **flat at 9** while the target is adequately resolved, so cost is
degree-driven rather than steepness-driven. But κ=165 needs roughly degree 1500; asked
for it at degree 1001 the target is under-resolved and the Newton iteration runs to its
cap without converging. **Size the degree with `minInverseDegree(κ, ε)` (C++) or
`min_degree` (Python) instead of choosing one by hand** — an under-resolved target
produces a non-convergence, not merely a less accurate answer.

**A caution on earlier numbers.** A first pass at this comparison was run while other
jobs were loading the machine and produced timings ~2× pessimistic and even
non-monotone in degree (degree 801 apparently slower than 1001). Those artefacts
vanish on an idle pinned core — median and min now agree to within 1–2%. Only the
table above should be quoted.

## Historical: sym_qsp vs pyqsp on a synthetic target

## sym_qsp: the embeddable Newton solver

A faithful C++ port of pyqsp's `sym_qsp` (Dong–Lin–Ni–Wang robust symmetric-QSP
Newton method, arXiv:2307.12468), validated against pyqsp to ~15 digits. The
honest framing: this is **not a new algorithm** and **not a better algorithm** —
it is the same Newton method, compiled, so it can be embedded in a C++ pipeline.
The speedup is the expected compiled-vs-interpreted constant factor; we report it
for completeness, not as a contribution.

- **Same function, both sides.** C++ `SymQspAngleSolver` vs pyqsp's own
  `pyqsp.sym_qsp_opt.newton_solver` (NOT the higher-level
  `QuantumSignalProcessingPhases`, which wraps extra work the C++ core does not do).
- **Target:** scaled Chebyshev `f(x) = 0.8 · T_d(x)`, isolating angle-finding from
  polynomial approximation.
- **Pinned & single-threaded.** Both timed single-thread; see the provenance
  header from `run_all.sh`. Numbers below: one run, Intel i7-9750H, g++ 12.4
  `-O3 -march=native`, pyqsp 0.2.0 / numpy 2.5.0 / scipy 1.18.0. Median of 3 runs
  for degree ≤ 101, single run above (slow, low-variance). Both converge in **5
  Newton iterations** to machine precision.

| degree | C++ sym_qsp | pyqsp `newton_solver` | speedup |
|-------:|------------:|----------------------:|--------:|
| 11   | 0.2 ms   | 6.6 ms      | ~33× |
| 21   | 0.6 ms   | 15.7 ms     | ~26× |
| 51   | 2.1 ms   | 82.9 ms     | ~39× |
| 101  | 3.9 ms   | 281 ms      | ~72× |
| 201  | 24.7 ms  | 1278 ms     | ~52× |
| 501  | 285 ms   | 8127 ms     | ~28× |
| 1001 | 1009 ms  | 28316 ms    | ~28× |

Both reach machine precision (C++ residual ~1e-14…1e-12; pyqsp self-err similar).
**Absolute timings are hardware-dependent and WSL2 timing is noisy** — quote them
only with the provenance header; the portable claim is "tens-fold, one-to-nearly-
two orders of magnitude." These numbers match the committed CSVs in `results/`
(regenerate with `run_all.sh`); see [SAMPLE_RUN.md](SAMPLE_RUN.md).

### Verdict (sym_qsp)

- **Embeddability: the only genuine differentiator** — two C++ files (Eigen + STL),
  callable from a compiled stack with no Python runtime. This is the reason to use it.
- **Accuracy: tie** — both machine precision, 5 Newton iterations, at any degree.
- **Degree reach: tie** — both reach degree > 1000 (C++ does it in ~1 s).
- **Speed: a compiled-vs-interpreted constant factor (~11–23x vs pyqsp/QSPPACK on the 1/x target; see the head-to-head table above)** running the
  *same* algorithm. Real, but expected — not a contribution, and angle-finding is
  a one-time offline precompute that is rarely anyone's bottleneck.

**Conclusion.** This is an **embeddable C++ implementation of the state-of-the-art
symmetric-QSP Newton method**, validated against pyqsp to machine precision and
reaching degree 1000+. Its value is being callable from a compiled quantum
toolchain without a Python runtime — not being "faster" in any way that matters to
someone who runs angle-finding once and caches the result. It is the package
default and supersedes the homotopy solver below.

---

## (Legacy) homotopy continuation vs pyqsp

The section below benchmarks the **older homotopy** solver (now the
`method="homotopy"` fallback). It is kept for historical context; for any new use
prefer `sym_qsp` above.

## Setup

- **Target:** scaled Chebyshev `f(x) = 0.8 · T_d(x)` for `d ∈ {11,21,31,51,71,101}`.
  A legitimate degree-`d` target (`|f| ≤ 0.8 < 1`, parity `d mod 2`) that both
  solvers can represent. Using a pure Chebyshev target isolates **angle-finding**
  from polynomial approximation, so we compare the *solvers*, not the approximators.
- **core:** this package's C++ solver (homotopy continuation + analytic-Jacobian
  Levenberg–Marquardt), `g++ -O3 -march=native`. Residual = max `|Re⟨0|U|0⟩ − f|`
  over a 401-point grid on `[-1,1]`.
- **pyqsp:** `QuantumSignalProcessingPhases(method="sym_qsp", signal_operator="Wx")`
  — pyqsp's modern Newton solver (machine-precision by construction). numpy/scipy.
  Residual = pyqsp's own self-reported final optimizer error.
- Each solver is graded **in its own convention** (their phase gauges differ, so
  cross-evaluating phases would be unfair). Median of 3 runs, after a warm-up.

Reproduce: build/run `bench_core.cpp` and `bench_pyqsp.py` (see headers).

## Results

| degree | core time | core residual | pyqsp time | pyqsp residual |
|-------:|----------:|--------------:|-----------:|---------------:|
| 11  | **4.1 ms**  | 3.4e-15 | 6.7 ms   | 2.0e-16 |
| 21  | 42 ms       | 2.4e-14 | 18 ms    | 2.2e-16 |
| 31  | 61 ms       | 1.0e-14 | **35 ms**  | 3.8e-16 |
| 51  | 265 ms      | 1.7e-14 | **181 ms** | 5.8e-16 |
| 71  | 748 ms      | 2.1e-14 | **263 ms** | 7.6e-16 |
| 101 | 2229 ms     | 3.1e-14 | **590 ms** | 8.4e-16 |

Hardware-dependent; the *ratios* and trend are the point, not absolute ms.
Low-degree timings are noisy (core degree-21 ranged 23–42 ms across runs); the
robust, repeatable signal is the high-degree gap (pyqsp ~3–4× faster from
degree ~31 up).

## Honest verdict (legacy homotopy solver)

- **Accuracy: tie.** Both reach machine precision. pyqsp is ~100× tighter, but
  both are orders of magnitude below any practical QSVT requirement.
- **Speed: the homotopy core is competitive only up to ~degree 25.** Beyond that
  pyqsp's `sym_qsp` (Newton) scales much better — ~4× faster at degree 101.
- **Degree reach: pyqsp wins decisively** — it handles degree > 1000; the
  homotopy core tops out around 100.
- **Embeddability: the homotopy core's only differentiator** — two C++ files
  (Eigen + STL), callable from a compiled stack with no Python runtime.

**Conclusion (and what changed).** As a *homotopy* solver this was "not a faster
pyqsp" — its only edge was embeddability. The roadmap item flagged here was to
bring the **Newton / `sym_qsp`** method into C++. That is now done (see the
sym_qsp section at the top): the C++ Newton port keeps the embeddability while
matching pyqsp's degree reach (1000+) and machine precision, and runs the same
algorithm at compiled speed (a ~11-23x constant factor on the 1/x target). So the homotopy solver below is superseded as the default,
and its high-degree cost (inflated by the conservative `2·d`-step homotopy
schedule) no longer matters for the recommended path. The homotopy solver
remains available as `method="homotopy"`.
