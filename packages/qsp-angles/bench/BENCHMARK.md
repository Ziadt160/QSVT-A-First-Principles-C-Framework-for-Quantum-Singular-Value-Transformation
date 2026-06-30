# Benchmark: qsp-angles C++ core vs pyqsp

**Question:** is this solver actually useful, or does `pyqsp` already do everything
better? This benchmark answers it honestly.

> **TL;DR — read the sym_qsp section first.** The package ships an **embeddable**
> C++ port of the **symmetric-QSP Newton method** (`sym_qsp`; Dong–Lin–Ni–Wang,
> arXiv:2307.12468 — the same algorithm pyqsp uses), and it is the **default**
> solver. It is machine-precision, reaches **degree > 1000**, and is callable from
> a compiled stack with no Python (Eigen + STL). Because it runs the *identical*
> algorithm as pyqsp's `newton_solver`, the speed difference is a
> compiled-vs-interpreted constant factor (**tens-fold; ~25–70× on the machine
> below**), not a better algorithm — the genuine differentiator is embeddability.

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
- **Speed: a compiled-vs-interpreted constant factor (~25–70× here)** running the
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
algorithm at compiled speed (a ~25–70× constant factor here). So the homotopy solver below is superseded as the default,
and its high-degree cost (inflated by the conservative `2·d`-step homotopy
schedule) no longer matters for the recommended path. The homotopy solver
remains available as `method="homotopy"`.
