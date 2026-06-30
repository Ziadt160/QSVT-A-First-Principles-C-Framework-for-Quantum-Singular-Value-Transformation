# Benchmark: qsp-angles C++ core vs pyqsp

**Question:** is this solver actually useful, or does `pyqsp` already do everything
better? This benchmark answers it honestly.

> **TL;DR — read the sym_qsp section first.** The package now ships a C++ port of
> the **symmetric-QSP Newton method** (`sym_qsp`, the same algorithm pyqsp uses),
> and it is the **default** solver. It is machine-precision, **19–100× faster than
> pyqsp**, reaches **degree > 1000**, and is **embeddable** (Eigen + STL, no
> Python). It supersedes the homotopy solver benchmarked below on every axis. The
> original homotopy-vs-pyqsp section is kept for context.

## sym_qsp: the embeddable Newton solver

This is the recommended path: a faithful C++ port of pyqsp's `sym_qsp`
(Dong–Lin–Ni–Wang robust symmetric-QSP Newton method, arXiv:2307.12468),
validated against pyqsp to ~15 digits. Both implementations run **the same
algorithm**, so this is a like-for-like race — and the compiled version wins
decisively while staying embeddable in a C++ stack.

- **Target:** scaled Chebyshev `f(x) = 0.8 · T_d(x)` (parity `d mod 2`, `|f| ≤ 0.8`),
  isolating angle-finding from approximation, same as below.
- **C++ sym_qsp:** `SymQspAngleSolver`, `g++ -O3 -march=native`. Residual = max
  `|Re⟨0|U|0⟩ − f|` over a 401-point grid on `[-1,1]` (Re convention).
- **pyqsp sym_qsp:** `QuantumSignalProcessingPhases(method="sym_qsp",
  signal_operator="Wx")` — pyqsp's own Newton solver. Residual = its self-reported
  final optimizer error. Both converge in **5 Newton iterations** to machine
  precision; only the wall-clock differs. Median of 3 runs after warm-up.

| degree | C++ sym_qsp | pyqsp sym_qsp | speedup |
|-------:|------------:|--------------:|--------:|
| 101  | **4.1 ms**   | 413 ms    | ~100× |
| 201  | **39 ms**    | 2147 ms   | ~55×  |
| 501  | **523 ms**   | 11372 ms  | ~22×  |
| 1001 | **1392 ms**  | 26915 ms  | ~19×  |

Both reach machine precision (residual ~1e-14…1e-12) in 5 iterations; the C++ port
also handles degree **> 1000** comfortably (degree 1001 in ~1.4 s). Hardware
varies, but the trend — a 1–2 order-of-magnitude constant-factor win from
compiling the identical algorithm — is the point.

Reproduce: build/run `bench_symqsp_core.cpp` (CSV `degree,iters,time_ms,residual`;
see its header) and `bench_pyqsp.py`.

### Verdict (sym_qsp)

- **Accuracy: tie** — both machine precision, 5 Newton iterations, at any degree.
- **Speed: the C++ port wins by 19–100×** running the *same* algorithm — pure
  compiled-vs-interpreted constant factor.
- **Degree reach: tie-and-then-some** — both reach degree > 1000; the C++ port
  does it in ~1.4 s.
- **Embeddability: only the C++ port has it** — two C++ files (Eigen + STL),
  callable from a compiled stack with no Python runtime.

**Conclusion.** The C++ `sym_qsp` port is **the fastest QSP angle solver we know
of**: machine-precision, 19–100× faster than pyqsp, degree 1000+, *and*
embeddable. It is now the package default and supersedes the homotopy solver.

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
sym_qsp section at the top): the C++ Newton port keeps the embeddability **and**
beats pyqsp on speed (19–100×) while matching its degree reach (1000+) and
machine precision. So the homotopy solver below is superseded as the default,
and its high-degree cost (inflated by the conservative `2·d`-step homotopy
schedule) no longer matters for the recommended path. The homotopy solver
remains available as `method="homotopy"`.
