# Benchmark: qsp-angles C++ core vs pyqsp

**Question:** is this solver actually useful, or does `pyqsp` already do everything
better? This benchmark answers it honestly.

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

## Honest verdict

- **Accuracy: tie.** Both reach machine precision. pyqsp is ~100× tighter, but
  both are orders of magnitude below any practical QSVT requirement.
- **Speed: the core is competitive only up to ~degree 25.** Beyond that pyqsp's
  `sym_qsp` (Newton) scales much better — ~4× faster at degree 101.
- **Degree reach: pyqsp wins decisively** — it handles degree > 1000; this core
  tops out around 100.
- **Embeddability: the core's only genuine differentiator** — it's two C++ files
  (Eigen + STL), callable directly from a compiled stack with no Python runtime.

**Conclusion.** This solver is *not* "a faster pyqsp." Its honest value is
**embeddability in C++/compiled pipelines**, where it is competitive up to
~degree 100. To become genuinely state-of-the-art (match pyqsp's speed *and*
degree reach while staying embeddable), the core needs the Newton / `sym_qsp`
(or Fejér–Riesz completion) method in C++ — see the project roadmap. The core's
high-degree cost is also inflated by a conservative `2·d`-step homotopy schedule,
which is tunable independently of that larger algorithmic change.
