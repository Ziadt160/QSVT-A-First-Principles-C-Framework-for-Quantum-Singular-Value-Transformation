# Porting the inverse nonlinear FFT (NLFT) to the C++ core — scope and design

**Motivation.** `nlft-qsp` (MIT, Python) overtakes this project's symmetric-QSP Newton
core between degree 1001 and 1501 and is 3.8× faster by degree 4001
(`bench/results/crossover.txt`). Its advantage is algorithmic, not linguistic: an
inverse nonlinear Fourier transform instead of Newton iteration. Since this project's
measured edge over *Python implementations of the same algorithm* is 11–23×, a C++ NLFT
would plausibly be fastest at every degree — and unlike the competition, embeddable.

## Effort: smaller than it looks

An earlier estimate of "2–4 weeks" was wrong. The algorithmic core of `nlft-qsp` is:

| module | lines | role |
|---|---:|---|
| `solvers/weiss.py` | 112 | complementary polynomial `a` with \|a\|²+\|b\|²=1 |
| `solvers/half_cholesky.py` | 61 | Ni–Ying half-Cholesky inverse NLFT |
| `solvers/layer_stripping.py` | 38 | layer-stripping reference (slow, for validation) |
| `solvers/nlfft.py` | 53 | divide-and-conquer driver |
| `nlft.py` | 65 | NLFT sequence representation |

≈330 lines of actual algorithm. The polynomial/FFT infrastructure it leans on
(`poly.py`, 386 lines) is largely already present in this repo: `SymQspAngleSolver.cpp`
ships a self-contained radix-2 + Bluestein FFT, which is exactly what the Weiss step
needs. Realistic scope: **2–5 days**, not weeks.

## The one thing that changes the strategy: NLFT is O(1/η), our Newton is not

From `weiss.py`:

```python
eta = 1 - b.sup_norm(...)
N = next_power_of_two(int(d/eta)) // 2   # exponential search on N
```

The Weiss step's FFT length starts at **N ≈ d/η**, where **η = 1 − sup|P|**. The
docstring for `xqsp_solve_laurent` states it outright: *"The time required by the
algorithm to compute the phase factors will scale with 1/η."*

This matters because QSVT matrix inversion *wants* `sup|p|` as close to 1 as possible:
the subnormalization `c` sets the post-selection success probability as `c²`, so shaving
harder to speed up NLFT directly costs repetitions. Our benchmark used a 0.999 shave
(η = 1e-3), which at degree 1001 forces N ≈ 10⁶ — i.e. we measured `nlft-qsp` in a
regime its own documentation calls expensive.

**The Newton core has a structural advantage precisely where QSVT needs it.** Its cost
is η-independent (iteration count is flat at 9 across κ=10…110). So for targets pressed
against the `|p| ≤ 1` boundary — which is the accuracy-optimal choice for linear
solving, since the subnormalization sets success probability — Newton is the right
algorithm regardless of degree. `bench/eta_fairness.py` quantifies the η dependence.

### Precision is not like-for-like, and the direction is counter-intuitive

`nlft-qsp` defaults to the **mpmath arbitrary-precision** backend, not double:
`machine_eps() = 1.08e-19`. Its Weiss completions reach residual ~1e-19 where a
double-precision implementation is bounded near ~1e-14. So the published shootout
compared our double against their extended precision — they deliver *more accuracy*
per solve than the timing alone suggests.

The obvious inference — that this made them look artificially slow, so the crossover is
earlier than measured — was tested and is **false**. Re-timed on their numpy (double)
backend, `chebqsp_solve` is *not* faster; it is slower at every degree measured
(degree 801: 13.7 s on numpy vs 4.96 s on the mpmath default; degree 51: 3.07 s vs
2.35 s). Their mpmath path is evidently the better-optimised one. The crossover
measured in `bench/results/crossover.txt` therefore stands and is not an artefact of
the precision mismatch.

What this *does* mean for the port: matching `nlft-qsp`'s accuracy would require
extended precision in the C++ port too. A double-precision port targets ~1e-14, which
is ample for QSP angle finding (our Newton core delivers the same) but should be stated
rather than implied.

**Design consequence:** do not replace the Newton solver. Ship **both**, and dispatch on
(degree, η). Rough rule from current data: Newton for small η (≲1e-2) or degree ≲1500;
NLFT otherwise. The dispatch rule should be measured, not guessed.

## Outcome (steps 1–6 complete)

The port works end-to-end and reproduces `nlft_qsp.chebqsp_solve` to ~1e-12 on real
1/x QSVT targets. The measured behaviour confirms the dispatch thesis and also shows
where this port is *not* yet competitive.

Identical near-minimax 1/x targets, single-threaded, pinned (seconds):

| κ | deg | η | C++ Newton | C++ NLFT | nlft-qsp (py) |
|--:|----:|--:|-----------:|---------:|--------------:|
| 10 | 79  | 1e-3 | **0.067** | 2.857 | 2.174 |
| 10 | 79  | 1e-2 | 0.061 | **0.096** | – |
| 10 | 79  | 5e-2 | 0.053 | **0.027** | – |
| 30 | 247 | 1e-3 | **0.856** | 10.090 | 8.434 |
| 30 | 247 | 1e-2 | 0.873 | **0.512** | – |
| 30 | 247 | 5e-2 | 0.718 | **0.124** | – |
| 60 | 505 | 1e-3 | **5.431** | 22.351 | 15.168 |
| 60 | 505 | 1e-2 | 5.193 | **1.204** | – |
| 60 | 505 | 5e-2 | 5.070 | **0.268** | – |

**1. The dispatch thesis is confirmed.** Newton's cost is flat in η (5.43 → 5.07 s at
degree 505 while η moves 50×), NLFT's swings **83×** over the same range. The crossover
sits near **η ≈ 3e-3**: below it Newton wins, above it NLFT wins by up to 19×. Shipping
both and dispatching on η is the right design, and the rule is now measured rather than
guessed.

**2. This matters for the flagship use case, in Newton's favour.** QSVT matrix inversion
wants `sup|p|` as close to 1 as possible, because the subnormalization sets the
post-selection success probability as `c²`. That is the small-η regime — exactly where
Newton wins. The decision not to replace the Newton solver is vindicated by data.

**3. Honest gap: this port is slower than the Python reference at small η**
(22.4 s vs 15.2 s at degree 505). The cause is identified: `__riemann_hilbert_weiss`
uses `nlfft.inlft`, the **divide-and-conquer** recursion of arXiv:2505.12615
(`nlfft_recurse`, O(n log² n)), whereas this port implements **half-Cholesky**
(arXiv:2410.06409, O(n²)). Half-Cholesky was chosen because a 2×m QR is a single
Givens rotation and needs no linear-algebra dependency — correct, but asymptotically
the slower of the two. Porting `nlfft_recurse` is the follow-up that would make the
C++ NLFT path beat the Python one outright; it needs polynomial `sharp()`,
`truncate`, and shift operations that `LaurentPoly` already provides.

## Status

| step | state | validation |
|---|---|---|
| 1. Laurent polynomials + FFT | **done** (`LaurentPoly`, `FftCore.hpp`) | all 7 conventions vs nlft-qsp, ≤2.3e-14 |
| 2. Schwarz transform | **done** (landed with step 1) | exact match |
| 3. Weiss completion | **done** (`WeissCompletion`) | `a` matches reference to 1.9e-14, deg ≤255 |
| 4. Half-Cholesky inverse NLFT | **done** (`InverseNlft`) | `F` matches reference to 2.9e-14, deg ≤63 |
| 5. Phase-factor layer | **remaining** | — |
| 6. Dispatch + re-benchmark | **remaining** | — |

The algorithmic core is complete. What is left is convention plumbing, which is
small but is exactly where silently-wrong angles come from, so it wants its own
validation pass rather than being rushed:

- `XQSPPhaseFactors.from_nlfs(F)` is simply **`phi_k = arctan(Im(F_k))`** (and it
  requires `F` to be purely imaginary — check that, do not assume it).
- `.iX()` applies a conjugation to move between the `(Q, -iP)` and `(P, iQ)`
  pictures; `xqsp_solve(mode='qsp')` premultiplies the target by `-1j` for the same
  reason.
- The entry point takes a **Chebyshev** target, so it needs
  `ChebyshevTExpansion.to_laurent()` then `laurent_to_analytic()`, and finally a
  conversion from the XQSP convention into this project's `Re<0|U|0>` Wx convention
  (the existing `SymQspAngleSolver` already carries an analogous ±π/4 adapter).

Acceptance test for step 5 is already specified below: phases must reproduce the
target through `SymQspAngleSolver::response` to ≤1e-12, cross-checked against
`nlft_qsp.chebqsp_solve` on identical Chebyshev coefficients.

## Implementation order (each step independently validatable)

1. **Laurent polynomial type + roots-of-unity evaluation.** `eval_at_roots_of_unity`,
   `sup_norm`, `l2_norm`, truncation, conjugation. Reuse the existing FFT. Validate
   coefficient-wise against `nlft_qsp.poly.Polynomial` on random inputs.
2. **Schwarz transform** (`R.schwarz_transform()`), the analytic projection that turns
   `log(1−|b|²)/2` into the outer function's exponent. Validate pointwise on the circle.
3. **Weiss completion** — steps 1–2 assembled, with the exponential search on `N` and
   the `WEISS_MAX_ATTEMPTS` non-convergence guard. **Acceptance test:**
   `‖ a·a* + b·b* − 1 ‖₂ < 100·machine_eps` for random `b` at degrees 51…1001, and
   coefficient agreement with `weiss.complete` to ~1e-12.
4. **Half-Cholesky inverse NLFT** producing the phase factors from `(a, b)`.
   **Acceptance test:** phases reproduce the target polynomial through the existing
   `SymQspAngleSolver::response`, residual ≤ 1e-12, cross-checked against
   `nlft_qsp.chebqsp_solve` on identical Chebyshev coefficients.
5. **Dispatch + benchmark.** Extend `bench/shootout_clean.py` with the new backend and
   re-measure the crossover, now C++ vs C++.

## Risks

- **Precision.** `nlft-qsp` supports an `mpmath` backend, implying parts of the method
  can need more than double precision. If step 3 or 4 stalls above some degree in
  double, that bounds the port — measure it and document the ceiling rather than
  claiming an unbounded range.
- **Attribution.** `nlft-qsp` is MIT. A port must carry its copyright notice and cite
  the underlying papers (Alexis–Mnatsakanyan–Thiele; Ni–Ying).
- **The 1/η wall is inherited, not fixed.** A C++ NLFT is still O(1/η); it will not
  rescue the small-η regime. That regime stays Newton's.
