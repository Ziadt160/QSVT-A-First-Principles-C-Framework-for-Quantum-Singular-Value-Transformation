# A homotopy + analytic-Jacobian QSP angle solver in C++

*Draft technical note / blog outline. The numbers below come from this
repository's `QspAngleSolver` and its tests; reproduce with the high-degree
solver test and the `bench/` harness.*

## 1. Why angle finding is the crux of QSVT

Quantum Signal Processing (QSP) implements a degree-`d` polynomial transform of a
scalar by interleaving a fixed signal rotation `W(x)` with `Z`-rotations:

```
U(x, Phi) = e^{i phi_0 Z} * prod_{k=1..d} [ W(x) e^{i phi_k Z} ],   W(x) = e^{i arccos(x) X}.
```

The `(0,0)` entry is a degree-`d` polynomial `P(x)` of definite parity, and
`Re<0|U|0>` realises a chosen real target `f(x)` with `|f| <= 1`. Quantum
Singular Value Transformation (QSVT) lifts this to a block-encoded operator:
the *same* phases `Phi`, with projector-controlled rotations replacing the
`Z`-rotations, apply `P` to the singular values of a matrix. So **finding the
phases `Phi` for a target `f` is the one classical pre-computation that every
QSVT application depends on** — matrix inversion, Hamiltonian simulation,
eigenvalue thresholding all reduce to "approximate this `f`, then find its
phases."

The difficulty: the map `Phi -> f` is highly nonlinear, the phases are
ill-conditioned at high degree, and useful targets (`1/x`, `sign`, `cos(tx)`)
push toward `|f| = 1` and need degree `~ tens to hundreds`.

## 2. The landscape of solvers

- **Optimization (Dong–Meng–Whaley–Lin, 2021).** Minimise the distance between
  the QSP-produced polynomial and the target over Chebyshev nodes, using the
  *symmetric* QSP ansatz (`phi_k = phi_{d-k}`). Robust; double precision reaches
  degree > 10,000. Needs a good initialisation and a Jacobian.
- **Root-finding / Fejér–Riesz (Haah, 2019; Ying, 2022).** Complete `f` to a
  complementary polynomial via spectral factorisation of `1 - |f|^2`, then strip
  phases by decimation. Non-iterative, but high-degree root-finding is
  numerically delicate.
- **"Machine-precision" halving (Chao–Ding–Gilyén–Huang–Szegedy, 2020).** A
  divide-and-conquer scheme justified by an algebraic uniqueness theorem; stable
  to thousands of angles in double precision.

`pyqsp` (Python) and `QSPPACK` (MATLAB) implement these. **No established C++
implementation exists** — which is the gap this solver fills.

## 3. This solver

A damped Gauss–Newton (Levenberg–Marquardt) optimiser, with two ideas that make
it reach high degree reliably in plain C++/Eigen.

### 3.1 Homotopy continuation

Cold-starting LM at `Phi = 0` stalls around degree 9 and fails near `|f| = 1`.
Instead, exploit a known exact solution: `Phi = 0` produces the Chebyshev
polynomial `T_d`. Morph the target along a straight path and warm-start each
step:

```
f_s(x) = (1 - s) * T_d(x) + s * f(x),   s : 0 -> 1.
```

At `s = 0` the solution is exact; each increment is a small perturbation solved
from the previous solution. This keeps every sub-problem near a known optimum,
so the solver tracks the solution branch all the way to the (hard) target.

### 3.2 Exact analytic Jacobian

Finite differences make the Jacobian cost `O(d)` residual evaluations per LM
iteration (`O(d^2)` work) and limit precision. Instead, differentiate the QSP
product analytically. Writing the factor list
`M = [E(phi_0), W, E(phi_1), W, ..., E(phi_d)]` (with `E(phi) = e^{i phi Z}`) and
prefix/suffix products `pre`, `suf`,

```
dU/dphi_k = pre[before E(phi_k)] * (i Z E(phi_k)) * suf[after],
```

so all `d+1` derivatives come from one `O(d)` sweep. Folding onto the symmetric
parameters (`dg/dfree_j = sum_{min(k,d-k)=j} dg/dphi_k`) gives the full Jacobian
in `O(m d)` for `m` nodes — about a `d`-fold speedup over finite differences.

### 3.3 A subtlety worth recording

`Phi = 0` is a **stationary point** of the symmetric response: `g = x cos(2 phi)`
for `d = 1`, whose gradient at `phi = 0` is exactly zero. Finite differences hide
this with a tiny nonzero value that *barely* escapes; the exact gradient is
precisely 0, so Gauss–Newton cannot move. The fix is to start the homotopy a hair
off the stationary point. (This is the kind of bug an exact Jacobian *exposes*
rather than causes — a good argument for analytic derivatives.)

## 4. Results

Achievable degree-`d` targets, solved to machine precision (this repo):

| target               | degree | residual | time (`-O3 -march=native`) |
|----------------------|--------|----------|----------------------------|
| `0.7 T_31`           | 31     | ~5e-15   | ~50 ms                     |
| `0.6 T_51`           | 51     | ~5e-15   | ~190 ms                    |
| `0.8 T_71`           | 71     | ~1e-14   | ~480 ms                    |
| `0.7 T_101`          | 101    | ~1e-14   | ~1.7 s                     |
| `0.97 T_11` (near 1) | 11     | ~2e-15   | fast                       |

The analytic Jacobian gives roughly a 10x speedup at degree 101 (16 s -> 1.7 s)
versus the finite-difference version. Downstream, this directly enables a
degree-25 QSVT matrix inversion (~2% on the spectrum) and an exact-to-1e-14
Hamiltonian simulation `e^{-iHt}` at degree 21.

## 5. Honest comparison and limits

- Against `pyqsp` / `QSPPACK`: this solver matches their accuracy in the regime
  it covers, and is the first such solver in **C++** (embeddable, no Python). It
  does **not** yet reach the `degree > 1000` regime of `sym_qsp` /
  root-finding — the homotopy LM is `O(steps * iterations * m * d)` and grows
  with degree.
- For very high degree / arbitrary precision, the principled next step is the
  non-iterative **Fejér–Riesz completion + decimation**, which is
  machine-precision by construction. That would close the gap to the state of
  the art and make the "first C++ high-degree QSP solver" claim unambiguous.

## References

- Low & Chuang, *Quantum Signal Processing* (qubitization), arXiv:1610.06546.
- Gilyén, Su, Low, Wiebe, *QSVT and beyond*, STOC 2019, arXiv:1806.01838.
- Martyn, Rossi, Tan, Chuang, *Grand Unification of Quantum Algorithms*,
  PRX Quantum 2021, arXiv:2105.02859.
- Dong, Meng, Whaley, Lin, *Efficient phase-factor evaluation in QSP*,
  PRA 2021, arXiv:2002.11649.
- Haah, *Product decomposition of periodic functions in QSP*, Quantum 2019,
  arXiv:1806.10236.
- Chao, Ding, Gilyén, Huang, Szegedy, *Finding angles for QSP with machine
  precision*, arXiv:2003.02831.
