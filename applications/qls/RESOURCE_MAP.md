# QSVT matrix inversion: a measured resource map over (κ, ε)

What a QSVT linear solve actually costs, computed with **real angles** rather than
asymptotic bounds, across 28 (condition number, target error) configurations.

Produced by [`resource_phase_diagram.py`](resource_phase_diagram.py) —
**468 s for all 28 solves on a laptop**, degrees from 9 to 1985.
Data: [`resource_phase_diagram.csv`](resource_phase_diagram.csv) ·
Plot: [`resource_phase_diagram.png`](resource_phase_diagram.png)

## Measured scaling (ε = 1e-3, per doubling of κ)

| quantity | exponent | meaning |
|---|---|---|
| polynomial degree | **κ^1.03** | block-encoding queries per QSVT application |
| queries / accepted sample | **κ^3.05** | end-to-end, post-selected, no amplification |
| queries / accepted sample, with AA | **κ^2.04** | with amplitude amplification |

Degree grows **linearly** in κ — the optimal `O(κ log 1/ε)` rate, confirming the
near-minimax construction rather than the `O(κ²)` of the Childs–Kothari–Somma closed
form. The end-to-end cost is a factor of κ² worse than the degree because the
subnormalization `c ≈ 1/2κ` shrinks the post-selection success probability as `c²`;
amplitude amplification recovers exactly one factor of κ, giving Θ(κ²).

The practical reading: **at κ=128, ε=1e-6 a single accepted sample costs ~1.1×10⁸
block-encoding queries** (~4.6×10⁵ with amplitude amplification). Degree alone
understates the true cost by κ² — a gap that asymptotic tables tend to hide.

## Where angle-finding is reliable (and where it is not)

25 of 28 solves reached machine precision (residual ≤ 1.3e-13). Three did not, all at
the **loosest** tolerance:

| κ | ε | degree | residual |
|--:|--:|-------:|---------:|
| 16 | 1e-2 | 93  | 7.0e-05 |
| 32 | 1e-2 | 191 | 2.4e-03 |
| 64 | 1e-2 | 391 | 2.1e-03 |

Counter-intuitively the *easier* accuracy target is the harder QSP problem: at loose ε
the minimal-degree near-minimax fit equioscillates with larger amplitude, so `|p|` sits
closer to the QSP feasibility boundary `|p| ≤ 1` and the Newton iteration is solved
near that boundary. Every ε ≤ 1e-3 row converged cleanly.

**Mitigation, already implemented:** `solve_angles()` verifies the residual and bumps
the degree until the solve is clean. This sweep deliberately reports the *minimal*
degree without that guard, so the boundary is visible rather than papered over.

## Why this map did not already exist

It needs a few hundred angle solves at degrees into the thousands:

- **pyqsp / QSPPACK** — 1–70 s per solve at these degrees, so this sweep is hours.
- **PennyLane** — cannot do it at all: `poly_to_angles` takes coefficients in the
  monomial basis, and converting a degree-79 Chebyshev inversion polynomial overflows
  double precision (coefficients reach ~10²⁵), so a target with a true sup-norm of
  0.999 is rejected as violating `|P(x)| ≤ 1`.
- **Qiskit** — ships no QSP angle solver.

Here it is 468 s on a laptop, which is what makes the map routine to regenerate at a
different (κ, ε) grid, a different target function, or a different accuracy model.

## Caveats

- Costs are in **block-encoding queries**, the hardware-independent unit. Converting to
  gates requires a block-encoding cost model; for a local Hamiltonian see
  [`lcu_be_validate.cpp`](lcu_be_validate.cpp).
- `queries/sample` uses the conservative `d/c²`, i.e. it assumes `‖A⁻¹b‖ ≈ 1`. A `b`
  aligned with small singular values succeeds more often; one aligned with large ones
  less.
- Single run per configuration, single-threaded, pinned (Intel i7-9750H). Wall-clock is
  indicative; the degree and query columns are exact and machine-independent.
