# QSVT Quantum Linear Systems (Phase 1)

Building a benchmarked QSVT solver for `A x = b` (prepare `|x> ∝ A⁻¹|b>`).
See [../../docs/phase1-qls-scope.md](../../docs/phase1-qls-scope.md) for the full scope.

## W1 — the `1/x` approximation (done, classically validated)

[`oneoverx_approx.py`](oneoverx_approx.py) builds an odd polynomial approximating
`c/x` on `D = [1/κ, 1] ∪ [−1, −1/κ]`, scaled to `|p| ≤ 1` (a valid QSP target), and
measures the degree needed vs condition number `κ`.

**Result (run it: `python oneoverx_approx.py`).** A least-squares fit in the odd
Chebyshev basis reaches relative error `ε` with degree scaling **~κ^1.04**
(i.e. the optimal `O(κ log 1/ε)`), versus the Childs–Kothari–Somma closed form
`(1-(1-x²)^b)/x` at **~κ²**:

| κ | fit degree (ε=1e-3) | CKS baseline |
|--:|--:|--:|
| 8  | 73  | 877    |
| 16 | 145 | 3,529  |
| 32 | 293 | 14,141 |
| 64 | 579 | 56,581 |

The fit's degrees (≈100s) are exactly the regime where the `sym_qsp` solver is
fast and reaches degree 1000+ — so the angle-solver work directly enables
ill-conditioned inversion. The polynomial's Chebyshev coefficients feed `sym_qsp`
unchanged; the reported subnormalization `c` (≈ 1/(2κ)) sets the un-amplified
success probability for W2.

## Next (W2)

Wire this polynomial through `QsvtPipeline` (block-encode normalized Hermitian
`A`, apply QSVT, post-select, read out `|x>` on Qrack), validate fidelity vs exact
`A⁻¹b`, and start the resource table. See the scope doc.
