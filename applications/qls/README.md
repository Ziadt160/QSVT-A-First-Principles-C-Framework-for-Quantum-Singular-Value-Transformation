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

## W2 — the QLS pipeline, dense validation (done)

[`qls_dense.py`](qls_dense.py) wires the W1 polynomial all the way through:
`p ≈ c/x → sym_qsp angles → QSVT block P(A) → solve A x = b`, validated with exact
linear algebra (no simulator). The QSVT block is built to match the project's
verified construction (`src/QsvtPipeline.cpp`); the linear-system operator is the
Hermitian part `f(A) = (P(A)+P(A)†)/2 ≈ c·A⁻¹` (the circuit realises this via an
LCU of `U_Phi` and `U_Phi†`, +1 ancilla). Angles come from *this* project's
zero-dependency solver through its C ABI (built with `g++`, no Eigen — the sym_qsp
core is dependency-free).

**Result (run it: `python qls_dense.py`).** Across n = 1–4 qubits (up to 16×16)
and κ = 4, 8, 16 at ε = 1e-3:

| metric | result |
|---|---|
| state fidelity `\|<x_exact\|x_qsvt>\|` | **1.000000** (every case) |
| operator rel-error `‖f(A) − cA⁻¹‖ / ‖cA⁻¹‖` | ~5e-4 (≤ ε) |
| sym_qsp angle residual (degree 35–145) | ~1e-14 |
| un-amplified success probability | 0.04–0.38 (shrinks with κ, dim) |

So the **math of the pipeline is correct end to end**: fidelity 1.0 vs the exact
solution, operator error bounded by the approximation error, and the angle solver
comfortably handling the high degrees ill-conditioning demands. The success
probability is finite but < 1 — quantifying exactly why **amplitude amplification**
is the Phase-2 efficiency step.

## W3 / GPU — resource scaling and the gate-count wall

[`resource_scaling.cpp`](resource_scaling.cpp) compiles the inversion circuit at
growing system size (degree 25; pure gate synthesis, no simulator). The CNOT
count grows ~**4.3× per system qubit** (the `~0.58·4ⁿ⁺¹ × degree` law):

| system qubits | total qubits | CNOTs | gates | depth |
|--:|--:|--:|--:|--:|
| 1 | 2 | 100 | 351 | 251 |
| 2 | 3 | 700 | 1,926 | 1,425 |
| 3 | 4 | 3,400 | 8,826 | 6,675 |
| 4 | 5 | 14,800 | 37,626 | 28,875 |
| 5 | 6 | 61,600 | 155,226 | 120,075 |
| 6 | 7 | 251,200 | 630,426 | 489,675 |

Extrapolating: n=8 ≈ 4M gates, n=10 ≈ 70M, n=12 ≈ 1B.

**Does the GPU help? No — for this (dense) block-encoding.** The framework builds
with Qrack and runs on this machine's GTX 1650 (CUDA): the `qsvt_app` demo
executes all decompositions + applications to ~1e-14 on the GPU.
[`qsvt_sim_timing.cpp`](qsvt_sim_timing.cpp) times the actual circuit on the GPU —
n=1 (100 CNOTs, 2 qubits) runs in ~30 ms (mostly CUDA context init). But the GPU
**cannot** rescue large n here: at n=8 you'd apply ~4M *sequential* gates to a
statevector of only 2⁹ = 512 amplitudes — the GPU is starved (too few amplitudes
to parallelize over, kernel-launch overhead dominates) while the sequential gate
count is the real cost. GPU statevector simulation wins in the opposite regime —
**many qubits, few gates** — which requires the **sparse block-encodings of
Phase 2**, not the dense Shannon decomposition used here.

> Known gap (honest): `qsvt_sim_timing` segfaults at n ≥ 2 — the QSVT
> circuit-on-Qrack execution path needs hardening before gate-level confirmation
> at n ≥ 2. The dense (operator-level) W2 validation above already confirms the
> math; this is a simulator-robustness fix for proper W2b.

## Phase 2 — sparse block-encoding: breaking the gate-count wall

[`lcu_block_encoding.py`](lcu_block_encoding.py) is the spike for the fix. For a
matrix written as a sum of few unitaries `A = Σ_k c_k P_k` (a local Hamiltonian),
the **LCU (PREPARE + SELECT)** construction block-encodes `A/α` (`α = Σ|c_k|`)
in `O(L · poly(n))` gates instead of the dense `~0.58·4ⁿ`:

```
PREPARE |0>_a = Σ_k sqrt(|c_k|/α) |k>        SELECT = Σ_k |k><k|_a ⊗ (c_k/|c_k|)P_k
U = (PREPARE† ⊗ I) SELECT (PREPARE ⊗ I)   =>   <0|_a U |0>_a = A/α
```

**Result (run it: `python lcu_block_encoding.py`).** For a 1D transverse-field
Ising chain (`L = 2n−1` terms), validated densely:

| n | LCU CNOTs | dense CNOTs | speedup | block error |
|--:|--:|--:|--:|--:|
| 4  | 68  | 594       | 9×    | 3e-16 |
| 6  | 136 | 9,503     | 70×   | 1e-17 |
| 8  | 174 | 152,000   | 874×  | 6e-16 |
| 10 | 282 | 2,433,000 | 8,627× | — (dense check capped at n≤8) |

The block-encoding is exact (`block == A/α` to ~1e-16) and the gate count grows
**~linearly** in n, vs `4ⁿ` for the dense path. This is the construction that
makes large n feasible — and `282 gates on 15 qubits` is the **few-gates,
many-qubits** regime where GPU statevector simulation (the Qrack/CUDA backend)
finally wins.

## Next

- **Phase 2 (C++):** implement LCU PREPARE/SELECT as native-gate primitives
  (the framework's `Lcu` already does the Pauli decomposition), wire it into
  `QsvtPipeline` as an alternative to the dense block-encoding, and confirm on
  Qrack at n ≫ 6 (where dense dies and the GPU helps).
- **W2b:** harden the QSVT circuit-on-Qrack path (fix the n ≥ 2 segfault).
- **W3/W4:** T-count estimate, full sweeps + the degree-vs-κ and cost curves for
  the paper. See the [scope doc](../../docs/phase1-qls-scope.md).
