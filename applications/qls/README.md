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

**Idealized estimate** (`python lcu_block_encoding.py`, an analytic gate model)
vs the **actual C++ native-gate synthesis** (`LcuBlockEncoding`, decomposed to
`{single-qubit, CNOT}` and validated by `denseCircuit`). For a 1D transverse-field
Ising chain (`L = 2n−1` terms):

| n | est. CNOTs | **C++ CNOTs** | dense CNOTs | C++ vs dense | block err |
|--:|--:|--:|--:|--:|--:|
| 2 | 24  | 42  | 37     | 0.9×  | 2e-15 |
| 4 | 68  | 172 | 594    | 3.5×  | 6e-15 |
| 5 | 117 | 324 | 2,376  | 7.3×  | 1e-14 |
| 6 | 136 | 390 | 9,503  | 24.4× | 1e-14 |

The native-gate block-encoding is **exact** (`block == A/α` to ~1e-14) and **poly(n)**
(42→390, ~linear vs `4ⁿ`). The synthesis was optimized (see below) to ~2–3× the
idealized estimate; it beats the dense path from n≈4, and poly(n) vs `4ⁿ` means the
advantage then explodes (24.4× at n=6, ~2000×+ by n=10). This is the
few-gates / many-qubits regime where the Qrack/CUDA GPU finally wins.

Implemented in [`include/LcuBlockEncoding.hpp`](../../include/LcuBlockEncoding.hpp) /
[`src/LcuBlockEncoding.cpp`](../../src/LcuBlockEncoding.cpp); validated by
[`lcu_be_validate.cpp`](lcu_be_validate.cpp): PREPARE (uniformly-controlled Ry
state-prep) + SELECT (multi-controlled Paulis), all from `{single, CNOT}`.

**Synthesis optimization (~40% CNOT reduction):** the first version rebuilt the
multi-control Toffoli ladder for every Pauli factor of a term; now a weight-`w`
term shares ONE AND-ladder across its `w` factors, and a unitary-preserving
peephole pass cancels the X-mask CNOTs between consecutive terms. Same exact
block, ~40% fewer CNOTs (n=6: 636 → 390). Further gains available via unary
iteration (Babbush et al.) — amortizing the SELECT control logic across all terms.

### Head-to-head vs Qiskit ([`bench_vs_qiskit.py`](bench_vs_qiskit.py))

Qiskit (2.4.2) has no sparse block-encoding primitive, so a Qiskit user
block-encodes a local Hamiltonian by synthesizing the **dense dilation unitary**
with `qs_decomposition` (near-optimal generic synthesis, `~0.48·4ⁿ` CNOTs). Our
LCU is poly(n):

| n (system qubits) | **our LCU** CNOTs | Qiskit dense-dilation CNOTs |
|--:|--:|--:|
| 2 | 42  | 19    |
| 3 | 126 | 95    |
| 4 | 172 | **423** |
| 5 | 324 | **1,783** |

Qiskit wins at trivial size; **we pull ahead at n=4 and the gap is unbounded**
(poly(n) vs `4ⁿ` — ~thousands× by n=10). Honest caveats: (a) the two are
*different* block-encodings — ours carries an `α` subnormalization and a
`⌈log L⌉`-qubit ancilla; the gate-cost *scaling* is the point; (b) for a **generic**
(non-sparse) unitary, Qiskit's QSD is **~1.4× better** than our from-scratch
synthesizer (n=5: 423 vs 592) — Qiskit leads there, and we say so. The win is
specifically on *structured / local* operators, which is what QSVT actually consumes.

## Sparse QSVT pipeline — dense end-to-end validation (done)

[`lcu_qsvt_dense.py`](lcu_qsvt_dense.py) proves the *whole sparse pipeline* connects,
densely: a local Hamiltonian `A = Σ c_k P_k` → LCU block-encoding `U` → QSVT with
this project's sym_qsp angles → the target polynomial applied to `A/α`.

Key fact used: for Hermitian `A` (real coeffs) the LCU `U = PREP† SELECT PREP` is
**Hermitian and `U² = I`** — a reflection. The qubitization **walk operator**
`W = U·(2Π−I)` then has a genuine rotation action, so the standard QSVT sequence
`R(φ₀)·Π_k[W·R(φ_k)]` with the Wx-convention sym_qsp phases applies the polynomial.
Verified: `Re(block) = 0.7·T₃(A/α)` to **5.4e-16**. This is the integration all the
pieces (LCU + sym_qsp + QSVT) were built for — it unblocks the C++/Qrack capstone.

## Next

- **Phase 2 (C++) capstone:** wire `LcuBlockEncoding` + the walk operator + sym_qsp
  angles into a QSVT run (dense math now validated above), then run on **Qrack** at
  n ≫ 6 — where the poly(n) LCU circuit + the GPU finally pay off together.
- **W2b:** harden the QSVT circuit-on-Qrack path (fix the n ≥ 2 segfault).
- **W3/W4:** T-count estimate, full sweeps + the degree-vs-κ and cost curves for
  the paper. See the [scope doc](../../docs/phase1-qls-scope.md).
