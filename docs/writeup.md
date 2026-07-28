# An embeddable QSVT toolkit: a fast angle solver, a sparse block-encoding that beats dense synthesis, and real-molecule ground states

*Draft technical write-up — edit freely before posting. All numbers below come from
committed, reproducible harnesses in this repository.*

I built a from-scratch, C++ implementation of **Quantum Singular Value
Transformation (QSVT)** — the framework that unifies Hamiltonian simulation,
quantum linear systems, ground-state estimation, and more (Gilyén–Su–Low–Wiebe
2019; Martyn et al. "Grand Unification of Quantum Algorithms" 2021). Along the way
a few pieces turned out to be genuinely useful, and a few claims I'd have liked to
make don't survive scrutiny. This post is an honest tour of both.

## TL;DR — what's actually here

- **`qsp-angles`** — a QSP phase-factor (angle) solver in dependency-free C++.
  Machine precision to **degree 1000+**, cross-validated against `pyqsp`, and
  **13.7–68.7× faster** (narrowing with degree) running the *same* algorithm. It's the only **embeddable**
  one: link it into a compiled quantum toolchain with no Python — C ABI, Rust
  crate, ctypes, and Qiskit adapters included.
- **A sparse (LCU) block-encoding** compiled to native gates that uses **poly(n)**
  gates where dense unitary synthesis needs **~4ⁿ**. Head-to-head vs Qiskit's
  `qs_decomposition` on a local Hamiltonian: **324 vs 1,783 CNOTs at n=5**, and the
  gap is *unbounded* with n.
- **The full pipeline, validated on real data:** the **H₂ molecule** ground-state
  energy to **chemical accuracy** (vs the literature FCI), and the **Fermi–Hubbard**
  model to **sub-μHa** across interaction strength.

And the honest flip side, up front: the speedups are a compiled-vs-interpreted
constant on a *one-time* precompute; the applications are validated *densely at
small size*; and for a *generic* unitary, Qiskit's synthesizer beats mine. More on
each below — the honesty is the point.

## 1. The angle solver, and why "embeddable" matters more than "fast"

Every QSVT application reduces to one classical pre-computation: given a target
polynomial `f(x)`, find the phase sequence `Φ` that makes a QSP circuit realise
`f`. The community tools are `pyqsp` (Python) and `QSPPACK` (MATLAB). I ported the
symmetric-QSP Newton method (Dong–Lin–Ni–Wang 2023 — the algorithm behind pyqsp's
`sym_qsp`) to C++, with a hand-rolled radix-2 + Bluestein FFT so the core needs
**no external dependency at all**.

Running the identical algorithm, compiled, is tens of times faster than the Python
original — but I want to be clear that **this is not the interesting part**.
Angle-finding is a one-time, offline computation you do once per polynomial and
cache; nobody's pipeline is bottlenecked on it. The genuinely useful property is
**embeddability**: `pyqsp` and `QSPPACK` *structurally cannot* be linked into a
compiled C++/Rust quantum compiler or simulator without dragging in a runtime.
This one can — `#include` it, or call it over a flat C ABI. To my knowledge it's
the first standalone C++ QSP angle solver.

It's cross-validated against pyqsp to machine precision on random generic targets,
and packaged (installable from a checkout; not yet on PyPI), plus a Rust crate and a pure-C example.

## 2. The sparse block-encoding — the one real scaling win

QSVT applies a polynomial to the eigenvalues of a matrix `A` that you've
*block-encoded* into a unitary. The naive route — synthesize the block-encoding
dilation as a generic unitary — costs `~0.48·4ⁿ` gates (Qiskit's
`qs_decomposition` is near-optimal at this). That exponential wall is why dense
QSVT dies around 6–8 qubits.

For a matrix written as a sum of few Paulis — which is exactly what a local
Hamiltonian is — the **LCU (PREPARE + SELECT)** construction block-encodes it in
**poly(n)** gates. I implemented it end-to-end down to `{single-qubit, CNOT}`
(Möttönen state prep, multi-controlled Paulis with a shared AND-ladder
optimization), validated exact (`block = A/α` to ~1e-14). Head-to-head against
Qiskit synthesizing the dense dilation of the same Hamiltonian:

| n (system qubits) | LCU (this work) | Qiskit dense-dilation |
|--:|--:|--:|
| 4 | 172 CNOTs | 423 |
| 5 | 324 | 1,783 |
| →10 (extrapolated) | ~600 | ~2,000,000 |

Qiskit wins at trivial size (its generic synthesis is excellent); we pull ahead at
n=4 and the gap is unbounded — poly(n) vs `4ⁿ`. The honest caveats: the two are
*different* block-encodings (mine carries a subnormalization `α` and a `⌈log L⌉`
ancilla), and this win is specifically on **structured/local** operators — which
is exactly what QSVT consumes.

## 3. The full pipeline, on real data

The pieces connect: local Hamiltonian → LCU block-encoding → qubitization walk
operator → QSVT with `sym_qsp` angles → the target polynomial applied to `A/α`,
validated end to end (densely) to ~1e-16. On top of that:

- **Quantum linear systems** (`Ax = b` via QSVT matrix inversion): state
  **infidelity 7e-10 – 2e-7** vs the exact solution (operator error ~5e-4), condition numbers 4–16.
- **Ground-state energy** with the near-optimal Lin–Tong eigenstate filter:
  - **H₂** (STO-3G, the first molecule run on a quantum computer, O'Malley et al.
    2016): −1.137 Ha, **chemical accuracy** (0.036 mHa vs the literature FCI).
  - **Fermi–Hubbard** (the canonical strongly-correlated model): exact ground
    energy to **sub-μHa** across `U/t`, with the filter degree growing as the Mott
    gap shrinks — the honest resource–difficulty relationship.

**The caveat I won't bury:** these are validated at 2–4 qubits with dense linear
algebra. "H₂ to chemical accuracy" is a *correctness* demonstration that the
pipeline reproduces the eigenvalue you can already get classically — not a
chemistry result that competes with mature tools at scale. The sparse
circuit-on-a-simulator run at large n is the next step, not a finished one.

## What I'm *not* claiming

- No new algorithm — this is a careful, validated *implementation* of published
  methods.
- The speed advantage is compiled-vs-interpreted, on a non-bottleneck step.
- For generic (non-structured) unitaries, Qiskit's synthesizer is ~1.4× better.
- The applications are small and dense; scaling them is future work.

If you're building a **compiled quantum toolchain** and want QSP angles without a
Python runtime, or an open, honestly-validated reference for the QSVT →
block-encoding → applications path, this might be useful. If you want a
production chemistry stack, use Qiskit/OpenFermion.

## Reproduce it

```bash
pip install .                               # the embeddable angle solver (from a checkout)
python applications/qls/bench_vs_qiskit.py  # the Qiskit head-to-head
python applications/ground_state/h2_molecule.py     # H2 ground-state energy
python applications/ground_state/fermi_hubbard.py   # Hubbard across U/t
```

Every benchmark and validation harness is in the repo. Feedback, and especially
a real downstream user for the embeddable solver, very welcome.

*Built on Qrack (simulator), and standing on pyqsp / QSPPACK / the QSVT literature
— see the repository README for full references.*
