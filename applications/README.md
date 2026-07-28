# QSVT applications

Worked, validated applications of the QSVT toolkit. They all consume the **same
primitives** — a sparse **LCU block-encoding** (poly(n) gates), the fast
**`sym_qsp`** angle solver, and a **qubitization QSVT** sequence — differing only
in the target polynomial. Everything here is validated against exact linear
algebra (and, for the physical models, against known/analytic ground truth).

## The shared pipeline

```
local Hamiltonian A = Σ c_k P_k
     │  LCU block-encoding  (applications/qls: poly(n) gates, breaks the 4ⁿ wall)
     ▼
   U  (Hermitian, U² = I)  →  qubitization walk  W = U·(2Π − I)
     │  QSVT sequence with sym_qsp angles for the target polynomial f
     ▼
   <0|_a U_Φ |0>_a  =  f(A/α)       (validated to ~1e-16)
```

## [`qls/`](qls) — Quantum linear systems, and the pipeline internals

- **`Ax = b`** via QSVT matrix inversion — state **infidelity 7e-10 – 2e-7** vs the exact
  solution (up to 16×16, κ = 4–16). The `1/x` approximation scales as the optimal
  `O(κ log 1/ε)`.
- The **sparse block-encoding** (`LcuBlockEncoding`), its **~40%-optimized**
  native-gate synthesis, the **dense end-to-end QSVT-on-LCU validation**, and the
  honest **gate-count / GPU scaling analysis** (why the dense path hits a `4ⁿ`
  wall and the sparse one doesn't) all live here.

## [`ground_state/`](ground_state) — Ground-state energy on real data

The market flagship. Filter onto the ground state with the near-optimal **Lin–Tong
eigenstate filter**, then read the energy — or **estimate** it via a threshold
binary search. Demonstrated on two real systems:

- **H₂ molecule** (STO-3G, O'Malley et al. 2016): ground energy to **chemical
  accuracy** vs the literature FCI.
- **Fermi–Hubbard model** (strongly-correlated electrons): exact ground energy to
  **sub-μHa** across interaction strength `U/t`, with the filter degree growing as
  the Mott gap shrinks (the honest resource–difficulty relationship).

## Running

The Python spikes self-build the `sym_qsp` C ABI (via `g++`, no Eigen needed) and
drive it over `ctypes` — no wheel build, no Qrack required. C++ validators compile
against `include/` + `src/`. See each subdirectory's README for commands.
