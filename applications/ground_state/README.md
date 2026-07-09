# QSVT ground-state energy estimation (flagship application)

The market flagship for the QSVT vertical: estimate the **ground-state energy** of
a local (chemistry/materials) Hamiltonian. This is the application quantum
computing sells itself on — drug discovery, catalysis, batteries, materials — and
it consumes exactly the primitives already built here:

- a local Hamiltonian `H = Σ_k c_k P_k` → the **LCU block-encoding** (poly(n) gates,
  `../qls` Phase 2), *not* the dense 4ⁿ path;
- an **eigenstate-filtering polynomial** applied via QSVT → the fast **sym_qsp**
  angle solver (high degree, machine precision).

## Spike 1 — the eigenstate filter (done, classically validated)

[`eigenstate_filter.py`](eigenstate_filter.py) is the algorithmic heart. Shift `H`
so the ground energy is at 0, then apply a polynomial that is ~1 at the ground
eigenvalue and ~0 on the rest of the spectrum; applying it to (almost) any initial
state projects onto the ground state. **Algorithmic improvement:** use the
near-optimal **Lin–Tong eigenstate filter** (arXiv:2002.12508), degree
`~ (1/Δ)·log(1/ε)`, rather than a naive `~1/Δ²` filter.

**Result (run it: `python eigenstate_filter.py`).** 1D antiferromagnetic
Heisenberg chain, validated against exact diagonalization:

| n | gap Δ | filter degree | ground fidelity | energy error | success prob |
|--:|--:|--:|--:|--:|--:|
| 4 | 2.64 | 22  | 0.9994 | 4.6e-3 | 1.7e-2 |
| 6 | 1.97 | 48  | 0.9994 | 6.0e-3 | 1.2e-2 |
| 8 | 1.57 | 104 | 0.9991 | 1.1e-2 | 2.8e-4 |

The filter reaches ~0.999 ground-state fidelity, the energy converges to `E₀`, and
the degree scales as **gap⁻¹·¹⁹ ≈ the optimal 1/Δ** — the near-optimal filter is the
algorithmic win over naive projectors.

**Honest caveats (scoped in from the start):**
- **Success probability ~ initial-state overlap** (1.7e-2 → 2.8e-4 as n grows): a
  random state has exponentially small overlap on the ground state. The filter is
  correct, but efficient preparation needs a good initial state (e.g. a
  mean-field / product state) + amplitude amplification. Ground-state *preparation*
  is a genuinely hard, separate problem; we scope the demo as **energy estimation
  given a reasonable initial state**, at small validatable sizes.
- **Energy accuracy is tied to fidelity/degree** — tighter energy needs higher
  degree (which sym_qsp handles).
- Chemistry is crowded; the angle here is the **open-source, embeddable QSVT
  pipeline**, not beating anyone's chemistry accuracy.

## Spike 2 — energy *estimation* without knowing E₀ (done)

[`energy_estimation.py`](energy_estimation.py) is the market deliverable: *find*
E₀, not just filter given it. Binary-search a threshold μ; at each μ apply the
QSVT projector onto eigenvalues < μ (`(I − sign(H−μ))/2`, a sign filter of degree
`~ β/w`) and measure the initial state's spectral weight below μ. The weight is ~0
for μ < E₀ and jumps once μ crosses E₀, so binary search on the threshold locates
E₀. Validated vs exact diagonalization (Heisenberg, n=6, E₀ = −9.9743):

| filter width w | ~degree (β/w) | E₀ estimate | abs error |
|--:|--:|--:|--:|
| 0.40  | 259   | −10.165 | 1.9e-1 |
| 0.20  | 518   | −10.070 | 9.5e-2 |
| 0.10  | 1,035 | −10.022 | 4.8e-2 |
| 0.05  | 2,069 | −9.998  | 2.4e-2 |
| 0.025 | 4,138 | −9.986  | 1.2e-2 |

The accuracy–cost tradeoff is clean: **energy error ~ β/degree** (halve the error →
~2× degree), and those degrees are in sym_qsp's fast range. Honest caveats: (a) the
overlap `|<g|ψ0>|²` must exceed the detection threshold (here o₀ ≈ 4e-2) — the
success-probability limit again; (b) simple thresholding gives `~1/ε` scaling;
Heisenberg-limited methods (QPE-style) do better and are a natural extension.

## Real molecule — H₂ ground-state energy (done)

[`h2_molecule.py`](h2_molecule.py) runs the whole pipeline on **real chemistry
data**: the H₂ molecule in the STO-3G basis, reduced to a 2-qubit qubit
Hamiltonian (O'Malley et al., *Scalable Quantum Simulation of Molecular Energies*,
PRX 6 031007, 2016 — the first molecule simulated on a quantum computer; R=0.735 Å).
The electronic coefficients plus the nuclear-repulsion constant give the total
molecular Hamiltonian — a sum of 5 Pauli terms, exactly the LCU input.

Pipeline: H₂ Hamiltonian → LCU block-encoding → Lin–Tong eigenstate filter
(sym_qsp angles) → QSVT (qubitization walk operator) → ground state → energy.

**Result (`python h2_molecule.py`):**

| quantity | value |
|---|---|
| QSVT-pipeline ground energy | **−1.137306 Ha** |
| vs exact diagonalization | 0.00000 mHa (pipeline exact) |
| vs literature FCI (−1.137270) | **0.036 mHa — chemical accuracy** |
| filter degree / fidelity | 36 / 0.9999999999 |
| block-encoding | 5 Pauli terms, 3 ancilla, poly(n) gates |

The H₂ **ground-state energy — a real, physical, experimentally-relevant quantity —
computed end to end through our QSVT pipeline to chemical accuracy.** This is the
market-facing demonstration: the same machinery that breaks the gate-count wall
(LCU) and finds the phases fast (sym_qsp) computes a real molecule's energy.

## Next

- **Dissociation curve:** repeat across bond lengths R for the H₂ binding curve
  (the canonical chemistry figure); add LiH.
- **Wire the pipeline in C++** and run on Qrack (capstone; the poly(n) LCU circuit
  + GPU regime), with resource + accuracy tables vs (n, gap, ε) for the paper.
