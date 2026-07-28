# Phase 1 — QSVT Quantum Linear Systems (QLS) solver: scope

**Goal.** Turn the toy degree-25 matrix inversion into a *benchmarked* QSVT
quantum-linear-systems solver, and produce one citable, falsifiable claim:

> For an n-qubit Hermitian `A` with condition number `κ`, the pipeline prepares
> `|x> ∝ A⁻¹|b>` to fidelity `≥ 1 − ε` using a degree `d ≈ O(κ log(1/ε))`
> circuit, with CNOT / T / depth / qubit cost measured and validated against
> exact linear algebra for n up to 5–6.

This *uses and justifies* the sym_qsp work: ill-conditioned systems need high
degree, which the fast Newton solver makes feasible (and the old homotopy solver
could not reach).

## Scope

**In:** Hermitian `A` (normalized contraction); a principled, classically
validated `1/x` approximation; the QLS driver on top of `QsvtPipeline`; resource
estimation incl. a first T-count; a validation + benchmark harness over
`(n, κ, ε, degree)`.

**Out (deferred on purpose, named for credibility):** general non-Hermitian `A`
(Hermitian dilation → Phase 2); amplitude amplification to boost success
probability (measure & report it instead); large-scale `|b>` state prep (use a
simple/known `b`); hardware execution.

## Workstreams

- **W1 — `1/x` approximation (the front door; accuracy crux; DO FIRST).**
  Odd polynomial approximating `c/x` on `[1/κ, 1] ∪ [−1, −1/κ]`, scaled so
  `|p| ≤ 1` on `[−1, 1]`. Expose `degree(κ, ε)`. Validate *classically* before any
  quantum step, so approximation error is decoupled from QSVT error. Reusable as
  the Phase-2 minimax front door. Output: Chebyshev coefficients that feed sym_qsp.
- **W2 — QLS driver at scale.** Extend `compileMatrixFunction` / `QsvtPipeline`:
  block-encode normalized `A`, apply QSVT with the W1 polynomial, run on Qrack,
  post-select the ancilla, read out `|x>`. Track the subnormalization `α` and
  report the (un-amplified) success probability honestly. Scale n = 1 → 5–6.
- **W3 — Resource estimation.** Extend the CNOT/depth counter with qubit count,
  depth, and a first **T-count** (rotation→T, Ross–Selinger-style estimate). Cost
  as a function of `(n, κ, ε, degree)` — this table is the headline result.
- **W4 — Validation + benchmark.** Classical ground truth `A⁻¹b`; fidelity /
  2-norm error of the QSVT `|x>`. Sweep fidelity-vs-degree, **degree-vs-κ**
  (verify the `O(κ log 1/ε)` law), cost-vs-(n, κ). Reproducible script + committed
  CSVs, same discipline as the angle benchmark.

## Risks (named) + mitigations

- **Circuit blow-up at scale** (~0.58·4ⁿ CNOTs per block-encoding × degree): sim
  likely caps at n ≈ 5–6. → Report it as a *finding* (cost curve + simulability
  ceiling), not a failure; it motivates Phase-2 synthesis work.
- **Tiny success probability without AA** (~1/(ακ)²): → report conditional
  fidelity + measured success prob; flag AA as the Phase-2 fix.
- **Approximation vs QSVT error entanglement:** → W1 validates the polynomial
  classically first, so error is attributable.
- **T-count estimate accuracy:** → label as an estimate with the method cited;
  approximate-but-honest beats absent.

## Timeline (~8 weeks, solo)

- Wk 1–2: **W1** — gate: approximation error matches theory; degree-vs-κ measured.
- Wk 3–4: **W2** (n ≤ 3, Hermitian) — gate: `|x>` fidelity ≥ 1−ε on a 2-qubit κ≈5 system.
- Wk 5–6: **W3 + scale W2 to n ≈ 5–6** — gate: resource table populated; ceiling found.
- Wk 7–8: **W4** — gate: the degree-vs-κ law and cost curves demonstrated, reproducible.

## Definition of done

A committed `applications/qls/` with the solver, the reproducible
benchmark + validation harness, CSVs/plots, and a short writeup of the
degree-vs-κ and cost-vs-(n, κ) results — the raw material for the JOSS paper's
central figure — plus the reusable `1/x` front door.
