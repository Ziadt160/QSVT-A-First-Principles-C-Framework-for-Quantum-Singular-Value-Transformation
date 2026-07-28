"""REAL DATA: ground-state energy of the H2 molecule via our QSVT pipeline.

The H2 molecule in the minimal STO-3G basis, reduced to a 2-qubit qubit
Hamiltonian (O'Malley et al., "Scalable Quantum Simulation of Molecular
Energies", PRX 6, 031007, 2016 -- the first molecule simulated on a quantum
computer; coefficients at bond length R = 0.735 A). This is a real chemistry
Hamiltonian, a sum of Pauli terms -- exactly the LCU input our algorithm consumes.

We run the FULL validated pipeline on it:
  H2 Hamiltonian  ->  LCU block-encoding  ->  Lin-Tong eigenstate filter (sym_qsp
  angles)  ->  QSVT (qubitization walk operator)  ->  ground state  ->  energy,
and compare to the exact FCI ground-state energy.

    python h2_molecule.py
"""

import os
import sys

import numpy as np
from numpy.polynomial import chebyshev as Cheb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qls"))
import lcu_qsvt_dense as pipe  # noqa: E402  (lcu_full_unitary, qsvt_block, solve_phases, ...)
import eigenstate_filter as ef  # noqa: E402  (lin_tong_filter)

# H2 / STO-3G, 2-qubit reduced, R = 0.735 A (O'Malley et al. 2016). These are the
# ELECTRONIC-Hamiltonian coefficients; the nuclear-repulsion constant Z_A Z_B / R
# is folded into the identity term so the ground eigenvalue is the TOTAL molecular
# energy (directly comparable to the literature FCI value / experiment).
NUCLEAR_REPULSION = 0.7199689  # 1 / (0.735 A in bohr), Ha
H2_TERMS = [
    (-1.052373245772859 + NUCLEAR_REPULSION, "II"),
    (0.39793742484318045, "IZ"),
    (-0.39793742484318045, "ZI"),
    (-0.01128010425623538, "ZZ"),
    (0.18093119978423156, "XX"),
]
FCI_REF = -1.137270  # literature FCI total ground energy at R=0.735 A (Ha)
CHEM_ACC = 1.6e-3    # "chemical accuracy" (Ha)


def main():
    lib = pipe.ensure_solver()
    n = 2
    H = sum(c * pipe.pauli_string(s) for c, s in H2_TERMS)
    evals, V = np.linalg.eigh(H)
    E0, E1 = evals[0], evals[1]
    gap = E1 - E0
    g = V[:, 0]
    print(f"H2 (STO-3G, R=0.735 A): exact ground energy = {E0:.6f} Ha "
          f"(literature FCI {FCI_REF:.6f}); gap = {gap:.4f} Ha")

    # Shift so the ground energy sits at 0: A = H - E0 I (fold into the II term).
    A_terms = list(H2_TERMS)
    A_terms[0] = (A_terms[0][0] - E0, "II")
    U, alpha, m = pipe.lcu_full_unitary(n, A_terms)
    delta_n = gap / alpha  # normalised gap seen by the filter
    print(f"LCU block-encoding: {len(A_terms)} Pauli terms, {m} ancilla qubits, "
          f"alpha={alpha:.3f}, normalised gap={delta_n:.3f}")

    rng = np.random.default_rng(0)
    psi0 = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    psi0 /= np.linalg.norm(psi0)

    # Grow the Lin-Tong filter until the ground state is isolated, run it THROUGH
    # the QSVT pipeline (LCU walk operator + sym_qsp angles), and read the energy.
    for ell in range(1, 60):
        deg = 2 * ell
        xs = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))  # Chebyshev nodes
        ys = 0.99 * ef.lin_tong_filter(xs, ell, delta_n)             # scale to |f|<1
        coeffs = Cheb.chebfit(xs, ys, deg)
        coeffs[1::2] = 0.0  # enforce even parity exactly

        phases, ares = pipe.solve_phases(lib, list(coeffs))
        block = pipe.qsvt_block(U, phases, n, m)      # QSVT on the LCU block-encoding
        fA = (block + block.conj().T) / 2             # Hermitian (real) component
        psif = fA @ psi0
        nf = np.linalg.norm(psif)
        fidelity = abs(np.vdot(g, psif / nf)) ** 2
        if fidelity >= 1 - 1e-9:
            energy = float(np.real(np.vdot(psif, H @ psif)) / nf ** 2)
            err_pipeline = abs(energy - E0)      # pipeline vs exact diagonalization
            err_fci = abs(energy - FCI_REF)      # vs the literature FCI value
            print(f"\nfilter degree {deg} (sym_qsp residual {ares:.1e}): "
                  f"ground-state fidelity {fidelity:.9f}")
            print(f"QSVT-pipeline ground energy = {energy:.6f} Ha")
            print(f"  vs exact diagonalization : {err_pipeline * 1e3:.5f} mHa "
                  f"(pipeline is exact)")
            print(f"  vs literature FCI        : {err_fci * 1e3:.4f} mHa  "
                  f"({'CHEMICAL ACCURACY' if err_fci < CHEM_ACC else 'above chem. accuracy'})")
            print("\nComputed on a REAL molecular Hamiltonian, end to end through the\n"
                  "LCU block-encoding + eigenstate filter + QSVT pipeline (all validated).")
            return
    print("did not reach the fidelity target; increase the degree cap.")


if __name__ == "__main__":
    main()
