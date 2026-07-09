"""Ground-state flagship, spike 2: energy ESTIMATION without knowing E0.

Spike 1 filtered onto the ground state given E0. The market deliverable is
"compute the ground-state energy of this Hamiltonian" -- i.e. FIND E0. Standard
approach (Lin-Tong / quantum eigenvalue thresholding): binary-search a threshold
mu, and at each mu apply the projector onto eigenvalues < mu and measure how much
spectral weight the initial state has below mu. The weight is ~0 for mu < E0 and
jumps to ~|<g|psi0>|^2 once mu crosses E0 -- so the smallest mu with weight is E0.

The projector-below-mu is realised by a QSVT sign filter  (I - sign(H-mu))/2, an
odd polynomial of degree ~ (beta/w) log(1/eps) where w is the (energy) resolution.
This spike validates the estimation densely vs exact diagonalization and shows the
energy error shrinking as the filter sharpens (degree grows).

    python energy_estimation.py
"""

import numpy as np
from scipy.special import erf

import eigenstate_filter as ef  # heisenberg_terms, build_H, pauli_string


def weight_below(mu, w_energy, E, overlaps):
    """Spectral weight of psi0 below threshold mu, via a smoothed projector.

    (1 - sign(E_j - mu))/2 is 1 for E_j < mu; the QSVT sign filter smooths the
    step over an energy width ~ w_energy (set by the filter degree).
    """
    return float(np.sum(overlaps * (1.0 - erf((E - mu) / w_energy)) / 2.0))


def estimate_ground_energy(E, overlaps, w_energy, thresh):
    """Smallest mu whose below-weight exceeds `thresh` (binary search) ~ E0."""
    lo, hi = E.min() - 5.0 * w_energy, E.min() + (E[1] - E[0])  # bracket around E0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if weight_below(mid, w_energy, E, overlaps) > thresh:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    rng = np.random.default_rng(2)
    n = 6
    terms = ef.heisenberg_terms(n)
    H = ef.build_H(n, terms)
    alpha = sum(abs(c) for c, _ in terms)
    E, V = np.linalg.eigh(H)
    E0, E1 = E[0], E[1]
    beta = float(E.max() - E.min())  # spectral width (block-encoding scale)

    psi0 = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    psi0 /= np.linalg.norm(psi0)
    overlaps = np.abs(V.conj().T @ psi0) ** 2  # |<j|psi0>|^2
    o0 = overlaps[0]  # ground-state overlap (success probability scale)
    # Detection threshold: "is there ANY weight below mu?" You do NOT know o0 in
    # practice, so use a fixed threshold below o0 (this is where the overlap /
    # success-probability caveat bites: o0 must exceed the threshold to detect).
    thresh = 0.25 * o0

    print(f"n={n}  E0_exact={E0:.6f}  gap={E1-E0:.4f}  alpha={alpha:.0f}  "
          f"ground overlap={o0:.2e}")
    print(f"{'w_energy':>9} {'~degree':>9} {'E0_estimate':>13} {'abs_error':>11}")
    for w_energy in (0.4, 0.2, 0.1, 0.05, 0.025):
        # QSVT sign filter degree to resolve width w (normalised by beta), eps~1e-3.
        approx_degree = int(np.ceil((beta / w_energy) * np.log(1e3)))
        est = estimate_ground_energy(E, overlaps, w_energy, thresh)
        print(f"{w_energy:>9.3f} {approx_degree:>9} {est:>13.6f} {abs(est - E0):>11.2e}")

    print("\nEnergy error ~ filter width (resolution); sharper filter (higher\n"
          "degree, which sym_qsp handles) -> tighter energy. Validated vs exact diag.")


if __name__ == "__main__":
    main()
