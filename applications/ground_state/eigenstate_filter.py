"""Ground-state flagship, spike 1: the eigenstate-filtering polynomial (the
algorithmic heart of QSVT ground-state energy estimation), validated densely.

A local Hamiltonian H = Σ_k c_k P_k (here a 1D Heisenberg chain, a native LCU
input) is block-encoded, shifted so its ground energy sits at 0, and a polynomial
`p` is applied that is ~1 at the ground eigenvalue and ~0 on the rest of the
spectrum. Applying p(H) to (almost) any initial state projects onto the ground
state. The cost is set by the filter DEGREE, which is set by the spectral gap.

Algorithmic improvement: use the near-optimal **Lin-Tong eigenstate filter**
(arXiv:2002.12508), degree ~ (1/Δ)·log(1/ε), instead of a naive filter. This
spike validates fidelity + energy error vs EXACT diagonalization and measures the
degree-vs-gap scaling (the resource driver). No quantum yet; angles/QSVT come next.

    python eigenstate_filter.py
"""

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def pauli_string(s):
    m = np.array([[1]], dtype=complex)
    for ch in s:
        m = np.kron(m, PAULI[ch])
    return m


def heisenberg_terms(n):
    """1D antiferromagnetic Heisenberg (open chain): H = Σ_i (XX+YY+ZZ)_{i,i+1}."""
    terms = []
    for i in range(n - 1):
        for p in "XYZ":
            s = ["I"] * n
            s[i] = s[i + 1] = p
            terms.append((1.0, "".join(s)))
    return terms


def build_H(n, terms):
    dim = 2 ** n
    H = np.zeros((dim, dim), complex)
    for c, s in terms:
        H += c * pauli_string(s)
    return H


def cheb_T(l, y):
    """Chebyshev T_l(y), numerically robust for |y| > 1 (grows as cosh)."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    inside = np.abs(y) <= 1.0
    out[inside] = np.cos(l * np.arccos(y[inside]))
    yo = y[~inside]
    sign = np.where(yo < 0, (-1.0) ** l, 1.0)
    out[~inside] = sign * np.cosh(l * np.arccosh(np.abs(yo)))
    return out


def lin_tong_filter(x, l, delta):
    """Even degree-2l polynomial: 1 at x=0, ≤ ~2 exp(-2 l delta) for |x| >= delta."""
    y = (2.0 * x ** 2 - 1.0 - delta ** 2) / (1.0 - delta ** 2)
    y0 = (-1.0 - delta ** 2) / (1.0 - delta ** 2)
    return cheb_T(l, np.atleast_1d(y)) / cheb_T(l, np.atleast_1d(np.array(y0)))[0]


def main():
    rng = np.random.default_rng(1)
    eps = 1e-3
    print(f"{'n':>3} {'E0_exact':>10} {'gap':>8} {'alpha':>7} {'gap_norm':>9} "
          f"{'deg':>5} {'fidelity':>10} {'E_err':>10} {'succ_prob':>10}")
    ns, gaps_norm, degs = [], [], []
    for n in (4, 6, 8):
        terms = heisenberg_terms(n)
        H = build_H(n, terms)
        alpha = sum(abs(c) for c, _ in terms)  # LCU subnormalization
        w, V = np.linalg.eigh(H)
        E0, E1 = w[0], w[1]
        gap = E1 - E0
        g = V[:, 0]  # exact ground state

        # Shift so the ground energy is at 0, normalize to spectrum in [0, 1].
        M = H - E0 * np.eye(2 ** n)
        beta = float(np.max(w) - E0)  # ||M||
        Meig = (w - E0) / beta  # normalized eigenvalues, ground at 0
        gap_norm = gap / beta

        # Random initial state (small overlap on |g>, like the real setting).
        psi0 = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
        psi0 /= np.linalg.norm(psi0)

        # Grow the Lin-Tong filter degree until ground-state fidelity >= 1 - eps.
        for l in range(1, 4000):
            filt = lin_tong_filter(Meig, l, gap_norm)  # p(eigenvalue) per mode
            psif = V @ (filt * (V.conj().T @ psi0))  # p(M/beta) |psi0>
            norm = np.linalg.norm(psif)
            fidelity = abs(np.vdot(g, psif / norm)) ** 2
            if fidelity >= 1 - eps:
                energy = float(np.real(np.vdot(psif, H @ psif)) / norm ** 2)
                succ = norm ** 2
                deg = 2 * l
                print(f"{n:>3} {E0:>10.5f} {gap:>8.4f} {alpha:>7.1f} {gap_norm:>9.4f} "
                      f"{deg:>5} {fidelity:>10.6f} {abs(energy-E0):>10.2e} {succ:>10.2e}")
                ns.append(n); gaps_norm.append(gap_norm); degs.append(deg)
                break

    # Degree-vs-gap scaling: Lin-Tong predicts degree ~ 1/gap_norm.
    if len(degs) >= 2:
        m, _b = np.polyfit(np.log(gaps_norm), np.log(degs), 1)
        print(f"\nfilter degree scales ~ gap_norm^{m:.2f}  "
              f"(Lin-Tong optimal: ~ gap^-1, i.e. exponent -1)")


if __name__ == "__main__":
    main()
