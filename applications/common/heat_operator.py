"""Implicit-Euler operator for the 1-D Dirichlet heat equation.

A = I + r*L0 with L0 = tridiag(-1, 2, -1) on N = 2^n interior points and
r = alpha*dt/dx^2 the CFL-like ratio. alpha cancels, so the conditioning
kappa(A) -> 1 + 4r depends only on r and N.
"""

import numpy as np


def heat_matrix(n, r=0.4):
    """A = I + r*L0 for N = 2^n interior points."""
    N = 2 ** n
    L0 = (np.diag(2.0 * np.ones(N))
          + np.diag(-1.0 * np.ones(N - 1), 1)
          + np.diag(-1.0 * np.ones(N - 1), -1))
    return np.eye(N) + r * L0
