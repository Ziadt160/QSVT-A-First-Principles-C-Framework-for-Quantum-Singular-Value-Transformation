"""When angle caching stops working: adaptive time stepping.

The standard objection to an embeddable angle solver is that phase factors are
computed once, offline, and cached -- PennyLane's own documentation tells you to
generate them in MATLAB. That is true when the problem has ONE (kappa, degree).

It stops being true the moment the timestep adapts. An adaptive implicit-Euler
solver chooses dt from a local error estimate, so the CFL ratio r = alpha*dt/dx^2
changes every step, kappa = 1+4r changes with it, the required polynomial degree
changes with kappa, and each new (kappa, degree) needs its own phase factors --
none of which are known before the run starts, because dt depends on the solution.

This measures how often that happens on a genuine adaptive solve, and what the
angle work costs at each solver's measured rate.

*** RESULT: NEGATIVE. This workload does NOT justify in-loop angle generation. ***

Measured on the 1-D heat equation at n=6, horizon T=1, tolerance 2e-4:
118 accepted steps, but kappa only ranges 1.80-3.59, which maps to degrees 13-27
and just **3 distinct (kappa, degree) pairs**. Three phase-factor solves are
trivially precomputed and cached, so the caching objection survives here intact.

The reason is structural, not a quirk of the tolerance: for implicit Euler on a
diffusion operator kappa = 1 + 4r is bounded and varies slowly, and the required
degree depends on kappa only logarithmically-ish over that narrow band. The
parameter space collapses to a handful of points no matter how many steps run.

What this rules out, and what it does not: it rules out "adaptive time stepping
on a well-conditioned parabolic PDE" as the motivating workload. It says nothing
about problems where the operator itself changes -- nonlinear PDEs whose Jacobian
is rebuilt each step, advection-dominated regimes where kappa spans orders of
magnitude, or parameter studies over many operators. Those remain plausible, and
remain unmeasured. Until one is measured, the honest case for embeddability is a
DEPLOYMENT argument (no interpreter anywhere in a compiled stack) rather than a
throughput one.

    python adaptive_stepping.py
"""

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import qsp_bridge as Q  # noqa: E402

ALPHA = 0.01
EPS_POLY = 1e-3          # angle accuracy target
TOL = 2e-4               # local truncation tolerance driving the adaptation


def heat_matrix(n, r):
    N = 2 ** n
    L0 = (np.diag(2.0 * np.ones(N))
          + np.diag(-1.0 * np.ones(N - 1), 1)
          + np.diag(-1.0 * np.ones(N - 1), -1))
    return np.eye(N) + r * L0


def step(A, u):
    return np.linalg.solve(A, u)


def main():
    n = 6
    N = 2 ** n
    dx = 1.0 / (N + 1)
    x = np.arange(1, N + 1) * dx
    u = np.exp(-((x - 0.5) ** 2) / (2 * 0.05 ** 2))   # sharp initial pulse

    dt = 0.2 * dx * dx / ALPHA
    t, T = 0.0, 1.0
    kappas, steps = [], 0

    # --- adaptive integration: step-doubling error estimate picks dt ---------
    while t < T and steps < 400:
        r = ALPHA * dt / (dx * dx)
        A = heat_matrix(n, r)
        big = step(A, u)
        Ah = heat_matrix(n, r / 2.0)
        small = step(Ah, step(Ah, u))
        err = float(np.linalg.norm(big - small) / max(1e-30, np.linalg.norm(small)))

        if err > TOL and dt > 1e-6:
            dt *= 0.5                     # reject, retry smaller
            continue
        u, t = small, t + dt
        kappas.append(1.0 + 4.0 * r)
        steps += 1
        if err < TOL / 8:
            dt *= 1.8                     # accept and grow

    kappas = np.array(kappas)
    # Each distinct (kappa, degree) needs its own phase factors. Bin kappa at the
    # resolution that actually changes the required degree.
    degrees = []
    for kap in kappas:
        d, _ = Q.min_degree(float(max(kap, 1.05)), EPS_POLY)
        degrees.append(d)
    degrees = np.array(degrees)
    distinct = sorted(set(zip(np.round(kappas, 3), degrees)))

    print(f"adaptive implicit-Euler heat solve, n={n} ({N} points), horizon T={T}")
    print(f"  accepted steps            : {steps}")
    print(f"  kappa range               : {kappas.min():.2f} .. {kappas.max():.2f}")
    print(f"  degree range              : {degrees.min()} .. {degrees.max()}")
    print(f"  DISTINCT (kappa, degree)  : {len(distinct)}")
    print(f"  -> phase-factor solves needed at run time: {len(distinct)}")
    print("     (dt is chosen from the solution, so none of these are known in advance)")

    # --- what that angle work costs ----------------------------------------
    t0 = time.perf_counter()
    for kap, d in distinct:
        co, _c, _r = Q.approx_inverse(float(max(kap, 1.05)), int(d))
        Q.solve_phases((co * 0.999).tolist())
    ours = time.perf_counter() - t0

    # Measured per-solve rates at these degrees, from bench/results/shootout_clean.txt.
    mean_deg = float(np.mean([d for _k, d in distinct]))
    per_solve = {"pyqsp": 0.33, "QSPPACK": 0.28, "nlft-qsp": 2.35}  # degree ~51
    print(f"\n  angle work, this solver   : {ours:.2f} s  "
          f"({ours/len(distinct)*1000:.0f} ms per solve, mean degree {mean_deg:.0f})")
    for k, v in per_solve.items():
        print(f"  same work via {k:<9}: {v*len(distinct):.1f} s  "
              f"(at its measured {v:.2f} s/solve)")

    print("\n  Caching does not remove this: the parameters are produced by the")
    print("  integration itself. A solver that cannot be called from inside the")
    print("  timestep loop has to round-trip to an interpreter at every change.")


if __name__ == "__main__":
    main()
