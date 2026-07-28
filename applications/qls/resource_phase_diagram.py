"""QSVT matrix-inversion resource map over (kappa, epsilon).

For every (condition number, target relative error) pair this computes what a QSVT
linear solve actually costs:

  * d          -- minimal odd degree of the near-minimax 1/x polynomial, which is
                  also the number of block-encoding QUERIES per QSVT application;
  * c          -- the subnormalization, p(x) ~ c/x with |p| <= 1. The un-amplified
                  post-selection success probability is ~(c * ||A^-1 b||)^2, so c
                  sets the repetition count;
  * queries/sample -- the honest end-to-end cost. Without amplitude amplification
                  a sample costs ~d / c^2 queries; with AA, ~d / c;
  * t_angles   -- wall-clock to actually produce the phases.

Why this map does not already exist: producing it needs a few hundred angle solves
at degrees into the thousands. With pyqsp (~1-70 s per solve) that is hours; with
PennyLane it is not possible at all, because its monomial-basis `poly_to_angles`
rejects a valid inversion polynomial once the degree passes ~79 (the Chebyshev ->
monomial conversion overflows double precision). Here it is minutes on a laptop,
which is the point.

    python resource_phase_diagram.py            # writes CSV + PNG next to this file
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import qsp_bridge as Q  # noqa: E402

KAPPAS = [2, 4, 8, 16, 32, 64, 128]
EPSILONS = [1e-2, 1e-3, 1e-4, 1e-6]
SAFETY = 0.999


def main():
    rows = []
    print(f"{'kappa':>6} {'eps':>7} {'degree':>7} {'c':>10} {'resid':>10} "
          f"{'t_angles_s':>11} {'q/sample':>12} {'q/sample_AA':>12}")
    for eps in EPSILONS:
        for kap in KAPPAS:
            d, _rel = Q.min_degree(float(kap), eps)
            if d is None:
                print(f"{kap:>6} {eps:>7.0e}  (not reachable)")
                continue
            coeffs, c, relerr = Q.approx_inverse(float(kap), d)
            coeffs = coeffs * SAFETY
            c *= SAFETY
            t0 = time.perf_counter()
            phases, resid = Q.solve_phases(coeffs.tolist())
            t_ang = time.perf_counter() - t0

            # Cost of one accepted sample. ||A^-1 b|| >= 1 for ||b||=1, so c is the
            # binding factor; report the conservative c-only form.
            q_plain = d / (c ** 2)
            q_aa = d / c

            rows.append((kap, eps, d, c, resid, t_ang, q_plain, q_aa, relerr))
            print(f"{kap:>6} {eps:>7.0e} {d:>7} {c:>10.3e} {resid:>10.1e} "
                  f"{t_ang:>11.3f} {q_plain:>12.3e} {q_aa:>12.3e}", flush=True)

    # ---- CSV -------------------------------------------------------------
    csv_path = os.path.join(HERE, "resource_phase_diagram.csv")
    with open(csv_path, "w") as fh:
        fh.write("kappa,epsilon,degree,subnormalization_c,angle_residual,"
                 "angle_time_s,queries_per_sample,queries_per_sample_AA,"
                 "poly_rel_err\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]:.0e},{r[2]},{r[3]:.6e},{r[4]:.3e},"
                     f"{r[5]:.4f},{r[6]:.6e},{r[7]:.6e},{r[8]:.3e}\n")
    print(f"\nwrote {csv_path}")
    print(f"total angle-solve wall-clock: {sum(r[5] for r in rows):.1f} s "
          f"for {len(rows)} solves")

    # ---- plot ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for eps in EPSILONS:
            sub = [r for r in rows if r[1] == eps]
            k = [r[0] for r in sub]
            axes[0].plot(k, [r[2] for r in sub], "o-", label=f"eps={eps:.0e}")
            axes[1].plot(k, [r[6] for r in sub], "o-", label=f"eps={eps:.0e}")
            axes[2].plot(k, [r[5] for r in sub], "o-", label=f"eps={eps:.0e}")
        for ax, ttl, yl in (
            (axes[0], "Polynomial degree = block-encoding queries", "degree d"),
            (axes[1], "End-to-end cost per accepted sample", "queries / sample"),
            (axes[2], "Wall-clock to produce the angles", "seconds"),
        ):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("condition number  kappa")
            ax.set_ylabel(yl)
            ax.set_title(ttl, fontsize=10)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("QSVT matrix inversion: resource map over (kappa, epsilon)",
                     fontsize=12)
        fig.tight_layout()
        png = os.path.join(HERE, "resource_phase_diagram.png")
        fig.savefig(png, dpi=140)
        print(f"wrote {png}")
    except Exception as exc:  # plotting is optional
        print(f"(plot skipped: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
