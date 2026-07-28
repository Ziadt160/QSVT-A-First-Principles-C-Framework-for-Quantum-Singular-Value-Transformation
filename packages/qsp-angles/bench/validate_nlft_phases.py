"""Validate the full C++ NLFT angle chain (step 5) against nlft_qsp.chebqsp_solve.

Compares the final phase arrays element-wise. Any error anywhere in the chain --
Chebyshev-to-Laurent, the analytic reduction, the -i premultiply, Weiss, the
inverse NLFT, from_nlfs, or iX -- shows up here.

    python validate_nlft_phases.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "applications", "common"))

import nlft_qsp  # noqa: E402
import qsp_bridge as Q  # noqa: E402

SRC = r"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "NlftAngleSolver.hpp"
int main(int argc, char** argv) {
    std::vector<double> c;
    for (int i = 1; i < argc; ++i) c.push_back(std::atof(argv[i]));
    const std::vector<double> phi = qsvt::nlftChebQspPhases(c);
    std::printf("%zu", phi.size());
    for (double p : phi) std::printf(" %.17g", p);
    std::printf("\n");
    return 0;
}
"""


def build():
    cpp = "/tmp/validate_nlft_phases.cpp"
    with open(cpp, "w") as fh:
        fh.write(SRC)
    exe = "/tmp/validate_nlft_phases_bin"
    subprocess.run(
        ["g++", "-O2", "-std=c++17", "-I" + os.path.join(ROOT, "include"), cpp,
         os.path.join(ROOT, "src", "LaurentPoly.cpp"),
         os.path.join(ROOT, "src", "WeissCompletion.cpp"),
         os.path.join(ROOT, "src", "InverseNlft.cpp"),
         os.path.join(ROOT, "src", "NlftAngleSolver.cpp"), "-o", exe],
        check=True,
    )
    return exe


def main():
    exe = build()
    ok = True
    print(f"{'case':>22} {'deg':>5} {'n_phi':>6} {'max|phi_cpp - phi_ref|':>24}")

    cases = []
    # Simple exact Chebyshev targets.
    for d in (3, 5, 9):
        c = [0.0] * (d + 1)
        c[d] = 0.8
        cases.append((f"0.8*T_{d}", c))
    # The real QSVT workload: near-minimax 1/x at moderate kappa.
    for kap, d in ((4.0, 21), (10.0, 41)):
        co, _c, _r = Q.approx_inverse(kap, d)
        cases.append((f"1/x kappa={kap:.0f} d={d}", (co * 0.999).tolist()))

    for name, c in cases:
        res = subprocess.run([exe] + [repr(float(x)) for x in c],
                             capture_output=True, text=True)
        if res.returncode != 0:
            print(f"{name:>22} {len(c)-1:>5} {'-':>6} {'C++ FAILED':>24}  "
                  f"{res.stderr.strip()[:40]}")
            ok = False
            continue
        parts = res.stdout.split()
        got = np.array([float(x) for x in parts[1:]])

        ref = np.array([float(x) for x in nlft_qsp.chebqsp_solve(list(c)).phi])
        if got.size != ref.size:
            print(f"{name:>22} {len(c)-1:>5} {got.size:>6} "
                  f"{'SIZE MISMATCH vs ' + str(ref.size):>24}")
            ok = False
            continue
        worst = float(np.max(np.abs(got - ref)))
        good = worst < 1e-9
        ok &= good
        print(f"{name:>22} {len(c)-1:>5} {got.size:>6} {worst:>24.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
