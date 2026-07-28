"""Step 6: does the C++ NLFT beat the C++ Newton core, and by how much?

Three solvers, identical 1/x QSVT targets:
  * C++ Newton  (sym_qsp, the current default)   -- eta-independent, ~O(d^2.8)
  * C++ NLFT    (this port)                      -- O(1/eta) Weiss + O(n^2) inverse
  * nlft-qsp    (Python, mpmath default backend) -- the reference

The eta sweep is the point: NLFT's Weiss step sizes its FFT as N ~ d/eta, so the
shave applied to the target should dominate its cost while leaving Newton flat.

    python bench_nlft_vs_newton.py
"""
import contextlib
import io
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "applications", "common"))

import qsp_bridge as Q  # noqa: E402
import nlft_qsp  # noqa: E402

SRC = r"""
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "NlftAngleSolver.hpp"
int main(int argc, char** argv) {
    std::vector<double> c;
    for (int i = 1; i < argc; ++i) c.push_back(std::atof(argv[i]));
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> phi;
    try { phi = qsvt::nlftChebQspPhases(c); }
    catch (const std::exception& e) { std::printf("ERR %s\n", e.what()); return 2; }
    const double s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("OK %.6f %zu\n", s, phi.size());
    return 0;
}
"""


def build():
    cpp = "/tmp/bench_nlft.cpp"
    with open(cpp, "w") as fh:
        fh.write(SRC)
    exe = "/tmp/bench_nlft_bin"
    subprocess.run(
        ["g++", "-O3", "-std=c++17", "-I" + os.path.join(ROOT, "include"), cpp,
         os.path.join(ROOT, "src", "LaurentPoly.cpp"),
         os.path.join(ROOT, "src", "WeissCompletion.cpp"),
         os.path.join(ROOT, "src", "InverseNlft.cpp"),
         os.path.join(ROOT, "src", "NlftAngleSolver.cpp"), "-o", exe],
        check=True,
    )
    return exe


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fn(*a, **kw)


def main():
    exe = build()
    print("Identical near-minimax 1/x targets. Times in seconds.\n")
    print(f"{'kappa':>6} {'deg':>5} {'shave':>7} {'eta':>7} | {'C++ Newton':>11} "
          f"| {'C++ NLFT':>18} | {'nlft-qsp (py)':>14}")
    print("-" * 88)

    for kap, d in ((10.0, 79), (30.0, 247), (60.0, 505)):
        for shave in (0.999, 0.99, 0.95):
            base, _c, _r = Q.approx_inverse(kap, d)
            coeffs = (base * shave).tolist()

            t0 = time.perf_counter()
            Q.solve_phases(coeffs)
            t_newton = time.perf_counter() - t0

            res = subprocess.run([exe] + [repr(float(x)) for x in coeffs],
                                 capture_output=True, text=True)
            parts = res.stdout.split()
            if res.returncode == 0 and parts and parts[0] == "OK":
                s_nlft = f"{float(parts[1]):.3f}"
            else:
                msg = " ".join(parts[1:])[:14] if parts else "crash"
                s_nlft = f"ERR {msg}"

            if shave == 0.999:  # the Python reference is slow; sample once
                t0 = time.perf_counter()
                try:
                    quiet(nlft_qsp.chebqsp_solve, coeffs)
                    s_py = f"{time.perf_counter() - t0:.3f}"
                except Exception as e:
                    s_py = f"ERR:{type(e).__name__}"
            else:
                s_py = "-"

            print(f"{kap:>6.0f} {d:>5} {shave:>7.3f} {1-shave:>7.0e} | "
                  f"{t_newton:>11.3f} | {s_nlft:>18} | {s_py:>14}", flush=True)

    print("\nBENCH_DONE")


if __name__ == "__main__":
    main()
