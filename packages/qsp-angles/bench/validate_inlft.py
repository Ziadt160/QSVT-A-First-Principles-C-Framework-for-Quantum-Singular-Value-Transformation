"""Validate the C++ Half-Cholesky inverse NLFT (step 4) against nlft-qsp.

The C++ side runs the whole chain in double precision: Weiss completion with the
ratio c ~ b/a, then inverseNlft. The reference runs the same chain on nlft-qsp's
default (mpmath) backend. Agreement is checked on the recovered sequence F.

    python validate_inlft.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

from nlft_qsp.poly import Polynomial  # noqa: E402
from nlft_qsp.solvers import weiss  # noqa: E402
from nlft_qsp.solvers.half_cholesky import inlft  # noqa: E402

SRC = r"""
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "InverseNlft.hpp"
#include "WeissCompletion.hpp"
using qsvt::Complexd; using qsvt::LaurentPoly;
int main(int argc, char** argv) {
    const int n = std::atoi(argv[1]);
    const double scale = std::atof(argv[2]);
    std::mt19937 rng(4242u + n);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<Complexd> cf;
    for (int i = 0; i <= n; ++i) cf.emplace_back(uni(rng), uni(rng));
    LaurentPoly b(cf, 0);
    const double s = b.supNorm(4096);
    for (auto& v : b.coeffs) v *= scale / s;
    std::printf("B %d %zu", b.supportStart, b.coeffs.size());
    for (const auto& v : b.coeffs) std::printf(" %.17g %.17g", v.real(), v.imag());
    std::printf("\n");
    const qsvt::WeissResult w = qsvt::weissComplete(b, -1.0, true);
    const qsvt::NlftSequence F = qsvt::inverseNlft(b, w.c);
    std::printf("F %d %zu", F.supportStart, F.coeffs.size());
    for (const auto& v : F.coeffs) std::printf(" %.17g %.17g", v.real(), v.imag());
    std::printf("\nRESID %.17g\n", w.residual);
    return 0;
}
"""


def build():
    cpp = "/tmp/validate_inlft.cpp"
    with open(cpp, "w") as fh:
        fh.write(SRC)
    exe = "/tmp/validate_inlft_bin"
    subprocess.run(
        ["g++", "-O2", "-std=c++17", "-I" + os.path.join(ROOT, "include"), cpp,
         os.path.join(ROOT, "src", "LaurentPoly.cpp"),
         os.path.join(ROOT, "src", "WeissCompletion.cpp"),
         os.path.join(ROOT, "src", "InverseNlft.cpp"), "-o", exe],
        check=True,
    )
    return exe


def parse(lines):
    out = {}
    for ln in lines:
        p = ln.split()
        if p[0] in ("B", "F"):
            start, n = int(p[1]), int(p[2])
            out[p[0]] = (start, [complex(float(p[3 + 2 * i]), float(p[4 + 2 * i]))
                                 for i in range(n)])
        else:
            out[p[0]] = float(p[1])
    return out


def main():
    exe = build()
    ok = True
    print(f"{'deg':>5} {'sup|b|':>8} {'weiss resid':>13} {'max|F_cpp-F_ref|':>18}")
    for n, scale in ((7, 0.9), (15, 0.9), (31, 0.9), (63, 0.9)):
        out = parse(subprocess.run([exe, str(n), str(scale)], check=True,
                                   capture_output=True, text=True).stdout.splitlines())
        bstart, bc = out["B"]
        fstart, fc = out["F"]

        b = Polynomial(list(bc), support_start=bstart)
        _a, c_ref = weiss.ratio(b)
        F_ref = inlft(b, c_ref)

        ref = [complex(F_ref[k]) for k in
               range(F_ref.support_start, F_ref.support_start + len(F_ref.coeffs))]
        worst = max(abs(x - y) for x, y in zip(fc, ref)) if len(fc) == len(ref) else float("inf")
        good = worst < 1e-9
        ok &= good
        print(f"{n:>5} {scale:>8.2f} {out['RESID']:>13.2e} {worst:>18.2e}  "
              f"{'PASS' if good else 'FAIL'}"
              + ("" if len(fc) == len(ref) else f"  (len {len(fc)} vs {len(ref)})"))

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
