"""Validate the C++ Weiss completion (NLFT port, step 3) against nlft-qsp.

Acceptance test is the defining property: || a a* + b b* - 1 ||_2 must reach
~machine precision. We also compare the coefficients of `a` against nlft-qsp's
`weiss.complete` -- `a` is the unique outer positive-mean solution, so the two
implementations must agree, not merely both be valid.

    python validate_weiss.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

from nlft_qsp.poly import Polynomial  # noqa: E402
from nlft_qsp.solvers import weiss  # noqa: E402

SRC = r"""
#include <cstdio>
#include <random>
#include <vector>
#include "WeissCompletion.hpp"
using qsvt::Complexd; using qsvt::LaurentPoly;
int main(int argc, char** argv) {
    const int n = std::atoi(argv[1]);
    const double scale = std::atof(argv[2]);
    // Deterministic analytic b with support [0, n].
    std::mt19937 rng(4242u + n);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<Complexd> c;
    for (int i = 0; i <= n; ++i) c.emplace_back(uni(rng), uni(rng));
    LaurentPoly b(c, 0);
    // Normalise so sup|b| = scale < 1.
    const double s = b.supNorm(4096);
    for (auto& v : b.coeffs) v *= scale / s;
    std::printf("B %d %zu", b.supportStart, b.coeffs.size());
    for (const auto& v : b.coeffs) std::printf(" %.17g %.17g", v.real(), v.imag());
    std::printf("\n");
    const qsvt::WeissResult r = qsvt::weissComplete(b);
    std::printf("A %d %zu", r.a.supportStart, r.a.coeffs.size());
    for (const auto& v : r.a.coeffs) std::printf(" %.17g %.17g", v.real(), v.imag());
    std::printf("\n");
    std::printf("RESID %.17g\nFFT %zu\n", r.residual, r.fftSize);
    return 0;
}
"""


def build():
    cpp = "/tmp/validate_weiss.cpp"
    with open(cpp, "w") as fh:
        fh.write(SRC)
    exe = "/tmp/validate_weiss_bin"
    subprocess.run(
        ["g++", "-O2", "-std=c++17", "-I" + os.path.join(ROOT, "include"), cpp,
         os.path.join(ROOT, "src", "LaurentPoly.cpp"),
         os.path.join(ROOT, "src", "WeissCompletion.cpp"), "-o", exe],
        check=True,
    )
    return exe


def parse(lines):
    out = {}
    for ln in lines:
        p = ln.split()
        if p[0] in ("B", "A"):
            start, n = int(p[1]), int(p[2])
            out[p[0]] = (start, [complex(float(p[3 + 2 * i]), float(p[4 + 2 * i]))
                                 for i in range(n)])
        else:
            out[p[0]] = float(p[1])
    return out


def main():
    exe = build()
    ok = True
    print(f"{'deg':>5} {'sup|b|':>8} {'C++ resid':>12} {'nlft resid':>12} "
          f"{'max|a_cpp-a_ref|':>18} {'FFT N':>9}")
    for n, scale in ((15, 0.9), (31, 0.9), (63, 0.95), (127, 0.9), (255, 0.9)):
        res = subprocess.run([exe, str(n), str(scale)], check=True,
                             capture_output=True, text=True)
        out = parse(res.stdout.splitlines())
        bstart, bc = out["B"]
        astart, ac = out["A"]

        b = Polynomial(list(bc), support_start=bstart)
        a_ref = weiss.complete(b)

        # Defining property, computed independently in Python.
        chk = a_ref * a_ref.conjugate() + b * b.conjugate()
        chk = chk - Polynomial([1], 0)
        ref_resid = float(chk.l2_norm())

        worst = 0.0
        lo = min(astart, a_ref.support_start)
        hi = max(astart + len(ac), a_ref.support_start + len(a_ref.coeffs))
        for k in range(lo, hi):
            x = ac[k - astart] if astart <= k < astart + len(ac) else 0j
            y = (complex(a_ref[k])
                 if a_ref.support_start <= k < a_ref.support_start + len(a_ref.coeffs)
                 else 0j)
            worst = max(worst, abs(x - y))

        good = out["RESID"] < 1e-11 and worst < 1e-9
        ok &= good
        print(f"{n:>5} {scale:>8.2f} {out['RESID']:>12.2e} {ref_resid:>12.2e} "
              f"{worst:>18.2e} {int(out['FFT']):>9}  {'PASS' if good else 'FAIL'}")

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
