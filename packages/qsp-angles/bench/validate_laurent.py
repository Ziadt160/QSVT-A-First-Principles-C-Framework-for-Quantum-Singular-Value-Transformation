"""Validate the C++ LaurentPoly (NLFT port, step 1) against nlft-qsp.

Builds and runs validate_laurent.cpp, then recomputes every quantity with
nlft-qsp's own Polynomial and compares. Any convention mismatch (FFT sign, FFT
normalization, frequency shift, Schwarz projection, conjugation) shows up here as
a coefficient-level disagreement, which is the point of validating this layer
before Weiss completion is built on top of it.

    python validate_laurent.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

import nlft_qsp  # noqa: E402
from nlft_qsp.poly import Polynomial  # noqa: E402

TOL = 1e-12


def build_and_run():
    exe = "/tmp/validate_laurent_bin"
    subprocess.run(
        ["g++", "-O2", "-std=c++17", "-I" + os.path.join(ROOT, "include"),
         os.path.join(HERE, "validate_laurent.cpp"),
         os.path.join(ROOT, "src", "LaurentPoly.cpp"), "-o", exe],
        check=True,
    )
    return subprocess.run([exe], check=True, capture_output=True,
                          text=True).stdout.splitlines()


def parse(line):
    """'TAG start n re im re im ...' -> (tag, start, [complex])."""
    parts = line.split()
    tag = parts[0]
    if tag in ("SUPNORM", "L2"):
        return tag, None, float(parts[1])
    if tag == "EVAL16":
        n = int(parts[1])
        vals = [complex(float(parts[2 + 2 * i]), float(parts[3 + 2 * i]))
                for i in range(n)]
        return tag, None, vals
    start = int(parts[1])
    n = int(parts[2])
    vals = [complex(float(parts[3 + 2 * i]), float(parts[4 + 2 * i]))
            for i in range(n)]
    return tag, start, vals


def cmp_poly(name, got_start, got, ref: Polynomial):
    """Compare on exponents, not array indices -- supports differ legitimately."""
    lo = min(got_start, ref.support_start)
    hi = max(got_start + len(got), ref.support_start + len(ref.coeffs))
    worst = 0.0
    for k in range(lo, hi):
        a = got[k - got_start] if got_start <= k < got_start + len(got) else 0j
        b = complex(ref[k]) if ref.support_start <= k < ref.support_start + len(ref.coeffs) else 0j
        worst = max(worst, abs(a - b))
    ok = worst < TOL
    print(f"  {name:10s} max|C++ - nlft-qsp| = {worst:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    out = {t: (s, v) for t, s, v in (parse(l) for l in build_and_run())}
    start, coeffs = out["POLY"]
    p = Polynomial(list(coeffs), support_start=start)
    print(f"reference polynomial: support_start={start}, {len(coeffs)} coefficients\n")

    ok = True

    ref_eval = list(p.eval_at_roots_of_unity(16))
    got_eval = out["EVAL16"][1]
    worst = max(abs(complex(a) - complex(b)) for a, b in zip(got_eval, ref_eval))
    print(f"  {'eval@16':10s} max|C++ - nlft-qsp| = {worst:.3e}  "
          f"{'PASS' if worst < TOL else 'FAIL'}")
    ok &= worst < TOL

    from nlft_qsp.solvers.weiss import laurent_approximation
    s, v = out["APPROX"]
    ok &= cmp_poly("approx", s, v, laurent_approximation(ref_eval))

    s, v = out["SCHWARZ"]
    ok &= cmp_poly("schwarz", s, v, p.schwarz_transform())

    s, v = out["CONJ"]
    ok &= cmp_poly("conjugate", s, v, p.conjugate())

    s, v = out["PROD"]
    ok &= cmp_poly("p * p*", s, v, p * p.conjugate())

    ref_sup = float(p.sup_norm(1024))
    got_sup = out["SUPNORM"][1]
    print(f"  {'sup_norm':10s} |C++ - nlft-qsp| = {abs(got_sup - ref_sup):.3e}  "
          f"{'PASS' if abs(got_sup - ref_sup) < TOL else 'FAIL'}")
    ok &= abs(got_sup - ref_sup) < TOL

    ref_l2 = float(p.l2_norm())
    got_l2 = out["L2"][1]
    print(f"  {'l2_norm':10s} |C++ - nlft-qsp| = {abs(got_l2 - ref_l2):.3e}  "
          f"{'PASS' if abs(got_l2 - ref_l2) < TOL else 'FAIL'}")
    ok &= abs(got_l2 - ref_l2) < TOL

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
