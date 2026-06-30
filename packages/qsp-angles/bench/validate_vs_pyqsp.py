"""Independent cross-validation of the C++ sym_qsp solver against pyqsp.

This is the "verify us, don't trust us" artifact. It generates random GENERIC
multi-coefficient targets (many nonzero Chebyshev modes, both parities, a range
of degrees) with a FIXED seed, solves each with pyqsp's own `newton_solver`
(the reference oracle), and writes the target coefficients to a manifest. The
C++ `validate` driver solves the SAME coefficients. Each solver is graded in its
own convention against the same target polynomial, so the comparison is
convention-neutral.

Run standalone (pyqsp side only):
    python validate_vs_pyqsp.py --manifest cases.txt

Run the full cross-check (build ./validate first; see run_all.sh):
    python validate_vs_pyqsp.py --manifest cases.txt --cpp-bin ./validate

Exit code is non-zero if any solver fails to reach the tolerance, so this
doubles as a CI gate.
"""

import argparse
import contextlib
import io
import subprocess
import sys

import numpy as np
from numpy.polynomial import chebyshev as C

# (degree, parity) cases: both parities across a range of degrees. Generic
# targets (every allowed Chebyshev mode populated), unlike single-mode c*T_d.
CASES = [(7, 1), (8, 0), (21, 1), (20, 0), (41, 1), (40, 0), (101, 1), (100, 0)]
SEED = 12345
SCALE = 0.9  # rescale each target so max|f| = 0.9 < 1 (a valid QSP target)
TOL = 1e-10  # residual bound; generic targets reach ~1e-14..1e-13 here


def _wx_im(x, full):
    """Im<0|U(x)|0> for full phases in the Wx convention (pyqsp's target part)."""
    s = np.sqrt(max(0.0, 1.0 - x * x))
    W = np.array([[x, 1j * s], [1j * s, x]], dtype=complex)
    E = lambda p: np.array([[np.exp(1j * p), 0], [0, np.exp(-1j * p)]], dtype=complex)
    U = E(full[0])
    for p in full[1:]:
        U = U @ W @ E(p)
    return U[0, 0].imag


def generate_and_run_pyqsp(manifest_path):
    """Generate cases, solve with pyqsp, write the manifest. Returns dict."""
    from pyqsp.sym_qsp_opt import newton_solver

    rng = np.random.default_rng(SEED)
    xs = np.linspace(-1.0, 1.0, 801)
    pyqsp_resid = {}
    lines = []
    for (d, parity) in CASES:
        n = d // 2 + 1
        coef = rng.standard_normal(n)
        fullc = np.zeros(d + 1)
        for i in range(n):
            fullc[parity + 2 * i] = coef[i]
        mx = np.max(np.abs(C.chebval(xs, fullc)))
        coef = coef / mx * SCALE
        fullc = fullc / mx * SCALE

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _red, _err, _it, obj = newton_solver(coef.copy(), parity, crit=1e-12, maxiter=100)
        full = np.asarray(obj.full_phases, dtype=float)
        tgt = C.chebval(xs, fullc)
        im_res = max(abs(_wx_im(x, full) - t) for x, t in zip(xs, tgt))
        pyqsp_resid[(d, parity)] = im_res

        lines.append(f"{parity} {d} {n}")
        lines.extend(f"{c:.17g}" for c in coef)

    with open(manifest_path, "w") as fo:
        fo.write("\n".join(lines) + "\n")
    return pyqsp_resid


def run_cpp(cpp_bin, manifest_path):
    """Run the C++ validator on the manifest; return {(deg,parity): residual}."""
    out = subprocess.run(
        [cpp_bin, manifest_path], capture_output=True, text=True, check=True
    ).stdout
    cpp_resid = {}
    for line in out.splitlines()[1:]:  # skip header
        deg, parity, resid, _conv = line.split(",")
        cpp_resid[(int(deg), int(parity))] = float(resid)
    return cpp_resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="cases.txt")
    ap.add_argument("--cpp-bin", default=None, help="path to the built C++ validate driver")
    args = ap.parse_args()

    pyqsp_resid = generate_and_run_pyqsp(args.manifest)
    cpp_resid = run_cpp(args.cpp_bin, args.manifest) if args.cpp_bin else {}

    print(f"\nGeneric multi-coefficient targets, seed={SEED}, |f|={SCALE}, tol={TOL:.0e}")
    header = "degree parity   pyqsp_resid" + ("     cpp_resid   verdict" if cpp_resid else "")
    print(header)
    ok = True
    for (d, parity) in CASES:
        pr = pyqsp_resid[(d, parity)]
        row = f"{d:6d} {parity:6d}   {pr:.3e}"
        if cpp_resid:
            cr = cpp_resid[(d, parity)]
            passed = (pr < TOL) and (cr < TOL)
            ok = ok and passed
            row += f"     {cr:.3e}   {'PASS' if passed else 'FAIL'}"
        else:
            ok = ok and (pr < TOL)
        print(row)

    if cpp_resid:
        print("\nVALIDATION " + ("PASSED — C++ and pyqsp both reach machine precision."
                                  if ok else "FAILED — a solver exceeded the tolerance."))
    else:
        print("\n(pyqsp-only run; pass --cpp-bin to cross-check the C++ solver.)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
