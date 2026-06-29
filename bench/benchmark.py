#!/usr/bin/env python3
"""Compare this framework against Qiskit (unitary decomposition) and PennyLane
(QSP angle finding) on identical inputs.

The C++ `qsvt_bench` binary writes the inputs (random unitaries, target
polynomials) and our own results to a JSON file. This script reads that file,
runs the reference libraries on the *same* inputs, and prints comparison tables.

Usage:  python3 benchmark.py bench_input.json
"""
import json
import sys
import time

import numpy as np


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def global_phase_diff(A, B):
    """Frobenius distance between A and B after removing a global phase."""
    idx = np.unravel_index(np.argmax(np.abs(B)), B.shape)
    ph = A[idx] / B[idx]
    ph /= abs(ph)
    return np.linalg.norm(A - ph * B)


def qsp_unitary(angles, x, sgn, zfac):
    """QSP product for a candidate convention (Wx): signal e^{i sgn arccos(x) X},
    processing e^{i zfac phi Z}."""
    a = np.arccos(np.clip(x, -1.0, 1.0))
    s = np.sin(a)
    W = np.array([[np.cos(a), 1j * sgn * s], [1j * sgn * s, np.cos(a)]])
    U = np.array([[np.exp(1j * zfac * angles[0]), 0],
                  [0, np.exp(-1j * zfac * angles[0])]])
    for phi in angles[1:]:
        E = np.array([[np.exp(1j * zfac * phi), 0],
                      [0, np.exp(-1j * zfac * phi)]])
        U = U @ W @ E
    return U


def timed_median(fn, repeats=7, warmup=2):
    """Median wall-clock (seconds) of fn() over `repeats`, after `warmup` runs."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def qsp_residual(angles, coeffs):
    """Best-fit max error of Re<0|U|0> vs the target polynomial, searching over
    standard QSP sign/scale conventions (so we don't hard-code one library's)."""
    xs = np.linspace(-0.99, 0.99, 201)
    target = np.array([np.polyval(list(reversed(coeffs)), x) for x in xs])
    best = np.inf
    best_conv = None
    for sgn in (+1, -1):
        for zfac in (1, -1, 2, -2):
            vals = np.array([qsp_unitary(angles, x, sgn, zfac)[0, 0].real
                             for x in xs])
            err = np.max(np.abs(vals - target))
            if err < best:
                best, best_conv = err, (sgn, zfac)
    return best, best_conv


# --------------------------------------------------------------------------
# Qiskit: unitary -> gates
# --------------------------------------------------------------------------
def qiskit_decompose(data):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    qsd = None
    for path in ("qiskit.synthesis", "qiskit.synthesis.unitary.qsd"):
        try:
            mod = __import__(path, fromlist=["qs_decomposition"])
            qsd = getattr(mod, "qs_decomposition")
            break
        except Exception:
            pass

    print("\n=== Unitary decomposition: ours vs Qiskit ===")
    if qsd is not None:
        print("(both = Quantum Shannon Decomposition; ours C++/Eigen, Qiskit qs_decomposition Python/NumPy)")
        algo = "qs_decomposition"
    else:
        print("(Qiskit: transpile to [u, cx], optimization_level=3)")
        algo = "transpile"
    print(f"{'n':>2} {'dim':>4} | {'CNOT(ours)':>10} {'CNOT(qk)':>9} {'ratio':>6} "
          f"| {'err(ours)':>9} {'err(qk)':>9} | {'t(ours)ms':>10} {'t(qk)ms':>9} {'speedup':>8}")
    print("-" * 100)

    def decompose(U, n):
        if qsd is not None:
            return transpile(qsd(U), basis_gates=["u", "cx"], optimization_level=0)
        qc = QuantumCircuit(n)
        qc.append(UnitaryGate(U), range(n))
        return transpile(qc, basis_gates=["u", "cx"], optimization_level=3)

    for u in data["unitaries"]:
        n, dim = u["n"], u["dim"]
        U = np.array(u["real"]) + 1j * np.array(u["imag"])
        o = u["ours"]

        # Time the core decomposition only (warmup + median), to match ours.
        t_qk = timed_median(lambda: (qsd(U) if qsd is not None else decompose(U, n)))
        qc = decompose(U, n)

        ops = qc.count_ops()
        qk_cx = ops.get("cx", 0)
        err = global_phase_diff(np.asarray(Operator(qc).data), U)
        ratio = (o["cnots"] / qk_cx) if qk_cx else float("nan")
        ours_ms = o["time_us"] / 1e3
        qk_ms = t_qk * 1e3
        speedup = qk_ms / ours_ms if ours_ms > 0 else float("nan")

        print(f"{n:>2} {dim:>4} | {o['cnots']:>10} {qk_cx:>9} {ratio:>6.2f} "
              f"| {o['recon_err']:>9.1e} {err:>9.1e} | {ours_ms:>10.3f} {qk_ms:>9.3f} "
              f"{speedup:>7.1f}x")


# --------------------------------------------------------------------------
# PennyLane: QSP angle finding
# --------------------------------------------------------------------------
def pennylane_qsp(data):
    import pennylane as qml

    print("\n=== QSP angle finding: ours vs PennyLane ===")
    print("(ours C++ Levenberg-Marquardt; PennyLane qml.poly_to_angles)")
    print(f"{'deg':>3} | {'resid(ours)':>11} {'resid(PL)':>11} | "
          f"{'t(ours)ms':>10} {'t(PL)ms':>9} {'speedup':>8} | {'conv':>6}")
    print("-" * 76)

    for q in data["qsp"]:
        d = q["degree"]
        coeffs = q["coeffs"]
        o = q["ours"]

        try:
            t_pl = timed_median(lambda: qml.poly_to_angles(coeffs, "QSP"))
            pl_angles = np.array(qml.poly_to_angles(coeffs, "QSP"))
            pl_resid, conv = qsp_residual(pl_angles, coeffs)
            pl_ms = t_pl * 1e3
            conv_s = f"{conv[0]:+d},{conv[1]:+d}"
        except Exception as e:
            pl_resid, pl_ms, conv_s = float("nan"), float("nan"), "err"
            print(f"   PennyLane error for deg {d}: {e}")

        ours_ms = o["time_us"] / 1e3
        speedup = pl_ms / ours_ms if ours_ms > 0 else float("nan")
        print(f"{d:>3} | {o['residual']:>11.2e} {pl_resid:>11.2e} | "
              f"{ours_ms:>10.3f} {pl_ms:>9.3f} {speedup:>7.1f}x | {conv_s:>6}")
    print("\n(resid = max error of the achieved polynomial vs target on [-1,1]; "
          "speedup = t_PL / t_ours.)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bench_input.json"
    with open(path) as f:
        data = json.load(f)

    print(f"Benchmark inputs: {len(data['unitaries'])} unitaries, "
          f"{len(data['qsp'])} QSP targets (from {path})")

    try:
        qiskit_decompose(data)
    except Exception as e:
        print(f"\nQiskit benchmark skipped: {e}")

    try:
        pennylane_qsp(data)
    except Exception as e:
        print(f"\nPennyLane benchmark skipped: {e}")


if __name__ == "__main__":
    main()
