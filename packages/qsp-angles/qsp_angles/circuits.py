"""Circuit adapters: turn solved QSP phases into runnable circuits.

Given the phases from :func:`qsp_angles.target2angles` (the Wx convention, where
``Re<0|U(x)|0> = f(x)``), this module emits the single-qubit **QSP** circuit as
OpenQASM 3 or a Qiskit ``QuantumCircuit``, so a quantum developer can simulate or
run it directly.

Gate convention (exact, no global phase in Qiskit's convention):

* signal  ``W(x) = e^{i arccos(x) X}``  ==  ``rx(-2 * arccos(x))``
* phase   ``E(phi) = e^{i phi Z}``      ==  ``rz(-2 * phi)``

and ``U(x) = E(phi_0) prod_{k>=1} [ W(x) E(phi_k) ]``.

For the full **QSVT** of a matrix function, use the same phases but replace
``W(x)`` with your block-encoding and ``E(phi)`` with projector-controlled
ancilla rotations -- that lifts the scalar QSP here to singular values. This
module covers the scalar QSP circuit; the block-encoding is application-specific.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

__all__ = ["gate_sequence", "qsp_qasm3", "to_qiskit"]

# A gate is ("rz", angle) for a fixed phase rotation, or ("w", None) for the
# signal W(x) (its angle depends on the signal value x, filled in later).
Gate = Tuple[str, "float | None"]


def gate_sequence(phases: Sequence[float]) -> List[Gate]:
    """Return the QSP gate list in **circuit order** (first gate applied first).

    ``U = E(phi_0) W E(phi_1) ... W E(phi_d)`` as a matrix product means the
    rightmost factor ``E(phi_d)`` acts on the state first, so the circuit lists
    the reversed factor order. (The phase sequence is symmetric, so this equals
    the forward order, but we build it correctly regardless.)
    """
    phi = list(phases)
    if len(phi) < 1:
        raise ValueError("need at least one phase")
    d = len(phi) - 1
    seq: List[Gate] = [("rz", -2.0 * phi[d])]
    for k in range(d - 1, -1, -1):
        seq.append(("w", None))
        seq.append(("rz", -2.0 * phi[k]))
    return seq


def qsp_qasm3(phases: Sequence[float], x: float) -> str:
    """OpenQASM 3 for the single-qubit QSP circuit at signal value ``x``.

    ``x`` must lie in [-1, 1]; ``W(x)`` becomes ``rx(-2*arccos(x))``.
    """
    if not -1.0 <= x <= 1.0:
        raise ValueError(f"signal x must be in [-1, 1], got {x}")
    w_angle = -2.0 * math.acos(x)
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', "qubit[1] q;"]
    for kind, angle in gate_sequence(phases):
        if kind == "rz":
            lines.append(f"rz({angle:.17g}) q[0];")
        else:  # signal
            lines.append(f"rx({w_angle:.17g}) q[0];")
    return "\n".join(lines) + "\n"


def to_qiskit(phases: Sequence[float]):
    """Return a parameterized 1-qubit Qiskit ``QuantumCircuit`` for the QSP.

    The circuit has a single ``Parameter("signal_angle")`` equal to
    ``arccos(x)``; bind it to evaluate at a signal value ``x``:

    >>> qc, theta = to_qiskit(phases)            # doctest: +SKIP
    >>> bound = qc.assign_parameters({theta: math.acos(0.3)})  # doctest: +SKIP

    Requires the optional ``qiskit`` dependency (``pip install .[qiskit]``).
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
    except ImportError as exc:  # pragma: no cover - exercised only without qiskit
        raise ImportError(
            "to_qiskit requires qiskit; install it with `pip install .[qiskit]` "
            "or `pip install qiskit`."
        ) from exc

    theta = Parameter("signal_angle")  # = arccos(x)
    qc = QuantumCircuit(1)
    for kind, angle in gate_sequence(phases):
        if kind == "rz":
            qc.rz(angle, 0)
        else:  # W(x) = rx(-2 * arccos(x)) = rx(-2 * theta)
            qc.rx(-2.0 * theta, 0)
    return qc, theta
