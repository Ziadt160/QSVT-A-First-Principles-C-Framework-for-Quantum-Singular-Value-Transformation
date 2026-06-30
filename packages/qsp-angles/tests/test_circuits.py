import math

import numpy as np
import pytest

from qsp_angles import circuits


# --- reference matrices in the QSP (Wx) convention -------------------------
def _W(x):
    th = math.acos(max(-1.0, min(1.0, x)))
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 1j * s], [1j * s, c]], dtype=complex)  # e^{i th X}


def _E(phi):
    return np.array([[np.exp(1j * phi), 0], [0, np.exp(-1j * phi)]], dtype=complex)


def _U_def(phases, x):
    """U = E(phi_0) W E(phi_1) ... W E(phi_d) -- the QSP definition."""
    U = _E(phases[0])
    for k in range(1, len(phases)):
        U = U @ _W(x) @ _E(phases[k])
    return U


# --- the same gates qsp_qasm3 / to_qiskit emit ------------------------------
def _rz(a):
    return np.array([[np.exp(-1j * a / 2), 0], [0, np.exp(1j * a / 2)]], dtype=complex)


def _rx(a):
    c, s = math.cos(a / 2), math.sin(a / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _U_circuit(phases, x):
    th = math.acos(max(-1.0, min(1.0, x)))
    U = np.eye(2, dtype=complex)
    for kind, angle in circuits.gate_sequence(phases):
        g = _rz(angle) if kind == "rz" else _rx(-2.0 * th)  # W(x) = rx(-2 arccos x)
        U = g @ U  # circuit order: first gate applied first
    return U


# Includes ASYMMETRIC phase sets so a reversed gate order cannot pass by accident.
@pytest.mark.parametrize(
    "phases",
    [[0.7], [0.1, 0.5, 0.3], [0.2, -0.4, 0.1, 0.6], [0.1, 0.2, 0.3, 0.2, 0.1]],
)
def test_circuit_matches_qsp_definition(phases):
    for x in (-0.9, -0.3, 0.0, 0.5, 0.85):
        assert np.allclose(_U_circuit(phases, x), _U_def(phases, x), atol=1e-12)


def test_qasm3_structure():
    qasm = circuits.qsp_qasm3([0.1, 0.5, 0.3], 0.3)
    assert "OPENQASM 3.0" in qasm
    assert 'include "stdgates.inc";' in qasm
    assert qasm.count("rx(") == 2  # one W per degree (degree 2)
    assert qasm.count("rz(") == 3  # degree + 1 phase rotations


def test_qasm3_rejects_out_of_range_signal():
    with pytest.raises(ValueError):
        circuits.qsp_qasm3([0.1, 0.2, 0.1], 1.5)


def test_to_qiskit_optional():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    phases = [0.2, -0.4, 0.1, 0.6]
    qc, theta = circuits.to_qiskit(phases)
    assert qc.num_qubits == 1
    x = 0.4
    bound = qc.assign_parameters({theta: math.acos(x)})
    U = Operator(bound).data
    assert np.allclose(U, _U_def(phases, x), atol=1e-10)
