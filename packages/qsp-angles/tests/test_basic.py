import numpy as np
import pytest

import qsp_angles as qa


def test_linear_target():
    r = qa.target2angles(lambda x: 0.7 * x, degree=1)
    assert r.converged
    assert r.residual < 1e-6
    for x in (-0.9, -0.3, 0.0, 0.4, 0.85):
        assert abs(qa.response(x, r.phases) - 0.7 * x) < 1e-6


def test_chebyshev_degree_5():
    # 0.6 * T_5(x), an odd degree-5 target with |f| <= 1.
    f = lambda x: 0.6 * np.cos(5 * np.arccos(np.clip(x, -1.0, 1.0)))
    r = qa.target2angles(f, degree=5)
    assert r.converged
    assert len(r.phases) == 6  # degree + 1


def test_poly2angles_from_coeffs():
    # Monomial coefficients [0, 0.5] == 0.5 * x.
    r = qa.poly2angles([0.0, 0.5])
    assert r.converged
    assert abs(qa.response(0.5, r.phases) - 0.25) < 1e-6


def test_pyqsp_compatible_entrypoint():
    phases = qa.QuantumSignalProcessingPhases([0.0, 0.7], signal_operator="Wx")
    assert isinstance(phases, np.ndarray)
    assert abs(qa.response(0.6, phases) - 0.7 * 0.6) < 1e-6


def test_unsupported_convention_raises():
    with pytest.raises(NotImplementedError):
        qa.QuantumSignalProcessingPhases([0.0, 0.7], signal_operator="Wz")
