import numpy as np
import pytest

import qsp_angles as qa


def _T(d, x):
    """Chebyshev T_d(x) via cos(d*arccos(x)) (x clipped to [-1, 1])."""
    return np.cos(d * np.arccos(np.clip(x, -1.0, 1.0)))


# The homotopy fallback is the only Eigen-dependent piece and is off by default,
# so parametrize over whatever this build actually shipped.
METHODS = ["sym_qsp"] + (["homotopy"] if qa.HAS_HOMOTOPY else [])


def test_core_importable():
    # The compiled extension must be present and importable.
    import qsp_angles._core as core

    assert hasattr(core, "sym_qsp_poly_to_angles")
    assert hasattr(core, "response")
    # poly_to_angles exists iff this build included the homotopy fallback.
    assert hasattr(core, "poly_to_angles") == qa.HAS_HOMOTOPY


@pytest.mark.skipif(qa.HAS_HOMOTOPY, reason="build includes the homotopy fallback")
def test_homotopy_absence_is_actionable():
    # Asking for a solver this build lacks must say how to get it, not KeyError.
    with pytest.raises(ValueError, match="QSP_ANGLES_WITH_HOMOTOPY"):
        qa.target2angles(lambda x: 0.7 * x, degree=1, method="homotopy")


@pytest.mark.parametrize("method", METHODS)
def test_methods_converge_odd_degree5(method):
    # Both solvers must converge on the same odd degree-5 target 0.6*T_5.
    f = lambda x: 0.6 * _T(5, x)
    r = qa.target2angles(f, degree=5, method=method)
    assert r.converged
    assert r.residual < 1e-6
    assert len(r.phases) == 6  # degree + 1
    for x in (-0.9, -0.3, 0.0, 0.4, 0.85):
        assert abs(qa.response(x, r.phases) - f(x)) < 1e-6


@pytest.mark.parametrize("method", METHODS)
def test_methods_converge_even_degree2(method):
    # Both solvers must converge on the same even degree-2 target 0.5*T_2.
    f = lambda x: 0.5 * _T(2, x)
    r = qa.poly2angles([0.0, 0.0, 0.5], method=method)  # 0.5 * T_2
    assert r.converged
    assert r.residual < 1e-6
    assert len(r.phases) == 3  # degree + 1
    for x in (-0.8, -0.2, 0.0, 0.3, 0.9):
        assert abs(qa.response(x, r.phases) - f(x)) < 1e-6


def test_sym_qsp_is_default_method():
    # The default (no method=) must route to sym_qsp and agree with an explicit
    # method="sym_qsp" call to the bit.
    f = lambda x: 0.6 * _T(5, x)
    r_default = qa.target2angles(f, degree=5)
    r_sym = qa.target2angles(f, degree=5, method="sym_qsp")
    assert np.allclose(np.asarray(r_default), np.asarray(r_sym), atol=0.0, rtol=0.0)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        qa.target2angles(lambda x: 0.7 * x, degree=1, method="bogus")
    with pytest.raises(ValueError, match="method"):
        qa.poly2angles([0.0, 0.7], method="newton")  # not a registered name


def test_linear_target():
    r = qa.target2angles(lambda x: 0.7 * x, degree=1)
    assert r.converged
    assert r.residual < 1e-6
    for x in (-0.9, -0.3, 0.0, 0.4, 0.85):
        assert abs(qa.response(x, r.phases) - 0.7 * x) < 1e-6


def test_chebyshev_degree_5():
    # 0.6 * T_5(x), an odd degree-5 target with |f| <= 1.
    f = lambda x: 0.6 * _T(5, x)
    r = qa.target2angles(f, degree=5)
    assert r.converged
    assert len(r.phases) == 6  # degree + 1


def test_even_degree_target():
    # Degree-2 EVEN target: 0.5 * T_2(x) = 0.5 * (2x^2 - 1). |f| <= 0.5 <= 1,
    # even parity matches degree % 2 == 0.
    f = lambda x: 0.5 * _T(2, x)
    r = qa.target2angles(f, degree=2)
    assert r.converged
    assert len(r.phases) == 3  # degree + 1
    for x in (-0.8, -0.2, 0.0, 0.3, 0.9):
        assert abs(qa.response(x, r.phases) - f(x)) < 1e-6


def test_poly2angles_from_coeffs():
    # Chebyshev coefficients [0, 0.5] == 0.5 * T_1(x) == 0.5 * x.
    r = qa.poly2angles([0.0, 0.5])
    assert r.converged
    assert abs(qa.response(0.5, r.phases) - 0.25) < 1e-6


def test_chebyshev_basis_is_default():
    # Chebyshev coeffs [0, 0, 1] mean T_2(x) = 2x^2 - 1, NOT the monomial x^2.
    # Under a (wrong) monomial reading these coeffs would be x^2, so this test
    # FAILS if the default basis is monomial.
    r = qa.poly2angles([0.0, 0.0, 1.0])  # default basis="chebyshev" -> T_2
    assert r.converged
    for x in (-0.9, -0.4, 0.0, 0.3, 0.85):
        achieved = qa.response(x, r.phases)
        assert abs(achieved - _T(2, x)) < 1e-6  # matches T_2
        # And it is NOT x^2 (they differ by more than the tolerance somewhere).
    # Spot-check a point where T_2 and x^2 differ a lot: at x=0, T_2=-1, x^2=0.
    assert abs(qa.response(0.0, r.phases) - (-1.0)) < 1e-6
    assert abs(qa.response(0.0, r.phases) - 0.0) > 0.5


def test_monomial_basis_explicit():
    # Same coeffs read in the monomial basis mean x^2.
    r = qa.poly2angles([0.0, 0.0, 1.0], basis="monomial")
    assert r.converged
    for x in (-0.9, -0.4, 0.0, 0.3, 0.85):
        assert abs(qa.response(x, r.phases) - x * x) < 1e-6


def test_numpy_chebyshev_object_honored():
    cheb = np.polynomial.Chebyshev([0.0, 0.0, 1.0])  # T_2
    r = qa.poly2angles(cheb)
    assert r.converged
    assert abs(qa.response(0.0, r.phases) - (-1.0)) < 1e-6  # T_2(0) == -1


def test_numpy_polynomial_object_honored():
    poly = np.polynomial.Polynomial([0.0, 0.0, 1.0])  # x^2 (monomial)
    r = qa.poly2angles(poly)
    assert r.converged
    assert abs(qa.response(0.0, r.phases) - 0.0) < 1e-6  # 0^2 == 0


def test_trailing_zeros_trimmed():
    # Trailing zeros must not inflate the inferred degree; [0, 0.5, 0, 0] is
    # still degree 1 (0.5 * T_1).
    r = qa.poly2angles([0.0, 0.5, 0.0, 0.0])
    assert r.converged
    assert len(r.phases) == 2  # degree 1 -> 2 phases


def test_parity_violation_raises():
    # Degree 2 (even required) but an ODD target (0.7 * x) -> ValueError.
    with pytest.raises(ValueError, match="parity"):
        qa.target2angles(lambda x: 0.7 * x, degree=2)


def test_magnitude_violation_raises():
    # |f| > 1 on [-1, 1] -> ValueError.
    with pytest.raises(ValueError, match=r"\|f"):
        qa.target2angles(lambda x: 2.0 * x, degree=1)


def test_validate_can_be_disabled():
    # With validate=False a parity-mismatched solve is attempted (it simply will
    # not converge well, but it must not raise the validation ValueError).
    try:
        qa.target2angles(lambda x: 0.7 * x, degree=2, validate=False)
    except ValueError as exc:  # pragma: no cover - must not be a validation error
        assert "parity" not in str(exc)


def test_tol_behavior():
    # A clean target converges under the default tol but a comically tight tol
    # flips converged to False (residual is finite, ~1e-12+).
    r = qa.target2angles(lambda x: 0.7 * x, degree=1, tol=1e-6)
    assert r.converged
    r_strict = qa.target2angles(lambda x: 0.7 * x, degree=1, tol=1e-30)
    assert not r_strict.converged
    # The reported residual is the same raw value regardless of tol.
    assert r.residual == r_strict.residual


def test_pyqsp_compatible_entrypoint():
    phases = qa.QuantumSignalProcessingPhases([0.0, 0.7], signal_operator="Wx")
    assert isinstance(phases, np.ndarray)
    # Chebyshev [0, 0.7] == 0.7 * T_1 == 0.7 * x.
    assert abs(qa.response(0.6, phases) - 0.7 * 0.6) < 1e-6


def test_unsupported_convention_raises():
    with pytest.raises(NotImplementedError):
        qa.QuantumSignalProcessingPhases([0.0, 0.7], signal_operator="Wz")


def test_qspphases_raises_on_nonconvergence(monkeypatch):
    # If the solve does not converge, the wrapper must raise RuntimeError rather
    # than silently returning phases (it discards the result object).
    import qsp_angles as mod

    class _Fake:
        phases = np.zeros(2)
        residual = 1.0
        converged = False

    monkeypatch.setattr(mod, "poly2angles", lambda *a, **k: _Fake())
    with pytest.raises(RuntimeError, match="did not converge"):
        mod.QuantumSignalProcessingPhases([0.0, 0.7], signal_operator="Wx")
