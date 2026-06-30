/* qsp_angles.h - stable C ABI for the qsp-angles solver.
 *
 * This is the embeddable entry point: a flat `extern "C"` interface with no C++
 * types in the signatures, so it links cleanly from C, Rust (FFI), Julia
 * (ccall), Go (cgo), or any compiled toolchain -- no Python runtime, no C++ ABI
 * concerns. The implementation (qsp_angles.cpp) wraps the validated
 * SymQspAngleSolver (symmetric-QSP Newton method, Dong-Lin-Ni-Wang
 * arXiv:2307.12468).
 *
 * Conventions: phases are the (degree+1) FULL symmetric phase sequence in the
 * Wx convention, i.e. they satisfy  Re<0|U(x)|0> = f(x)  where
 *   U(x) = e^{i phi_0 Z} prod_k [ W(x) e^{i phi_k Z} ],  W(x) = e^{i arccos(x) X}.
 * The target f must have |f| <= 1 on [-1,1] and definite parity (degree mod 2).
 */
#ifndef QSP_ANGLES_H
#define QSP_ANGLES_H

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. All entry points return QSP_OK (0) on success. Exceptions never
 * cross the ABI boundary; they are converted to QSP_ERR_INTERNAL. */
typedef enum qsp_status {
    QSP_OK = 0,
    QSP_ERR_NULL = 1,        /* a required pointer argument was NULL          */
    QSP_ERR_DEGREE = 2,      /* degree < 1                                    */
    QSP_ERR_NCOEFFS = 3,     /* ncoeffs != degree + 1                         */
    QSP_ERR_BUFFER = 4,      /* out_cap < degree + 1                          */
    QSP_ERR_NOT_CONVERGED = 5, /* solve ran but residual exceeded tolerance   */
    QSP_ERR_INTERNAL = 6     /* an unexpected internal/C++ error              */
} qsp_status;

/* Library version string, e.g. "0.1.0". Never NULL. */
const char* qsp_version(void);

/* Human-readable name for a status code, e.g. "QSP_ERR_BUFFER". Never NULL. */
const char* qsp_status_string(qsp_status status);

/* Number of phases produced for a given degree (= degree + 1), or 0 if
 * degree < 1. Use this to size out_phases before calling a solver. */
int qsp_num_phases(int degree);

/* Solve from the target's Chebyshev coefficients.
 *
 *   degree       polynomial degree d (>= 1)
 *   cheb_coeffs  full Chebyshev coefficients c_0..c_d of the target
 *                f(x) = sum_k c_k T_k(x); off-parity coefficients should be ~0
 *   ncoeffs      length of cheb_coeffs (must equal degree + 1)
 *   out_phases   caller-allocated, capacity >= degree + 1; receives the phases
 *   out_cap      capacity of out_phases
 *   out_len      [out] number of phases written (= degree + 1); may be NULL
 *   out_residual [out, optional] achieved max|Re<0|U|0> - f| on [-1,1]; may be NULL
 *   out_converged[out, optional] 1 if residual < 1e-6, else 0; may be NULL
 *
 * Returns QSP_OK even if the solve is imprecise (check out_residual /
 * out_converged); returns an error only for bad arguments or internal failure.
 */
qsp_status qsp_solve_chebyshev(int degree,
                               const double* cheb_coeffs, int ncoeffs,
                               double* out_phases, int out_cap, int* out_len,
                               double* out_residual, int* out_converged);

/* A real target callback f(x) with an opaque user context. */
typedef double (*qsp_target_fn)(double x, void* ctx);

/* Solve from a target callback (defined on [-1,1], |f| <= 1, parity = degree%2).
 * Same out_* semantics as qsp_solve_chebyshev. `ctx` is passed through to `f`. */
qsp_status qsp_solve_callback(int degree, qsp_target_fn f, void* ctx,
                              double* out_phases, int out_cap, int* out_len,
                              double* out_residual, int* out_converged);

/* Evaluate the achieved QSP polynomial Re<0|U(x)|0> at x for the given phases.
 * Returns 0.0 if phases is NULL or len < 1. */
double qsp_response(double x, const double* phases, int len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* QSP_ANGLES_H */
