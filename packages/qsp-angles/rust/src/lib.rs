//! Safe Rust bindings for **qsp-angles** — an embeddable QSP/QSVT angle
//! (phase-factor) solver.
//!
//! The heavy lifting is a C++ port of the symmetric-QSP Newton method
//! (Dong–Lin–Ni–Wang, arXiv:2307.12468), linked statically via `build.rs`.
//! There is no Python in the loop, which is the point: this is meant to be
//! embedded directly in a compiled (Rust) quantum toolchain.
//!
//! Phases are the `degree + 1` full symmetric sequence in the **Wx convention**
//! (`Re<0|U(x)|0> = f(x)`); the target must satisfy `|f| <= 1` on `[-1, 1]` and
//! have parity `degree % 2`.
//!
//! ```no_run
//! // 0.8 * T_5(x): full Chebyshev coefficients c_0..c_5.
//! let sol = qsp_angles::solve_chebyshev(5, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.8]).unwrap();
//! assert!(sol.converged);
//! let y = qsp_angles::response(0.3, &sol.phases); // Re<0|U(0.3)|0>
//! ```

use std::ffi::CStr;
use std::fmt;
use std::os::raw::{c_char, c_double, c_int};

extern "C" {
    fn qsp_version() -> *const c_char;
    fn qsp_status_string(status: c_int) -> *const c_char;
    fn qsp_num_phases(degree: c_int) -> c_int;
    fn qsp_solve_chebyshev(
        degree: c_int,
        cheb_coeffs: *const c_double,
        ncoeffs: c_int,
        out_phases: *mut c_double,
        out_cap: c_int,
        out_len: *mut c_int,
        out_residual: *mut c_double,
        out_converged: *mut c_int,
    ) -> c_int;
    fn qsp_response(x: c_double, phases: *const c_double, len: c_int) -> c_double;
}

/// A successful solve: the phase sequence and how well it hit the target.
#[derive(Debug, Clone)]
pub struct Solution {
    /// `degree + 1` full symmetric phases (Wx convention).
    pub phases: Vec<f64>,
    /// Worst-case `|Re<0|U|0> - f|` over a grid on `[-1, 1]`.
    pub residual: f64,
    /// Whether `residual < 1e-6`.
    pub converged: bool,
}

/// An error returned by the underlying C ABI (carries the status code).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QspError(pub i32);

impl QspError {
    /// The symbolic name of the status code, e.g. `"QSP_ERR_BUFFER"`.
    pub fn name(&self) -> &'static str {
        // Safe: qsp_status_string returns a static C string, never NULL.
        unsafe { CStr::from_ptr(qsp_status_string(self.0)).to_str().unwrap_or("QSP_ERR_UNKNOWN") }
    }
}

impl fmt::Display for QspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "qsp-angles error {} ({})", self.0, self.name())
    }
}

impl std::error::Error for QspError {}

/// Number of phases produced for a given degree (`degree + 1`).
pub fn num_phases(degree: usize) -> usize {
    // Safe FFI call with a trivially-valid argument.
    unsafe { qsp_num_phases(degree as c_int) as usize }
}

/// Library version string (e.g. `"0.1.0"`).
pub fn version() -> &'static str {
    // Safe: qsp_version returns a static C string, never NULL.
    unsafe { CStr::from_ptr(qsp_version()).to_str().unwrap_or("?") }
}

/// Solve from the target's full Chebyshev coefficients `c_0..c_degree`
/// (`f(x) = Σ c_k T_k(x)`). `cheb_coeffs.len()` must equal `degree + 1`.
pub fn solve_chebyshev(degree: usize, cheb_coeffs: &[f64]) -> Result<Solution, QspError> {
    let n = degree + 1;
    if cheb_coeffs.len() != n {
        return Err(QspError(3)); // QSP_ERR_NCOEFFS
    }
    let mut phases = vec![0.0f64; n];
    let mut len: c_int = 0;
    let mut residual: c_double = 0.0;
    let mut converged: c_int = 0;
    // Safe: pointers/lengths are consistent and outlive the call.
    let status = unsafe {
        qsp_solve_chebyshev(
            degree as c_int,
            cheb_coeffs.as_ptr(),
            n as c_int,
            phases.as_mut_ptr(),
            n as c_int,
            &mut len,
            &mut residual,
            &mut converged,
        )
    };
    if status != 0 {
        return Err(QspError(status));
    }
    phases.truncate(len as usize);
    Ok(Solution { phases, residual, converged: converged != 0 })
}

/// Evaluate the achieved QSP polynomial `Re<0|U(x)|0>` at `x`.
pub fn response(x: f64, phases: &[f64]) -> f64 {
    // Safe: slice pointer + length are consistent.
    unsafe { qsp_response(x, phases.as_ptr(), phases.len() as c_int) }
}
