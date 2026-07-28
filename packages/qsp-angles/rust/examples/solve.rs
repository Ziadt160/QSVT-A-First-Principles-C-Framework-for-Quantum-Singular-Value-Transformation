// cargo run --example solve
//
// Solves for the QSP phases of 0.8*T_5(x) and verifies them — a Rust client of
// the embeddable C++ solver, with no Python anywhere.

fn main() {
    println!("qsp-angles {}", qsp_angles::version());

    // 0.8 * T_5(x): full Chebyshev coefficients c_0..c_5.
    let coeffs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.8];
    let sol = qsp_angles::solve_chebyshev(5, &coeffs).expect("solve failed");
    println!(
        "degree 5 -> {} phases, residual = {:.2e}, converged = {}",
        sol.phases.len(),
        sol.residual,
        sol.converged
    );

    // Verify: Re<0|U(x)|0> reproduces 0.8*T_5(x).
    let mut max_err = 0.0f64;
    for i in 0..=20 {
        let x = -1.0 + 2.0 * i as f64 / 20.0;
        let got = qsp_angles::response(x, &sol.phases);
        let want = 0.8 * (5.0 * x.clamp(-1.0, 1.0).acos()).cos();
        max_err = max_err.max((got - want).abs());
    }
    println!("max |Re<0|U|0> - 0.8*T_5| over grid = {:.2e}", max_err);
    assert!(max_err < 1e-10, "phases did not reproduce the target");
}
