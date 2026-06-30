# qsp-angles (Rust)

Safe Rust bindings for the **qsp-angles** QSP/QSVT angle (phase-factor) solver —
a C++ port of the symmetric-QSP Newton method (Dong–Lin–Ni–Wang, arXiv:2307.12468),
linked statically with **no Python runtime**. This is the path for embedding QSP
angle-finding directly in a compiled Rust quantum toolchain.

## Use

```rust
// 0.8 * T_5(x): full Chebyshev coefficients c_0..c_5.
let sol = qsp_angles::solve_chebyshev(5, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.8])?;
assert!(sol.converged);                       // residual < 1e-6
let y = qsp_angles::response(0.3, &sol.phases); // Re<0|U(0.3)|0>
```

Phases are the `degree + 1` full symmetric sequence in the **Wx convention**
(`Re<0|U(x)|0> = f(x)`). The target must satisfy `|f| <= 1` and parity `degree % 2`.

## Build requirements

`build.rs` compiles the C++ core via the [`cc`](https://crates.io/crates/cc)
crate, so you need:

- a C++17 compiler (`g++`/`clang++`), and
- **Eigen** headers (header-only). It looks at `$EIGEN_INCLUDE`, then
  `/usr/include/eigen3` (Debian/Ubuntu `libeigen3-dev`).

```bash
# default Eigen location
cargo run --example solve

# explicit Eigen location
EIGEN_INCLUDE=/path/to/eigen cargo run --example solve
```

The C++ sources are pulled from the sibling `../capi` and `../src` directories of
this repository; published as a standalone crate they would be vendored in.

## Status

The underlying C ABI is verified (the C and ctypes clients reproduce `0.8*T_5` to
~1e-15). These Rust bindings are a thin, idiomatic wrapper over that ABI; the FFI
signatures mirror [`capi/qsp_angles.h`](../capi/qsp_angles.h). They have not yet
been compiled in CI (no Rust toolchain on the dev box) — `cargo test` / `cargo
run --example solve` on a machine with Rust + Eigen is the remaining check.

## API

- `solve_chebyshev(degree, &[f64]) -> Result<Solution, QspError>` — coefficients
  `c_0..c_degree`.
- `response(x, &[f64]) -> f64` — achieved `Re<0|U(x)|0>`.
- `num_phases(degree) -> usize`, `version() -> &str`.
- `Solution { phases, residual, converged }`, `QspError(code)` with `.name()`.
