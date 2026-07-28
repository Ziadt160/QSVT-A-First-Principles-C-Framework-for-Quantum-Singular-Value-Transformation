// Compiles the qsp-angles C++ core + C ABI into a static lib the crate links.
//
// NO SYSTEM DEPENDENCIES. The C ABI wraps SymQspAngleSolver, which is
// stdlib-only C++17, so `cargo build` works on a bare machine with just a C++17
// compiler -- no Eigen, no headers to install, no network. The `cc` crate links
// the C++ standard library automatically.
//
// (The Eigen-dependent homotopy fallback, QspAngleSolver, is not part of the C
// ABI and is deliberately not compiled here. It used to be, which forced every
// Rust consumer to install Eigen for code the ABI never called.)

use std::path::PathBuf;

fn main() {
    // Paths relative to this crate (packages/qsp-angles/rust/).
    let capi = PathBuf::from("../capi");
    let src = PathBuf::from("../src");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&capi) // qsp_angles.h
        .include(&src) // SymQspAngleSolver.hpp
        .opt_level(3)
        .file(capi.join("qsp_angles.cpp"))
        .file(src.join("SymQspAngleSolver.cpp"))
        .compile("qsp_angles_c");

    for f in [
        "../capi/qsp_angles.h",
        "../capi/qsp_angles.cpp",
        "../src/SymQspAngleSolver.hpp",
        "../src/SymQspAngleSolver.cpp",
        "../src/QspResult.hpp",
    ] {
        println!("cargo:rerun-if-changed={f}");
    }
}
