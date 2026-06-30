// Compiles the qsp-angles C++ core + C ABI into a static lib the crate links.
//
// Eigen (header-only) must be findable: set EIGEN_INCLUDE to its include dir, or
// rely on the /usr/include/eigen3 default (Debian/Ubuntu `libeigen3-dev`). The
// `cc` crate links the C++ standard library automatically.

use std::path::PathBuf;

fn main() {
    // Paths relative to this crate (packages/qsp-angles/rust/).
    let capi = PathBuf::from("../capi");
    let src = PathBuf::from("../src");
    let eigen = std::env::var("EIGEN_INCLUDE").unwrap_or_else(|_| "/usr/include/eigen3".into());

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&capi) // qsp_angles.h
        .include(&src) // QspAngleSolver.hpp, SymQspAngleSolver.hpp
        .include(&eigen) // Eigen + unsupported/Eigen/FFT
        .opt_level(3)
        .file(capi.join("qsp_angles.cpp"))
        .file(src.join("SymQspAngleSolver.cpp"))
        .file(src.join("QspAngleSolver.cpp"))
        .compile("qsp_angles_c");

    for f in [
        "../capi/qsp_angles.h",
        "../capi/qsp_angles.cpp",
        "../src/SymQspAngleSolver.hpp",
        "../src/SymQspAngleSolver.cpp",
        "../src/QspAngleSolver.hpp",
        "../src/QspAngleSolver.cpp",
    ] {
        println!("cargo:rerun-if-changed={f}");
    }
    println!("cargo:rerun-if-env-changed=EIGEN_INCLUDE");
}
