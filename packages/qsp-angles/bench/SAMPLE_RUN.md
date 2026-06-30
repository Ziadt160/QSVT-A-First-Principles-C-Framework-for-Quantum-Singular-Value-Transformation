# Sample run

A captured run of `bash bench/run_all.sh`. Reproduce it yourself — the point of
this file is to show the *shape* of the output (and that the validation passes),
not to be authoritative on timing for your hardware.

## Provenance

```
date_utc: 2026-06-30T13:24:49Z
host:     Linux 6.6.87.2-microsoft-standard-WSL2 x86_64
cpu:      Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
compiler: g++ (Ubuntu 12.4.0) 12.4.0
cxxflags: -O3 -march=native -DNDEBUG -std=c++17
threads:  pinned to 1 (OMP/OpenBLAS/MKL)
python:   3.12.3   |  numpy 2.5.0   scipy 1.18.0   pyqsp (installed)
```

## C++ sym_qsp benchmark

```
degree,iters,time_ms,residual
11,   5,   0.2,  5.06e-15
21,   5,   0.5,  1.30e-14
51,   5,   1.2,  4.64e-14
101,  5,   3.8,  6.97e-14
201,  5,  24.2,  9.52e-13
501,  5, 271.9,  8.26e-12
1001, 5, 955.4,  8.91e-12
```

## pyqsp sym_qsp timing baseline (same algorithm, Python)

```
degree, pyqsp_time_ms
11,     5.2
21,    14.1
51,    70.6
101,  254.6
```

At degree 101 the C++ port is ~3.8 ms vs pyqsp's ~255 ms on this machine (~67×;
the gap is wider at higher degree — see BENCHMARK.md for the dedicated
newton_solver-vs-newton_solver table going to degree 1001).

## Independent cross-validation vs pyqsp (the part that matters)

Random generic multi-coefficient targets, seed=12345, |f|=0.9, tol=1e-10. Each
solver graded in its own convention against the same target polynomial.

```
degree parity   pyqsp_resid     cpp_resid   verdict
     7      1     8.882e-16     1.443e-15   PASS
     8      0     1.110e-15     1.693e-15   PASS
    21      1     2.359e-15     3.886e-15   PASS
    20      0     4.316e-15     5.551e-15   PASS
    41      1     2.998e-15     5.385e-15   PASS
    40      0     4.996e-15     1.998e-14   PASS
   101      1     2.159e-14     1.932e-14   PASS
   100      0     7.716e-15     2.059e-14   PASS

VALIDATION PASSED — C++ and pyqsp both reach machine precision.
```
