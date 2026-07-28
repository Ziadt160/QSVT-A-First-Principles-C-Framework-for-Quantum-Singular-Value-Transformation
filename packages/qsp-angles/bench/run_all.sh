#!/usr/bin/env bash
# One-command, pinned reproduction of the qsp-angles benchmark + validation.
#
#   bash bench/run_all.sh            # from the package root
#
# Emits a provenance header (so the numbers are interpretable), runs the C++
# sym_qsp benchmark, the pyqsp timing baseline, and the independent
# cross-validation against pyqsp. Timings are pinned to a SINGLE THREAD on both
# sides so the comparison is fair (C++ is single-threaded; numpy/scipy BLAS is
# forced single-threaded here too). Correctness (residuals) is hardware-
# independent; absolute timings are not — report your own provenance.
set -euo pipefail

# Fair, reproducible timing: single thread on both sides.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
SRC="$PKG/src"
OUT="$HERE/results"
mkdir -p "$OUT"
BUILD="$(mktemp -d)"
EIGEN="${EIGEN_INCLUDE:-/usr/include/eigen3}"
CXX="${CXX:-g++}"
CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++17 -I$SRC -I$EIGEN"

echo "=== provenance ==="            | tee "$OUT/provenance.txt"
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "host:     $(uname -srm)"
  echo "cpu:      $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //' || echo n/a)"
  echo "compiler: $($CXX --version | head -1)"
  echo "cxxflags: $CXXFLAGS"
  echo "threads:  pinned to 1 (OMP/OpenBLAS/MKL)"
  echo "python:   $(python3 --version 2>&1)"
  python3 - <<'PY' 2>/dev/null || true
import importlib
from importlib import metadata
for m in ("pyqsp", "numpy", "scipy"):
    try:
        importlib.import_module(m)
        try:
            ver = metadata.version(m)        # distribution version (pyqsp lacks __version__)
        except metadata.PackageNotFoundError:
            ver = "?"
        print(f"{m:8} {ver}")
    except Exception as e:
        print(f"{m:8} (missing: {e})")
PY
} | tee -a "$OUT/provenance.txt"

echo; echo "=== build ==="
$CXX $CXXFLAGS "$HERE/bench_symqsp_core.cpp" "$SRC/SymQspAngleSolver.cpp" "$SRC/QspAngleSolver.cpp" -o "$BUILD/bench_symqsp"
$CXX $CXXFLAGS "$HERE/validate.cpp"          "$SRC/SymQspAngleSolver.cpp" "$SRC/QspAngleSolver.cpp" -o "$BUILD/validate"
echo "built bench_symqsp + validate"

echo; echo "=== C++ sym_qsp benchmark ==="     | tee "$OUT/cpp_sym_qsp.csv"
"$BUILD/bench_symqsp"                           | tee -a "$OUT/cpp_sym_qsp.csv"

if python3 -c "import pyqsp" 2>/dev/null; then
  echo; echo "=== pyqsp sym_qsp timing baseline ==="
  python3 "$HERE/bench_pyqsp.py" 2>/dev/null | tee "$OUT/pyqsp_sym_qsp.csv" || echo "(pyqsp timing skipped)"

  echo; echo "=== independent cross-validation vs pyqsp ==="
  python3 "$HERE/validate_vs_pyqsp.py" --manifest "$BUILD/cases.txt" --cpp-bin "$BUILD/validate" \
    | tee "$OUT/validation.txt"
else
  echo; echo "pyqsp not installed; skipping baseline + cross-validation."
  echo "install with: pip install pyqsp"
fi

echo; echo "Results written to $OUT/"
