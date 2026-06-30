#!/usr/bin/env python3
"""Guard against drift between the vendored QSP solver and the repo-root canonical source.

The standalone ``qsp-angles`` package vendors four C++ files:

    packages/qsp-angles/src/QspAngleSolver.hpp
    packages/qsp-angles/src/QspAngleSolver.cpp
    packages/qsp-angles/src/SymQspAngleSolver.hpp
    packages/qsp-angles/src/SymQspAngleSolver.cpp

These are copies of the repo-root canonical sources:

    include/QspAngleSolver.hpp
    src/QspAngleSolver.cpp
    include/SymQspAngleSolver.hpp
    src/SymQspAngleSolver.cpp

Only two differences are *intentional* and are normalized away here before
comparison:

  1. A leading "VENDORED from ..." comment block in each vendored file.
  2. The Eigen include path: ``<eigen3/Eigen/Dense>`` (canonical) vs
     ``<Eigen/Dense>`` (vendored, so the wheel builds against system or fetched
     Eigen). The ``<unsupported/Eigen/FFT>`` include (used by the sym_qsp solver)
     is identical in both and needs no rewrite.

If the algorithmic bodies diverge in any other way, this script prints a unified
diff of the *normalized* contents and exits non-zero.

It can be run from any working directory: the repo root is located by walking up
from this file until a directory containing both ``include/QspAngleSolver.hpp``
and ``packages/qsp-angles`` is found.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the repo root.

    The root is the first ancestor that contains both the canonical header
    (``include/QspAngleSolver.hpp``) and the vendored package directory
    (``packages/qsp-angles``).
    """
    for candidate in [start, *start.parents]:
        if (candidate / "include" / "QspAngleSolver.hpp").is_file() and (
            candidate / "packages" / "qsp-angles"
        ).is_dir():
            return candidate
    raise SystemExit(
        "check_vendored_sync: could not locate the repo root (looked for a "
        "directory containing both include/QspAngleSolver.hpp and "
        "packages/qsp-angles) starting from "
        f"{start}"
    )


# The leading vendored-comment block, line-for-line, as it appears at the top of
# each vendored file (after #pragma once for the header, and at the very top for
# the .cpp). These lines are removed before comparison.
_VENDORED_COMMENT_MARKERS = (
    "// VENDORED from the QSVT_Project root",
    "// The ONLY change vs the canonical source is the Eigen include path",
    "// (<eigen3/Eigen/Dense> -> <Eigen/Dense>) so the standalone wheel builds",
    "// (<eigen3/Eigen/Dense> -> <Eigen/Dense>). Keep in sync with the root.",
    "// against either a system Eigen or a fetched one. Keep in sync with the root.",
    # Extra marker lines used by the vendored SymQspAngleSolver.cpp, whose
    # comment notes the unchanged unsupported-FFT include.
    "// (<eigen3/Eigen/Dense> -> <Eigen/Dense>). The unsupported FFT include",
    "// (<unsupported/Eigen/FFT>) is unchanged -- it resolves against both a",
    "// system Eigen and a FetchContent Eigen. Keep in sync with the root.",
)


def normalize(text: str) -> str:
    """Normalize a source file's text for body-only comparison.

    Removes the leading vendored-comment block lines and rewrites the Eigen
    include so the canonical and vendored variants compare equal.
    """
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop the known vendored-comment-block lines anywhere they appear in the
        # leading comment (they are unique marker strings, so this is safe).
        if any(stripped.startswith(marker) for marker in _VENDORED_COMMENT_MARKERS):
            continue
        # Normalize the Eigen include path.
        normalized = line.replace("<eigen3/Eigen/Dense>", "<Eigen/Dense>")
        out_lines.append(normalized.rstrip())
    # Collapse runs of "empty" lines to a single one, and drop empty lines at the
    # very top of the file (or immediately after a leading `#pragma once`), so
    # that removing the vendored comment block does not leave spurious
    # blank-line / bare-comment differences. An "empty" line is either truly
    # blank or a bare "//" comment separator (the vendored block is bracketed by
    # such separators).
    def is_empty(s: str) -> bool:
        return s == "" or s.strip() == "//"

    collapsed = []
    prev_empty = False
    for line in out_lines:
        empty = is_empty(line)
        if empty and prev_empty:
            continue
        collapsed.append(line)
        prev_empty = empty

    # Drop a leading run of empties; keep a leading `#pragma once` and drop
    # empties immediately following it.
    result = []
    seen_pragma = False
    for idx, line in enumerate(collapsed):
        if not result and not seen_pragma:
            if line.strip() == "#pragma once":
                result.append(line)
                seen_pragma = True
                continue
            if is_empty(line):
                continue  # strip leading empties before any content
        if seen_pragma and len(result) == 1 and is_empty(line):
            continue  # strip empties right after the leading #pragma once
        result.append(line)

    return "\n".join(result) + "\n"


def compare(canonical: Path, vendored: Path, label: str) -> bool:
    """Return True if the normalized bodies match; print a diff and return False otherwise."""
    if not canonical.is_file():
        print(f"check_vendored_sync: canonical source missing: {canonical}")
        return False
    if not vendored.is_file():
        print(f"check_vendored_sync: vendored source missing: {vendored}")
        return False

    canon_norm = normalize(canonical.read_text(encoding="utf-8"))
    vend_norm = normalize(vendored.read_text(encoding="utf-8"))

    if canon_norm == vend_norm:
        print(f"  OK   {label}: vendored body matches canonical.")
        return True

    print(f"  DRIFT  {label}: vendored body differs from canonical (after "
          "normalizing the vendored comment block and Eigen include).")
    diff = difflib.unified_diff(
        canon_norm.splitlines(keepends=True),
        vend_norm.splitlines(keepends=True),
        fromfile=f"canonical:{canonical}",
        tofile=f"vendored:{vendored}",
    )
    sys.stdout.writelines(diff)
    return False


def main() -> int:
    here = Path(__file__).resolve()
    root = find_repo_root(here.parent)
    pkg = root / "packages" / "qsp-angles"

    print(f"check_vendored_sync: repo root = {root}")
    pairs = [
        (root / "include" / "QspAngleSolver.hpp", pkg / "src" / "QspAngleSolver.hpp", "QspAngleSolver.hpp"),
        (root / "src" / "QspAngleSolver.cpp", pkg / "src" / "QspAngleSolver.cpp", "QspAngleSolver.cpp"),
        (root / "include" / "SymQspAngleSolver.hpp", pkg / "src" / "SymQspAngleSolver.hpp", "SymQspAngleSolver.hpp"),
        (root / "src" / "SymQspAngleSolver.cpp", pkg / "src" / "SymQspAngleSolver.cpp", "SymQspAngleSolver.cpp"),
    ]

    all_ok = True
    for canonical, vendored, label in pairs:
        if not compare(canonical, vendored, label):
            all_ok = False

    if all_ok:
        print("check_vendored_sync: OK -- vendored sources are in sync.")
        return 0
    print(
        "check_vendored_sync: FAILED -- vendored sources have drifted from the "
        "repo-root canonical files. Re-vendor src/QspAngleSolver.{hpp,cpp} from "
        "the repo root (keeping only the vendored comment block and the "
        "<Eigen/Dense> include change)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
