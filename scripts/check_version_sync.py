#!/usr/bin/env python3
"""Fail if the package version is not consistent across its sources.

LOFOP ships one version from four files, and a release is wrong if any of
them disagrees:

* ``lofop/version.py`` -- the runtime source of truth (``lofop.__version__``)
* ``pyproject.toml`` -- what the built wheel reports
* ``cpp/CMakeLists.txt`` -- what the C++ package exports to CMake consumers,
  and what vcpkg and Conan recipes pin against
* ``cpp/include/lofop/lofop.hpp`` -- ``lofop::kVersion``, what a linked C++
  binary reports at runtime through ``lofop_version()``

The last two matter as much as the first two: a C++ consumer resolving
``find_package(lofop 1.3)`` is matching against the CMake project version, so
a drift there is a broken dependency resolution rather than a cosmetic bug.
Run this locally or in CI; it exits non-zero and prints every mismatch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _version_py() -> str:
    text = (ROOT / "lofop" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        raise SystemExit("could not find __version__ in lofop/version.py")
    return match.group(1)


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        # Fallback for 3.9/3.10 without the stdlib TOML parser: read the
        # version line from the [project] table directly.
        match = re.search(r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']', text)
        if not match:
            raise SystemExit("could not find version in pyproject.toml") from None
        return match.group(1)
    return tomllib.loads(text)["project"]["version"]


def _cmake_version() -> str:
    """The C++ package version CMake consumers resolve against."""
    text = (ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*project\s*\(\s*lofop\s+VERSION\s+([0-9][^\s)]*)", text)
    if not match:
        raise SystemExit("could not find project(lofop VERSION ...) in cpp/CMakeLists.txt")
    return match.group(1)


def _header_version() -> str:
    """``lofop::kVersion``, reported at runtime by a linked C++ binary."""
    header = ROOT / "cpp" / "include" / "lofop" / "lofop.hpp"
    match = re.search(
        r'kVersion\s*=\s*"([^"]+)"', header.read_text(encoding="utf-8")
    )
    if not match:
        raise SystemExit("could not find kVersion in cpp/include/lofop/lofop.hpp")
    return match.group(1)


def main() -> int:
    sources = {
        "lofop/version.py": _version_py(),
        "pyproject.toml": _pyproject_version(),
        "cpp/CMakeLists.txt": _cmake_version(),
        "cpp/include/lofop/lofop.hpp": _header_version(),
    }
    distinct = set(sources.values())
    if len(distinct) > 1:
        print("version mismatch:", file=sys.stderr)
        for source, value in sources.items():
            print(f"  {source:<30} {value}", file=sys.stderr)
        return 1
    print(f"version OK: {distinct.pop()} (4 sources agree)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
