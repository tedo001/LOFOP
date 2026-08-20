# Packaging and Releasing

LOFOP ships to two different ecosystems:

**Python**
- **PyPI** — `pip install lofop`
- **AUR** (Arch Linux) — `yay -S lofop`

**C++** (the `cpp/` SDK)
- **Source, by tag** — CMake `FetchContent` / CPM, no registry involved
- **Prebuilt archives** — attached to each GitHub Release
- **vcpkg / Conan** — recipes in `packaging/`

This document is for maintainers publishing releases. End users only need the install commands
above (see the [README](../README.md) / [MANUAL](../MANUAL.md)).

## Building the distributions

The package builds a standard wheel and sdist with no custom steps:

```bash
pip install build twine
python -m build              # writes dist/lofop-<ver>.tar.gz and .whl
python -m twine check dist/* # validates metadata
```

The sdist bundles the C++ ops source (`lofop/csrc/*.cpp`), the built-in configs, the license, and
the docs, so a source build is self-contained.

## Publishing to PyPI

Releases publish automatically via `.github/workflows/release.yml` when a version tag is pushed.
It uses **PyPI Trusted Publishing (OIDC)** — no API token is stored in the repo.

**One-time setup** (PyPI account required):

1. On https://pypi.org, register the `lofop` project name (or create a *pending publisher*).
2. Under the project's *Publishing* settings, add a trusted publisher:
   - Owner: `tedo001`, Repository: `LOFOP`, Workflow: `release.yml`, Environment: `pypi`.
3. In the GitHub repo, create an environment named `pypi` (Settings -> Environments).

**Each release:**

```bash
# 1. Bump the version in ALL FOUR sources, commit, merge to main:
#      lofop/version.py, pyproject.toml,
#      cpp/CMakeLists.txt (project VERSION), cpp/include/lofop/lofop.hpp (kVersion)
#    Then verify:  python scripts/check_version_sync.py
# 2. Tag and push:
git tag v1.3.0      # match the version in pyproject.toml
git push origin v1.3.0
```

The workflow builds, checks, and publishes. `pip install lofop` serves it within a minute.

To publish manually instead (needs a PyPI token):

```bash
python -m build
python -m twine upload dist/*
```

## Publishing to the AUR

The AUR package lives in [`packaging/aur/`](../packaging/aur/) and sources the PyPI sdist, so it
tracks the pip release. **Publish to PyPI first**, then:

```bash
cd packaging/aur
updpkgsums                                  # fills the real sha256 from the PyPI sdist
makepkg --printsrcinfo > .SRCINFO           # regenerate metadata
makepkg -si                                 # optional: build+install locally to test

# Push to the AUR (requires an AUR account + registered SSH key):
git clone ssh://aur@aur.archlinux.org/lofop.git aur-lofop
cp PKGBUILD .SRCINFO aur-lofop/
cd aur-lofop && git commit -am "lofop 1.2.1" && git push
```

After that, `yay -S lofop` (or any AUR helper) installs it. Optional features map to AUR
optdepends: `python-pytorch` for models/training, `python-onnx`/`python-onnxruntime` for export,
`gcc` for the native C++ ops.

## Version bumping

The version is defined in two places that must stay in sync: `lofop/version.py` (`__version__`)
and `pyproject.toml` (`version`). Update both, and the AUR `pkgver`, for each release.


## Releasing the C++ package

C++ has no PyPI. There is no registry you push a build to; the two central
indexes (vcpkg and ConanCenter) store a *recipe* that downloads the tarball
GitHub generates for your tag and verifies its hash. **The GitHub Release is
the distribution point** — the registries only point at it.

That makes the release ladder additive rather than either/or:

### Tier 0 — source by tag (works with no extra infrastructure)

Because `cpp/` installs a proper CMake package config, consumers can already
depend on a tag directly:

```cmake
include(FetchContent)
FetchContent_Declare(lofop
  GIT_REPOSITORY https://github.com/tedo001/LOFOP.git
  GIT_TAG v1.3.0
  SOURCE_SUBDIR cpp)
FetchContent_MakeAvailable(lofop)
target_link_libraries(my_app PRIVATE lofop::lofop)
```

Pushing the tag is the entire release step. Nothing else is required.

### Tier 1 — prebuilt archives on the GitHub Release

`.github/workflows/release.yml` builds the SDK on Linux, macOS, and Windows,
runs its tests, installs into a staging tree, and attaches
`lofop-<tag>-<platform>.tar.gz` / `.zip` to the GitHub Release. Consumers who
do not want to compile unpack one of those and point `CMAKE_PREFIX_PATH` at it.

This is automatic on every `v*` tag. It also produces the artifacts the next
two tiers are verified against.

### Tier 2 — your own vcpkg registry (self-published, no review)

`packaging/vcpkg/ports/lofop/` holds a complete port. To publish it yourself,
push that `ports/` tree plus a `versions/` database to a separate repository;
consumers then add it to their `vcpkg-configuration.json`:

```json
{
  "registries": [
    {
      "kind": "git",
      "repository": "https://github.com/tedo001/lofop-vcpkg-registry",
      "baseline": "<commit sha>",
      "packages": ["lofop"]
    }
  ]
}
```

and install with `vcpkg install lofop` or `vcpkg install lofop[onnxruntime]`.
No Microsoft review is involved.

### Tier 3 — the central registries (curated, reviewed)

- **vcpkg**: open a PR to `microsoft/vcpkg` adding `ports/lofop/` (the files in
  `packaging/vcpkg/`) plus a `versions/l-/lofop.json` entry.
- **ConanCenter**: open a PR to `conan-io/conan-center-index` adding
  `recipes/lofop/all/conanfile.py` (from `packaging/conan/`) and a `config.yml`.

Both are curated and generally expect a library to show some adoption, so
expect review and possibly a wait. Tiers 0–2 are not blocked on either.

### Refreshing the hashes

Both recipes pin the release tarball by digest, so each new version needs the
hash refreshed:

```bash
curl -sL https://github.com/tedo001/LOFOP/archive/refs/tags/v1.3.0.tar.gz | sha512sum  # vcpkg
curl -sL https://github.com/tedo001/LOFOP/archive/refs/tags/v1.3.0.tar.gz | sha256sum  # conan
```

Put the first in `portfile.cmake` (`SHA512`) and the second in
`conandata.yml` (`sha256`).

### Note on GitHub Packages

GitHub Packages does not support vcpkg or Conan — it covers npm, NuGet, Maven,
RubyGems, and containers. For C++, GitHub's role in this pipeline is Releases
(hosting the tarball and archives) and Actions (building and attaching them);
a custom vcpkg registry is simply another GitHub repository.
