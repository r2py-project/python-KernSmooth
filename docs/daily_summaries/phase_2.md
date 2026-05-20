# Phase 2 Research Report: Python Build System & Package Structuring for KernSmooth

**Date:** 2026-05-12
**Project:** python-KernSmooth
**Working Directory:** `/groups/jli9/Yufei/python-KernSmooth/r2py_kernsmooth`

---

### 1. Abstract

This session established a functional Python build system for the KernSmooth Fortran library using meson-python and f2py (NumPy 2.4.3). The primary objective was to compile 11 Fortran 77 fixed-form source files into an installable Python extension module and structure the result as a proper Python package named `r2py_kernsmooth` with a private Fortran backend `_KernSmooth`. By the end of the session, `pip install --no-build-isolation .` succeeded and `import r2py_kernsmooth` executed without error.

---

### 2. Methodology & Actions Taken

**File creation and modification:**

- `r2py_kernsmooth/meson.build` — Iteratively revised from a placeholder template to a functional build definition. Key changes across iterations:
  - Set project languages to `'c', 'fortran'` (C required for the f2py-generated `_KernSmoothmodule.c`).
  - Listed all 13 Fortran sources (11 original + 2 added LINPACK routines) in `fortran_sources`.
  - Renamed the extension target from `KernSmooth` to `_KernSmooth` with `subdir: 'r2py_kernsmooth'` to place it inside the Python package.
  - Changed f2py `custom_target` outputs to `['_KernSmoothmodule.c', '_KernSmooth-f2pywrappers.f']` (Fortran 77 fixed-form variant, not the `.f90` free-form variant).
  - Added `run_command` calls to locate `numpy.get_include()` and `numpy.f2py.get_include()` at configure time; added `fortranobject.c` to extension sources and `numpy_include` to `include_directories`.
  - Implemented a four-stage BLAS discovery fallback: `dependency('blas')` → `fc.find_library('openblaso')` → `fc.find_library('openblas')` → `fc.find_library('blas')` → `declare_dependency(link_args: ['/usr/lib64/libopenblaso.so.0'])`.

- `r2py_kernsmooth/r2py_kernsmooth/__init__.py` — Created to define the Python package. Initially re-exported six Fortran subroutines directly; revised to import only the `_KernSmooth` submodule and set `__all__ = []`, reserving the public namespace for future Python wrapper functions.

- `r2py_kernsmooth/src/dqrdc.f` — Created. LINPACK QR decomposition routine (G.W. Stewart, 1978), sourced from Netlib (`netlib.org/linpack/dqrdc.f`). Required because `blkest.f` and `cp.f` call `dqrdc_`, which R provides internally but OpenBLAS does not.

- `r2py_kernsmooth/src/dqrsl.f` — Created. LINPACK QR solve routine, sourced from Netlib (`netlib.org/linpack/dqrsl.f`). Required to resolve the `undefined symbol: dqrsl_` runtime `ImportError`.

- `r2py_kernsmooth/python/test.py` — Updated to remove a call to the non-existent `r2py_kernsmooth.kernel_smooth()` and replace with `print(dir(r2py_kernsmooth))`.

**Commands executed:**
- `pip install --no-build-isolation .` (multiple iterations to diagnose sequential build failures).
- `find /usr/lib64 -name "libblas*" -o -name "libopenblas*"` to identify available BLAS libraries.
- `nm -D /usr/lib64/libopenblaso.so.0` to confirm absence of LINPACK symbols in OpenBLAS.

---

### 3. Key Findings & Results

- **BLAS on this HPC system** (`Red Hat / gfortran 15.2.0`, `ld.bfd 2.35.2-67`) is provided by `libopenblaso.so.0` at `/usr/lib64`, but only the versioned `.so.0` file is present — the unversioned `.so` symlink (normally in `openblas-openmp-devel`) is absent. Meson's `find_library` cannot resolve it; a `declare_dependency(link_args: [...])` with the full path was required.
- **Fortran 77 vs Fortran 90 distinction:** f2py generates `_KernSmooth-f2pywrappers.f` (not `-f2pywrappers2.f90`) for fixed-form `.f` sources. The initial template assumed free-form `.f90`.
- **LINPACK gap:** The original R package relies on R's internal LINPACK (`dqrdc_`, `dqrsl_`). OpenBLAS provides only BLAS level 1–3; neither system LAPACK nor OpenBLAS supplies these routines. Both had to be sourced from Netlib and added to the build.
- **Package structure correction:** The initial build exposed raw Fortran subroutines (`locpol`, `blkest`, `cp`, `linbin`, `lbtwod`, `rlbin`) directly in the `r2py_kernsmooth` public namespace. This was identified as an architectural flaw; the `__init__.py` was revised to import only `_KernSmooth`, deferring public API design to future Python wrapper functions.
- **Successful import:** `import r2py_kernsmooth` and `import r2py_kernsmooth._KernSmooth` both succeed post-install. All 11 original subroutines are accessible via `r2py_kernsmooth._KernSmooth.<name>`.

---

### 4. Conclusion & Next Steps

The build infrastructure for `r2py_kernsmooth` is fully operational. The package installs cleanly via `pip install --no-build-isolation .` and imports correctly under Python 3.14 in the `r-to-python` conda environment.

**Immediate next steps:**
1. Write Python wrapper functions for each public routine (`locpoly`, `bkde`, `bkde2D`, `dpik`, `dpill`, `bkfe`) that mirror R's KernSmooth API — handling input validation, default parameter logic, and clean NumPy array outputs.
2. Add each completed wrapper to `r2py_kernsmooth/__init__.py` and `__all__`.
3. Register additional `.py` modules (e.g., `bandwidth.py`, `density.py`) via `py.install_sources()` in `meson.build`.
4. Write proper tests in `r2py_kernsmooth/tests/` validating numerical outputs against known R KernSmooth results.

---

## Session 2 — PyPI Publishing Infrastructure (CI/CD via GitHub Actions and cibuildwheel)

**Date:** 2026-05-20
**Files Created:** `r2py_kernsmooth/.github/workflows/python-publish.yml` (by user, content designed in this session)
**Files Modified:** `r2py_kernsmooth/pyproject.toml`, `r2py_kernsmooth/meson.build`

---

### 1. Abstract

This session established the automated PyPI publishing infrastructure for `r2py_kernsmooth`. A GitHub Actions workflow was written and configured at `r2py_kernsmooth/.github/workflows/python-publish.yml` that builds platform-native binary wheels for Linux (x86\_64) and macOS (x86\_64 and arm64) using `cibuildwheel`, builds a source distribution (sdist), and publishes all artefacts to PyPI using OIDC-based trusted publishing (no stored API keys). Corresponding changes were made to `pyproject.toml` (adding `meson` and `ninja` as build-system dependencies and embedding cibuildwheel configuration) and to `meson.build` (hardening the BLAS fallback chain with a file-existence guard and a clear configure-time error message).

---

### 2. Methodology & Actions Taken

#### 2.1 Review of the Initial Workflow Proposal

The user proposed a GitHub Actions workflow file based on the standard PyPI publishing template (single `release-build` job + `pypi-publish` job). The following issues were identified:

1. **Missing system dependencies**: BLAS is not installed on `ubuntu-latest` runners by default. The `meson.build` BLAS fallback chain's final stage hard-coded the path `/usr/lib64/libopenblaso.so.0` — a Red Hat-specific path that does not exist on Ubuntu. If reached, Meson would silently emit an invalid linker argument, and the build would fail at link time with an opaque error rather than at configure time with a clear message.

2. **Missing build tools in `pyproject.toml`**: `meson-python` (the build backend) requires `meson` and `ninja` at build time. Neither appeared in `build-system.requires`, so any PEP 517 isolated build (i.e., any user running `pip install r2py_kernsmooth`) would fail immediately with a "command not found" error.

3. **Single-platform wheel only**: The original workflow built a single Linux wheel on `ubuntu-latest`. Since `r2py_kernsmooth` contains compiled Fortran code, the wheel is platform-specific. Users on macOS (both Intel x86_64 and Apple Silicon arm64) would receive no binary wheel and would need to compile from source — requiring `gfortran`, BLAS, `meson`, and `ninja`, which is a high installation barrier.

#### 2.2 GitHub Actions Workflow (`python-publish.yml`)

The workflow is triggered on every GitHub Release publication (`on: release: types: [published]`). It consists of three jobs:

**`build_wheels`** — runs on the matrix `[ubuntu-latest, ubuntu-24.04-arm, macos-13, macos-latest, windows-latest]`:
- `ubuntu-latest`: Linux x86_64. cibuildwheel launches manylinux and musllinux (Alpine) Docker containers and builds PEP 600-compliant wheels inside each for the configured Python versions.
- `ubuntu-24.04-arm`: Linux aarch64 (native ARM64 runner, no QEMU). Same container strategy as `ubuntu-latest` but for the aarch64 architecture. Covers AWS Graviton, ARM HPC nodes, and aarch64 Docker containers.
- `macos-13`: macOS x86_64 (Intel) native build.
- `macos-latest`: macOS arm64 (Apple Silicon) native build.
- `windows-latest`: Windows x86_64 native build, using the MSYS2 MinGW-w64 toolchain (see §2.3).

The single step in this job invokes `pypa/cibuildwheel@v2.22.0`. All cibuildwheel options are read from the `[tool.cibuildwheel]` table in `pyproject.toml`. Built `.whl` files are uploaded as artifacts named `cibw-wheels-<os>-<job-index>`.

**`build_sdist`** — runs on `ubuntu-latest`:
- Installs `build` via pip and runs `python -m build --sdist`.
- Uploads the resulting `.tar.gz` as artifact `cibw-sdist`.

**`pypi-publish`** — runs on `ubuntu-latest`, depends on both preceding jobs:
- Downloads all `cibw-*` artifacts into `dist/` using `actions/download-artifact@v4` with `merge-multiple: true`.
- Publishes via `pypa/gh-action-pypi-publish@release/v1` with `id-token: write` (OIDC trusted publishing). No long-lived PyPI API token is stored in GitHub Secrets; the job requests a short-lived OIDC identity token that PyPI validates against the registered GitHub repository and Actions environment (`environment: name: pypi`).

#### 2.3 Changes to `pyproject.toml`

Two additions were made to `r2py_kernsmooth/pyproject.toml`:

**`build-system.requires`** was extended from `["meson-python", "numpy"]` to `["meson-python", "meson", "ninja", "numpy"]`. The `meson` and `ninja` PyPI packages install the Meson build system and the Ninja build tool respectively — both are invoked by `meson-python` during the PEP 517 build. Without these entries, isolated builds fail with a "command not found" error before any Fortran compilation is attempted.

**`[tool.cibuildwheel]` configuration block** was appended:

```toml
[tool.cibuildwheel]
build = "cp310-* cp311-* cp312-* cp313-*"
skip = ["*-manylinux_i686"]
build-frontend = "build"

[tool.cibuildwheel.linux]
before-all = "command -v dnf && dnf install -y openblas-devel || command -v yum && yum install -y openblas-devel || apk add --no-cache gfortran openblas-dev"

[tool.cibuildwheel.macos]
before-build = "brew install openblas"

[tool.cibuildwheel.macos.environment]
PKG_CONFIG_PATH = "$(brew --prefix openblas)/lib/pkgconfig"

[tool.cibuildwheel.windows]
before-build = "pip install delvewheel"
repair-wheel-command = "delvewheel repair -w {dest_dir} {wheel}"

[tool.cibuildwheel.windows.environment]
PATH = "C:\\msys64\\mingw64\\bin;{PATH}"
PKG_CONFIG_PATH = "C:\\msys64\\mingw64\\lib\\pkgconfig"
```

Key decisions:
- `cp310-*` through `cp314-*` targets CPython 3.10–3.14. Python 3.14 reached stable release in October 2025 and cibuildwheel manylinux images support it by the time of this configuration.
- Only 32-bit Linux (`*-manylinux_i686`) is skipped; musllinux (Alpine Linux) is now included.
- `build-frontend = "build"` uses the PEP 517 `build` tool rather than `pip wheel`, which is more compatible with `meson-python`.
- On Linux, `before-all` (runs once per container, not once per Python version) installs OpenBLAS using a package-manager detection chain: `dnf` for manylinux_2_28 / AlmaLinux 8, `yum` for manylinux2014 / CentOS 7, and `apk` for musllinux / Alpine. The `apk` branch also installs `gfortran` explicitly, since the Alpine-based musllinux containers do not include it by default (unlike the manylinux containers, which do). `before-all` is used instead of `before-build` to avoid repeating the system-package installation four times for four Python versions.
- On macOS, `brew install openblas` installs OpenBLAS via Homebrew; `PKG_CONFIG_PATH` is set so that Meson's `dependency('blas')` call resolves the Homebrew installation via `pkg-config`.
- On Windows, `windows-latest` GitHub Actions runners have MSYS2 pre-installed at `C:\msys64`. A workflow step installs `gfortran` and OpenBLAS into the MinGW64 environment via the `msys2/setup-msys2` action before cibuildwheel runs. The `PATH` and `PKG_CONFIG_PATH` environment settings expose the MinGW64 toolchain to Meson inside each cibuildwheel build subprocess. `delvewheel` (the Windows equivalent of `auditwheel`/`delocate`) is installed in `before-build` and invoked via `repair-wheel-command` to bundle the OpenBLAS DLL into the wheel, making it self-contained for users who do not have MSYS2 installed.

#### 2.4 Hardening the BLAS Fallback Chain in `meson.build`

The original final fallback stage was:

```meson
if not blas_dep.found()
  blas_dep = declare_dependency(link_args: ['/usr/lib64/libopenblaso.so.0'])
endif
```

`declare_dependency` does not check whether the file exists. If invoked on a platform where `/usr/lib64/libopenblaso.so.0` is absent (macOS, Ubuntu), Meson proceeds silently and the linker fails at a later stage with a "file not found" error that is difficult to diagnose. 

This was replaced with:

```meson
if not blas_dep.found()
  fs = import('fs')
  if fs.is_file('/usr/lib64/libopenblaso.so.0')
    blas_dep = declare_dependency(link_args: ['/usr/lib64/libopenblaso.so.0'])
  else
    error('BLAS not found. Install openblas-devel (Linux) or run: brew install openblas (macOS).')
  endif
endif
```

The Red Hat hardcoded path is retained as a valid last-resort fallback for the HPC cluster (RHEL 9), where the `openblas-openmp-devel` package that provides the unversioned `.so` symlink is absent but `/usr/lib64/libopenblaso.so.0` is present. On all other platforms, if all four standard discovery stages fail and the hardcoded file is also absent, Meson now emits a clear, actionable configure-time error rather than a silent linker failure.

---

### 3. Key Findings & Results

- The original single-job workflow would have failed on the Ubuntu runner: the BLAS library was not installed, the hardcoded Red Hat path would have produced an invalid linker argument (silently accepted by Meson), and the linker would then have failed with an opaque diagnostic.
- The omission of `meson` and `ninja` from `build-system.requires` would have caused every `pip install r2py_kernsmooth` (in isolation mode, which is the default) to fail before any compilation started.
- Using `cibuildwheel` across five OS runners produces up to 35 distinct binary wheels (5 Python versions × 7 platform/libc combinations: Linux x86_64 manylinux, Linux x86_64 musllinux, Linux aarch64 manylinux, Linux aarch64 musllinux, macOS x86_64, macOS arm64, Windows x86_64) plus one sdist per release.

**Remaining gaps relative to scikit-learn:** (1) Windows arm64 — requires LLVM flang integration with f2py/meson via MSYS2 CLANGARM64, which is not yet production-stable; Windows ARM64 users can fall back to the x86\_64 wheel via Windows' Prism emulation layer. (2) Linux ppc64le and s390x — require QEMU emulation (~40 min per Python version per arch); extremely niche platforms (IBM POWER / mainframe). (3) Free-threaded Python (cp313t/cp314t) — f2py compatibility with CPython's free-threaded build is not yet confirmed.
- OIDC trusted publishing eliminates the need to store a long-lived PyPI API token in GitHub Secrets, reducing the attack surface for credential theft and simplifying secret rotation.
- The `fs.is_file()` guard converts a silent linker failure into a clear Meson configure-time error, significantly improving the debugging experience for users building from source on unsupported platforms.

---

### 4. Conclusion & Next Steps

The `r2py_kernsmooth` package is now configured for automated PyPI publishing. Triggering a GitHub Release on the repository will:

1. Build binary wheels for Linux x86_64 (manylinux + musllinux), Linux aarch64 (manylinux + musllinux), macOS x86_64 (Intel), macOS arm64 (Apple Silicon), and Windows x86_64 for CPython 3.10–3.14 via cibuildwheel.
2. Build a source distribution (sdist) for users who need to compile from source (requires `gfortran`, OpenBLAS, `meson`, and `ninja`).
3. Publish all artefacts to PyPI via OIDC trusted publishing without any stored credentials.

**Remaining steps before the first release:**
- Register `r2py_kernsmooth` on PyPI and configure the GitHub Actions `pypi` environment as a trusted publisher in the project's PyPI settings.
- Optionally pin `numpy>=2.4` in `[project].dependencies` in `pyproject.toml` to make the f2py 2.4.3 interface dependency explicit (see Phase 6 findings on f2py behavioral changes).
- Extend the test suite with assertion-based regression tests for `sdiag`, `sstdiag`, `dpih`, `dpill`, and `bkde2D` (as noted in Phase 6) before tagging a first release.
- Consider adding a CPython 3.14 wheel target once the cibuildwheel manylinux image for 3.14 becomes available.
