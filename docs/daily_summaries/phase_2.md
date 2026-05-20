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
