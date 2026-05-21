# Phase 5 Session Report — python-KernSmooth

**Date:** 2026-05-18
**File Modified:** `r2py_kernsmooth/r2py_kernsmooth/__init__.py`

---

## Session 1 — Type Annotation Completion and Namespace Privacy

### 1. Abstract

This session focused on two sequential improvements to `r2py_kernsmooth/r2py_kernsmooth/__init__.py`: (1) completing incomplete type annotations across all 14 public function prototypes, and (2) investigating and applying the standard Python convention for restricting the package's public namespace. The session concluded with a fully annotated public API and a single `del atexit` statement added to remove the one init-time-only import from the module namespace.

---

### 2. Methodology & Actions Taken

**Task 1 — Type Annotation Completion**

- Reviewed all function signatures in `r2py_kernsmooth/r2py_kernsmooth/__init__.py` to identify annotations missing dtype or container subtypes.
- Added `from typing import Any` to the stdlib import block.
- Updated every `np.ndarray` parameter and return annotation to `np.ndarray[Any, np.dtype[np.float64]]`, reflecting the exclusive use of 64-bit floating-point arrays throughout the module.
- Updated all bare `dict` return types to parameterized forms:
  - `rlbin`, `locpoly`, `sdiag`, `sstdiag`, `bkde`, `bkde2D` → `dict[str, np.ndarray[Any, np.dtype[np.float64]]]`
  - `blkest` → `dict[str, np.float64]` (scalar values extracted from float64 arrays)
- Replaced bare `tuple` and `list` annotations with fully parameterized forms: `tuple[float, float]` and `list[tuple[float, float]]`.
- Added `| None` to all optional parameters previously annotated without it (`bandwidth`, `range_x`, `y`, `degree`).
- Functions affected: `linbin`, `rlbin`, `bkfe`, `blkest`, `cpblock`, `linbin2D`, `locpoly`, `sdiag`, `sstdiag`, `bkde`, `bkde2D`, `dpih`, `dpik`, `dpill`.

**Task 2 — Namespace Privacy Investigation and Resolution**

- Identified that importing `numpy as np`, `math`, `sys`, etc. at the module level exposes those names as attributes of the `r2py_kernsmooth` package (e.g., `r2py_kernsmooth.np`).
- **Attempt 1:** Renamed all imports to underscore-prefixed aliases (`import numpy as _np`, `import math as _math`, etc.) and performed a global replace of all references throughout the file. Confirmed via runtime test that `hasattr(r2py_kernsmooth, 'np')` returned `False`.
- **Reversal:** Following a web research query on the conventions of major scientific Python packages (scipy, numpy, pandas), found that underscore-aliasing of dependency imports is not the standard practice. Reverted all aliases to their conventional forms.
- **Finding:** Major packages use `del` to remove init-time-only imports, and rely on `__all__` as the authoritative public API contract. Names used inside function bodies cannot be deleted without causing `NameError` at call time.
- **Final action:** Added `del atexit` on the line immediately following `atexit.register(_on_unload)`, the sole location where `atexit` is used.

---

### 3. Key Findings & Results

- **14 function signatures** were updated with complete type annotations; 0 bare `np.ndarray`, `dict`, `tuple`, or `list` annotations remain in any public function prototype.
- The proper numpy dtype annotation form is `np.ndarray[Any, np.dtype[np.float64]]`, requiring `from typing import Any`; `NDArray[np.float64]` from `numpy.typing` was considered but rejected by the user in favour of the `np.ndarray` base type.
- `del` is only applicable to names used exclusively during module initialization. Of all imports in the file (`atexit`, `math`, `sys`, `warnings`, `Any`, `np`, `norm`, `beta_dist`), only `atexit` qualifies — all others are referenced inside function bodies.
- Module-level `__getattr__` (PEP 562) cannot hide names already present in the module's `__dict__`; it is only invoked for absent attributes.
- The `__all__` list (already present, covering all 13 public functions) remains the definitive public API contract per established Python convention.

---

### 4. Conclusion

`r2py_kernsmooth/r2py_kernsmooth/__init__.py` received complete, parameterized type annotations on all public functions and a `del atexit` statement removing the one safely deletable import from the module namespace. The remaining dependency imports (`np`, `math`, `sys`, `warnings`, `norm`, `beta_dist`, `Any`) were intentionally retained, consistent with the standard practice of relying on `__all__` for public API enforcement. No further namespace-privacy actions were possible without restructuring function bodies to avoid module-level globals.

---
---

## Session 2 — Code Quality Audit and Systematic Refactoring

### 1. Abstract

This session performed a thorough, in-depth review of `r2py_kernsmooth/r2py_kernsmooth/__init__.py` to evaluate conformance with standard Python packaging conventions. Seven distinct categories of defects were identified — ranging from dead R-porting artifacts and a silent crash bug to structural code duplication and PEP 8 violations — and all were subsequently resolved in a single comprehensive rewrite of the file.

---

### 2. Methodology & Actions Taken

**Step 1 — Full File Audit**

The complete text of `r2py_kernsmooth/r2py_kernsmooth/__init__.py` was read and analyzed function by function. The audit examined:
- Import usage across the entire module, not just at the import site.
- Every function signature for correctness of type annotations and default values.
- Each function body for correctness relative to its declared interface.
- Cross-function patterns for repeated logic blocks.
- Compliance with PEP 8 line-length limits (≤ 88 characters for signatures).
- Presence of comments carried over from the original R source (`KernSmooth` R package, M. P. Wand, 1997–2009).

Seven defect categories were identified (detailed in Section 3).

**Step 2 — Rewrite and Application of All Fixes**

`r2py_kernsmooth/r2py_kernsmooth/__init__.py` was rewritten in full. The specific changes applied are itemized below.

*Removals:*
- `import atexit` (line 1) — rendered unused after removing `_on_unload` and its registration.
- `import sys` (line 3) — rendered unused after removing `_on_attach`, its sole caller.
- `_on_attach(libname, pkgname)` (lines 509–510) — a direct port of R's `.onAttach` hook; Python import machinery has no equivalent and never calls it. The function was dead code since the initial port.
- `_on_unload(libpath="")` (lines 513–519) — an explicitly documented no-op (`pass` body); its `atexit` registration was therefore also a no-op.
- `atexit.register(_on_unload)` (line 522) — wasted an atexit slot for zero effect.
- `del atexit` (line 523) — non-standard namespace cleanup that had been added in Session 1 to clean up after the above, now also moot.
- The R changelog comment `# remove unused 'q' 2007-07-10` inside `cpblock` — an R-era source-control note with no meaning in a Python file.
- The multi-line R-centric comment block inside `_on_unload` explaining R's `library.dynam.unload` semantics — removed with the function.

*Additions:*
- `_resolve_choice(val, choices, param_name)` — a private helper that encapsulates the repeated prefix-match validation idiom (exact match → unique prefix match → ambiguous prefix error → no-match error). Replaces three identical inline blocks in `bkde` (for `kernel`), `dpih` (for `scalest`), and `dpik` (for both `kernel` and `scalest`), reducing each call site to a single line.
- `_discretize_bandwidth(bandwidth, M, delta, Q, tau)` — a private helper returning `(hdisc, Lvec, indic, Q)` that encapsulates the 27-line bandwidth-discretization block (scalar path, vector-of-length-M path, error path) previously copy-pasted verbatim into `locpoly`, `sdiag`, and `sstdiag`. The helper also corrects a robustness defect in the scalar-detection logic: the original code used `elif len(bandwidth) == 1`, which raises `TypeError` for 0-d numpy arrays (produced by `np.asarray` on a plain Python float or `np.float64` scalar); the helper instead gates on `bw.ndim == 0 or len(bw) == 1`, mirroring the pattern already used correctly in `bkde2D`.

*Bug fix — `bandwidth=None` crash in `locpoly`, `sdiag`, `sstdiag`:*
- All three functions declared `bandwidth: np.ndarray[...] | None = None` yet their bodies called `np.asarray(bandwidth, dtype=np.float64)` unconditionally, converting `None` to `array(nan)` (a 0-d array), then immediately called `len(bandwidth)` on the result, raising `TypeError: len() of unsized object`. The type annotation advertised `None` as valid; the implementation crashed silently.
- Fix: a guard `if bandwidth is None: raise ValueError("'bandwidth' must be specified")` was inserted as the first statement of each function body, converting an opaque `TypeError` from deep inside numpy into a clear, attributable `ValueError` at the function boundary.

*Signature reformatting:*
- All function signatures with lines exceeding 88 characters were reformatted to multi-line style with one parameter per line. Affected functions: `linbin`, `rlbin`, `bkfe`, `blkest`, `cpblock`, `linbin2D`, `locpoly`, `sdiag`, `sstdiag`, `bkde`, `bkde2D`, `dpih`, `dpik`, `dpill`.

---

### 3. Key Findings & Results

**Dead code and unused imports from R porting**

`_on_attach` and `_on_unload` were direct transliterations of R's `.onAttach` and `.onUnload` hooks. Python's import system provides no hook mechanism equivalent to R's package attach/detach events. Neither function was reachable from any Python code path, making `import sys` — used only in `_on_attach` — an unused import as well. These artifacts had been present since the initial port without effect on runtime behavior, but added confusion and namespace noise.

**Silent crash bug in three public functions**

The `bandwidth=None` crash in `locpoly`, `sdiag`, and `sstdiag` was a latent defect introduced when type annotations were added in Session 1. Adding `| None` to the `bandwidth` annotation (to reflect the default value of `None`) created the impression that `None` was a semantically meaningful input, while the function body made no attempt to handle it. In practice, all three functions are always called with a concrete bandwidth from `dpill`, so the bug was unreachable in the current internal call graph — but would have been encountered immediately by any external caller following the type annotation.

**Structural code duplication**

The bandwidth-discretization block and the prefix-match validation block each appeared in identical form across three functions. The duplication was exact (copy-paste), meaning any future fix or behavioral change would require three synchronized edits. Extraction into `_discretize_bandwidth` and `_resolve_choice` eliminates this obligation. The `_discretize_bandwidth` extraction also fixed the 0-d array robustness issue as a side effect.

**Line-length violations**

The most extreme signatures (`locpoly`, `sdiag`, `sstdiag`) reached 210+ characters on a single line.

**Conformance assessment**

The following aspects were confirmed as standard and correct prior to this session:
- `__all__` defined at module top, listing all 14 public names.
- Relative private import `from . import _KernSmooth` for the compiled Fortran extension.
- `warnings.warn(..., UserWarning, stacklevel=2)` with correct `stacklevel`.
- Lowercase generic annotations (`dict[str, ...]`, `tuple[float, float]`) appropriate for Python 3.10+.
- `float | None` union syntax (Python 3.10+), consistent with the runtime version (CPython 3.14, as evidenced by `__pycache__/__init__.cpython-314.pyc`).

---

### 4. Conclusion & Next Steps

All seven defect categories identified in the audit have been resolved. `r2py_kernsmooth/r2py_kernsmooth/__init__.py` now:
- Contains no dead code or unused imports.
- Raises a clear `ValueError` (rather than a cryptic `TypeError`) when `bandwidth=None` is passed to `locpoly`, `sdiag`, or `sstdiag`.
- Has no duplicated bandwidth-discretization or kernel/scalest-validation logic.
- Handles the 0-d numpy array edge case correctly in the scalar-bandwidth path.
- Conforms to PEP 8 line-length limits across all function signatures.
- Contains no R-era changelog comments or R-specific hook documentation.

The `_resolve_choice` and `_discretize_bandwidth` helpers are private (underscore-prefixed) and absent from `__all__`, preserving the public API surface unchanged.

**Suggested next steps:**
- Expand `r2py_kernsmooth/tests/test.py` beyond the single `bkde` smoke test to cover all 14 public functions, including boundary conditions such as `bandwidth` near zero, `level` at 0 and 5 for `dpih`/`dpik`, and the binned vs. unbinned code paths.
- Validate numerical agreement between `r2py_kernsmooth` outputs and the reference R `KernSmooth` package (v2.23-22) on a shared dataset for each function.
- Consider adding `__version__ = "2.23"` to `__init__.py` to expose the mirrored R package version programmatically.

---
---

## Session 3 — R-Source Alignment Audit: Unused Parameters and Error Handling

### 1. Abstract

This session investigated two questions raised by inspection of the ported `r2py_kernsmooth/r2py_kernsmooth/__init__.py`: (1) why the `drv` and `kernel` parameters appear in the signatures of `sdiag` and `sstdiag` yet are never referenced in their bodies, and (2) whether all error behaviors in the Python module exactly match those defined in `KernSmooth/R/all.R`. The first question was answered by examining both the Fortran source and the R reference implementation. The second revealed three discrepancies that were corrected: the `_resolve_choice` error message format, an invented `bandwidth=None` guard in `locpoly`/`sdiag`/`sstdiag`, and the bandwidth positivity check pattern in `locpoly`.

---

### 2. Methodology & Actions Taken

**Task 1 — Investigation of Unused `drv` and `kernel` Parameters**

- Read `r2py_kernsmooth/src/sdiag.f` and `r2py_kernsmooth/src/sstdiag.f` to understand why `drv` is absent from their Fortran interfaces.
- Read `r2py_kernsmooth/src/locpoly.f` for contrast, noting its `idrv` parameter.
- Read `KernSmooth/R/all.R` to confirm R's `sdiag` and `sstdiag` carry the same unused parameters in identical signatures.

**Task 2 — Error Handling Audit**

- Extracted every `stop()`, `warning()`, and `match.arg()` call from `KernSmooth/R/all.R` (16 total error/warning emissions across 8 functions).
- Compared each against the corresponding `raise`/`warn` in `r2py_kernsmooth/r2py_kernsmooth/__init__.py`.
- Applied three targeted fixes to align the Python behavior with the R source.

---

### 3. Key Findings & Results

**Unused `drv` and `kernel` parameters in `sdiag`/`sstdiag`**

Two independent sources confirm these parameters are intentional non-operative artifacts inherited from the R source:

1. **Fortran source**: `sdiag.f` and `sstdiag.f` hardcode the Gaussian kernel via `exp(-(delta*j/hdisc(i))**2/2)` with no kernel-type argument. Both subroutines always extract `Smat(1,1)` — the [1,1] entry of the inverted local design matrix — hardcoding derivative order zero; there is no `idrv` parameter. By contrast, `locpoly.f` selects the requested derivative coefficient via `cvest(k) = Tvec(idrv+1)`.

2. **R source**: R's `sdiag` and `sstdiag` in `KernSmooth/R/all.R` carry `drv = 0L` and `kernel = "normal"` in their signatures but never reference either in their bodies. The Python port faithfully reproduces this design; both parameters exist purely for API symmetry with `locpoly`.

**Error handling discrepancies found and corrected**

A systematic comparison of every `stop()`, `warning()`, and `match.arg()` call in `KernSmooth/R/all.R` against the Python module revealed three discrepancies:

1. **`_resolve_choice` error message**: Python generated two distinct custom messages — one for ambiguous partial matches and one for non-matching values — neither matching R's `match.arg` format. R's `match.arg` raises the same message for both cases: `'arg' should be one of "c1", "c2", ...`. Python's `_resolve_choice` was rewritten to produce this single unified message for both failure paths. The now-unused `param_name` parameter was removed from the function signature and all four call sites.

2. **Invented `ValueError` in `locpoly`, `sdiag`, `sstdiag`**: The Session 2 rewrite added `if bandwidth is None: raise ValueError("'bandwidth' must be specified")` to all three functions. No corresponding `stop()` call exists in the R source. In R, a missing required argument causes a language-level error (`argument "bandwidth" is missing, with no default`), not a user-defined message. This invented guard was removed from all three functions to eliminate error behavior that has no R counterpart.

3. **`locpoly` positivity check pattern**: Session 2 left `locpoly` with `if np.any(np.asarray(bandwidth) <= 0):` as an unconditional positivity guard. R's equivalent is `if (!missing(bandwidth) && any(bandwidth <= 0)) stop(...)`. The correct Python translation mirrors the `!missing()` sentinel idiom documented in the `missing.md` conversion guide: `if bandwidth is not None and np.any(np.asarray(bandwidth) <= 0):`. The check was corrected accordingly.

The following error and warning behaviors were confirmed as correctly matching R's explicit `stop()`/`warning()` calls: `bkde` (bandwidth positivity, canonical check, binning-grid warning), `bkde2D` (bandwidth positivity, binning-grid warning), `bkfe` (bandwidth positivity, binning-grid warning), `dpih` (level range, scale-estimate zero), `dpik` (level range, scale-estimate zero), `locpoly` (bandwidth positivity, scalar/length error, binning-grid error), `sdiag` and `sstdiag` (scalar/length error).

---

### 4. Conclusion

All Python error behaviors in `r2py_kernsmooth/r2py_kernsmooth/__init__.py` now correspond exactly to the explicit `stop()` and `warning()` calls in `KernSmooth/R/all.R`. No invented error messages are present, and no R-defined messages are missing. The `!missing(bandwidth) && condition` pattern in R is consistently translated as `bandwidth is not None and condition` across all functions (`bkfe`, `bkde`, `bkde2D`, `locpoly`) that test bandwidth before use. The next priority remains comprehensive test coverage against R's reference output.
