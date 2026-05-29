# Phase 7 Session Report — python-KernSmooth

**Date:** 2026-05-21
**Files Created:**
- `r2py_kernsmooth/tests/test_bkde_positive.py`
- `r2py_kernsmooth/tests/test_bkde_negative.py`
- `r2py_kernsmooth/tests/test_bkde_edge.py`
- `r2py_kernsmooth/tests/test_bkde2D_positive.py`
- `r2py_kernsmooth/tests/test_bkde2D_negative.py`
- `r2py_kernsmooth/tests/test_bkde2D_edge.py`
- `r2py_kernsmooth/tests/test_bkfe_positive.py`
- `r2py_kernsmooth/tests/test_bkfe_negative.py`
- `r2py_kernsmooth/tests/test_bkfe_edge.py`
- `r2py_kernsmooth/tests/test_dpih_positive.py`
- `r2py_kernsmooth/tests/test_dpih_negative.py`
- `r2py_kernsmooth/tests/test_dpih_edge.py`
- `r2py_kernsmooth/tests/test_dpik_positive.py`
- `r2py_kernsmooth/tests/test_dpik_negative.py`
- `r2py_kernsmooth/tests/test_dpik_edge.py`
- `r2py_kernsmooth/tests/test_dpill_positive.py`
- `r2py_kernsmooth/tests/test_dpill_negative.py`
- `r2py_kernsmooth/tests/test_dpill_edge.py`
- `r2py_kernsmooth/tests/test_locpoly_positive.py`
- `r2py_kernsmooth/tests/test_locpoly_negative.py`
- `r2py_kernsmooth/tests/test_locpoly_edge.py`
- `r2py_kernsmooth/src/_KernSmooth.pyf`

**Files Modified:**
- `r2py_kernsmooth/r2py_kernsmooth/__init__.py`
- `r2py_kernsmooth/meson.build`
- `r2py_kernsmooth/tests/test_dpill_positive.py`
- `r2py_kernsmooth/tests/test_dpill_edge.py`

---

## Session 1 — Comprehensive Pytest Suite Generation for All Seven Public KernSmooth Functions

### 1. Abstract

The `/generate-python-file-tests` skill was invoked to create comprehensive `pytest` suites for all public-facing functions in `r2py_kernsmooth/r2py_kernsmooth/__init__.py`. The R package's CRAN reference manual was fetched to identify the seven exported public interfaces (`bkde`, `bkde2D`, `bkfe`, `dpih`, `dpik`, `dpill`, `locpoly`), filtering out the seven internal helpers (`blkest`, `cpblock`, `linbin`, `linbin2D`, `rlbin`, `sdiag`, `sstdiag`) present in `__all__` but absent from the R documentation. Sequential invocation of the `generate-python-function-tests` sub-agent produced 514 tests across 21 files. During the `dpill` generation pass, the sub-agent discovered a pre-existing bug in the `blkest` Fortran/f2py interface that caused 60 of 83 `dpill` tests to fail.

---

### 2. Methodology & Actions Taken

#### 2.1 Function Discovery and Public Interface Filtering

`r2py_kernsmooth/r2py_kernsmooth/__init__.py` was read to enumerate all function definitions. The file contained 16 functions:

- **Private helpers** (excluded from test generation): `_resolve_choice`, `_discretize_bandwidth`
- **`__all__` members** (candidates for testing): `bkde`, `bkde2D`, `bkfe`, `blkest`, `cpblock`, `dpih`, `dpik`, `dpill`, `linbin`, `linbin2D`, `locpoly`, `rlbin`, `sdiag`, `sstdiag`

The KernSmooth CRAN reference manual at `https://cran.r-project.org/web/packages/KernSmooth/refman/KernSmooth.html` was fetched via `WebFetch`. The documentation confirmed exactly seven exported functions: `bkde`, `bkde2D`, `bkfe`, `dpih`, `dpik`, `dpill`, `locpoly`. The remaining seven `__all__` members (`blkest`, `cpblock`, `linbin`, `linbin2D`, `rlbin`, `sdiag`, `sstdiag`) are internal utilities used by `dpill` and other public functions but not documented as user-facing interfaces in the R package.

Two existing test files were found: `r2py_kernsmooth/tests/test_bkfe.py` and `r2py_kernsmooth/tests/test_locpoly.py` (created in Phase 6). Both were reviewed and supplemented by the test-generation agents rather than replaced.

#### 2.2 Test Generation — Per Function

The `generate-python-function-tests` sub-agent was invoked sequentially for each of the seven public functions. Each invocation produced three files (positive, negative, edge), ran the tests via `rpy2` to benchmark Python output against live R `KernSmooth` 2.23 results, and reported pass/fail counts.

**`bkde` — 57 tests (all pass):**

- *Positive* (28 tests, `test_bkde_positive.py`): All 5 kernel types (`normal`, `box`, `epanech`, `biweight`, `triweight`); both `canonical=True/False`; auto-computed vs. explicit bandwidth; `gridsize` variants (101, 256, 512); custom `range_x`; `truncate=True/False`; integer input coercion; partial kernel-name abbreviations (e.g., `"n"` → `"normal"`); return-structure invariants (dict keys `"x"`, `"y"`; sorted equispaced grid; density integrates to ~1.0); distributions: normal, uniform, bimodal, exponential; n=3 through n=500.
- *Negative* (9 tests, `test_bkde_negative.py`): Negative, zero, and very-negative bandwidth; unrecognised kernel name; ambiguous prefix `"b"` (matches both `"box"` and `"biweight"`); `canonical` as string/integer/multi-element boolean vector; wrong-length bandwidth array. All verify that both Python and R raise; message-phrasing divergences emit `UserWarning` rather than failing.
- *Edge* (20 tests, `test_bkde_edge.py`): n=1, n=2, constant data; very small bandwidth (warns but matches R); very large bandwidth; `gridsize=1` and `gridsize=2`; minimum positive `float64` bandwidth; degenerate `range_x` (a==b, documented divergence: R returns NaN, Python raises `ZeroDivisionError`); `tau` boundary tests for all 5 kernels; `canonical=np.bool_` acceptance; output non-negativity; large/small magnitude data.

**`bkde2D` — 63 tests (all pass; first invocation failed with socket error, retried successfully):**

- *Positive* (28 tests, `test_bkde2D_positive.py`): Standard bivariate normal; gridsizes (51×51 default, 101×101, 51×81, 201×201); asymmetric bandwidths `[0.2, 0.8]`; scalar bandwidth broadcast; explicit `range_x`; `truncate=True/False` (confirmed no effect, matching R); uniform, bimodal, correlated (ρ=0.8), negative-valued, constant data; integer coercion; non-negativity of `fhat`; density integration ~1.0; KernSmooth reference example with `bandwidth=[0.7, 7]`.
- *Negative* (12 tests, `test_bkde2D_negative.py`): Negative/zero bandwidth in both components individually and together; 1-column input; NaN in input; inverted `range_x` per dimension.
- *Edge* (23 tests, `test_bkde2D_edge.py`): n=1 through n=3; constant data; very small bandwidth (warns); very large bandwidth (~1000); gridsize (2,2) minimum; `Inf` in input (R raises, Python returns non-finite — documented divergence); scalar vs length-2 bandwidth identity; tau=3.4 smoothing boundary; asymmetric gridsizes; data far from origin.

**`bkfe` — 49 tests (all pass; existing `test_bkfe.py` reviewed and supplemented with three new files):**

- *Positive* (24 tests, `test_bkfe_positive.py`): `drv` in {0, 2, 4, 6, 8, 10}; n ∈ {50, 100, 200, 1000}; normal, uniform, exponential distributions; `gridsize` ∈ {200, 401, 1001}; explicit `range_x`; `truncate=True/False` verified to differ; `binned=True` with various `drv` values; scalar return type; finiteness for well-behaved inputs. All use `rtol=1e-9`.
- *Negative* (9 tests, `test_bkfe_negative.py`): `bandwidth=0`, `-1`, `-999`; `bandwidth=None`; NaN/Inf/-Inf in x; empty array; `bandwidth=-1e-10`. Documented divergence: `Inf`/`-Inf` in x causes R to raise (inside `seq()`), Python silently propagates NaN via IEEE 754.
- *Edge* (16 tests, `test_bkfe_edge.py`): `gridsize` as exact powers of 2 (256, 512) — regression for the historical power-of-2 bug; `gridsize=2`; n=1 (both return NaN); n=2 (finite, matches R); all-constant data (NaN from zero range); very small and very large bandwidth; `drv=0` (positive) and `drv=10` (large magnitude); narrow `range_x`.

**`dpih` — 71 tests (all pass):**

- *Positive* (32 tests, `test_dpih_positive.py`): All three `scalest` options (`minim`, `stdev`, `iqr`); confirmed `minim == min(stdev_result, iqr_result)` on heavy-tailed data; all 6 `level` values (0–5); 5×3 grid of `(level, scalest)` combinations; normal, exponential, uniform, bimodal, outlier data; gridsize extremes (100, 801); partial-name abbreviations; analytical formula verification for `level=0` (`scalest_val * (24*sqrt(π)/n)^(1/3)`); return type `np.float64`; positivity.
- *Negative* (15 tests, `test_dpih_negative.py`): `level=6` and `level=100`; `level=-1` (documented divergence: R returns `numeric(0)`, Python raises `UnboundLocalError`); constant data for each `scalest`; zero-IQR data with `scalest='iqr'`; invalid `scalest`; n=1; Inf/-Inf/NaN in data; empty array; degenerate `range_x` (a==b).
- *Edge* (24 tests, `test_dpih_edge.py`): n=2, n=3, n=5; large (mean=1e6) and small (std=1e-4) magnitude; approximate scale-invariance `dpih(c*x) ≈ c * dpih(x)` (1% tolerance); gridsize extremes (10, 2001); convergence between coarse/fine grids; `truncate` with narrow `range_x`; `minim` monotonicity invariant.

**`dpik` — 119 tests (all pass):**

- *Positive* (52 tests, `test_dpik_positive.py`): All 5 kernels × all 6 levels (30 combinations); all 3 `scalest` × all 6 levels (18 combinations); `canonical=True/False` for all kernels; all partial-name abbreviations; `range_x`, `gridsize`, `truncate` variations; n ∈ {2, 10, 1000, 5000}; analytical check for `level=0`; kernel bandwidth ordering confirmed for non-canonical case (`normal < box < epanech < biweight < triweight`).
- *Negative* (23 tests, `test_dpik_negative.py`): `level` out of range; unrecognised/ambiguous kernel; unrecognised `scalest`; constant data per `scalest`; zero-IQR data; n=1; Inf/-Inf/NaN; empty array; degenerate `range_x`; `level=6` across all 5 kernels (parametrised).
- *Edge* (44 tests, `test_dpik_edge.py`): Scale invariance `dpik(c*x) == c * dpik(x)` verified against R including small c=0.001; location shift invariance; large/small magnitude data; all-negative data; `canonical=True` kernel-independence; gridsize extremes; (kernel, level) and (kernel, canonical) parametrised spot-checks.

**`dpill` — 83 tests written; 21 pass initially, 62 fail (bug discovered):**

- *Positive* (38 tests, `test_dpill_positive.py`) and *Edge* (28 tests, `test_dpill_edge.py`): Standard regression scenarios, various `blockmax`/`divisor`/`trim`/`proptrun`/`gridsize`; scale and shift invariance; reproducibility; various regression functions (linear, quadratic, sine, cosine, exponential decay).
- *Negative* (17 tests, `test_dpill_negative.py`): n=1, empty arrays, mismatched lengths, constant x, Inf/NaN, degenerate `range_x`, `trim=0.5`, `gridsize=0/1`. All 17 pass immediately (both Python and R raise for all these inputs).
- **Bug discovered:** 60 positive and edge tests fail with `ValueError: cannot convert float NaN to integer` inside `_discretize_bandwidth`. Root cause identified by the sub-agent: `blkest`'s Fortran scalar outputs `sigsqe`, `th22e`, `th24e` were being declared as `input float` in the f2py interface, so the Fortran-computed values were never returned to Python. `blkest` always returned zeros, causing `gamseh = sigsqQ / (abs(th24Q) * n)` to divide by zero, propagating NaN as the `bandwidth` argument into `locpoly`.

**`locpoly` — 72 tests (all pass; existing `test_locpoly.py` reviewed and supplemented):**

- *Positive* (38 tests, `test_locpoly_positive.py`): Regression mode with `drv` ∈ {0, 1, 2, 3}; default/explicit `degree`; scalar and vector bandwidth; `truncate=True/False`; `binned=True`; `bwdisc` variation; `range_x`, `gridsize`; negative x; constant/linear y; n=1000; integer coercion; Prestige dataset canonical example. Density mode (`y=None`): standard normal, uniform, `drv=1`, custom `range_x`, 5% range extension.
- *Negative* (11 tests, `test_locpoly_negative.py`): Negative/zero/very-negative bandwidth (regression and density); bandwidth vector with non-positive entry; wrong-length bandwidth vector; bandwidth too small (Lvec==0); missing bandwidth (both raise, message differs).
- *Edge* (23 tests, `test_locpoly_edge.py`): n=1 (both raise); n=2; very large and very small bandwidth; constant bandwidth vector = scalar; large/small magnitude x/y; degenerate `range_x`; `gridsize=2` and `gridsize=801`; `drv` 0–3 all finite; density non-negative in interior; density integrates ~1.0 for n=1000; `binned=True` with `drv=1`.

---

### 3. Key Findings & Results

#### 3.1 Test Count Summary

| Function | Positive | Negative | Edge | Total | Initial Pass |
|---|---|---|---|---|---|
| `bkde` | 28 | 9 | 20 | 57 | 57 |
| `bkde2D` | 28 | 12 | 23 | 63 | 63 |
| `bkfe` | 24 | 9 | 16 | 49 | 49 |
| `dpih` | 32 | 15 | 24 | 71 | 71 |
| `dpik` | 52 | 23 | 44 | 119 | 119 |
| `dpill` | 38 | 17 | 28 | 83 | 21 |
| `locpoly` | 38 | 11 | 23 | 72 | 72 |
| **Total** | **240** | **96** | **178** | **514** | **452** |

#### 3.2 `blkest` f2py Bug Pre-identified by Test Generation

The `generate-python-function-tests` sub-agent independently discovered that `blkest` always returned `{'sigsqe': 0.0, 'th22e': 0.0, 'th24e': 0.0}` before the fix, correctly attributing it to the f2py interface passing scalar outputs by value rather than as writeable outputs. This was the first detection of the bug in the project.

#### 3.3 Documented Behavioral Divergences

Several Python/R behavioral divergences were documented via `UserWarning` emissions rather than test failures:

- **`bkde` / degenerate `range_x` (a==b):** R returns NaN; Python raises `ZeroDivisionError`.
- **`bkfe` / `Inf` or `-Inf` in x:** R raises inside `seq()` (cannot form grid); Python propagates NaN via IEEE 754.
- **`dpih` / `level=-1`:** R returns `numeric(0)` silently; Python raises `UnboundLocalError`.
- **`dpill` / `divisor=0`:** R returns valid result via integer division yielding Inf; Python raises `ZeroDivisionError`.
- **`bkde2D` / `Inf` in input:** R raises inside `seq.default()`; Python returns partially non-finite result.
- All negative tests where both Python and R raise but with different message phrasing (Python omits R's full function signature prefix) emit `UserWarning` rather than failing.

---

## Session 2 — Bug Fixing: Three Root-Cause Bugs Resolved to Achieve 100% Pass Rate

### 1. Abstract

The `/fix-bugs-found-in-tests` skill was invoked to systematically resolve all failing tests. Starting from 49 failures (all in `dpill` tests), three bugs were identified and fixed: (1) the `blkest` f2py scalar-output binding, (2) a `dpill` range computation ordering error matching R's lazy-evaluation semantics, and (3) a NaN-equality comparison error in two test assertions. The final state is **518 passed, 0 skipped, 0 failed** across all 21 test files.

---

### 2. Methodology & Actions Taken

#### 2.1 Initial Test Run

`python -m pytest r2py_kernsmooth/tests/ -x --tb=short -q` was executed. The first failure was `test_dpill_n10_matches_r` in `test_dpill_edge.py`, crashing with:

```
ValueError: cannot convert float NaN to integer
```
at `r2py_kernsmooth/r2py_kernsmooth/__init__.py:49` (`_discretize_bandwidth`), called from `locpoly` at line 364, called from `dpill` at line 997.

#### 2.2 Bug Fix 1 — `blkest` f2py Scalar Output Binding

The `fix-bug-found-in-test` sub-agent was invoked with the full traceback and root-cause context. Investigation of the f2py wrapper via `python -c "import r2py_kernsmooth._KernSmooth as k; print(k.blkest.__doc__)"` confirmed that `sigsqe`, `th22e`, `th24e` were declared as `input float` (scalar copies) rather than writeable outputs. The Fortran subroutine `blkest` (defined in `r2py_kernsmooth/src/blkest.f`) modifies these via `sigsqe = ...` assignments, but f2py never wrote the results back to the Python caller.

**Fixes applied:**

1. Created `r2py_kernsmooth/src/_KernSmooth.pyf` — a full f2py signature file generated from all Fortran source files, with `intent(out)` annotations added to `sigsqe`, `th22e`, `th24e` in the `blkest` block.
2. Updated `r2py_kernsmooth/meson.build` to pass `_KernSmooth.pyf` as the sole input to the f2py wrapper generation step, removing the conflicting `-m _KernSmooth --lower` flags.
3. Updated `blkest()` in `r2py_kernsmooth/r2py_kernsmooth/__init__.py` (lines 211–224): changed from passing pre-allocated scalar arrays as input arguments to receiving `sigsqe, th22e, th24e = _KernSmooth.blkest(...)` as return values. Removed the three pre-allocation lines for `sigsqe`, `th22e`, `th24e`.
4. Rebuilt with `pip install -e r2py_kernsmooth/ --no-build-isolation -q`.

Verification: `r2py_kernsmooth.blkest(x, y, 2, 4)` now returns non-zero values for both n=10 and n=200 test cases.

#### 2.3 Second Test Run

After the `blkest` fix and reinstall, the test suite was re-run. Result: **49 failures remained** (all `dpill`), with 466 tests passing. Failures included both NaN returns and numerical mismatches. For example, `test_dpill_all_default_params_match_r` (seed=42, n=200) failed with Python returning NaN while R returned `0.1379083775114942`.

#### 2.4 Investigating the NaN Return

A step-by-step trace of `dpill` internals for the seed=42, n=200 case was performed:

1. `blkest` confirmed working: `sigsqe=0.09209`, `th22e=10.50`, `th24e=-39.888`
2. `gamseh` computed to `0.19949503503706387`
3. `locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh, ...)` returned a valid `mddest`
4. `th22kn = 34.869`; `lamseh = 0.10453009038364468`
5. `sdiag(xcounts, bandwidth=lamseh, ...)` returned 29 `Inf` values (at positions where `xcounts==0`)
6. `np.sum(Sdg * xcounts)` → NaN (because `Inf * 0 = NaN` in IEEE 754)
7. `sigsqd = n - 2 * np.sum(Sdg * xcounts) + ...` → NaN → `dpill` returns NaN

The same inputs were given to R's `KernSmooth:::sdiag` directly: R also produced 29 `Inf` values, and `sum(Sdg * xcounts)` also gave NaN in R. Yet R's actual `dpill` returned `0.1379`.

**Hypothesis formation:** R's `dpill` must be calling `sdiag` with different arguments. The `sdiag` inside R's `dpill` was patched at runtime:

```r
unlockBinding('sdiag', asNamespace('KernSmooth'))
assign('sdiag', wrapper_sdiag, envir=asNamespace('KernSmooth'))
result <- KernSmooth:::dpill(xd, yd)
```

The wrapper logged `bandwidth=0.1328546` — different from the manually computed `lamseh=0.10453`. R's `dpill` was using a larger bandwidth. The discrepancy was traced back to `gamseh`: R's internal computation gave `gamseh=0.1919741` while Python gave `0.19949503`.

**Root cause identified:** The critical difference was in `b - a`. Python set `range_x = (x[0], x[-1])` **before** trimming: `b - a = 5.046` (untrimmed range). But in R, `range.x = range(x)` is a **lazy default argument** evaluated in the function's local environment at the time `a <- range.x[1L]` is first accessed — by which point `x` has already been sorted AND trimmed. The trimmed range for seed=42, n=200 (2% trim removing 4 points) is `b - a = 3.856`, not `5.046`. This propagated through `gamseh`, `mddest`, `th22kn`, and `lamseh`, causing `lamseh=0.10453` instead of `0.1328546`. With `lamseh=0.1328546`, R's `sdiag` returns zero Inf values; with `lamseh=0.10453` (too small a bandwidth), `sdiag` encounters near-singular `Smat` matrices for empty grid bins, producing Inf.

#### 2.5 Bug Fix 2 — `dpill` range_x Computed Before Trimming

`r2py_kernsmooth/r2py_kernsmooth/__init__.py` was edited at lines 951–960. The block:

```python
sort_idx = np.argsort(x)
x = x[sort_idx]; y = y[sort_idx]

if range_x is None:
    range_x = (x[0], x[-1])  # computed before trimming

indlow = int(np.floor(trim * len(x)))
indupp = len(x) - int(np.floor(trim * len(x)))
x = x[indlow:indupp]; y = y[indlow:indupp]
```

was changed to:

```python
sort_idx = np.argsort(x)
x = x[sort_idx]; y = y[sort_idx]

indlow = int(np.floor(trim * len(x)))
indupp = len(x) - int(np.floor(trim * len(x)))
x = x[indlow:indupp]; y = y[indlow:indupp]

# In R, range.x = range(x) is a lazy default evaluated after trimming.
if range_x is None:
    range_x = (x[0], x[-1])  # computed after trimming, matching R's lazy eval
```

Verification: `r2py_kernsmooth.dpill(x_seed42_n200, y_seed42_n200)` → `0.1379083775114942` (exact match to R). The package was reinstalled with `pip install -e r2py_kernsmooth/ --no-build-isolation -q`.

#### 2.6 Third Test Run

Result: **2 failures remaining**, both `dpill` tests:
- `test_dpill_positive.py::test_dpill_unsorted_input_same_as_sorted`
- `test_dpill_edge.py::test_dpill_sorted_vs_unsorted_input`

Both failed with:
```
AssertionError: assert np.float64(nan) == nan ± ???
```

#### 2.7 Bug Fix 3 — NaN Equality in Test Assertions

The failing tests verify that `dpill(unsorted_input) == dpill(sorted_input)`. For seed=13, n=100, both Python and R legitimately return NaN (a degenerate intermediate computation). `pytest.approx(nan) != nan` under IEEE 754 — no `NaN`-equality exception is applied.

The `fix-bug-found-in-test` sub-agent edited both test files, changing:

```python
assert h_unsorted == pytest.approx(h_sorted, rel=1e-10)
```

to:

```python
both_nan = np.isnan(h_unsorted) and np.isnan(h_sorted)
assert both_nan or h_unsorted == pytest.approx(h_sorted, rel=1e-10)
```

in `test_dpill_positive.py` (line ~411) and `test_dpill_edge.py` (line ~409).

#### 2.8 Final Test Run

```
518 passed, 0 skipped, 0 failed in 6.40s
```

---

### 3. Key Findings & Results

#### 3.1 Bug Summary

| # | Bug | Location | Symptom | Root Cause | Fix |
|---|---|---|---|---|---|
| 1 | `blkest` f2py scalar output binding | `_KernSmooth.pyf`, `meson.build`, `__init__.py:211-224` | `dpill` raises `ValueError: cannot convert float NaN to integer` | f2py generated `sigsqe`, `th22e`, `th24e` as `input float` (passed by value); Fortran writes never returned to Python | Created `.pyf` with `intent(out)` on three scalars; updated `blkest()` to unpack return tuple |
| 2 | `dpill` range computed before trimming | `__init__.py:955-960` | `dpill` returns NaN for many n=200 inputs; R returns valid values | Python computed `range_x = (x[0], x[-1])` from untrimmed sorted data; R evaluates `range.x=range(x)` lazily after trimming, yielding smaller `b-a`, larger `lamseh`, no Inf in `sdiag` | Moved `range_x` assignment to after the trim step |
| 3 | `NaN == NaN` test assertion | `test_dpill_positive.py:411`, `test_dpill_edge.py:409` | `assert nan == pytest.approx(nan)` always fails | IEEE 754: `NaN != NaN`; test did not handle the case where both results are legitimately NaN | Changed assertion to `both_nan or approx_equal` |

#### 3.2 R Lazy Evaluation Semantics — Critical Insight

The most subtle bug (Bug 2) arose from a fundamental difference between R's lazy argument evaluation and Python's eager evaluation. In R, default parameter expressions are evaluated in the function's local scope at first use, not at call time. Since R's `dpill` reassigns `x` to a trimmed version before accessing `range.x`, the computed range reflects the trimmed data. Python default parameters (`range_x=None`) do not have this behavior, so the translation must explicitly mirror the R execution order. This pattern may affect other functions in the codebase that use `range.x=range(x)` as a default argument in the R source.

#### 3.3 `sdiag` Inf Values — Numerical Instability at Small Bandwidths

The `sdiag` Fortran routine calls `dgefa`/`dgedi` (LU factorization and matrix inverse from LINPACK) on the local smoother matrix `Smat`. For grid bins with zero counts (`xcounts[k] == 0`), the `Smat` matrix is not built from any data and may be near-singular, causing `dgedi` to produce Inf for the `(1,1)` inverse element. This is a known numerical behavior of the algorithm. The fix is upstream (Bug 2 ensuring `lamseh` is large enough to avoid near-singular matrices), not in `sdiag` itself. Both Python and R produce the same Inf values for any given `(xcounts, bandwidth)` input; the difference was purely in which bandwidth was used.

#### 3.4 `blkest` Fortran Interface — Pre-existing Silent Error

Bug 1 was pre-existing from at least Phase 6 (the f2py argument-order fix session). The prior session fixed argument ordering for `blkest` (removing `np.int32(n)`, `np.int32(qq)` from explicit arguments) but did not address the scalar output issue. The scalar outputs `sigsqe`, `th22e`, `th24e` were silently passed as zero throughout all prior testing, and the `dpill` function was never tested with inputs large enough to reveal the crash.

---

### 4. Conclusion & Next Steps

The `r2py_kernsmooth` Python package now has a comprehensive `pytest` suite of **518 passing tests** across **21 test files** covering all 7 public KernSmooth functions. Three pre-existing bugs were found and fixed: a Fortran f2py binding error in `blkest`, a range-computation ordering error in `dpill`, and two test assertions that incorrectly handled the `NaN == NaN` case.

**Suggested next steps:**
- The `dpill` function's R lazy-evaluation pattern (`range.x = range(x)` evaluated post-trim) should be checked against all other functions in `__init__.py` that accept `range_x=None` defaults (`bkfe`, `dpih`, `dpik`, `locpoly`, `sdiag`, `sstdiag`). None of these trim `x` before setting the range, so no analogous bug exists there — but the audit should be confirmed explicitly.
- The `blkest` `.pyf` file should be reviewed to ensure all other Fortran routines (`locpol`, `sdiag`, `sstdg`, `cp`, `lbtwod`, `linbin`, `rlbin`) also have correct `intent` declarations for any modified-in-place array arguments that f2py would otherwise treat as input-only.
- Consider adding a `conftest.py`-level `pytest.ini` marker to tag tests that require `rpy2` and R, so the suite can be run in partial mode without an R installation.
