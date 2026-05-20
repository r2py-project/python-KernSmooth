# Phase 3 Research Report: R Language Dependency Extraction and R-to-Python Conversion Guides

**Date:** 2026-05-13
**Project:** python-KernSmooth
**Working Directory:** `/groups/jli9/Yufei/python-KernSmooth`

---

### 1. Abstract

This session focused on systematically cataloguing every R standard-library (language) dependency used in `KernSmooth/R/all.R` and producing a structured R-to-Python conversion guide for each one. Starting from a pre-computed structural analysis JSON (`structural_analysis/R/all.json`), the session extracted 370 language-dependency call sites across 46 unique functions, organised them into a sorted combined CSV table, and generated 46 individual Markdown conversion guides covering the complete translation surface from R to Python for this codebase.

---

### 2. Methodology & Actions Taken

**Step 1 — Language dependency extraction (`/extract-r-folder-language-dependencies`)**

The `extract-r-file-language-dependencies` subagent was invoked on the single R source file `KernSmooth/R/all.R`, paired with its pre-calculated structural analysis at `structural_analysis/R/all.json`. The agent resolved each language dependency call site (function name, enclosing R function, line number, full call body) and returned CSV-formatted output. The result was saved to `language_dependency_analysis/R/all.csv` — a 370-row, 4-column CSV (`language_dependency`, `function_name`, `line_number`, `call_body`) covering 16 R functions: `bkde`, `bkde2D`, `bkfe`, `blkest`, `cpblock`, `dpih`, `dpik`, `dpill`, `linbin`, `linbin2D`, `locpoly`, `rlbin`, `sdiag`, `sstdiag`, `.onAttach`, `.onUnload`.

**Step 2 — CSV combination script (`language_dependency_analysis/combine_csvs.py`)**

A Python script was written (not executed) that:
- Recursively globs all `.csv` files under `language_dependency_analysis/R/` using `glob.glob(..., recursive=True)`.
- Reads each CSV with `pandas.read_csv`, inserts a `file_path` column (relative path of the source R file within the `R/` folder, `.csv` extension replaced with `.R`) between `language_dependency` and `function_name` using `df.insert`.
- Concatenates all frames, sorts by `["language_dependency", "file_path", "function_name", "line_number"]`, and saves to `language_dependency_analysis/combined_table.csv`.

The script was iteratively refined to add the `file_path` column after user feedback on the initial version.

**Step 3 — Conversion guide generation (`/generate-language-dependency-conversion-guides`)**

The combined table CSV was parsed to identify 46 unique `language_dependency` values. Per-dependency CSV subsets were extracted using `pandas`. All 46 `generate-language-dependency-conversion-guide` subagents were launched in a single parallel batch, each receiving the base folder path (`KernSmooth/R/`) and the relevant CSV subset. Each agent read the R source file to examine context before producing its guide. Upon completion, 46 Markdown files were saved to `language_dependency_analysis/conversion_guides/` with filenames matching the dependency name (e.g., `fft.md`, `missing.md`, `as.double.md`).

**Files created this session:**
| File | Description |
|---|---|
| `language_dependency_analysis/R/all.csv` | 370-row language dependency table for `all.R` |
| `language_dependency_analysis/combine_csvs.py` | Pandas script to merge, enrich, and sort dependency CSVs |
| `language_dependency_analysis/conversion_guides/*.md` | 46 R-to-Python conversion guides (one per unique dependency) |

---

### 3. Key Findings & Results

**Coverage:** All 46 unique R language dependencies in `KernSmooth/R/all.R` were assigned conversion guides. No dependency was skipped or failed.

**Critical translation nuances identified across the 46 guides:**

- **FFT normalisation mismatch (`fft.md`):** R's `fft(..., inverse=TRUE)` is unnormalised (returns the raw DFT sum). NumPy's `np.fft.ifft` / `np.fft.ifft2` normalises by 1/N by default. The R source explicitly divides by `P` or `P1*P2` post-`Re()`; these divisions must be **removed** when porting to NumPy.

- **Fortran FFI type coercions (`as.double.md`, `as.integer.md`):** R's `as.double(rep(0, n))` pre-allocation pattern maps to `np.zeros(n, dtype=np.float64)`, and `as.integer(Lvec)` on a vector maps to `Lvec.astype(np.int32)`. The 32-bit constraint (`np.int32`) is non-obvious but required because standard Fortran `INTEGER` is 32-bit.

- **Sample variance denominator (`var.md`):** R's `var(x)` uses Bessel's correction (`n-1` denominator). NumPy defaults to `n` (population variance). All occurrences of `sqrt(var(x))` must be translated as `np.std(x, ddof=1)`.

- **Optional argument detection (`missing.md`):** R's `missing(param)` has no Python built-in equivalent. The idiomatic replacement is the `None` sentinel default pattern (`def f(x, param=None)` + `if param is None:`).

- **Negative base exponentiation (`sqrt.md`):** R silently computes real-valued odd roots of negative numbers (e.g., `(-3*sqrt(2/pi) / psi6hat)^(1/7)`). Python's `**` operator raises `ValueError` for this case; explicit `math.copysign(abs(x)**(1/n), x)` is required.

- **`library.dynam.unload` (`library.dynam.unload.md`):** Has no Python equivalent. CPython does not support unloading native extension modules safely. The `.onUnload` hook should be omitted entirely in the Python port, or replaced with `atexit.register(fn)` for generic cleanup only.

- **`match.arg` (`match.arg.md`):** R's built-in partial string matching and validation has no standard Python equivalent. A reusable `_match_arg(value, choices, arg_name)` helper is required, supporting both exact and unambiguous-prefix matching.

- **R `switch` on strings (`switch.md`):** Translates cleanly to a Python dictionary lookup (`dict[key]`). Eager evaluation of all branches (acceptable given cheap scalar computations) or lazy evaluation via `dict` of `lambda` functions are both valid patterns.

**Column-major array ordering:** R's `matrix(data, nrow, ncol)` and Fortran both use column-major (Fortran-order) storage. NumPy defaults to row-major (C-order). `matrix.md` explicitly documents that `np.reshape(..., order='F')` is required when reshaping Fortran output arrays.

---

### 4. Conclusion & Next Steps

The language dependency analysis for `KernSmooth/R/all.R` is complete. All 46 R standard-library dependencies have been catalogued with precise location data and paired with actionable R-to-Python conversion guides covering type semantics, indexing differences, and library equivalences.

**Immediate next steps:**
1. Execute `language_dependency_analysis/combine_csvs.py` once source data from additional R files (if any) is available, to produce the unified `combined_table.csv`.
2. Begin implementing Python wrapper functions for the core public routines (`bkde`, `bkde2D`, `locpoly`, `dpik`, `dpill`, `bkfe`), using the conversion guides as the primary reference for each R dependency encountered.
3. Pay particular attention to: FFT normalisation (remove `/P` divisors), Fortran type contracts (`np.float64` / `np.int32`), Bessel-corrected variance (`ddof=1`), and optional-parameter sentinel patterns (`None` defaults).
4. Port `blkest`, `cpblock`, `linbin`, `linbin2D`, `rlbin` (internal helper functions calling `.Fortran()`) as thin wrappers over `r2py_kernsmooth._KernSmooth` Fortran subroutines, with `np.asarray(..., dtype=np.float64)` / `np.zeros(..., dtype=np.int32)` replacing `as.double` / `as.integer` / `double` / `integer` buffer allocations.
