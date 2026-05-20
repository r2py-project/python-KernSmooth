# Research Report — KernSmooth Static Analysis (Phase 1)

**Date:** 2026-05-12  
**Working directory:** `python-KernSmooth/`  
**Package under analysis:** KernSmooth 2.23-26

---

### 1. Abstract

This session conducted a complete static structural analysis of the R package KernSmooth (v2.23-26), a recommended R package implementing kernel smoothing methods from Wand & Jones (1995). Two primary artefacts were produced: a per-function dependency classification stored in `structural_analysis/R/all.json`, and a comprehensive architecture document stored in `docs/architecture-analysis.md`. The package's internal call graph, Fortran integration layer, and public API surface are now fully documented.

---

### 2. Methodology & Actions Taken

**Step 1 — R dependency analysis (`structural_analysis/R/all.json`)**

The `/analyze-r-folder-dependencies` skill was invoked against `KernSmooth/R/`. A recursive scan identified one R source file: `KernSmooth/R/all.R`. The `analyze-r-file-dependencies` sub-agent was dispatched against this file and returned a structured JSON classifying every function's calls into three categories: `language_dependencies` (R builtins and `stats` imports), `internal_dependencies` (calls to other functions defined within `all.R`), and `external_dependencies` (`.Fortran()` calls to compiled Fortran routines). The result was written to `structural_analysis/R/all.json`.

**Step 2 — Architecture synthesis (`docs/architecture-analysis.md`)**

The following source files were read in parallel to provide input for synthesis:

| File | Purpose |
|---|---|
| `KernSmooth/DESCRIPTION` | Package metadata, authorship, license, dependencies |
| `KernSmooth/NAMESPACE` | Exported symbols, Fortran library registration, stats imports |
| `structural_analysis/R/all.json` | Per-function dependency classification (produced in Step 1) |
| `structural_analysis/dependency_levels.csv` | Function call-depth hierarchy (pre-existing) |
| `KernSmooth/man/bkde.Rd` | Density estimation API and algorithm description |
| `KernSmooth/man/bkde2D.Rd` | 2D density estimation API |
| `KernSmooth/man/bkfe.Rd` | Kernel functional estimator API |
| `KernSmooth/man/dpih.Rd` | Histogram bin-width selector API |
| `KernSmooth/man/dpik.Rd` | Kernel density bandwidth selector API |
| `KernSmooth/man/dpill.Rd` | Local linear regression bandwidth selector API |
| `KernSmooth/man/locpoly.Rd` | Local polynomial estimator API |

A check for `KernSmooth/vignettes/` confirmed the directory does not exist; there are no vignette sources. The synthesized document was written to `docs/architecture-analysis.md`.

---

### 3. Key Findings & Results

**Package scope and runtime dependencies**

KernSmooth has zero runtime dependencies on contributed CRAN packages. It imports only five functions from the base `stats` package (`dbeta`, `dnorm`, `fft`, `quantile`, `var`). `MASS` and `carData` appear in `Suggests` but are referenced only in `man/` examples.

**API surface**

Seven functions are exported: `bkde`, `bkde2D`, `bkfe`, `dpih`, `dpik`, `dpill`, `locpoly`. No S3 methods are registered. The public API is flat and entirely procedural.

**Internal function inventory**

`R/all.R` defines 16 functions in a single file. Nine are internal (unexported): `linbin`, `linbin2D`, `rlbin`, `blkest`, `cpblock`, `sdiag`, `sstdiag`, `.onAttach`, `.onUnload`.

**Three-level dependency hierarchy**

| Level | Functions |
|---|---|
| 0 (entry points) | `bkde`, `bkde2D`, `dpih`, `dpik`, `dpill`, `.onAttach`, `.onUnload` |
| 1 (mid-tier helpers) | `bkfe`, `blkest`, `cpblock`, `linbin2D`, `locpoly`, `sdiag`, `sstdiag` |
| 2 (binning primitives) | `linbin`, `rlbin` |

**Most critical internal function**

`linbin` (1D linear binning, backed by Fortran `F_linbin`) is called by 7 distinct functions: `bkde`, `bkfe`, `dpih`, `dpik`, `locpoly`, `sdiag`, and `sstdiag`. It is the single computational bottleneck through which all density and functional estimation paths pass.

**Fortran integration**

Eight internal R wrapper functions delegate their numerical loops to Fortran via `.Fortran()`. The `NAMESPACE` directive `useDynLib(KernSmooth, .registration = TRUE, .fixes = "F_")` registers all entry points at load time with a `F_` prefix. The LINPACK-derived linear algebra routines (attributed to Cleve Moler) reside in `src/d*` and underpin `locpoly`, `sdiag`, and `sstdiag`.

**Exported function `bkfe` is also an internal dependency**

`bkfe` is unique in that it is both exported and called internally by `dpih` and `dpik`. It computes density functionals of the form ∫ f(x) f^(r)(x) dx, which are the inputs required by the plug-in bandwidth selectors.

**`dpill` has the most complex internal fan-out**

`dpill` orchestrates six internal helpers (`rlbin`, `cpblock`, `blkest`, `locpoly`, `sdiag`, `sstdiag`) to produce a single bandwidth estimate, making it the most internally coupled exported function.

---

### 4. Conclusion & Next Steps

Phase 1 static analysis of the KernSmooth R layer is complete. Both output artefacts (`structural_analysis/R/all.json` and `docs/architecture-analysis.md`) are in their final state. The Fortran source layer (`src/`) has not yet been analysed; a natural Phase 2 task would be applying the `analyze-c-folder-dependencies` (or equivalent Fortran) analysis to the compiled source files in `KernSmooth/src/` to complete the full cross-language dependency map. Additionally, the Python re-implementation under `r2py_kernsmooth/` has not yet been cross-referenced against the R architecture established here.
