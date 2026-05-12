# KernSmooth Package — Architecture Analysis

## 1. Package Metadata

| Field | Value |
|---|---|
| **Package** | KernSmooth |
| **Version** | 2.23-26 |
| **Release date** | 2024-12-10 (CRAN publication 2025-01-01) |
| **Priority** | recommended (ships as part of the base R distribution) |
| **License** | Unlimited |
| **Compilation** | Required (`NeedsCompilation: yes`) |
| **Primary author** | Matt Wand (`aut`) |
| **Contributors** | Cleve Moler (`ctb`, LINPACK Fortran routines in `src/d*`); Brian Ripley (`trl`, `cre`, `ctb`, R port and ongoing maintenance) |
| **Maintainer** | Brian Ripley <Brian.Ripley@R-project.org> |
| **Repository** | CRAN |

The package implements the statistical methods described in:
> Wand, M. P. and Jones, M. C. (1995). *Kernel Smoothing*. Chapman and Hall, London.

It is accordingly scoped to the techniques from that monograph and is not intended to be a general-purpose smoothing toolkit.

---

## 2. External Dependencies

### Hard dependencies (`Depends`)
- **R ≥ 2.5.0** — minimum runtime.
- **stats** — four functions are explicitly imported from `stats`: `dbeta`, `dnorm`, `fft`, `quantile`, `var`. All are standard statistical primitives; the package has no dependency on any non-base contributed package at runtime.

### Soft dependencies (`Suggests`)
- **MASS** — used only in documentation examples (the `geyser` dataset).
- **carData** — listed as a suggested package; not referenced in any exported function or documented example in the current source tree.

### No S3 method dispatch
The `NAMESPACE` file registers no S3 methods. The package exports a flat set of seven functions and does not participate in R's method dispatch system.

---

## 3. Exported API Surface

The `NAMESPACE` declaration exports exactly **seven** functions:

```
export(bkde, bkde2D, bkfe, dpih, dpik, dpill, locpoly)
```

These seven functions constitute the entire public interface. They divide naturally into three functional groups.

### 3.1 Density Estimation

| Function | Signature summary | Purpose |
|---|---|---|
| `bkde` | `(x, kernel, canonical, bandwidth, gridsize, range.x, truncate)` | Binned kernel density estimate for univariate data. Returns a list of `(x, y)` coordinates. Five kernel shapes are supported: normal (default), box, Epanechnikov, biweight, triweight. Linear binning followed by FFT convolution is used for speed. |
| `bkde2D` | `(x, bandwidth, gridsize, range.x, truncate)` | Binned kernel density estimate for bivariate data. The kernel is fixed at the standard bivariate normal. Returns grid vectors `x1`, `x2`, and a density matrix `fhat`. FFT-based convolution over the induced mesh is used. |

### 3.2 Bandwidth / Bin-width Selection

| Function | Signature summary | Purpose |
|---|---|---|
| `dpik` | `(x, scalest, level, kernel, canonical, gridsize, range.x, truncate)` | Direct plug-in bandwidth selector for kernel *density* estimation (Sheather & Jones 1991 / Wand & Jones 1995 §3.6). Returns a single scalar bandwidth. |
| `dpih` | `(x, scalest, level, gridsize, range.x, truncate)` | Direct plug-in *bin-width* selector for histograms (Wand 1995). Analogous to `dpik` but targets the optimal histogram bin width rather than a smoothing bandwidth. Returns a single scalar width. |
| `dpill` | `(x, y, blockmax, divisor, trim, proptrun, gridsize, range.x, truncate)` | Direct plug-in bandwidth selector for *local linear regression* (Ruppert, Sheather & Wand 1995). Uses Mallows' Cp to select the number of blocks for an initial quartic parametric fit. Returns a single scalar bandwidth. |

### 3.3 Regression / General Smoothing

| Function | Signature summary | Purpose |
|---|---|---|
| `locpoly` | `(x, y, drv, degree, kernel, bandwidth, gridsize, bwdisc, range.x, binned, truncate)` | Local polynomial estimator. When `y` is omitted, estimates the density of `x`; when `y` is supplied, estimates the regression E[Y|X]. Supports arbitrary derivative order (`drv`) and polynomial degree (`degree`). Variable bandwidths (a vector of length `gridsize`) are allowed. |
| `bkfe` | `(x, drv, bandwidth, gridsize, range.x, binned, truncate)` | Binned kernel functional estimator. Estimates the integral of the density multiplied by its `drv`-th derivative — the density functionals that appear in the plug-in bandwidth formulas used by `dpik` and `dpih`. Not a smoothing function in itself; it is a building block for bandwidth selection. |

---

## 4. Internal Architecture

### 4.1 Complete Function Inventory

The package defines **16 functions** in `R/all.R`. The 7 exported functions are described above. The remaining 9 are internal (unexported):

| Function | Role | Fortran entry point |
|---|---|---|
| `linbin` | 1D linear binning of raw data onto a grid | `F_linbin` |
| `linbin2D` | 2D linear binning onto a grid mesh | `F_lbtwod` |
| `rlbin` | Linear binning for regression data (returns weighted counts) | `F_rlbin` |
| `blkest` | Block-wise parametric estimation for `dpill` initialisation | `F_blkest` |
| `cpblock` | Mallows Cp–based block count selection for `dpill` | `F_cp` |
| `sdiag` | Diagonal of the local polynomial smoother matrix | `F_sdiag` |
| `sstdiag` | Sum-of-squares diagonal of the local polynomial smoother matrix | `F_sstdg` |
| `.onAttach` | Package startup hook (prints startup message) | — |
| `.onUnload` | Package unload hook (unloads shared library) | — |

### 4.2 Dependency Levels

Functions are assigned a level reflecting their position in the internal call graph. Level 0 functions are the top-level entry points; higher levels are called by lower-level functions.

| Level | Functions | Character |
|---|---|---|
| **0** | `bkde`, `bkde2D`, `dpih`, `dpik`, `dpill`, `.onAttach`, `.onUnload` | Entry points; called directly by the user. |
| **1** | `bkfe`, `blkest`, `cpblock`, `linbin2D`, `locpoly`, `sdiag`, `sstdiag` | Mid-tier helpers called by one or more level-0 functions. |
| **2** | `linbin`, `rlbin` | Fundamental binning primitives; called by multiple level-1 functions. `linbin` is the most widely depended-upon function in the package. |

### 4.3 Internal Dependency Graph

The directed edges below represent "A calls B" relationships (internal calls only; Fortran and language builtins are excluded).

```
bkde      ──────────────────────────────► linbin
bkde2D    ──────────────────────────────► linbin2D
bkfe      ──────────────────────────────► linbin
dpih      ──────────────► bkfe ─────────► linbin
                    └──────────────────── linbin
dpik      ──────────────► bkfe ─────────► linbin
                    └──────────────────── linbin
dpill     ──────────────► rlbin
          │               cpblock
          │               blkest
          │               locpoly ────────► linbin
          │                       └──────── rlbin
          │               sdiag ──────────► linbin
          └───────────────sstdiag ─────────► linbin
locpoly   ──────────────────────────────► linbin
                                     └───► rlbin
sdiag     ──────────────────────────────► linbin
sstdiag   ──────────────────────────────► linbin
```

**Key observations:**

- `linbin` is the single most central function. It is called by `bkde`, `bkfe`, `dpih`, `dpik`, `locpoly`, `sdiag`, and `sstdiag` — seven distinct callers. Every density or functional estimation path passes through it.
- `rlbin` is the regression-specific analogue of `linbin`, used by `dpill` (indirectly via `locpoly`) and directly by `locpoly`.
- `bkfe` is the only exported function that is also an internal dependency: `dpih` and `dpik` both call it to evaluate the density functionals needed for plug-in bandwidth selection.
- `dpill` is the most internally complex exported function, orchestrating six internal helpers to produce its bandwidth estimate.
- `bkde2D` and `linbin2D` form an isolated sub-graph: `linbin2D` is called only by `bkde2D` and nowhere else.

### 4.4 Internal Dependency Counts (Summary)

| Metric | Value |
|---|---|
| Total functions | 16 |
| Exported (public) | 7 |
| Internal (unexported) | 9 |
| Functions with Fortran calls | 8 |
| Functions calling other internal functions | 8 |
| Functions with no internal callers (true leaf-only) | `bkde`, `bkde2D`, `.onAttach`, `.onUnload` |
| Most-called internal function | `linbin` (7 callers) |

---

## 5. Fortran Integration

The package is compiled (`NeedsCompilation: yes`) and bridges R to Fortran via `.Fortran()`. The `NAMESPACE` directive:

```r
useDynLib(KernSmooth, .registration = TRUE, .fixes = "F_")
```

registers all Fortran entry points at load time and applies the `F_` prefix to their R-level names, avoiding symbol conflicts. All eight Fortran calls use this naming scheme.

| R wrapper | Fortran symbol | Computational role |
|---|---|---|
| `linbin` | `F_linbin` | 1D linear binning |
| `linbin2D` | `F_lbtwod` | 2D linear binning |
| `rlbin` | `F_rlbin` | Regression linear binning |
| `blkest` | `F_blkest` | Block-wise polynomial fitting |
| `cpblock` | `F_cp` | Mallows Cp block count selection |
| `locpoly` | `F_locpol` | Local polynomial weighted least squares |
| `sdiag` | `F_sdiag` | Smoother matrix diagonal computation |
| `sstdiag` | `F_sstdg` | Sum-of-squares smoother diagonal computation |

The Fortran layer handles all numerically intensive loops: binning, convolutions, and local polynomial linear algebra. The R layer handles argument validation, grid construction, kernel weight computation (Gaussian / beta kernels), FFT-based convolution (via `stats::fft`), and result assembly. This clean separation means R is responsible for correctness and usability; Fortran is responsible for throughput.

LINPACK-derived routines (attributed to Cleve Moler in `DESCRIPTION`) reside in `src/d*` and underpin the linear algebra used in `locpoly`, `sdiag`, and `sstdiag`.

---

## 6. Algorithmic Design

### 6.1 Binned approximation as the unifying strategy

Every computationally non-trivial function in KernSmooth adopts the same two-phase strategy:

1. **Bin** the raw data onto an equally-spaced grid using linear binning (which preserves the first moment of the data within each bin interval, unlike simple histogram binning).
2. **Convolve** the bin counts with a kernel weight vector, either via FFT (`bkde`, `bkde2D`, `bkfe`) or via the Fortran local polynomial routines (`locpoly`, `sdiag`, `sstdiag`).

This approximation trades a small, controllable bias for a reduction in complexity from O(n·g) to O(n + g·log g), where n is the sample size and g is the grid size. Wand (1994) establishes the theoretical justification for this approximation in the multivariate case.

### 6.2 Plug-in bandwidth selection

`dpik`, `dpih`, and `dpill` all use the *direct plug-in* principle: unknown population quantities in the asymptotically optimal bandwidth formula are replaced by kernel estimates of the same quantities, evaluated at a pilot bandwidth. The pilots are derived from a Gaussian reference distribution. `bkfe` is the shared workhorse that evaluates the required density functionals (integrals of the form ∫ f(x) f^(r)(x) dx) for all three selectors.

### 6.3 Local polynomial estimation

`locpoly` implements local polynomial regression (and density estimation as a special case) by solving a weighted least-squares problem at each grid point. The degree of the local polynomial and derivative order are user-specified, making the function general. The Fortran routine `F_locpol` performs the actual coefficient computation; `linbin` and `rlbin` prepare the binned data it operates on.

---

## 7. Notes on Source Organisation

- **Single R file:** All 16 R functions reside in `R/all.R`. There is no per-function or per-topic file splitting.
- **No vignettes:** The package does not include a `vignettes/` directory. Narrative explanation of methods and worked examples exist only in the `man/` documentation files and in the referenced book.
- **Documentation coverage:** Every exported function has a corresponding `.Rd` file in `man/`. None of the nine internal functions are documented.
- **No S3/S4/R5 classes:** The package is entirely procedural. All inputs and outputs are base R types (numeric vectors, matrices, lists).
- **No CRAN package runtime dependencies:** At runtime the package depends only on R's `stats` package. `MASS` and `carData` appear only in `Suggests` and are used solely in `man/` examples.
