## Conversion Guide: `as.integer` in R to Python

---

### 1. Overview of `as.integer` in R

`as.integer` is a base R coercion function that converts its argument to integer type. It accepts numeric values (doubles, logicals, character strings representing numbers, or existing integers) and returns an integer value or integer vector, discarding any fractional part by truncation toward zero.

In the KernSmooth codebase its role is purely type-coercion: ensuring that values passed to `.Fortran()` FFI calls carry the correct storage type. Fortran routines require arguments to be passed with exact type contracts; R's `.Fortran()` interface enforces this. Every scalar that must arrive as a Fortran `INTEGER` argument is explicitly wrapped with `as.integer`.

---

### 2. Contextual Usage Analysis

The source file involved is `KernSmooth/R/all.R`.

All 46 occurrences of `as.integer` fall into two clearly separated categories:

**Category A — Scalar parameter coercion before a `.Fortran()` call**

The value being coerced is a single scalar (integer or very small integer-valued numeric). Examples: `n` (length of a vector), `q`/`qq` (polynomial degree and degree+1), `Nval`/`Nmax` (block count), `M`/`M1`/`M2` (grid sizes), `bwdisc`/`Q` (number of bandwidth discretisation points), `trun` (truncation flag 0 or 1), `drv` (derivative order), `degree` (polynomial degree), `pp` (degree+1), `ppp` (2*degree+1). Functions affected: `blkest`, `cpblock`, `linbin`, `linbin2D`, `locpoly`, `rlbin`, `sdiag`, `sstdiag`.

**Category B — Vector coercion before a `.Fortran()` call**

The value being coerced is a numeric vector, not a scalar. Examples: `Lvec` (a vector of L-values of length Q), `indic` (index vector of length M), `midpts` (midpoints vector of length Q), `ipvt` (pivot index vector of length pp). All four appear together in the same `.Fortran()` argument list across `locpoly`, `sdiag`, and `sstdiag`.

---

### 3. Python Conversion Strategy

The Python equivalent depends on whether the argument is a scalar or a vector:

- For **scalars**, use the built-in `int()`. It truncates toward zero, exactly matching `as.integer`. When the integer is passed into an `f2py`-generated or `ctypes`-wrapped Fortran routine, it must be typed as `np.int32` (matching Fortran `INTEGER` which is 32-bit by default).
- For **vectors**, use `numpy.ndarray.astype(np.int32)` because the Fortran routines expect 32-bit integers. NumPy arrays are the natural vectorized container.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Scalar derived from `len()` / `length()` — `n`, `M`, `M1`, `M2`, `trun`

**Locations:** `blkest` (line 265), `cpblock` (line 300), `linbin` (lines 572–574), `linbin2D` (lines 592–594), `rlbin` (lines 723–724)

```r
n <- length(X)           # integer scalar
M <- length(gpoints)     # integer scalar
trun <- if (truncate) 1L else 0L

.Fortran(F_linbin, as.double(X), as.integer(n),
         as.double(a), as.double(b), as.integer(M),
         as.integer(trun), double(M))
```

**Python Equivalent:**

```python
import numpy as np

n = len(X)
M = len(gpoints)
trun = np.int32(1) if truncate else np.int32(0)

args = (
    X.astype(np.float64),
    np.int32(n),
    np.float64(a),
    np.float64(b),
    np.int32(M),
    np.int32(trun),
    np.zeros(M, dtype=np.float64),
)
```

**Explanation:** `np.int32` is needed when passing to Fortran (matching Fortran `INTEGER` which is 32-bit). For high-level Python logic, a plain `int` is sufficient.

---

#### 4.2 Integer vector coercion — `Lvec`, `indic`, `midpts`, `ipvt`

**Locations:** `locpoly` (lines 703–707), `sdiag` (lines 800–804), `sstdiag` (lines 879–884)

```r
Lvec   <- floor(tau * hdisc / delta)   # numeric vector, length Q
indic  <- round(...)                   # numeric vector, length M
midpts <- rep(0, Q)                    # numeric vector, length Q
ipvt   <- rep(0, pp)                   # numeric vector, length pp

.Fortran(F_locpol, ...,
         as.integer(Lvec), as.integer(indic), as.integer(midpts),
         ..., as.integer(ipvt), ...)
```

**Python Equivalent:**

```python
import numpy as np

Lvec   = np.floor(tau * hdisc / delta).astype(np.int32)   # int32 array, length Q
indic  = np.round(...).astype(np.int32)                    # int32 array, length M
midpts = np.zeros(Q, dtype=np.int32)                       # int32 array, length Q
ipvt   = np.zeros(pp, dtype=np.int32)                      # int32 array, length pp
```

**Explanation:** R's `as.integer()` is fully vectorized and converts every element of an array in-place. The direct NumPy equivalent is `.astype(np.int32)` on an existing array, or `dtype=np.int32` at construction time. `np.int32` (32-bit) is preferred over `np.int64` (64-bit) because standard Fortran `INTEGER` is 32-bit.

---

### Summary Table

| R pattern | Context | Python equivalent |
|---|---|---|
| `as.integer(n)` where `n = length(X)` | Scalar from `len()` | `int(len(X))` for logic; `np.int32(len(X))` for Fortran FFI |
| `as.integer(q)` where `q` is a scalar parameter | Scalar, arithmetic result | `int(q)` for logic; `np.int32(q)` in FFI arg list |
| `as.integer(trun)` where `trun` is `0` or `1` | Boolean-derived scalar | `np.int32(1 if truncate else 0)` |
| `as.integer(Lvec)` where `Lvec` is a vector | Numeric vector → integer vector | `Lvec.astype(np.int32)` |
| `as.integer(midpts)`, `as.integer(ipvt)` | Pre-allocated zero vectors | `np.zeros(length, dtype=np.int32)` |
