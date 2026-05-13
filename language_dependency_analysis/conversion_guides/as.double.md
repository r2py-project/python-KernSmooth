## Conversion Guide: `as.double` (R) to Python

---

### 1. Overview of `as.double` in R

`as.double` is a base R coercion function that converts its argument to double-precision floating-point storage mode (equivalent to a 64-bit IEEE 754 `double`). It is an alias for `as.numeric` and is the canonical way to force R objects into the `double` type before passing them across the R/Fortran foreign-function interface (FFI).

**Typical inputs:** numeric scalars, integer vectors, numeric vectors, numeric matrices, or any coercible R object.

**Expected output:** an object of the same shape as the input, with all elements stored as 64-bit doubles.

**Critical context for this codebase:** every call to `as.double` in `all.R` occurs exclusively inside `.Fortran(...)` call argument lists. `.Fortran` is R's built-in mechanism for calling compiled Fortran subroutines. `as.double` guarantees the argument is a contiguous block of 64-bit doubles before the data is handed off to the Fortran layer.

---

### 2. Contextual Usage Analysis

All 59 occurrences of `as.double` in the CSV follow a single, uniform pattern: they appear as argument coercions inside `.Fortran(...)` calls spread across six internal helper functions — `blkest`, `cpblock`, `linbin`, `linbin2D`, `locpoly`, `rlbin`, `sdiag`, and `sstdiag`.

| Argument category | R object type before cast | Example variables |
|---|---|---|
| Input data vectors | Numeric vector (length `n`) | `x`, `y`, `X`, `Y` |
| Scalar boundary values | Length-1 numeric (scalar) | `a`, `b`, `a1`, `a2`, `b1`, `b2`, `delta`, `hdisc` |
| Pre-allocated work vectors (all zeros) | `rep(0, k)` | `xj`, `yj`, `coef`, `wk`, `qraux`, `fkap`, `ss`, `tt`, `work`, `det`, `Sdg`, `SSTd`, `curvest`, `Tvec` |
| Pre-allocated work matrices (all zeros) | `matrix(0, nrow, ncol)` | `Xmat`, `Smat`, `Umat` |
| Scalar accumulators | Scalar `0` | `sigsqe`, `th22e`, `th24e` |

---

### 3. Python Conversion Strategy

The direct Python equivalent is **`numpy.asarray(..., dtype=np.float64)`** (or the shorthand **`numpy.float64(...)`** for true scalars).

- R's `double` is IEEE 754 64-bit, identical to `numpy.float64`. This is a lossless, exact type mapping.
- `numpy.asarray(x, dtype=np.float64)` handles scalars, lists, arrays, and matrices uniformly.
- `math` module functions must not be used here: every argument being cast is either a vector or a pre-allocated buffer array.

The mapping rule:

| R source object | Python equivalent |
|---|---|
| Numeric vector / array | `np.asarray(x, dtype=np.float64)` |
| Numeric matrix | `np.asarray(X, dtype=np.float64)` (shape preserved) |
| Scalar `0` or scalar double | `np.float64(0.0)` or `float(0.0)` |
| Pre-allocated zero buffer (vector) | `np.zeros(n, dtype=np.float64)` replaces `as.double(rep(0, n))` entirely |
| Pre-allocated zero buffer (matrix) | `np.zeros((nrow, ncol), dtype=np.float64)` replaces `as.double(matrix(0, nrow, ncol))` |

---

### 4. Step-by-Step Conversion Examples

#### 4.1 `blkest` — Coercing input data vectors and pre-allocated work buffers

**Locations:** `KernSmooth/R/all.R`, function `blkest`, lines 265–269.

```r
out <- .Fortran(F_blkest,
    as.double(x),        # double vector, length n
    as.double(y),        # double vector, length n
    ...
    as.double(xj),       # double work vector, length n (zeros)
    as.double(Xmat),     # double work matrix, n x qq (zeros)
    as.double(sigsqe),   # double scalar (zero)
    ...
)
```

**Python Equivalent:**

```python
import numpy as np

x_d = np.asarray(x, dtype=np.float64)
y_d = np.asarray(y, dtype=np.float64)

# Pre-allocated work vectors — construct directly as float64 zeros
xj    = np.zeros(n, dtype=np.float64)
yj    = np.zeros(n, dtype=np.float64)
coef  = np.zeros(qq, dtype=np.float64)
wk    = np.zeros(n, dtype=np.float64)
qraux = np.zeros(qq, dtype=np.float64)
Xmat  = np.zeros((n, qq), dtype=np.float64)

sigsqe = np.float64(0.0)
th22e  = np.float64(0.0)
th24e  = np.float64(0.0)
```

**Explanation:**
- `np.asarray(x, dtype=np.float64)` coerces any array-like to a 64-bit double array.
- For zero-initialized work buffers, `np.zeros(...)` skips the two-step R pattern (allocate, then cast).
- For scalar accumulators, `np.float64(0.0)` produces a NumPy scalar of the correct type.

---

#### 4.2 `linbin` — Scalar boundary values and input vectors

**Locations:** `KernSmooth/R/all.R`, function `linbin`, lines 572–574.

```r
.Fortran(F_linbin,
    as.double(X),   # double vector, length n
    as.integer(n),
    as.double(a),   # double scalar (boundary value)
    as.double(b),   # double scalar (boundary value)
    ...
)
```

**Python Equivalent:**

```python
import numpy as np

X_d = np.asarray(X, dtype=np.float64)
a   = np.float64(gpoints[0])    # R's gpoints[1L] → Python gpoints[0]
b   = np.float64(gpoints[-1])   # R's gpoints[M] → Python gpoints[-1]
counts = np.zeros(M, dtype=np.float64)
```

**Explanation:**
- Scalar extraction: R's 1-based `gpoints[1L]` → Python's 0-based `gpoints[0]`.
- `np.float64(...)` ensures the scalar has the correct type for Fortran wrappers that expect a `double *`.

---

### Summary: Universal Translation Rule

Every `as.double(arg)` call falls into one of three categories:

| R pattern | Purpose | Python replacement |
|---|---|---|
| `as.double(x)` where `x` is an existing data vector/matrix | Type-safety cast before FFI call | `np.asarray(x, dtype=np.float64)` |
| `as.double(rep(0, n))` | Zero-initialize a double work vector | `np.zeros(n, dtype=np.float64)` |
| `as.double(matrix(0, r, c))` | Zero-initialize a double work matrix | `np.zeros((r, c), dtype=np.float64)` |
| `as.double(scalar_0)` | Scalar double accumulator | `np.float64(0.0)` or `0.0` |

The `dtype=np.float64` specification is mandatory in all cases to guarantee the 64-bit double precision that R's `as.double` enforces.
