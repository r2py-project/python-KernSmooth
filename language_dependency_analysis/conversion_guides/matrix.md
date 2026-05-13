## Conversion Guide: R `matrix()` to Python

---

### 1. Overview of `matrix` in R

`matrix()` is a base R function that constructs a two-dimensional array (a matrix) from a given data vector. Its full signature is:

```r
matrix(data = NA, nrow = 1, ncol = 1, byrow = FALSE, dimnames = NULL)
```

Key behaviours:

- **Single-argument form** — `matrix(data)`: wraps a vector into a column matrix (`length(data) × 1`).
- **Zero-fill form** — `matrix(0, nrow, ncol)`: creates a zero-filled `nrow × ncol` matrix.
- **Reshape form** — `matrix(data, nrow, ncol)`: reshapes an existing flat vector into an `nrow × ncol` matrix. R fills matrices **column-major** (down the first column, then the second column, etc.) by default.
- **Boolean-mask form** — `matrix(as.numeric(rp > 0), nrow(rp), ncol(rp))`: converts a logical matrix to 0/1 float matrix of the same dimensions.

---

### 2. Contextual Usage Analysis

All 16 usages appear in `KernSmooth/R/all.R` and fall into four structurally distinct patterns:

| Pattern | Lines | Functions |
|---|---|---|
| Single-argument column-vector wrap | 128 | `bkde2D` |
| Zero initialisation — two dimensions | 144, 150, 258, 294, 693, 694, 695, 792, 793, 869, 870, 871, 872 | `bkde2D`, `blkest`, `cpblock`, `locpoly`, `sdiag`, `sstdiag` |
| Reshape Fortran output into 2-D grid | 595 | `linbin2D` |
| Element-wise boolean mask (reshape in-place) | 162 | `bkde2D` |

---

### 3. Python Conversion Strategy

**`numpy`** is the correct and only appropriate equivalent for all four patterns:

- `numpy.zeros((nrow, ncol))` — zero-initialisation
- `numpy.reshape(data, (nrow, ncol), order='F')` — column-major (Fortran-order) reshape, matching R's default `byrow=FALSE`
- Wrapping a 1-D array into a column vector: `arr.reshape(-1, 1)` or `arr[:, np.newaxis]`
- Boolean masking: direct element-wise multiplication using `numpy` comparison operators — no explicit `matrix()` call is needed

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A — Single-argument column-vector wrap

**Location:** `bkde2D`, line 128.

```r
z <- matrix(dnorm(lvecid * facid) / h[id])    # (L[id]+1) x 1 column matrix
```

**Python Equivalent:**

```python
import numpy as np
from scipy.stats import norm

lvecid = np.arange(0, L_id + 1)
facid  = (b_id - a_id) / (h_id * (M_id - 1))
z = (norm.pdf(lvecid * facid) / h_id).reshape(-1, 1)  # shape: (L_id+1, 1)
```

**Explanation:** R's `matrix(vec)` with no dimension arguments produces an `(n × 1)` column matrix. In Python the equivalent is `.reshape(-1, 1)`.

---

#### 4.2 Pattern B — Zero-filled matrix initialisation

**Locations:** `locpoly` (lines 693–695), `sdiag`, `sstdiag`, `bkde2D`, `blkest`, `cpblock`.

```r
ss   <- matrix(0, M, ppp)   # M x ppp zero matrix
tt   <- matrix(0, M, pp)    # M x pp zero matrix
Smat <- matrix(0, pp, pp)   # pp x pp zero matrix
```

**Python Equivalent:**

```python
import numpy as np

ss   = np.zeros((M, ppp), dtype=np.float64)
tt   = np.zeros((M, pp),  dtype=np.float64)
Smat = np.zeros((pp, pp), dtype=np.float64)
```

**Explanation:** `matrix(0, nrow, ncol)` is R's idiomatic zero-allocation. `numpy.zeros((nrow, ncol))` is its direct counterpart. The explicit `dtype=np.float64` mirrors R's `as.double()` cast that these matrices receive before being passed to the Fortran routines.

---

#### 4.3 Pattern C — Reshape Fortran output into a 2-D grid

**Location:** `linbin2D`, line 595.

```r
matrix(out[[9L]], M1, M2)   # returns M1 x M2 matrix
```

**Python Equivalent:**

```python
import numpy as np

result = flat_result.reshape((M1, M2), order='F')   # column-major, matching R default
```

**Explanation:** R fills matrices column-by-column by default (`byrow=FALSE`). Fortran likewise stores arrays in column-major order. NumPy's default is row-major (`order='C'`), so `order='F'` must be specified explicitly to get the same memory layout.

---

#### 4.4 Pattern D — Element-wise boolean mask applied via reshape

**Location:** `bkde2D`, line 162.

```r
rp <- rp * matrix(as.numeric(rp > 0), nrow(rp), ncol(rp))
```

**Python Equivalent:**

```python
import numpy as np

# Option A: idiomatic NumPy boolean mask
rp = rp * (rp > 0).astype(np.float64)

# Option B: preferred NumPy idiom — element-wise maximum
rp = np.maximum(rp, 0.0)
```

**Explanation:** In Python the `matrix()` reshape step is entirely unnecessary because `numpy` comparison operators already return an array of the same shape as the operand. `(rp > 0)` produces a boolean array of shape `(M1, M2)` without any explicit dimension arguments.

---

### Summary Table

| R call | Python equivalent | Notes |
|---|---|---|
| `matrix(vec)` | `vec.reshape(-1, 1)` | Wraps 1-D array into column vector `(n, 1)` |
| `matrix(0, nrow, ncol)` | `np.zeros((nrow, ncol), dtype=np.float64)` | Zero buffer; use `float64` to match `as.double()` |
| `matrix(flat_vec, nrow, ncol)` | `flat_vec.reshape((nrow, ncol), order='F')` | Column-major fill — **must** specify `order='F'` for Fortran output |
| `rp * matrix(as.numeric(rp>0), nrow(rp), ncol(rp))` | `np.maximum(rp, 0.0)` | Non-negativity clip; no reshape needed in numpy |
