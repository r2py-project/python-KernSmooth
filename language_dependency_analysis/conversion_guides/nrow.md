## Conversion Guide: R `nrow` to Python

---

### 1. Overview of `nrow` in R

`nrow()` is a base R function that returns the number of rows in a two-dimensional object, specifically a **matrix** or **data frame**. It returns a single integer scalar. If the input has no dimension attribute (e.g., it is a plain vector), `nrow()` returns `NULL`.

- **Input:** A matrix or data frame.
- **Output:** A single integer — the count of rows.
- **Equivalent:** `nrow(x)` is semantically identical to `dim(x)[1]` in R.

---

### 2. Contextual Usage Analysis

All three usages of `nrow` in the codebase appear in `KernSmooth/R/all.R` and follow the same pattern: extracting the number of data observations from a two-column numeric matrix. The result is always stored as an integer scalar `n`, which is then passed to downstream Fortran routines or used in arithmetic normalization.

**Usage 1 — `bkde2D`, line 92:**
```r
n <- nrow(x)
```
`x` is the input data matrix to `bkde2D` — a numeric matrix with 2 columns where each row is a bivariate observation. `n` is subsequently used in `kapp <- ... / n` to normalize a kernel weight matrix.

**Usage 2 — `bkde2D`, line 162:**
```r
rp <- rp * matrix(as.numeric(rp > 0), nrow(rp), ncol(rp))
```
Here `rp` is an intermediate 2D matrix. `nrow(rp)` is passed as the first dimension argument to `matrix()`.

**Usage 3 — `linbin2D`, line 584:**
```r
n <- nrow(X)
```
`X` is the bivariate input data matrix passed to `linbin2D`. `n` is passed directly to a Fortran subroutine as the sample size.

---

### 3. Python Conversion Strategy

The primary Python equivalent is **`numpy`**. NumPy 2D arrays expose `.shape` as a tuple `(n_rows, n_cols)`, so `nrow(x)` translates directly to `x.shape[0]`.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 `nrow(x)` — Extract sample size from a bivariate data matrix

**Locations:** `bkde2D` (line 92), `linbin2D` (line 584)

```r
n <- nrow(x)
```

**Python Equivalent:**

```python
import numpy as np

# x is a 2D numpy array of shape (n_observations, 2)
n = x.shape[0]
```

**Explanation:**
- `x.shape` returns a tuple `(n_rows, n_cols)`. Index `[0]` gives the row count, which is the direct counterpart of R's `nrow(x)`.
- The result is a plain Python `int`, exactly as R's `nrow()` returns an integer scalar.

---

#### 4.2 `nrow(rp)` — Use row count as a matrix construction dimension

**Location:** `bkde2D` (line 162)

```r
rp <- rp * matrix(as.numeric(rp > 0), nrow(rp), ncol(rp))
```

**Python Equivalent:**

```python
import numpy as np

# rp is a 2D numpy array of shape (M1, M2)
rp = rp * (rp > 0).astype(np.float64)
# or equivalently:
rp = np.maximum(rp, 0.0)
```

**Explanation:**
- In R, `matrix(as.numeric(rp > 0), nrow(rp), ncol(rp))` creates a float matrix of the same shape filled with 1.0/0.0. The explicit `nrow`/`ncol` arguments are needed because `matrix()` requires explicit dimension arguments.
- In NumPy, `rp > 0` already produces a boolean array of identical shape to `rp`. Casting it with `.astype(np.float64)` gives the equivalent 0.0/1.0 mask, and the `*` operator applies element-wise broadcasting automatically — no shape dimensions need to be named explicitly.
- `np.maximum(rp, 0.0)` is the most idiomatic alternative that clips all negative entries to zero in a single vectorized call, making `nrow` entirely unnecessary.
