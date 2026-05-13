## Conversion Guide: R `length` to Python

---

### 1. Overview of `length` in R

`length` is a base R function that returns the number of elements in an object. Its signature is:

```r
length(x)
```

- **Input:** Any R object — atomic vectors (numeric, logical, character, etc.), lists, or factors.
- **Output:** A single non-negative integer scalar representing the total element count.

Key behaviors to be aware of:
- For a 1-D atomic vector, `length` returns the number of elements, equivalent to Python's `len()` on a list or `numpy` array's length on the first axis.
- For a matrix or data frame, `length` returns the **total** number of elements (rows times columns) — though in this codebase all calls apply `length` to 1-D vectors.
- `length(NULL)` returns `0`.
- The result is always a plain integer scalar.

---

### 2. Contextual Usage Analysis

Across all 27 call sites in `KernSmooth/R/all.R`, `length` is applied exclusively to **1-D vectors**. The vectors fall into three semantic categories:

**Category A — Raw data vectors (input samples):** Arguments named `x`, `X`, or `Y`. The result is stored as `n`, the sample size.

Lines: 18 (`bkde`), 242 (`blkest`), 278 (`cpblock`), 317 (`dpih`), 414 (`dpik`), 498–499–505 (`dpill`), 567 (`linbin`), 718 (`rlbin`).

**Category B — Grid / bin-count vectors:** Arguments named `gpoints`, `gpoints1`, `gpoints2`, or `gcounts`. The result is stored as `M` (grid size).

Lines: 30 (`bkde`), 192 (`bkfe`), 568 (`linbin`), 586–587 (`linbin2D`), 651 (`locpoly`), 719 (`rlbin`), 755 (`sdiag`), 833 (`sstdiag`).

**Category C — Bandwidth vectors (scalar-or-vector dispatch):** The `bandwidth` parameter may be either a scalar or a vector of length `M`. `length(bandwidth)` is compared against `M` and `1L` to dispatch into two distinct code paths.

Lines: 660 and 676 (`locpoly`), 765 and 781 (`sdiag`), 842 and 858 (`sstdiag`).

---

### 3. Python Conversion Strategy

The primary Python equivalent is **`len()`** from the Python standard library, used on `numpy.ndarray` objects (1-D arrays). This is the direct and idiomatic counterpart to R's `length` in all 27 occurrences because:

- Every call operates on a 1-D array.
- The result is always used as a plain integer scalar for arithmetic, comparisons, loop bounds, or Fortran interface arguments.
- `len(arr)` on a 1-D `numpy.ndarray` returns the size of the first axis, which equals the total element count — identical to R's `length`.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Category A — Sample size from a raw data vector

```r
n <- length(x)
```

**Python Equivalent:**

```python
n = len(x)
```

---

#### 4.2 Category B — Grid size from a grid-point or bin-count vector

```r
M  <- length(gpoints)
M1 <- length(gpoints1)
M2 <- length(gpoints2)
```

**Python Equivalent:**

```python
M  = len(gpoints)
M1 = len(gpoints1)
M2 = len(gpoints2)
```

---

#### 4.3 Category C — Bandwidth dispatch: scalar vs. vector

**Original R Context:**

```r
if (length(bandwidth) == M) {
    # variable-bandwidth path
} else if (length(bandwidth) == 1L) {
    # fixed-bandwidth path
} else {
    stop("'bandwidth' must be a scalar or an array of length 'gridsize'")
}
```

**Python Equivalent:**

```python
import numpy as np

bandwidth = np.atleast_1d(bandwidth)  # normalize scalar to array for uniform handling

if len(bandwidth) == M:
    # variable-bandwidth path
    hlow = np.sort(bandwidth)[0]        # R's sort(bandwidth)[1L] is 1-indexed
    hupp = np.sort(bandwidth)[M - 1]    # R's sort(bandwidth)[M] is 1-indexed
    hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))
elif len(bandwidth) == 1:
    # fixed-bandwidth path
    indic = np.ones(M, dtype=int)
    Q = 1
    hdisc = np.full(Q, bandwidth[0])
else:
    raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")
```

**Explanation:**
- `np.atleast_1d(bandwidth)` normalizes a Python float or 0-D array into a 1-D array so that `len()` always works.
- R uses 1-based indexing: `sort(bandwidth)[1L]` is the minimum, `sort(bandwidth)[M]` is the maximum. Python uses 0-based indexing: `np.sort(bandwidth)[0]` and `np.sort(bandwidth)[M - 1]`.

---

#### 4.4 Arithmetic inside `length` — `dpill` trimming indices

**Original R Context:**

```r
# R
indlow <- floor(trim * length(x)) + 1
indupp <- length(x) - floor(trim * length(x))
x <- x[indlow:indupp]
```

**Python Equivalent:**

```python
import numpy as np

n = len(x)
indlow = int(np.floor(trim * n))        # 0-based: first index to keep
indupp = n - int(np.floor(trim * n))    # 0-based exclusive stop
x = x[indlow:indupp]
```

**Explanation:** The critical nuance is the index arithmetic across the 1-based/0-based boundary. R's `x[indlow:indupp]` with `indlow = floor(trim*n) + 1` (1-based inclusive start) translates to Python's `x[indlow:indupp]` with `indlow = floor(trim*n)` (0-based inclusive start).
