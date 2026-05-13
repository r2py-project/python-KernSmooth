## Conversion Guide: R `round` to Python

---

### 1. Overview of `round` in R

`round(x, digits = 0)` rounds the numeric values in `x` to the specified number of decimal places. When `digits = 0` (the default), it rounds to the nearest integer.

Key behaviors:

- **Vectorized:** when `x` is a numeric vector, `round` applies element-wise and returns a numeric vector of the same length.
- **Banker's rounding (round half to even):** when a value is exactly halfway between two integers (e.g., 0.5, 1.5, 2.5), R rounds toward the nearest *even* integer, following IEC 60559. This differs from the "round half up" rule common in everyday usage.
- **Return type:** the result is a numeric (double) vector, even though its values are whole numbers. Downstream code in KernSmooth immediately converts these values via `as.integer(indic)` before passing them to Fortran, so the final consumed type is integer.

---

### 2. Contextual Usage Analysis

All three CSV rows contain an identical expression, appearing in the `indic` assignment block inside the `length(bandwidth) == M` branch of three closely related functions: `locpoly` (line 674), `sdiag` (line 779), and `sstdiag` (line 856), all in `KernSmooth/R/all.R`.

The full expression in each case is:

```r
indic <- if (Q > 1L) {
    lhdisc <- log(hdisc)
    gap <- (lhdisc[Q] - lhdisc[1L]) / (Q - 1)
    if (gap == 0) rep(1, M)
    else round(((log(bandwidth) - log(sort(bandwidth)[1L])) / gap) + 1)
} else rep(1, M)
```

The argument to `round` is a **numeric vector** of length `M`. The purpose is to map each bandwidth value to a 1-based index into `hdisc`, the discretized bandwidth grid. The result is immediately passed to Fortran as `as.integer(indic)`, so it must contain valid positive integers.

---

### 3. Python Conversion Strategy

`numpy.round()` (equivalently `numpy.ndarray.round()`) is the correct replacement:

- operates element-wise on NumPy arrays, matching R's vectorized semantics exactly,
- also implements **round half to even** (banker's rounding), matching R's IEC 60559 behavior,
- returns a `float64` array by default, which can then be cast to `int` via `.astype(int)`, matching R's subsequent `as.integer()` cast.

`math.round` must not be used here because it is scalar-only and would require a Python loop over `M` elements.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 `round` applied to an index-computation vector

**Locations:** `locpoly` (line 674), `sdiag` (line 779), `sstdiag` (line 856)

**Original R Context:**

```r
lhdisc <- log(hdisc)
gap    <- (lhdisc[Q] - lhdisc[1L]) / (Q - 1)
indic  <- if (Q > 1L) {
    if (gap == 0) rep(1, M)
    else round(((log(bandwidth) - log(sort(bandwidth)[1L])) / gap) + 1)
} else rep(1, M)
```

**Python Equivalent:**

```python
import numpy as np

lhdisc = np.log(hdisc)
gap    = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)   # 0-based indexing

if Q > 1:
    if gap == 0:
        indic = np.ones(M, dtype=np.int64)
    else:
        raw   = ((np.log(bandwidth) - np.log(np.sort(bandwidth)[0])) / gap) + 1
        indic = np.round(raw).astype(np.int64)
else:
    indic = np.ones(M, dtype=np.int64)
```

**Explanation:**

| Translation point | Detail |
|---|---|
| `round(x)` → `np.round(raw).astype(np.int64)` | `np.round` is element-wise on arrays and uses the same round-half-to-even rule as R. The `.astype(np.int64)` mirrors R's `as.integer()` cast before the Fortran call. |
| `sort(bandwidth)[1L]` → `np.sort(bandwidth)[0]` | R uses 1-based indexing; Python uses 0-based. The first element is index `0` in NumPy. |
| `lhdisc[Q]` → `lhdisc[Q - 1]` | Same 1-based to 0-based shift; `Q` is the last valid 1-based index in R, so it maps to `Q - 1` in Python. |
| `rep(1, M)` → `np.ones(M, dtype=np.int64)` | R's `rep(1, M)` creates a length-`M` vector of 1s. `np.ones` with integer dtype is the equivalent. |
| `log(bandwidth)` → `np.log(bandwidth)` | Both operate element-wise on a vector/array; direct substitution. |

**Important nuance — banker's rounding match:** both R's `round` and `np.round` use round-half-to-even, so the results are numerically identical for tie cases.
