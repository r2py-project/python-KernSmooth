## Conversion Guide: R `min` to Python

---

### 1. Overview of `min` in R

`min()` is a base R function that returns the minimum value from one or more numeric arguments. When passed a single atomic vector, it returns the smallest element as a length-1 scalar. When passed multiple arguments, it returns the smallest value across all of them.

Key signature: `min(..., na.rm = FALSE)`

- Input: one or more numeric scalars or vectors (or a mix).
- Output: always a single scalar — the global minimum across all supplied values.

This "reduce to a scalar" behavior distinguishes `min` from vectorized element-wise comparisons like `pmin`.

---

### 2. Contextual Usage Analysis

The 17 CSV rows fall into four distinct functional patterns:

| Pattern | Description | Locations |
|---|---|---|
| A | `min(x)` — find minimum of a data vector | `bkde:42`, `bkde2D:107`, `bkfe:176`, `locpoly:619,621`, `sdiag:736`, `sstdiag:816` |
| B | `min(scalar_expr1, scalar_expr2)` — take the smaller of two computed scalars | `bkde:54`, `bkde2D:125`, `bkfe:204`, `dpih:333`, `dpik:428`, `locpoly:684` |
| C | `min(bandwidth)` — find minimum of a bandwidth vector | `bkde2D:87` |
| D | `min(L)` — find minimum of a short integer vector | `bkde2D:134` |

**Note:** The CSV entry `min,all.R,sdiag,789,sum(Lvec)` is mislabeled — the actual source code at that line calls `sum()`, not `min()`. That entry belongs in the `sum` conversion guide, not here.

---

### 3. Python Conversion Strategy

- For **Pattern A and C and D** (minimum of a single array/vector): use `numpy.min(arr)` or equivalently `arr.min()`.
- For **Pattern B** (minimum of exactly two scalar expressions): use Python's built-in `min(a, b)`. Both operands are guaranteed to be Python/NumPy scalars at the call site.

`numpy` is the primary library because all numeric data in the Python port of KernSmooth will be represented as NumPy arrays, making `numpy.min` the natural drop-in for R's `min(x)`.

---

### 4. Step-by-Step Conversion Examples

#### Pattern A — Minimum of a data vector

**Original R Context:**

```r
range.x <- c(min(x) - tau*h, max(x) + tau*h)
```

**Python Equivalent:**

```python
import numpy as np

range_x = np.array([np.min(x) - tau * h, np.max(x) + tau * h])
```

---

#### Pattern B — Minimum of two scalar expressions

**Original R Context:**

```r
L <- min(floor(tau / delta), M)  # bkde, line 54

scalest <- min(
    (quantile(x, 3/4) - quantile(x, 1/4)) / 1.349,
    sqrt(var(x))
)  # dpih / dpik, lines 333 / 428
```

**Python Equivalent:**

```python
import numpy as np

L = min(int(np.floor(tau / delta)), M)

iqr_scale = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
stdev_scale = np.std(x, ddof=1)  # ddof=1 matches R's var()
scalest = min(iqr_scale, stdev_scale)
```

**Explanation:** When exactly two scalar arguments are passed, Python's built-in `min(a, b)` is idiomatic and sufficient.

Key translation nuance for `dpih`/`dpik`:
- R's `quantile(x, p)` defaults to type 7 interpolation; `numpy.quantile` also defaults to linear interpolation, so the results are equivalent.
- R's `var(x)` uses `n-1` (sample variance); the Python equivalent is `np.var(x, ddof=1)` or equivalently `np.std(x, ddof=1)`.

---

#### Pattern C — Minimum of a bandwidth vector (validation guard)

**Original R Context:**

```r
if (!missing(bandwidth) && min(bandwidth) <= 0)
    stop("'bandwidth' must be strictly positive")
```

**Python Equivalent:**

```python
import numpy as np

bandwidth = np.atleast_1d(np.asarray(bandwidth, dtype=float))
if np.min(bandwidth) <= 0:
    raise ValueError("'bandwidth' must be strictly positive")
```

---

#### Pattern D — Minimum of a short integer vector (zero-check guard)

**Original R Context:**

```r
if (min(L) == 0)
    warning("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
```

**Python Equivalent:**

```python
import numpy as np
import warnings

# L is a numpy array of length 2 (integer values)
if np.min(L) == 0:
    warnings.warn("Binning grid too coarse for current (small) bandwidth: "
                  "consider increasing 'gridsize'")
```
