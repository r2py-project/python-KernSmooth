## Conversion Guide: R `var` to Python

---

### 1. Overview of `var` in R

R's `var()` computes the **sample variance** of a numeric vector. For a numeric vector `x` of length `n`, it returns:

```
var(x) = (1/(n-1)) * sum((xi - x_bar)^2)
```

The denominator is `n - 1` (Bessel's correction), meaning it computes the **unbiased sample variance**, not the population variance.

Key defaults:
- `na.rm = FALSE` — missing values propagate to `NA` by default.
- Divisor is always `n - 1`.

---

### 2. Contextual Usage Analysis

All five usages occur in `KernSmooth/R/all.R` and fall into two recurring patterns:

**Pattern A — Standard deviation via `sqrt(var(x))`**

`var(x)` is immediately wrapped in `sqrt()` to obtain the sample standard deviation of the input data vector `x`. This appears in `bkde` (line 35), `dpih` (lines 331, 333), and `dpik` (lines 426, 428).

**Pattern B — Scale estimate selection in a `switch` block**

In both `dpih` and `dpik`, a `scalest` parameter selects among three scale estimators (`"stdev"`, `"iqr"`, `"minim"`). The `"stdev"` branch assigns `sqrt(var(x))` directly. The `"minim"` branch takes the minimum of the IQR-based estimate and `sqrt(var(x))`.

In all cases `x` is the raw numeric input data vector, and the result is a positive scalar used subsequently to standardize `x`.

---

### 3. Python Conversion Strategy

The primary Python equivalent is **`numpy.var(x, ddof=1)`** for variance, or more directly **`numpy.std(x, ddof=1)`** when the square root is immediately applied (which covers every usage here).

`numpy` is the correct choice because:
- `x` is always a numeric array (vector), not a scalar, so NumPy's vectorized implementation is appropriate.
- `numpy.std(ddof=1)` computes the exact same Bessel-corrected sample standard deviation as R's `sqrt(var(x))` in a single call.
- `numpy.var(ddof=1)` mirrors R's `var()` exactly (sample variance, divisor `n-1`) when the raw variance value is needed.
- `math.sqrt` / `statistics.variance` are scalar-only alternatives and are less idiomatic for array data.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 `bkde` — default bandwidth computation (line 35)

```r
h <- if (missing(bandwidth)) del0 * (243/(35*n))^(1/5) * sqrt(var(x))
     else if(canonical) del0 * bandwidth else bandwidth
```

**Python Equivalent:**

```python
import numpy as np

n = len(x)
if bandwidth is None:
    h = del0 * (243 / (35 * n)) ** (1 / 5) * np.std(x, ddof=1)
elif canonical:
    h = del0 * bandwidth
else:
    h = bandwidth
```

**Explanation:**
- `np.std(x, ddof=1)` directly replaces `sqrt(var(x))`. The `ddof=1` argument sets the divisor to `n - 1`, matching R's Bessel-corrected default.
- `bandwidth is None` is the idiomatic Python replacement for R's `missing(bandwidth)`.

---

#### 4.2 `dpih` and `dpik` — scale estimate selection (lines 331, 333, 426, 428)

```r
scalest <- switch(scalest,
    "stdev" = sqrt(var(x)),
    "iqr"   = (quantile(x, 3/4) - quantile(x, 1/4)) / 1.349,
    "minim" = min((quantile(x, 3/4) - quantile(x, 1/4)) / 1.349, sqrt(var(x)))
)
```

**Python Equivalent:**

```python
import numpy as np

std_x = np.std(x, ddof=1)
iqr_x = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349

if scalest == "stdev":
    scalest = std_x
elif scalest == "iqr":
    scalest = iqr_x
elif scalest == "minim":
    scalest = min(iqr_x, std_x)
```

**Explanation:**
- `np.std(x, ddof=1)` replaces `sqrt(var(x))` for the same reasons as above.
- `np.quantile(x, 0.75)` replaces `quantile(x, 3/4)`. NumPy's default interpolation method (`"linear"`) matches R's `quantile` type 7 (R's default).
- Python's built-in `min()` replaces R's `min()` here because both arguments are already scalars.
- A shared helper function is recommended:

```python
def _scale_estimate(x, method: str) -> float:
    std_x = np.std(x, ddof=1)
    iqr_x = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    if method == "stdev":
        return std_x
    elif method == "iqr":
        return iqr_x
    elif method == "minim":
        return min(iqr_x, std_x)
    else:
        raise ValueError(f"Unknown scalest method: {method}")
```

---

### Summary Table

| R expression | Python equivalent | Notes |
|---|---|---|
| `var(x)` | `np.var(x, ddof=1)` | Scalar sample variance, Bessel-corrected (`n-1`) |
| `sqrt(var(x))` | `np.std(x, ddof=1)` | Sample standard deviation; preferred single-call form |
| `min(..., sqrt(var(x)))` | `min(..., np.std(x, ddof=1))` | Both args are scalars; Python built-in `min` is correct |

The critical keyword argument in every case is `ddof=1`. NumPy defaults to `ddof=0` (population variance/std), so omitting `ddof=1` would produce a systematically different result from R's `var()`.
