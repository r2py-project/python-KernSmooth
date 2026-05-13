## Conversion Guide: R `quantile` to Python

---

### 1. Overview of `quantile` in R

`quantile(x, probs)` is a base R function from the `stats` package that computes sample quantiles of a numeric vector `x` at the specified probability levels given by `probs`.

- **`x`**: A numeric vector of data values.
- **`probs`**: A numeric vector of probabilities in `[0, 1]`. A single scalar probability is also accepted.
- **Return value**: A named numeric vector (or scalar when a single probability is supplied) of the same length as `probs`, containing the estimated quantile values.
- **Default interpolation method**: R uses `type = 7` by default (linear interpolation between order statistics, the most common convention for continuous data).

---

### 2. Contextual Usage Analysis

All eight CSV rows resolve to exactly **two structurally identical code patterns**, one inside `dpih` (line 332-333) and one inside `dpik` (line 427-428). Both functions share the same `scalest` switch block for computing a scale estimate from the data vector `x`.

```r
# dpih (lines 332-333) and dpik (lines 427-428) — identical pattern:
"iqr"   = (quantile(x, 3/4) - quantile(x, 1/4)) / 1.349,
"minim" = min((quantile(x, 3/4) - quantile(x, 1/4)) / 1.349, sqrt(var(x)))
```

Key observations:
- `x` is the raw input data vector — a 1-D numeric array of observed data values.
- `quantile` is called with a **scalar probability** (`3/4` = 0.75 for Q3, `1/4` = 0.25 for Q1).
- The two calls always appear as a **subtracted pair** to form the IQR (interquartile range).
- The IQR is divided by `1.349`, which is the standard normal IQR, converting the raw IQR into a **robust standard deviation estimate**.
- The result is a **scalar float**.

---

### 3. Python Conversion Strategy

**Chosen library: `numpy.quantile`**

`numpy.quantile(a, q)` operates element-wise over arrays and accepts scalar or array-valued `q`, matching R's vectorized semantics. It defaults to **linear interpolation** (`method='linear'`), which corresponds exactly to R's `type = 7` default. It returns a scalar float when `q` is a scalar, matching the scalar output produced here.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 IQR-Based Scale Estimate (`"iqr"` branch)

**Locations:**
- `KernSmooth/R/all.R` — function `dpih`, line 332
- `KernSmooth/R/all.R` — function `dpik`, line 427

```r
"iqr" = (quantile(x, 3/4) - quantile(x, 1/4)) / 1.349
```

**Python Equivalent:**

```python
import numpy as np

# x: np.ndarray, shape (n,), dtype float
scalest_iqr = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
```

**Explanation:**
- `3/4` and `1/4` in R are floating-point literals; they translate directly to `0.75` and `0.25` in Python.
- `np.quantile(x, q)` with a scalar `q` returns a scalar `float64`, matching R's scalar output.
- The default `method='linear'` in NumPy corresponds to R's default `type=7`, so no extra argument is needed.

---

#### 4.2 Minimum-of-Two-Estimators Scale Estimate (`"minim"` branch)

**Locations:**
- `KernSmooth/R/all.R` — function `dpih`, line 333
- `KernSmooth/R/all.R` — function `dpik`, line 428

```r
"minim" = min((quantile(x, 3/4) - quantile(x, 1/4)) / 1.349, sqrt(var(x)))
```

**Python Equivalent:**

```python
import numpy as np

iqr_estimate = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
std_estimate  = np.std(x, ddof=1)          # sqrt(var(x)) with ddof=1 matches R's var()
scalest_minim = min(iqr_estimate, std_estimate)
```

**Explanation:**
- R's `var(x)` uses `n - 1` (Bessel-corrected) degrees of freedom by default, so the NumPy equivalent is `np.std(x, ddof=1)`. Using the default `np.std(x)` (which applies `ddof=0`) would produce a systematically smaller value and is incorrect here.
- Both `iqr_estimate` and `std_estimate` are scalar floats, so Python's built-in `min()` is a direct equivalent of R's `min()` called on two scalars.

---

### Summary Mapping Table

| R Expression | Python Equivalent | Notes |
|---|---|---|
| `quantile(x, 1/4)` | `np.quantile(x, 0.25)` | Default `method='linear'` matches R `type=7` |
| `quantile(x, 3/4)` | `np.quantile(x, 0.75)` | Default `method='linear'` matches R `type=7` |
| `quantile(x, 3/4) - quantile(x, 1/4)` | `np.quantile(x, 0.75) - np.quantile(x, 0.25)` | IQR; both return scalar floats |
| `(...) / 1.349` | `(...) / 1.349` | No translation needed |
| `sqrt(var(x))` (paired context) | `np.std(x, ddof=1)` | `ddof=1` matches R's Bessel-corrected `var()` |
| `min(iqr_est, std_est)` | `min(iqr_est, std_est)` | Scalar `min`; no change required |
