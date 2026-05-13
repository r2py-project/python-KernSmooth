## Conversion Guide: `mean` (R) to Python

---

### 1. Overview of `mean` in R

`mean` is a base R function that computes the arithmetic mean of a numeric vector. Its signature is:

```r
mean(x, trim = 0, na.rm = FALSE, ...)
```

- **`x`**: A numeric vector.
- **`trim`**: A fraction (0 to 0.5) of observations to trim from each end before computing the mean. Defaults to `0` (no trimming).
- **`na.rm`**: Logical; whether to strip `NA` values before computation. Defaults to `FALSE`.
- **Return value**: A single numeric scalar — the arithmetic mean of all elements in `x`.

In all usages present in this codebase, `mean` is called with a single argument (`mean(x)`), using only default parameters.

---

### 2. Contextual Usage Analysis

All four CSV entries share the same structural pattern, appearing in two functions:

**`dpih` (lines 339–340)** and **`dpik` (lines 433–434):**

```r
## Replace input data by standardised data for numerical stability:
sx <- (x - mean(x)) / scalest
sa <- (a - mean(x)) / scalest ; sb <- (b - mean(x)) / scalest
```

In both functions, `x` is the raw input data vector — a one-dimensional numeric array of observations. The call `mean(x)` computes its scalar arithmetic mean, which is then subtracted from every element of `x` (mean-centering / standardization), and also subtracted from scalar range boundary values `a` and `b`.

Key characteristics:
- `x` is always a **numeric vector** (1-D array), never a scalar.
- `mean(x)` is called **twice per function call**, once for element-wise subtraction from `x` and once for scalar subtraction from `a` and `b`.
- No `trim` or `na.rm` arguments are used; only the default arithmetic mean is needed.

---

### 3. Python Conversion Strategy

**Chosen library: `numpy`**

`numpy.mean()` is the correct Python equivalent because:

1. `x` is a numeric vector (array-like), not a scalar, so `statistics.mean` (scalar/sequence, not vectorized) is unsuitable.
2. `numpy.mean()` operates natively on NumPy arrays and returns a scalar float, matching R's behavior exactly.
3. The surrounding expressions — `(x - mean(x)) / scalest` — involve element-wise array arithmetic, which requires NumPy arrays. Using `numpy.mean()` integrates seamlessly.
4. `numpy.mean()` with default arguments computes the plain arithmetic mean with no trimming and no special `NA` handling, exactly matching `mean(x)` with R's defaults.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Mean-centering a data vector and its range boundaries

**Locations:**
- `KernSmooth/R/all.R`, function `dpih`, lines 339–340
- `KernSmooth/R/all.R`, function `dpik`, lines 433–434

**Original R Context:**

```r
sx <- (x - mean(x)) / scalest
sa <- (a - mean(x)) / scalest
sb <- (b - mean(x)) / scalest
```

**Python Equivalent:**

```python
import numpy as np

# x: np.ndarray, 1-D array of float observations
# a, b: float scalars (range boundaries)
# scalest: float scalar (scale estimate)

x_mean = np.mean(x)
sx = (x - x_mean) / scalest
sa = (a - x_mean) / scalest
sb = (b - x_mean) / scalest
```

**Explanation:**

- `np.mean(x)` computes the arithmetic mean over all elements of the 1-D NumPy array `x`, returning a scalar `float64`. This is a direct drop-in for R's `mean(x)`.
- The result is stored in `x_mean` once and reused for both array and scalar subtractions. This mirrors the two separate `mean(x)` calls in the R source (lines 339 and 340 / 433 and 434) while avoiding redundant computation.
- `x - x_mean` broadcasts the scalar subtraction across the entire array, matching R's implicit vectorization.
- No argument mapping is required beyond the function name itself: R's `mean(x)` with no extra arguments maps directly to `np.mean(x)` with no extra arguments.
- If the input data could contain `NaN` values and the intent is to ignore them (analogous to R's `mean(x, na.rm = TRUE)`), substitute `np.nanmean(x)` instead.
