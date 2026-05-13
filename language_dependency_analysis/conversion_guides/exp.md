## Conversion Guide: `exp` (R to Python)

---

### 1. Overview of `exp` in R

`exp` is a base R function that computes the natural exponential (e^x) of its input. It is fully vectorized: when given a numeric vector, it applies the operation element-wise and returns a numeric vector of the same length.

In this codebase the function never operates on a plain scalar in isolation. It is always called with the result of `seq(...)` as its argument, which produces a numeric vector, so the return value `hdisc` is always a numeric vector of length `Q`.

---

### 2. Contextual Usage Analysis

All three occurrences of `exp` are structurally identical. The surrounding pattern, reproduced across `locpoly` (line 663), `sdiag` (line 768), and `sstdiag` (line 845) in `KernSmooth/R/all.R`, is:

```r
hlow  <- sort(bandwidth)[1L]          # scalar: minimum bandwidth value
hupp  <- sort(bandwidth)[M]           # scalar: maximum bandwidth value
hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
```

The intent is to produce `Q` bandwidth values that are **logarithmically (geometrically) spaced** between `hlow` and `hupp`. Both `hlow` and `hupp` are positive scalars. `Q` is a positive integer (`bwdisc`, defaulting to 25). The output `hdisc` is a 1-D numeric vector of length `Q`.

---

### 3. Python Conversion Strategy

The direct Python equivalent is **`numpy.exp`** combined with **`numpy.linspace`**.

- `numpy.exp` is the vectorized counterpart to R's `exp`. It accepts any array-like input and applies e^x element-wise.
- `numpy.linspace(start, stop, num)` is the direct equivalent of R's `seq(start, stop, length.out = num)`.
- `math.exp` from the Python standard library is **not** appropriate here because the argument is always a vector produced by `seq`/`linspace`.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Logarithmically Spaced Bandwidth Discretisation

**Locations:**
- `KernSmooth/R/all.R`, function `locpoly`, line 663
- `KernSmooth/R/all.R`, function `sdiag`, line 768
- `KernSmooth/R/all.R`, function `sstdiag`, line 845

**Original R Context:**

```r
hlow  <- sort(bandwidth)[1L]
hupp  <- sort(bandwidth)[M]
hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
```

**Python Equivalent:**

```python
import numpy as np

# bandwidth is a 1-D numpy array of positive floats; Q is a positive int
hlow  = np.sort(bandwidth)[0]           # index 0 in Python (was [1L] in R)
hupp  = np.sort(bandwidth)[M - 1]       # index M-1 in Python (was [M] in R)
hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), num=Q))
```

**Explanation:**

| R construct | Python equivalent | Notes |
|---|---|---|
| `sort(bandwidth)[1L]` | `np.sort(bandwidth)[0]` | R uses 1-based indexing; Python uses 0-based. |
| `sort(bandwidth)[M]` | `np.sort(bandwidth)[M - 1]` | Same indexing shift. |
| `seq(start, stop, length.out = Q)` | `np.linspace(start, stop, num=Q)` | Both produce a closed-interval sequence of exactly `Q` evenly spaced values. |
| `log(x)` | `np.log(x)` | `np.log` handles both scalars and arrays. |
| `exp(vector_of_length_Q)` | `np.exp(array_of_length_Q)` | Core translation: `numpy.exp` applies e^x element-wise. |

The combined expression `np.exp(np.linspace(np.log(hlow), np.log(hupp), num=Q))` is numerically equivalent to R's `exp(seq(log(hlow), log(hupp), length.out = Q))` and produces a geometrically spaced grid between `hlow` and `hupp`.

This pattern is also available as `numpy.geomspace(hlow, hupp, num=Q)`, which is a convenient shorthand for exactly this pattern and can be used as a direct one-line replacement:

```python
hdisc = np.geomspace(hlow, hupp, num=Q)
```
