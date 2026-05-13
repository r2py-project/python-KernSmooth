## Conversion Guide: `dnorm` (R) to Python

---

### 1. Overview of `dnorm` in R

`dnorm` is R's standard normal probability density function (PDF). Its signature is:

```r
dnorm(x, mean = 0, sd = 1, log = FALSE)
```

- **`x`**: A numeric scalar or vector of quantile values at which the PDF is evaluated.
- **`mean`**: Mean of the normal distribution (default `0`).
- **`sd`**: Standard deviation (default `1`).
- **`log`**: If `TRUE`, returns the log of the density instead (default `FALSE`).
- **Return value**: A numeric vector of the same length as `x`, containing the probability density of the standard normal distribution evaluated at each element of `x`.

In all three usages in this codebase, `dnorm` is called with only its first positional argument, relying entirely on the defaults `mean = 0` and `sd = 1`.

---

### 2. Contextual Usage Analysis

All three call sites are in `KernSmooth/R/all.R` and follow a consistent pattern:

- The argument to `dnorm` is always a **numeric vector** produced by element-wise multiplication of an integer sequence with a scalar step factor.
- The result is immediately divided by a scalar bandwidth-derived factor as part of constructing discrete kernel weight vectors.

| File | Function | Line | Purpose |
|---|---|---|---|
| `all.R` | `bkde` | 60 | 1-D binned KDE — construct Gaussian kernel weights |
| `all.R` | `bkde2D` | 128 | 2-D binned KDE — per-dimension Gaussian kernel |
| `all.R` | `bkfe` | 212 | Kernel functional estimation — base Gaussian density |

---

### 3. Python Conversion Strategy

The direct equivalent is **`scipy.stats.norm.pdf`**:

```python
from scipy.stats import norm
norm.pdf(x)          # mean=0, sd=1 by default
```

`scipy.stats.norm.pdf` matches R's `dnorm` defaults exactly (`loc=0`, `scale=1`) and is fully vectorized over NumPy arrays. It returns a NumPy array of the same shape as the input.

An equally valid alternative using the explicit NumPy formula:

```python
import numpy as np
(1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
```

---

### 4. Step-by-Step Conversion Examples

#### 4.1 `bkde` — Standard Normal Kernel Weights for 1-D KDE (Line 60)

**Original R Context:**

```r
# lvec: integer vector 0:L
# delta: positive scalar = (b - a) / (h * (M - 1))
# n: positive integer (sample size)
# h: positive scalar (bandwidth)
kappa <- dnorm(lvec * delta) / (n * h)
```

**Python Equivalent:**

```python
import numpy as np
from scipy.stats import norm

lvec = np.arange(0, L + 1)
kappa = norm.pdf(lvec * delta) / (n * h)
```

**Explanation:**
- `0L:L` in R maps to `np.arange(0, L + 1)` in Python (inclusive upper bound in R → exclusive in Python).
- `norm.pdf(lvec * delta)` is vectorized: it receives a 1-D array and returns a 1-D array of the same length.
- The scalar division `/ (n * h)` broadcasts over the array exactly as in R.

---

#### 4.2 `bkde2D` — Standard Normal Kernel Weights for 2-D KDE (Line 128)

**Original R Context:**

```r
# Loop runs for id in {1, 2} — once per spatial dimension.
# lvecid: integer vector 0:L[id]
# facid: positive scalar = (b[id] - a[id]) / (h[id] * (M[id] - 1))
# h[id]: positive scalar bandwidth for dimension id
z <- matrix(dnorm(lvecid * facid) / h[id])
```

**Python Equivalent:**

```python
import numpy as np
from scipy.stats import norm

lvecid = np.arange(0, int(L[id]) + 1)
z = norm.pdf(lvecid * facid) / h_id          # shape: (L[id]+1,)
z = z.reshape(-1, 1)                         # column vector, matches R's matrix()
```

**Explanation:**
- R's `matrix(v)` with no `nrow`/`ncol` arguments produces a single-column matrix. The Python equivalent is `.reshape(-1, 1)` to obtain a 2-D column array.
- This column shape is important because the subsequent outer product `kapid[[1L]] %*% t(kapid[[2L]])` requires matching shapes.

---

#### 4.3 `bkfe` — Standard Normal Kernel for Derivative-Based Functional Estimation (Line 212)

**Original R Context:**

```r
# lvec: integer vector 0:L
# delta: positive scalar
# h: positive scalar bandwidth
# drv: non-negative integer (derivative order)
arg    <- lvec * delta / h
kappam <- dnorm(arg) / (h ^ (drv + 1))
```

**Python Equivalent:**

```python
import numpy as np
from scipy.stats import norm

lvec = np.arange(0, L + 1)
arg    = lvec * delta / h
kappam = norm.pdf(arg) / (h ** (drv + 1))
```

**Explanation:**
- `norm.pdf(arg)` evaluates the standard normal PDF at each element of the vector `arg`, returning an array of the same shape.
- R's `^` operator maps to Python's `**`.
