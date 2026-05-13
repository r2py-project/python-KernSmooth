## Conversion Guide: `dbeta` (R) to Python

---

### 1. Overview of `dbeta` in R

`dbeta(x, shape1, shape2, ncp = 0, log = FALSE)` is part of R's built-in `stats` package. It computes the **probability density function (PDF) of the Beta distribution** evaluated at `x`.

- **`x`**: A numeric vector of values at which to evaluate the density. Values outside `[0, 1]` return 0.
- **`shape1`** (`alpha`): First positive shape parameter of the Beta distribution.
- **`shape2`** (`beta`): Second positive shape parameter of the Beta distribution.
- **`ncp`**: Non-centrality parameter (defaults to 0, i.e., the standard central Beta distribution).
- **`log`**: If `TRUE`, returns the log of the density (defaults to `FALSE`).

The PDF is defined as: `f(x; alpha, beta) = x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta)`. `dbeta` is fully vectorized in R.

---

### 2. Contextual Usage Analysis

All four usages occur in the `bkde` function in `KernSmooth/R/all.R`, lines 62–68. They form a mutually exclusive `if/else if` chain that selects a kernel weighting function based on the `kernel` argument:

| Line | Kernel name | `shape1` | `shape2` | Effective kernel shape |
|------|-------------|----------|----------|----------------------|
| 62 | `"box"` | 1 | 1 | Uniform (Beta(1,1) = flat) |
| 64 | `"epanech"` | 2 | 2 | Epanechnikov (parabolic) |
| 66 | `"biweight"` | 3 | 3 | Biweight (quartic) |
| 68 | `"triweight"` | 4 | 4 | Triweight |

The argument to `dbeta` is always `0.5*(lvec*delta+1)`, where `lvec` is an integer vector `0:L` and `delta` is a positive scalar. The result is always a **numeric vector**, not a scalar.

---

### 3. Python Conversion Strategy

The primary Python equivalent is **`scipy.stats.beta.pdf(x, a, b)`** from `scipy.stats`:

- Accepts NumPy arrays as input and returns NumPy arrays (fully vectorized).
- Maps directly to R's `dbeta(x, shape1, shape2)` with the same argument order.
- Handles boundary values (`x` outside `[0, 1]`) by returning 0, matching R's behavior.

---

### 4. Step-by-Step Conversion Examples

Because all four usages follow an identical structural pattern — differing only in the integer shape parameters — they share one unified translation:

**Original R Context (all four usages follow this pattern):**

```r
# lvec: integer vector 0:L
# delta: positive scalar
# n: positive integer (sample size)
# h: positive scalar (bandwidth)

kappa <- 0.5 * dbeta(0.5*(lvec*delta + 1), shape1, shape2) / (n*h)
```

Where `shape1` and `shape2` are 1, 2, 3, or 4 for box, epanech, biweight, triweight respectively.

**Python Equivalent (unified dispatch):**

```python
import numpy as np
from scipy.stats import beta as beta_dist

# kernel_shape_params maps kernel name -> (shape1, shape2)
KERNEL_BETA_PARAMS = {
    "box":       (1, 1),
    "epanech":   (2, 2),
    "biweight":  (3, 3),
    "triweight": (4, 4),
}

lvec = np.arange(0, L + 1)               # integer array, equivalent to R's 0L:L
x_eval = 0.5 * (lvec * delta + 1)        # transform to Beta support [0, 1]

a, b = KERNEL_BETA_PARAMS[kernel]
kappa = 0.5 * beta_dist.pdf(x_eval, a, b) / (n * h)
```

**Explanation:**
- `scipy.stats.beta.pdf` is the one-to-one replacement for `dbeta`; argument order `(x, a, b)` matches R's `dbeta(x, shape1, shape2)`.
- The `log=FALSE` default in R is already the default behavior of `scipy.stats.beta.pdf`. If the log density were needed, use `beta_dist.logpdf(x_eval, a, b)`.
- R's `0L:L` becomes `np.arange(0, L + 1)` in Python (R's `:` operator is inclusive on both ends; Python's `arange` is exclusive at the upper bound).
- No zero-based/one-based indexing differences affect this computation because the `lvec` array is consumed as arithmetic values, not as indices.
