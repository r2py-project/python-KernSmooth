## Conversion Guide: `sqrt` (R to Python)

---

### 1. Overview of `sqrt` in R

`sqrt` is a base R function that computes the non-negative square root of its argument. Its signature is:

```r
sqrt(x)
```

- **Input:** A numeric scalar, vector, matrix, or any numeric array. R automatically vectorizes the operation across all elements.
- **Output:** A numeric scalar or numeric vector/array of the same shape as the input.
- For negative inputs, R returns `NaN` with a warning. For complex inputs, a separate `sqrt.default` dispatches to complex arithmetic.

In this codebase, `sqrt` is used exclusively with non-negative real-valued inputs (mathematical constants and non-negative summary statistics).

---

### 2. Contextual Usage Analysis

Across the 43 CSV rows, `sqrt` appears in four functions (`bkde`, `dpih`, `dpik`, `dpill`) and falls into two distinct semantic categories:

**Category A — `sqrt(var(x))`: Standard deviation of a data vector**

Found in `bkde` (line 35), `dpih` (lines 331, 333), and `dpik` (lines 426, 428). The argument is `var(x)`, where `x` is a 1-D numeric vector of observed data values. `sqrt(var(x))` produces a scalar standard deviation.

**Category B — `sqrt` of scalar mathematical constants**

Found throughout `dpih`, `dpik`, and `dpill`. The arguments are small positive constants:

| Expression | Numeric value | Context |
|---|---|---|
| `sqrt(pi)` | 1.7724538… | Gaussian normalization denominators |
| `sqrt(2)` | 1.4142135… | Bandwidth pilot formulas |
| `sqrt(2/pi)` | 0.7978845… | Normal kernel derivative integrals |
| `sqrt(3)` | 1.7320508… | `C3K` kernel constant in `dpill` |
| `sqrt(2*pi)` | 2.5066282… | `C3K` denominator in `dpill` |

---

### 3. Python Conversion Strategy

The recommended Python equivalent is **`numpy.sqrt`** (`numpy` imported as `np`).

Rationale:

- `numpy.sqrt` operates identically on both scalars and arrays, matching R's native vectorization.
- `numpy` is already the standard dependency for numerical array work in Python scientific code.
- `numpy.var(x)` (with `ddof=1` to match R's default unbiased estimator) provides a direct scalar equivalent of R's `var(x)`, so `sqrt(var(x))` maps cleanly to `np.std(x, ddof=1)`.
- For the constant expressions, `numpy.sqrt` applied to `numpy.pi` or a float literal produces the same IEEE 754 double-precision result as R's `sqrt`.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 `sqrt(var(x))` — Standard deviation as scale estimate

**Locations:** `bkde` (line 35), `dpih` (lines 331, 333), `dpik` (lines 426, 428)

```r
scalest <- sqrt(var(x))
```

**Python Equivalent:**

```python
import numpy as np

# np.var with ddof=1 matches R's var() (unbiased, n-1 denominator)
scalest_stdev = np.sqrt(np.var(x, ddof=1))

# Equivalent and more idiomatic:
scalest_stdev = np.std(x, ddof=1)
```

**Explanation:**
- R's `var(x)` uses `n-1` in the denominator by default. NumPy's `np.var` defaults to `n` (population variance), so `ddof=1` is mandatory to match R's behaviour.
- `np.std(x, ddof=1)` is the most direct one-liner equivalent of `sqrt(var(x))` and is preferred for clarity.

---

#### 4.2 `sqrt(pi)` — Square root of pi

**Locations:** `dpih` (line 350), `dpik` (line 443), `dpill` (lines 532, 533, 561)

```r
(24 * sqrt(pi) / n) ^ (1/3)
(3 * gamseh / (8 * sqrt(pi))) ^ (1/7)
```

**Python Equivalent:**

```python
import numpy as np

SQRT_PI = np.sqrt(np.pi)   # 1.7724538509055159

hpi = (24 * SQRT_PI / n) ** (1/3)
gamseh = (3 * gamseh / (8 * SQRT_PI)) ** (1/7)
```

---

#### 4.3 `sqrt(2)` — Square root of 2

**Locations:** `dpih` (lines 352, 356, 362, 370, 380), `dpik` (lines 445, 448, 453, 460, 469)

```r
alpha <- (2 / (3*n))^(1/5) * sqrt(2)
alpha <- (2 * (sqrt(2))^7 / (5*n))^(1/7)
```

**Python Equivalent:**

```python
import numpy as np

SQRT2 = np.sqrt(2.0)   # 1.4142135623730951

alpha = (2 / (3 * n)) ** (1/5) * SQRT2
alpha = (2 * SQRT2**7 / (5 * n)) ** (1/7)
```

---

#### 4.4 `sqrt(2/pi)` — Square root of 2/pi

**Locations:** `dpih` (lines 358, 364, 366, 372, 374, 376, 382, 384, 386, 388), `dpik` (lines 450, 455, 457, 462, 464, 466, 471, 473, 475, 477)

```r
alpha <- (sqrt(2/pi)  / (psi4hat*n))^(1/5)
alpha <- (-3*sqrt(2/pi) / (psi6hat*n))^(1/7)
```

**Python Equivalent:**

```python
import numpy as np

SQRT2_OVER_PI = np.sqrt(2.0 / np.pi)   # 0.7978845608028654

alpha = (SQRT2_OVER_PI / (psi4hat * n)) ** (1/5)
```

**IMPORTANT:** Several of these expressions have a negative numerator (e.g., `-3 * sqrt(2/pi) / psi6hat`). In R, raising a negative number to a fractional power such as `(1/7)` returns the real-valued odd root (a negative number). Python's `**` operator raises a `ValueError` for negative bases with fractional exponents. The correct Python idiom:

```python
import math

def odd_root(x, n):
    """Compute the real-valued n-th root of x for odd integer n."""
    return math.copysign(abs(x) ** (1.0 / n), x)

# Example: (-3 * SQRT2_OVER_PI / (psi6hat * n)) ** (1/7)
val = -3 * SQRT2_OVER_PI / (psi6hat * n)
alpha = odd_root(val, 7)
```

---

#### 4.5 Module-level constant recommendations

Because all Category B usages involve pure mathematical constants that are evaluated repeatedly, the recommended Python practice is to define them once at module scope:

```python
import numpy as np

# Pre-computed constants matching R's sqrt(.) calls in KernSmooth
SQRT2      = np.sqrt(2.0)              # sqrt(2)     ~ 1.41421356
SQRT3      = np.sqrt(3.0)              # sqrt(3)     ~ 1.73205081
SQRT_PI    = np.sqrt(np.pi)            # sqrt(pi)    ~ 1.77245385
SQRT2_PI   = np.sqrt(2.0 * np.pi)     # sqrt(2*pi)  ~ 2.50662827
SQRT2_OVER_PI = np.sqrt(2.0 / np.pi)  # sqrt(2/pi)  ~ 0.79788456
```
