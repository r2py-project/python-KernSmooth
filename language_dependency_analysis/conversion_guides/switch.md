## Conversion Guide: R `switch` to Python

---

### 1. Overview of `switch` in R

`switch` is a control-flow function that evaluates a single expression (the EXPR argument) and dispatches to one of several named branches, returning the value of the matching branch. Its signature is:

```r
switch(EXPR, ...)
```

Where `EXPR` is typically a character string, and the remaining named arguments form a lookup table mapping possible string values to return expressions. The return value is whatever expression the matching branch evaluates to — a scalar numeric in every usage here.

---

### 2. Contextual Usage Analysis

All four usages appear in `KernSmooth/R/all.R` and fall into exactly two functional patterns:

**Pattern A — Kernel canonical scaling factor (`del0`).**
Appears in `bkde` (line 23) and `dpik` (line 406). The switch key is `kernel`, a single character string validated upstream by `match.arg`. Each branch computes a specific floating-point constant used as a canonical bandwidth scaling factor.

**Pattern B — Scale estimate (`scalest`).**
Appears in `dpih` (line 330) and `dpik` (line 425). The switch key is `scalest`, a single character string validated upstream by `match.arg`. Each branch computes a scalar numeric derived from the input data vector `x`.

---

### 3. Python Conversion Strategy

The direct Python equivalent for R's character-keyed `switch` is a **dictionary lookup**. Since each branch returns a scalar `float` constant or a data-derived scalar computed from a NumPy array, a `dict` mapping strings to values is the idiomatic and efficient replacement.

A `dict`-based approach is preferred over `if/elif/else` chains because it mirrors the semantics of R's `switch` directly: it is a named table lookup with O(1) dispatch.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A — Kernel Canonical Scaling Factor

**Locations:**
- `KernSmooth/R/all.R` — function `bkde`, line 23
- `KernSmooth/R/all.R` — function `dpik`, line 406

```r
del0 <- switch(kernel,
               "normal"    = (1/(4*pi))^(1/10),
               "box"       = (9/2)^(1/5),
               "epanech"   = 15^(1/5),
               "biweight"  = 35^(1/5),
               "triweight" = (9450/143)^(1/5))
```

**Python Equivalent:**

```python
import math

_KERNEL_DEL0 = {
    "normal":    (1.0 / (4.0 * math.pi)) ** (1.0 / 10.0),
    "box":       (9.0 / 2.0) ** (1.0 / 5.0),
    "epanech":   15.0 ** (1.0 / 5.0),
    "biweight":  35.0 ** (1.0 / 5.0),
    "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
}

# bkde usage
del0 = _KERNEL_DEL0[kernel]

# dpik usage (respects the `canonical` guard)
del0 = 1.0 if canonical else _KERNEL_DEL0[kernel]
```

**Explanation:**
- R's named `switch` branches map one-to-one to dictionary key-value pairs. Defining the dictionary as a module-level constant avoids recomputing the same fixed constants on every function call.
- `math.pi` replaces R's `pi`. The exponent operators `^` in R become `**` in Python.
- The `(1/(4*pi))^(1/10)` in `bkde` and `1/((4*pi)^(1/10))` in `dpik` are algebraically identical, so both map to the same Python expression.
- The `if (canonical) 1 else switch(...)` guard in `dpik` translates directly to a Python ternary expression.

---

#### 4.2 Pattern B — Scale Estimate from Data

**Locations:**
- `KernSmooth/R/all.R` — function `dpih`, line 330
- `KernSmooth/R/all.R` — function `dpik`, line 425

```r
scalest <- switch(scalest,
                  "stdev"  = sqrt(var(x)),
                  "iqr"    = (quantile(x, 3/4) - quantile(x, 1/4)) / 1.349,
                  "minim"  = min((quantile(x, 3/4) - quantile(x, 1/4)) / 1.349, sqrt(var(x))))
```

**Python Equivalent:**

```python
import numpy as np

def _compute_scalest(scalest: str, x) -> float:
    stdev = float(np.std(x, ddof=1))                              # R var(x) uses ddof=1
    iqr   = float(np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349

    scale_map = {
        "stdev":  stdev,
        "iqr":    iqr,
        "minim":  min(iqr, stdev),
    }
    return scale_map[scalest]

# Usage
scalest_val = _compute_scalest(scalest, x)
if scalest_val == 0:
    raise ValueError("scale estimate is zero for input data")
```

**Explanation:**
- R's `var(x)` computes the sample variance with `n-1` in the denominator. The Python equivalent is `np.std(x, ddof=1)` (or `np.sqrt(np.var(x, ddof=1))`).
- R's `quantile(x, 3/4)` uses type 7 interpolation by default, which matches `np.quantile`'s default `method='linear'`.
- The `min(...)` in the `"minim"` branch operates on two scalars, so Python's built-in `min` is correct.
- All three branch values are computed eagerly before the dictionary is constructed. This is safe and correct since all three expressions are cheap scalar operations.

**Alternative: lazy evaluation with lambdas (for expensive branches):**

```python
scale_map = {
    "stdev":  lambda: float(np.std(x, ddof=1)),
    "iqr":    lambda: float(np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
    "minim":  lambda: min(
                  float(np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
                  float(np.std(x, ddof=1))
              ),
}
scalest_val = scale_map[scalest]()
```

This matches R's `switch` exactly in evaluation semantics: only the selected branch's expression is computed.
