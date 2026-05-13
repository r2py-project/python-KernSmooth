## Conversion Guide: `missing` in R to Python

---

### 1. Overview of `missing` in R

`missing(x)` is a base R function that tests whether a formal argument of the enclosing function was supplied by the caller. It returns `TRUE` if the argument was **not** passed (and has no evaluated default yet), and `FALSE` if it was explicitly supplied — even if the supplied value happens to be `NULL` or `NA`.

Key properties:

- It can only be called inside a function, and its argument must be the name of a formal parameter of that function.
- It is used almost exclusively to implement **optional parameters**: parameters that appear in the function signature without a default value (`= <expr>`) but whose presence or absence alters the function's behaviour.
- When `missing(x)` is `TRUE`, accessing `x` without a guard will raise an error.

---

### 2. Contextual Usage Analysis

Across all 13 call sites in `KernSmooth/R/all.R`, `missing` is used in exactly two recurring patterns:

**Pattern A — Validation guard (check before using)**
The parameter is tested first to prevent an illegal operation from being performed on an undefined value. If the argument was not supplied, the validation is skipped entirely.

Affected sites: `bkde` line 10, `bkde2D` line 87, `bkfe` line 173, `locpoly` line 610.

**Pattern B — Compute-or-assign default**
Inside the body, `missing` determines whether to compute a derived default or to use whatever the caller passed.

Affected sites: `bkde` lines 35 and 42, `bkde2D` line 104, `bkfe` line 176, `locpoly` lines 614, 617, and 618, `sdiag` line 736, `sstdiag` line 816.

---

### 3. Python Conversion Strategy

Python has no `missing()` built-in. The standard idiomatic replacement is the **sentinel default** pattern: declare the optional parameter with a default of `None` in the function signature, then test `if param is None:` inside the body. This faithfully replicates `missing`'s semantics because:

- `None` is never a valid value for any of the numeric parameters involved (bandwidth, range.x, degree, y), so it unambiguously signals "not supplied".
- The `is None` test mirrors `missing()` exactly: it is `True` only when the caller omitted the argument.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A — Validation guard before using a numeric parameter

**Locations:** `bkde` (line 10), `bkde2D` (line 87), `bkfe` (line 173), `locpoly` (line 610)

```r
# bkde / bkfe / locpoly — scalar bandwidth
if (!missing(bandwidth) && bandwidth <= 0)
    stop("'bandwidth' must be strictly positive")

# bkde2D — vector bandwidth
if (!missing(bandwidth) && min(bandwidth) <= 0)
    stop("'bandwidth' must be strictly positive")
```

**Python Equivalent:**

```python
import numpy as np

# Scalar bandwidth (bkde, bkfe, locpoly)
def bkde(x, kernel="normal", canonical=False, bandwidth=None,
         gridsize=401, range_x=None, truncate=True):
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

# Vector bandwidth (bkde2D)
def bkde2D(x, bandwidth=None, gridsize=(51, 51), range_x=None, truncate=True):
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")
```

**Explanation:** `!missing(bandwidth) && bandwidth <= 0` becomes `bandwidth is not None and bandwidth <= 0`. The short-circuit (`&&` / `and`) ensures the value check is only performed when the argument was actually supplied — identical semantics.

---

#### 4.2 Pattern B — Compute a derived numeric default

**Locations:** `bkde` line 35 (`bandwidth`), `bkfe` line 176 (`range.x`), `locpoly` line 614 (`degree`)

```r
# bkde — default bandwidth derived from data variance
h <- if (missing(bandwidth)) del0 * (243/(35*n))^(1/5) * sqrt(var(x))
     else if (canonical) del0 * bandwidth else bandwidth

# bkfe / sdiag / sstdiag — default range from data extremes
if (missing(range.x) && !binned) range.x <- c(min(x), max(x))

# locpoly — default degree is drv + 1
if (missing(degree)) degree <- drv + 1L
else degree <- as.integer(degree)
```

**Python Equivalent:**

```python
import numpy as np

def bkde(x, kernel="normal", canonical=False, bandwidth=None,
         gridsize=401, range_x=None, truncate=True):
    n = len(x)
    if bandwidth is None:
        h = del0 * (243 / (35 * n)) ** (1/5) * np.std(x, ddof=1)
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

def bkfe(x, drv, bandwidth=None, gridsize=401, range_x=None,
         binned=False, truncate=True):
    if range_x is None and not binned:
        range_x = (np.min(x), np.max(x))

def locpoly(x, y=None, drv=0, degree=None, kernel="normal",
            bandwidth=None, gridsize=401, bwdisc=25, range_x=None,
            binned=False, truncate=True):
    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)
```

**Explanation:** The R ternary `h <- if (missing(...)) A else B` becomes a standard `if param is None: ... else: ...` block. R's `var(x)` uses `n-1` (sample variance), so the Python equivalent is `np.std(x, ddof=1)`.

---

#### 4.3 Pattern B — Compute a derived range default (with conditional nesting)

**Locations:** `bkde` line 42 (`range.x`), `locpoly` lines 617–618 (`range.x` and `y` together)

```r
if (missing(range.x)) range.x <- c(min(x) - tau*h, max(x) + tau*h)

if (missing(range.x) && !binned)
    if (missing(y)) {
        extra <- 0.05 * (max(x) - min(x))
        range.x <- c(min(x) - extra, max(x) + extra)
    } else range.x <- c(min(x), max(x))
```

**Python Equivalent:**

```python
import numpy as np

# bkde
if range_x is None:
    tau = 4 if kernel == "normal" else 1
    range_x = (np.min(x) - tau * h, np.max(x) + tau * h)

# locpoly
if range_x is None and not binned:
    if y is None:
        extra = 0.05 * (np.max(x) - np.min(x))
        range_x = (np.min(x) - extra, np.max(x) + extra)
    else:
        range_x = (np.min(x), np.max(x))
```

---

### Summary Table

| R idiom | Python equivalent | Notes |
|---|---|---|
| `missing(param)` is `TRUE` | `param is None` | Sentinel `None` default in signature |
| `!missing(param)` is `TRUE` | `param is not None` | Argument was explicitly supplied |
| `fun <- function(x, opt_param, ...)` | `def fun(x, opt_param=None, ...)` | No bare undefaulted formals in Python |
| `if (!missing(p) && p <= 0) stop(...)` | `if p is not None and p <= 0: raise ValueError(...)` | Short-circuit preserved |
| `if (missing(p)) p <- <expr>` | `if p is None: p = <expr>` | In-body default computation |
