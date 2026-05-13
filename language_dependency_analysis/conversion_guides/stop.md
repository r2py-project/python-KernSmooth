## Conversion Guide: `stop` in R to Python

---

### 1. Overview of `stop` in R

`stop()` is a base R function that halts execution of the current expression or function and raises an error condition. It takes one or more character strings (which are concatenated into a single error message) and optionally a `call.` argument (default `TRUE`) that controls whether the call stack is included in the error message. When triggered, execution is immediately interrupted and the message is surfaced to the caller.

Signature:
```r
stop(..., call. = TRUE, domain = NULL)
```

Key properties:
- Unconditionally aborts the calling function.
- The message is always a character string.
- Used exclusively for input validation and fatal runtime errors (not warnings).

---

### 2. Contextual Usage Analysis

All thirteen `stop` calls in `KernSmooth/R/all.R` are pure **input-validation guards**. Three distinct message patterns appear:

| Pattern | Message text | Functions affected |
|---|---|---|
| A | `'bandwidth' must be strictly positive` | `bkde` (line 11), `bkde2D` (line 88), `bkfe` (line 174), `locpoly` (line 611) |
| B | `'bandwidth' must be a scalar or an array of length 'gridsize'` | `locpoly` (line 682), `sdiag` (line 787), `sstdiag` (line 864) |
| C | `Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'` | `locpoly` (line 685) |
| D | `Level should be between 0 and 5` | `dpih` (line 313), `dpik` (line 400) |
| E | `scale estimate is zero for input data` | `dpih` (line 335), `dpik` (line 430) |
| F | `'canonical' must be a length-1 logical vector` | `bkde` (line 31) |

---

### 3. Python Conversion Strategy

The direct Python equivalent of `stop("message")` is `raise ValueError("message")`. `ValueError` is the conventional built-in exception for signalling that a function received an argument of the correct type but an inappropriate value — exactly the semantic carried by every `stop` call in this codebase.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A — Bandwidth must be strictly positive (scalar)

```r
if (!missing(bandwidth) && bandwidth <= 0)
    stop("'bandwidth' must be strictly positive")
```

**Python Equivalent:**

```python
def bkde(x, kernel="normal", canonical=False, bandwidth=None,
         gridsize=401, range_x=None, truncate=True):
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")
```

---

#### 4.2 Pattern A — Bandwidth must be strictly positive (vector)

```r
if (!missing(bandwidth) && min(bandwidth) <= 0)
    stop("'bandwidth' must be strictly positive")
```

**Python Equivalent:**

```python
import numpy as np

def bkde2d(x, bandwidth=None, gridsize=(51, 51), range_x=None, truncate=True):
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")
```

---

#### 4.3 Pattern B — Bandwidth must be scalar or length-`gridsize` array

```r
if (length(bandwidth) == M) {
    # ... vector bandwidth logic
} else if (length(bandwidth) == 1L) {
    # ... scalar bandwidth logic
} else
    stop("'bandwidth' must be a scalar or an array of length 'gridsize'")
```

**Python Equivalent:**

```python
import numpy as np

bandwidth = np.atleast_1d(bandwidth)

if len(bandwidth) == M:
    pass   # vector bandwidth path
elif len(bandwidth) == 1:
    pass   # scalar bandwidth path
else:
    raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")
```

---

#### 4.4 Pattern D — Level must be between 0 and 5

```r
if (level > 5L) stop("Level should be between 0 and 5")
```

**Python Equivalent:**

```python
def dpih(x, scalest="minim", level=2, gridsize=401, range_x=None, truncate=True):
    if level > 5:
        raise ValueError("Level should be between 0 and 5")
```

---

#### 4.5 Pattern E — Scale estimate is zero

```r
if (scalest == 0) stop("scale estimate is zero for input data")
```

**Python Equivalent:**

```python
if scalest == 0:
    raise ValueError("scale estimate is zero for input data")
```

---

#### 4.6 Pattern F — `canonical` must be a length-1 logical vector

```r
if (length(canonical) != 1L || !is.logical(canonical))
    stop("'canonical' must be a length-1 logical vector")
```

**Python Equivalent:**

```python
if not isinstance(canonical, bool):
    raise ValueError("'canonical' must be a length-1 logical vector")
```

---

### Summary Table

| R call | Python equivalent | Notes |
|---|---|---|
| `stop("'bandwidth' must be strictly positive")` | `raise ValueError(...)` | Scalar comparison; `np.min()` for vector form |
| `stop("'bandwidth' must be a scalar or an array of length 'gridsize'")` | `raise ValueError(...)` | Trailing `else` in `if/elif/else` chain |
| `stop("Binning grid too coarse...")` | `raise ValueError(...)` | Condition uses `np.min()` on integer array |
| `stop("Level should be between 0 and 5")` | `raise ValueError(...)` | Pure scalar integer comparison |
| `stop("scale estimate is zero for input data")` | `raise ValueError(...)` | Pure scalar float comparison |
| `stop("'canonical' must be a length-1 logical vector")` | `raise ValueError(...)` | Type check via `isinstance` |
