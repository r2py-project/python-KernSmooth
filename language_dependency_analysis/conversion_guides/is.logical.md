## Conversion Guide: `is.logical` (R to Python)

---

### 1. Overview of `is.logical` in R

`is.logical` is a base R type-checking predicate. It takes a single argument and returns `TRUE` if that object is of type `logical` (i.e., R's boolean type, whose values are `TRUE`, `FALSE`, and `NA`), and `FALSE` otherwise.

- **Input:** Any R object.
- **Output:** A single logical scalar (`TRUE` or `FALSE`).
- **Typical use:** Argument validation — confirming that a parameter passed to a function is of the expected boolean type.

---

### 2. Contextual Usage Analysis

There is one usage of `is.logical` in the codebase, located in `bkde` in `KernSmooth/R/all.R` at line 30.

The full validation expression is:

```r
if (length(canonical) != 1L || !is.logical(canonical))
    stop("'canonical' must be a length-1 logical vector")
```

The parameter `canonical` has a default value of `FALSE` in the function signature (line 6), so it is expected to be a scalar boolean. The guard does two things in combination:

1. **`length(canonical) != 1L`** — rejects vectors of length other than 1.
2. **`!is.logical(canonical)`** — rejects any value that is not of logical type (e.g., integers `0`/`1`, strings `"TRUE"`/`"FALSE"`, or `NULL`).

The result is used immediately after on line 36 in a conditional expression: `if(canonical) del0 * bandwidth else bandwidth`.

---

### 3. Python Conversion Strategy

The appropriate Python equivalent is the built-in `isinstance()` function combined with Python's native `bool` type. No third-party library (NumPy, SciPy, etc.) is needed here, because:

- The R parameter `canonical` is a **scalar** boolean (default `FALSE`), not a vector.
- Python's `bool` is the direct semantic equivalent of R's `logical` scalar.
- If NumPy arrays were involved, `isinstance` can be extended to cover `np.bool_` as well.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Scalar boolean type check in `bkde`

**Locations:** `KernSmooth/R/all.R`, function `bkde`, line 30.

**Original R Context:**

```r
bkde <- function(x, kernel = "normal", canonical = FALSE, bandwidth, ...)
{
    if (length(canonical) != 1L || !is.logical(canonical))
        stop("'canonical' must be a length-1 logical vector")

    # canonical is then used as a condition:
    h <- if(canonical) del0 * bandwidth else bandwidth
}
```

**Python Equivalent:**

```python
import numpy as np

def bkde(x, kernel="normal", canonical=False, bandwidth=None, ...):
    if not isinstance(canonical, (bool, np.bool_)):
        raise TypeError("'canonical' must be a length-1 logical (bool) value")

    # canonical is then used as a condition (identical semantics):
    h = del0 * bandwidth if canonical else bandwidth
```

If the codebase needs to additionally accept a length-1 NumPy boolean array (mirroring R's allowance of a length-1 logical vector), extend the check:

```python
import numpy as np

def _is_scalar_bool(value):
    """Return True iff value is a scalar boolean (Python bool, numpy bool,
    or a length-1 numpy array of boolean dtype)."""
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray) and value.shape == (1,) and value.dtype == bool:
        return True
    return False

def bkde(x, kernel="normal", canonical=False, bandwidth=None, ...):
    if not _is_scalar_bool(canonical):
        raise TypeError("'canonical' must be a length-1 logical (bool) value")

    canonical_scalar = bool(canonical)  # normalize to plain Python bool
    h = del0 * bandwidth if canonical_scalar else bandwidth
```

**Explanation:**

| R construct | Python equivalent | Notes |
|---|---|---|
| `is.logical(canonical)` | `isinstance(canonical, (bool, np.bool_))` | R's `logical` maps to Python's `bool`. NumPy's `np.bool_` covers array-sourced boolean scalars. |
| `length(canonical) != 1L` | Covered by scalar `isinstance` check | In pure Python, a scalar `bool` has no length, so the scalar check already enforces length-1 semantics. |
| `stop(...)` | `raise TypeError(...)` or `raise ValueError(...)` | `TypeError` is most idiomatic for wrong-type arguments in Python. |
| `if(canonical) ... else ...` | `del0 * bandwidth if canonical else bandwidth` | Python's ternary expression works identically for a bool scalar. |

One key nuance: in R, integers `0L`/`1L` are **not** logical and would be rejected by `is.logical`. In Python, `bool` is a subclass of `int`, so `isinstance(1, bool)` returns `False` while `isinstance(True, bool)` returns `True` — this correctly mirrors R's strict type distinction without any extra work.
