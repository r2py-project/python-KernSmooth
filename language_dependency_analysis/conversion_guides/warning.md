## Conversion Guide: `warning` in R to Python

---

### 1. Overview of `warning` in R

`warning()` is a base R function that generates a diagnostic warning message during function execution. It does **not** halt execution — control continues to the next statement after the call. The function accepts one or more character strings that are concatenated into a single message. Its signature is:

```r
warning(..., call. = TRUE, immediate. = FALSE, noBreaks. = FALSE, domain = NULL)
```

Key behaviour relevant to these usages:
- A single string argument is the most common form.
- Execution continues after the call; it is a soft signal, not an error.
- The default attaches the calling function's name to the message (`call. = TRUE`).

---

### 2. Contextual Usage Analysis

All three occurrences in `KernSmooth/R/all.R` are structurally identical: a single scalar integer `L` (or in `bkde2D`, a 2-element integer vector) is checked against zero immediately after being computed from the bandwidth and grid parameters. If the grid is too coarse relative to the bandwidth, `L` evaluates to `0`, and the warning fires. Execution always continues regardless of whether the warning was issued.

| Location | Function | Trigger condition | `L` type |
|---|---|---|---|
| line 56 | `bkde` | `L == 0` | scalar integer |
| line 135 | `bkde2D` | `min(L) == 0` | 2-element numeric vector |
| line 207 | `bkfe` | `L == 0` | scalar integer |

The message string is identical across all three sites:

> `"Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"`

---

### 3. Python Conversion Strategy

The correct Python equivalent is the `warnings` standard-library module, specifically `warnings.warn()`. This maps precisely to R's `warning()` in every important respect:

- Execution continues after the call (unlike `raise`, which aborts).
- The message is a plain string.
- The default warning category `UserWarning` is the appropriate semantic match for a soft user-facing diagnostic.
- Python's warning filter infrastructure provides the same kind of opt-in suppression / promotion that R's `options(warn = ...)` provides.

`numpy` or `scipy` are not relevant here — `warning` is a control-flow/diagnostic primitive, not a mathematical operation.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Scalar grid-coarseness check — `bkde` and `bkfe`

**Locations:** `KernSmooth/R/all.R` — functions `bkde` (line 56) and `bkfe` (line 207)

```r
# bkde / bkfe (representative pattern)
if (L == 0)
    warning("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
```

**Python Equivalent:**

```python
import warnings
import math

L = min(int(math.floor(tau / delta)), M)

if L == 0:
    warnings.warn(
        "Binning grid too coarse for current (small) bandwidth: "
        "consider increasing 'gridsize'",
        UserWarning,
        stacklevel=2,
    )
```

**Explanation:**
- `warnings.warn(message, category, stacklevel)` is a direct drop-in for R's `warning(message)`.
- `UserWarning` is the standard category for application-level soft diagnostics.
- `stacklevel=2` makes the warning point to the *caller* of the enclosing Python function rather than to the `warnings.warn` line itself, mirroring R's `call. = TRUE` default.
- Execution falls through to the next statement exactly as in R.

---

#### 4.2 Vectorised (2-D) grid-coarseness check — `bkde2D`

**Location:** `KernSmooth/R/all.R` — function `bkde2D` (line 134–135)

```r
if (min(L) == 0)
    warning("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
```

**Python Equivalent:**

```python
import numpy as np
import warnings

# L is a numpy array of length 2 (integer values)
if np.min(L) == 0:
    warnings.warn(
        "Binning grid too coarse for current (small) bandwidth: "
        "consider increasing 'gridsize'",
        UserWarning,
        stacklevel=2,
    )
```

**Explanation:**
- `np.min(L)` reduces the short vector to a scalar for the comparison. R's `min(L)` on a vector maps to `np.min(L)`.
- The `warnings.warn` call itself is identical to the scalar case.

---

### Summary: R-to-Python mapping table

| R construct | Python equivalent | Notes |
|---|---|---|
| `warning("msg")` | `warnings.warn("msg", UserWarning, stacklevel=2)` | Non-fatal; execution continues |
| `call. = TRUE` (default) | `stacklevel=2` | Points warning at caller, not at `warn` line |
| `options(warn = 2)` — treat as error | `warnings.filterwarnings("error", category=UserWarning)` | Promotes to exception |
| `suppressWarnings(expr)` | `with warnings.catch_warnings(): warnings.simplefilter("ignore")` | Suppresses within a block |
