## Conversion Guide: `sort` in R to Python

---

### 1. Overview of `sort` in R

`sort` is a base R function that returns a sorted copy of a vector in ascending order by default. Its signature is:

```r
sort(x, decreasing = FALSE, na.last = NA, ...)
```

- **Input:** A vector (numeric, character, or logical). In the usages examined here, the input is always a numeric vector.
- **Output:** A new numeric vector of the same length as `x`, sorted in ascending order. The original vector is not modified.
- **Key behaviours:**
  - `decreasing = FALSE` (default) sorts lowest-to-highest.
  - `na.last = NA` (default) removes `NA` values from the result rather than placing them at either end.
  - The function is fully vectorized; it operates on and returns a vector, never a scalar.

---

### 2. Contextual Usage Analysis

All nine usages of `sort` appear in `KernSmooth/R/all.R` and are confined to three functions — `locpoly`, `sdiag`, and `sstdiag` — which share identical bandwidth-discretisation logic. The three call patterns are:

| Pattern | Lines (locpoly / sdiag / sstdiag) | Purpose |
|---|---|---|
| `sort(bandwidth)[1L]` | 661 / 766 / 843 | Extract the minimum bandwidth value (`hlow`) |
| `sort(bandwidth)[M]` | 662 / 767 / 844 | Extract the maximum bandwidth value (`hupp`) |
| `sort(bandwidth)[1L]` (inside `indic`) | 674 / 779 / 856 | Retrieve the minimum bandwidth again when computing the log-spacing index array |

In every case `bandwidth` is a numeric vector of length `M`. The results of `sort(bandwidth)[1L]` and `sort(bandwidth)[M]` are scalars used as the lower and upper bounds of a logarithmically spaced sequence.

---

### 3. Python Conversion Strategy

The primary Python equivalent is **NumPy**. `bandwidth` maps to a `numpy.ndarray` in Python, and NumPy provides:

- `numpy.sort(a)` — returns a sorted copy of an array, directly mirroring R's `sort()`.
- `numpy.min(a)` / `numpy.max(a)` — more idiomatic alternatives when only the extreme values are needed, which is the sole purpose of `sort` in this codebase.

Because the only elements ever accessed after sorting are index `[0]` (minimum) and `[-1]` (maximum), using `np.min` / `np.max` is both more readable and more efficient (O(n) vs O(n log n)).

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Extracting the minimum bandwidth (`hlow`)

**Locations:** `locpoly` line 661, `sdiag` line 766, `sstdiag` line 843.

```r
hlow <- sort(bandwidth)[1L]   # scalar double — the minimum
```

**Python Equivalent:**

```python
import numpy as np

hlow = np.min(bandwidth)          # preferred: O(n), returns a scalar float

# Alternative that directly mirrors the R idiom:
hlow = np.sort(bandwidth)[0]      # np.sort returns a sorted copy; [0] is the first element
```

**Explanation:** R uses 1-based indexing, so `[1L]` selects the first (smallest) element. Python uses 0-based indexing, so the equivalent subscript is `[0]`.

---

#### 4.2 Extracting the maximum bandwidth (`hupp`)

**Locations:** `locpoly` line 662, `sdiag` line 767, `sstdiag` line 844.

```r
hupp <- sort(bandwidth)[M]    # scalar double — the maximum
```

**Python Equivalent:**

```python
import numpy as np

M = len(bandwidth)
hupp = np.max(bandwidth)          # preferred: O(n), returns a scalar float

# Alternative:
hupp = np.sort(bandwidth)[M - 1]  # Python index M-1 == R index M (last element)
```

---

#### 4.3 Re-using the minimum bandwidth inside the log-spacing index computation

**Locations:** `locpoly` line 674, `sdiag` line 779, `sstdiag` line 856.

```r
indic <- round(
    ((log(bandwidth) - log(sort(bandwidth)[1L])) / gap) + 1
)
```

**Python Equivalent:**

```python
import numpy as np

indic = np.round(
    ((np.log(bandwidth) - np.log(np.min(bandwidth))) / gap) + 1
).astype(int)
```

**Explanation:**
- `sort(bandwidth)[1L]` is again replaced by `np.min(bandwidth)`.
- `np.log()` and `np.round()` operate element-wise on arrays, matching R's implicit vectorization.
- `.astype(int)` applies the integer cast that R's `round()` applies implicitly when the result is used as an index.

---

### Summary Table

| R expression | Python equivalent | Notes |
|---|---|---|
| `sort(bandwidth)[1L]` | `np.min(bandwidth)` | Minimum of a numeric vector |
| `sort(bandwidth)[M]` | `np.max(bandwidth)` | Maximum; `M` == `len(bandwidth)` |
| `sort(bandwidth)` (full sorted copy) | `np.sort(bandwidth)` | Ascending by default in both |
| `sort(bandwidth, decreasing=TRUE)` | `np.sort(bandwidth)[::-1]` | Descending; not used here but provided for completeness |
