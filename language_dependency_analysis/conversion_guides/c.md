## Conversion Guide: R `c()` to Python

---

### 1. Overview of `c` in R

R's `c()` (short for "combine" or "concatenate") is one of the most fundamental functions in the language. Its purpose is to combine its arguments into a single atomic vector (or list). Key behavioral properties:

- **Type coercion:** When mixing types, R coerces all elements to the most general type.
- **Flattening:** Nested `c()` calls are automatically flattened — `c(c(1,2), 3)` yields `c(1, 2, 3)`, a length-3 vector.
- **Vectorized output:** The result is always a flat, 1-D vector, never a multi-dimensional structure.

In the KernSmooth codebase `c()` plays three distinct roles:

1. **Constructing a fixed-length vector of string literals** (used as the allowed-values argument to `match.arg`).
2. **Constructing a two-element numeric vector** from two scalar expressions (ranges, coordinate pairs, paired scalars `sa`/`sb`).
3. **Concatenating a computed numeric vector with padding/reversed tail** to build a zero-padded, symmetrically-wrapped array for FFT convolution.

---

### 2. Contextual Usage Analysis

The 51 CSV rows reduce to four functionally distinct patterns:

| Pattern | Representative lines | Description |
|---|---|---|
| A – String vector for `match.arg` | 14, 403 | A character vector of kernel name literals |
| B – Two-scalar numeric vector | 42, 83, 100, 107, 110, 111, 176, 353–389, 428, 446–478, 736, 816 | Builds a length-2 `[min, max]`-style double vector from two scalars |
| C – Multi-part FFT padding | 73–75, 227–228, 129 | Concatenates a computed vector, a zero-padded middle section, and a reversed tail |
| D – Column extraction / interleave | 585 | Vertically concatenates two matrix columns into a single flat vector |

---

### 3. Python Conversion Strategy

**Primary library: `numpy`**

R vectors are the direct conceptual equivalent of 1-D NumPy arrays. `numpy` is chosen because:

- `numpy.array([...])` and `numpy.concatenate([...])` exactly replicate R's flat-vector semantics.
- All downstream consumers of these vectors in the translated code will be NumPy-based.
- For the string-list pattern (Pattern A), a plain Python `list` of strings is appropriate since no numeric computation is involved.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A — String Vector for Kernel-Name Validation

**Locations:** `bkde` (line 14), `dpik` (line 403)

```r
kernel <- match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
```

**Python Equivalent:**

```python
VALID_KERNELS = ["normal", "box", "epanech", "biweight", "triweight"]

if kernel not in VALID_KERNELS:
    raise ValueError(f"'kernel' must be one of {VALID_KERNELS}; got {kernel!r}")
```

**Explanation:** R's `c()` of strings maps directly to a Python `list` of strings.

---

#### 4.2 Pattern B — Two-Scalar Numeric Vector (Range / Coordinate Pair)

**Locations:** `bkde` (line 42), `bkde2D` (lines 100, 107, 110, 111), `bkfe` (line 176), `dpih` (lines 353–389), `dpik` (lines 446–478), `sdiag` (line 736), `sstdiag` (line 816)

```r
range.x <- c(min(x) - tau*h, max(x) + tau*h)
a <- range.x[1L]
b <- range.x[2L]
```

**Python Equivalent:**

```python
import numpy as np

range_x = np.array([np.min(x) - tau * h, np.max(x) + tau * h])
a = range_x[0]   # R's [1L] → Python [0]
b = range_x[1]   # R's [2L] → Python [1]
```

**Explanation:** R's 1-based `[1L]` / `[2L]` subscripts become 0-based `[0]` / `[1]` in Python.

---

#### 4.3 Pattern C — FFT Zero-Padding / Symmetric Wrap-Around Concatenation

**Locations:** `bkde` (lines 73–75), `bkfe` (lines 227–228), `bkde2D` (line 129)

```r
P <- 2^(ceiling(log(M + L + 1L) / log(2)))
kappa   <- c(kappa, rep(0, P-2L*L-1L), rev(kappa[-1L]))
gcounts <- c(gcounts, rep(0L, P-M))
```

**Python Equivalent:**

```python
import numpy as np

P = int(2 ** np.ceil(np.log2(M + L + 1)))

kappa = np.concatenate([
    kappa,
    np.zeros(P - 2*L - 1),
    kappa[1:][::-1]          # rev(kappa[-1L]): all but first, reversed
])

gcounts = np.concatenate([
    gcounts,
    np.zeros(P - M, dtype=int)
])
```

**Explanation:**
- R's `c(vec1, vec2, vec3)` with multiple vector arguments maps to `numpy.concatenate([arr1, arr2, arr3])`.
- R's `rep(0, k)` → `numpy.zeros(k)`.
- R's `rev(kappa[-1L])` removes the first element and reverses: in Python this is `kappa[1:][::-1]`. R's negative indexing (`-1L`) means "exclude element at position 1", which corresponds to Python's `[1:]` (skip index 0).

---

#### 4.4 Pattern D — Matrix Column Extraction into Flat Vector

**Locations:** `linbin2D` (line 585)

```r
n <- nrow(X)
X <- c(X[, 1L], X[, 2L])
# X is now a numeric vector of length 2n
```

**Python Equivalent:**

```python
import numpy as np

n = X.shape[0]
X_flat = np.concatenate([X[:, 0], X[:, 1]])
# X_flat is a 1-D array of length 2*n
```

**Explanation:**
- R's column selector `X[, 1L]` (1-based) becomes `X[:, 0]` in Python (0-based).
- `np.hstack` is an equivalent alternative: `np.hstack([X[:, 0], X[:, 1]])`.
