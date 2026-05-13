## Conversion Guide: R `ceiling` to Python

---

### 1. Overview of `ceiling` in R

`ceiling` is a base R mathematical function that rounds a numeric value **up** to the nearest integer — it returns the smallest integer that is greater than or equal to the input. It is the ceiling function in the mathematical sense.

- **Input:** A numeric scalar or numeric vector (integer or double).
- **Output:** A numeric vector of the same length as the input.
- **Vectorized:** Yes. When passed a vector, `ceiling` is applied element-wise, returning a vector of equal length.
- **Key distinction from `round`:** `ceiling` always rounds toward positive infinity regardless of the fractional part. For example, `ceiling(2.1)` returns `3`, not `2`.

---

### 2. Contextual Usage Analysis

All three usages of `ceiling` follow an identical pattern — computing the **smallest power of 2 that is greater than or equal to a derived sum** — which is a standard technique for FFT (Fast Fourier Transform) padding. The general form is:

```r
P <- 2^(ceiling(log(M+L+1L)/log(2)))
# or (bkde2D):
P <- 2^(ceiling(log(M+L)/log(2)))
```

This is equivalent to computing `2^ceil(log2(n))` — the next power of two at or above `n`.

| Location | `M` type | `L` type | Result `P` type |
|---|---|---|---|
| `bkde` (line 72) | Integer scalar (`gridsize`, default `401L`) | Integer scalar | Integer scalar |
| `bkde2D` (line 139) | Integer vector of length 2 (`gridsize = c(51L, 51L)`) | Numeric vector of length 2 | Numeric vector of length 2 |
| `bkfe` (line 226) | Integer scalar | Integer scalar | Integer scalar |

---

### 3. Python Conversion Strategy

The primary Python equivalent is **`numpy.ceil`**.

1. **Vectorization:** `numpy.ceil` operates element-wise on scalars, arrays, and any array-like input, matching R's inherent vectorization. In `bkde2D`, where `M` and `L` are 2-element arrays, `numpy.ceil` handles this natively without any code change.
2. **Return type:** `numpy.ceil` returns a `numpy.ndarray` (or a numpy scalar for scalar input), which integrates directly with subsequent numpy/scipy array operations including FFT.
3. **Logarithm base-2:** The R expression `log(x)/log(2)` (change-of-base formula) has a direct, more readable numpy equivalent: `numpy.log2(x)`.
4. **Next-power-of-2 idiom:** The full pattern `2^ceil(log2(n))` maps cleanly to `2**int(np.ceil(np.log2(n)))` for scalars or `2**np.ceil(np.log2(n)).astype(int)` for arrays.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Scalar case: `bkde` (line 72) and `bkfe` (line 226)

**Original R Context:**

```r
# M: integer scalar, L: integer scalar
P <- 2^(ceiling(log(M + L + 1L) / log(2)))
```

**Python Equivalent:**

```python
import numpy as np

# M: int, L: int
P = int(2 ** np.ceil(np.log2(M + L + 1)))
```

**Explanation:**
- `log(x)/log(2)` in R is replaced by the more idiomatic `np.log2(x)`.
- `ceiling(...)` maps directly to `np.ceil(...)`. `np.ceil` returns a `float64` numpy scalar, so wrapping with `int(...)` produces a plain Python integer for use as an array length.
- The `1L` suffix in R is simply an integer literal; it becomes `1` in Python.

---

#### 4.2 Vector case: `bkde2D` (line 139)

**Original R Context:**

```r
# M: integer vector length 2, L: numeric vector length 2
P <- 2^(ceiling(log(M + L) / log(2)))   # smallest powers of 2 >= M+L
P1 <- P[1L]
P2 <- P[2L]
```

**Python Equivalent:**

```python
import numpy as np

# M: np.ndarray of shape (2,) or list of 2 ints
# L: np.ndarray of shape (2,) or list of 2 floats
M = np.array(M)
L = np.array(L)

P = (2 ** np.ceil(np.log2(M + L))).astype(int)
P1 = P[0]
P2 = P[1]
```

**Explanation:**
- Because `M` and `L` are 2-element arrays, `np.log2` and `np.ceil` both apply element-wise automatically — no loop is required.
- `.astype(int)` converts the float array returned by `np.ceil` to an integer array.
- R uses 1-based indexing (`P[1L]`, `P[2L]`), which maps to Python's 0-based indexing (`P[0]`, `P[1]`).
- Note the absence of the `+1` offset here compared to the 1D cases, reflecting a different padding geometry for the 2D wrap-around kernel.

---

### Summary of the Core Translation

| R idiom | Python equivalent | Notes |
|---|---|---|
| `ceiling(x)` | `np.ceil(x)` | Element-wise for both scalars and arrays |
| `log(x)/log(2)` | `np.log2(x)` | Idiomatic base-2 log |
| `2^(ceiling(log(x)/log(2)))` scalar | `int(2 ** np.ceil(np.log2(x)))` | Cast to `int` for use as array size |
| `2^(ceiling(log(x)/log(2)))` vector | `(2 ** np.ceil(np.log2(x))).astype(int)` | Preserves element-wise vectorization |
| `P[1L]`, `P[2L]` (1-based) | `P[0]`, `P[1]` (0-based) | Index shift of 1 |
