## Conversion Guide: R `sum` to Python

---

### 1. Overview of `sum` in R

`sum()` is a base R function that computes the arithmetic sum of all values in one or more numeric (or logical) vectors. It returns a single scalar value. Its signature is:

```r
sum(..., na.rm = FALSE)
```

Key behaviours:

- Accepts one or more vectors (or scalars), flattening them before summing.
- When passed element-wise products such as `sum(a * b)`, it computes the dot product of two equal-length vectors.
- When passed a squared vector such as `sum(x^2)`, it computes the sum of squares.
- Returns a length-1 numeric (or integer) scalar.

---

### 2. Contextual Usage Analysis

Across the thirteen call sites in `KernSmooth/R/all.R`, `sum` is used in four distinct patterns:

| Pattern | Representative call | What is summed |
|---|---|---|
| A – plain vector sum | `sum(kappa)`, `sum(gcounts)`, `sum(Lvec)` | A 1-D numeric or integer array |
| B – weighted/dot-product sum | `sum(c(z, rev(z[-1L])) * facid * h[id])` | Element-wise product of arrays |
| C – weighted quadratic sum | `sum((mddest[llow:lupp]^2)*xcounts[llow:lupp])`, `sum(y^2)` | Sum of element-wise squares times weights |
| D – FFT-weighted sum | `sum(gcounts * (Re(fft(kappam*Gcounts, TRUE))/P)[1L:M])` | Element-wise product involving an FFT result |

---

### 3. Python Conversion Strategy

`numpy.sum()` (or equivalently `ndarray.sum()`) is the direct replacement for all usages here. The reasons are:

- All inputs are already vectors (or will be `numpy` arrays in a Python translation), so `math.sum` is not appropriate.
- `numpy.sum` reduces a 1-D array to a Python/NumPy scalar in exactly the same way R's `sum` reduces a vector.
- Element-wise vector operations (`*`, `**`) are native to NumPy arrays, so expressions like `sum(a^2 * b)` translate without restructuring into `np.sum(a**2 * b)`.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A – Plain vector sum

**Locations:** `bkde` (line 74), `bkfe` (line 198), `locpoly` (line 689), `sdiag` (line 789), `sstdiag` (line 866)

```r
tot <- sum(kappa) * (b - a) / (M - 1L) * n
n   <- sum(gcounts)
dimfkap <- 2L * sum(Lvec) + Q
```

**Python Equivalent:**

```python
import numpy as np

tot     = np.sum(kappa) * (b - a) / (M - 1) * n
n       = np.sum(gcounts)
dimfkap = 2 * np.sum(Lvec) + Q
```

**Explanation:** `np.sum(arr)` sums all elements of a NumPy array and returns a scalar, identical to R's `sum(vec)`.

---

#### 4.2 Pattern B – Weighted / dot-product sum

**Locations:** `bkde2D` (line 129), `dpill` (lines 556–557)

```r
tot    <- sum(c(z, rev(z[-1L])) * facid * h[id])
sigsqd <- n - 2*sum(Sdg*xcounts) + sum(SSTdg*xcounts)
```

**Python Equivalent:**

```python
import numpy as np

# bkde2D
z = z.ravel()   # ensure 1-D
symmetric_kernel = np.concatenate([z, z[1:][::-1]])
tot = np.sum(symmetric_kernel) * facid * h_id   # facid, h_id are scalars

# dpill
sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
sigsqn = np.sum(y**2) - 2 * np.sum(mest * ycounts) + np.sum((mest**2) * xcounts)
```

**Explanation:**
- R's `c(z, rev(z[-1L]))` is replicated by `np.concatenate([z, z[1:][::-1]])`. `z[-1L]` in R drops the first element (R uses 1-based indexing), which is `z[1:]` in Python.
- Element-wise multiplication `a * b` works identically in NumPy.

---

#### 4.3 Pattern C – Sum of squares (weighted)

**Locations:** `dpill` (line 540)

```r
th22kn <- sum((mddest[llow:lupp]^2) * xcounts[llow:lupp]) / n
```

**Python Equivalent:**

```python
import numpy as np

# Slicing: R's llow:lupp (1-based, inclusive) → Python [llow-1:lupp] (0-based, exclusive upper)
th22kn = np.sum(mddest[llow-1:lupp]**2 * xcounts[llow-1:lupp]) / n
```

**Explanation:**
- R's exponentiation operator `^` becomes `**` in Python/NumPy; `a**2` produces element-wise squares for NumPy arrays.
- R uses 1-based inclusive slicing (`llow:lupp`). In Python this is `[llow-1:lupp]`.

---

#### 4.4 Pattern D – FFT-weighted sum

**Locations:** `bkfe` (line 232)

```r
sum(gcounts * (Re(fft(kappam * Gcounts, TRUE)) / P)[1L:M]) / (n^2)
```

**Python Equivalent:**

```python
import numpy as np

# numpy.fft.ifft is the normalised inverse FFT; it absorbs the /P from R
ifft_result = np.real(np.fft.ifft(kappam * Gcounts))[:M]
result = np.sum(gcounts * ifft_result) / n**2
```

**Explanation:**
- R's `fft(..., inverse=TRUE)` computes an unnormalised inverse DFT (output = sum, not mean). `numpy.fft.ifft` is normalised (divides by N), so the division by `P` in R is absorbed automatically.
- `Re()` in R extracts the real part; `np.real()` is the direct equivalent.
- R's `[1L:M]` (first `M` elements, 1-based) becomes `[:M]` in Python.
