## Conversion Guide: R `log` to Python

---

### 1. Overview of `log` in R

R's built-in `log` computes the natural logarithm (base *e*) by default. Its signature is:

```r
log(x, base = exp(1))
```

- **`x`**: A numeric scalar or vector.
- **`base`**: The logarithm base; defaults to *e*. `log(x)/log(2)` is the change-of-base formula for log base 2.
- **Return value**: A numeric scalar or vector of the same length as `x`.

In the KernSmooth codebase, `log` is always called with its default base, so every occurrence is a natural logarithm.

---

### 2. Contextual Usage Analysis

All 21 occurrences are in `KernSmooth/R/all.R` and fall into two functionally distinct patterns.

**Pattern A — Change-of-base scalar arithmetic for FFT padding (`bkde`, `bkde2D`, `bkfe`)**

```r
P <- 2^(ceiling(log(M+L+1L)/log(2)))
```

This computes the smallest power of 2 at least as large as `M+L+1`. `log(2)` is a plain scalar constant; the division `log(x)/log(2)` is the change-of-base formula giving log base 2.

**Pattern B — Logarithmically spaced bandwidth discretization (`locpoly`, `sdiag`, `sstdiag`)**

- `hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))`: generates `Q` geometrically spaced bandwidth values.
- `lhdisc <- log(hdisc)`: applies the natural log to the entire `hdisc` vector (length `Q`).
- `round(((log(bandwidth) - log(sort(bandwidth)[1L]))/gap) + 1)`: applies the natural log element-wise to `bandwidth` (a numeric vector of length `M`).

---

### 3. Python Conversion Strategy

**Primary library: `numpy`**

All R `log` calls in this codebase operate on either scalars or vectors, but the vector cases (Pattern B) are the dominant ones. `numpy.log` is a universal function (ufunc) that handles both scalars and arrays uniformly, matching R's implicit vectorization exactly. Using `math.log` would require explicit Python loops for the vector cases.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A: Scalar change-of-base for next-power-of-2 padding

**Locations:** `bkde` (line 72), `bkfe` (line 226), `bkde2D` (line 139)

```r
P <- 2^(ceiling(log(M+L+1L)/log(2)))
```

**Python Equivalent:**

```python
import numpy as np

# Scalar case (bkde / bkfe)
P = int(2 ** np.ceil(np.log2(M + L + 1)))

# Vector case (bkde2D — M and L are each length-2 numpy arrays)
P = (2 ** np.ceil(np.log2(M + L))).astype(int)
```

**Explanation:**
- `np.log(x) / np.log(2)` directly mirrors R's `log(x)/log(2)` change-of-base formula. `np.log2(x)` is the idiomatic NumPy shorthand.
- `np.ceil` maps to R's `ceiling`.
- An explicit `int(...)` cast (or `.astype(int)`) is needed before using the result as an array size.

---

#### 4.2 Pattern B-1: Logarithmically spaced bandwidth grid

**Locations:** `locpoly` (line 663), `sdiag` (line 768), `sstdiag` (line 845)

```r
hlow  <- sort(bandwidth)[1L]
hupp  <- sort(bandwidth)[M]
hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
```

**Python Equivalent:**

```python
import numpy as np

hlow = np.sort(bandwidth)[0]       # 0-based index
hupp = np.sort(bandwidth)[M - 1]   # 0-based index
hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), num=Q))
# Or equivalently:
hdisc = np.geomspace(hlow, hupp, num=Q)
```

---

#### 4.3 Pattern B-2: Natural log of a vector (recovering the log grid)

**Locations:** `locpoly` (line 671), `sdiag` (line 776), `sstdiag` (line 853)

```r
lhdisc <- log(hdisc)
gap <- (lhdisc[Q] - lhdisc[1L]) / (Q - 1)
```

**Python Equivalent:**

```python
import numpy as np

lhdisc = np.log(hdisc)            # element-wise log over the array
gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)   # 0-based indexing
```

---

#### 4.4 Pattern B-3: Element-wise log of the full bandwidth vector for index mapping

**Locations:** `locpoly` (line 674), `sdiag` (line 779), `sstdiag` (line 856)

```r
indic <- round(((log(bandwidth) - log(sort(bandwidth)[1L])) / gap) + 1)
```

**Python Equivalent:**

```python
import numpy as np

indic = np.round(
    (np.log(bandwidth) - np.log(np.sort(bandwidth)[0])) / gap + 1
).astype(int)
```

**Explanation:**
- `np.log(bandwidth)` operates element-wise over the full length-`M` array, matching R's vectorized `log(bandwidth)`.
- `np.round` uses "round half to even" (banker's rounding), which matches R's `round` behavior for ties.
- Whether to include the `+1` offset depends on whether downstream Python code uses 1-based or 0-based indices into `hdisc`. If the Fortran routines expect 1-based indices, keep `+1`; if porting to pure Python with 0-based indexing, drop it.
