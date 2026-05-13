## Conversion Guide: `ncol` (R to Python)

---

### 1. Overview of `ncol` in R

`ncol` is a base R function that returns the number of columns of a matrix or data frame. It takes a single argument — any 2D object such as a `matrix`, `data.frame`, or `array` — and returns a single integer scalar. For 1D vectors it returns `NULL`. It is the column-axis counterpart of `nrow`.

```r
ncol(x)  # returns an integer: the number of columns of x
```

---

### 2. Contextual Usage Analysis

There is one usage of `ncol` in the codebase, located in `bkde2D` inside `KernSmooth/R/all.R` at line 162.

The relevant line is:

```r
rp <- rp * matrix(as.numeric(rp > 0), nrow(rp), ncol(rp))
```

`rp` at this point is a 2D numeric matrix of shape `(M1, M2)` — the result of a truncated inverse FFT. The expression constructs a same-shape indicator matrix to zero out negative values. `ncol(rp)` supplies the column dimension argument to the `matrix` constructor — it plays no independent role beyond dimension introspection.

---

### 3. Python Conversion Strategy

`numpy` is the direct and natural equivalent. A NumPy 2D array (ndarray) exposes its shape through the `.shape` attribute, and `ncol(rp)` maps to `rp.shape[1]`. More importantly, NumPy's broadcasting and boolean masking make the entire surrounding expression collapsible into a single vectorized line, eliminating the need to explicitly pass column count at all.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Dimension Introspection — `ncol(rp)` as a standalone call

```r
num_cols <- ncol(rp)   # integer, e.g. 51
```

**Python Equivalent:**

```python
import numpy as np

# rp is a 2D numpy array of shape (M1, M2)
num_cols = rp.shape[1]   # integer, e.g. 51
```

**Explanation:**
- R matrices store dimensions as `dim(x)` internally; `ncol(x)` is syntactic sugar for `dim(x)[2]`.
- NumPy stores dimensions in `ndarray.shape` as a tuple `(nrows, ncols)`. The column count is therefore `rp.shape[1]` (0-based index into the shape tuple — index 0 is rows, index 1 is columns).

---

#### 4.2 Full Line Conversion — the non-negativity clamp using `ncol`

**Location:** `KernSmooth/R/all.R`, function `bkde2D`, line 162.

```r
rp <- rp * matrix(as.numeric(rp > 0), nrow(rp), ncol(rp))
```

**Python Equivalent:**

```python
import numpy as np

# Option A: idiomatic NumPy boolean mask (direct translation)
rp = rp * (rp > 0).astype(float)

# Option B: preferred NumPy idiom — element-wise maximum
rp = np.maximum(rp, 0)
```

**Explanation:**

- `rp > 0` on a NumPy array produces a boolean array of the same shape — no reshape or explicit dimension arguments are needed because NumPy operations are inherently shape-preserving.
- `.astype(float)` converts `True`/`False` to `1.0`/`0.0`, matching R's `as.numeric(...)`.
- Option B (`np.maximum(rp, 0)`) is the most idiomatic and efficient NumPy form of a non-negativity clamp; it is equivalent and preferred over the mask-multiply approach.
- The key translation insight is that R's `matrix()` constructor requires explicit dimensions because R does not propagate shape automatically, whereas NumPy's element-wise operators broadcast over matching shapes without any dimension arguments — making `ncol` (and `nrow`) entirely unnecessary in Python for this pattern.
