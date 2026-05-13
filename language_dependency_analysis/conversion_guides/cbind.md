## Conversion Guide: `cbind` (R to Python)

---

### 1. Overview of `cbind` in R

`cbind` (column-bind) is a base R function that combines one or more R objects by columns into a single matrix or data frame. When applied to two numeric vectors of equal length, it produces a two-dimensional matrix where each input vector becomes one column. The result is an `n x k` matrix, where `n` is the length of the input vectors and `k` is the number of vectors supplied.

- **Typical inputs:** Two or more atomic vectors (integer or double) of equal length, or matrices.
- **Expected output:** A matrix with the inputs arranged as columns.

---

### 2. Contextual Usage Analysis

All three usages appear in `KernSmooth/R/all.R`, in the functions `blkest` (line 247), `cpblock` (line 282), and `dpill` (line 494). The pattern is structurally identical across all three sites:

1. Two 1D numeric vectors — the predictor `x` (or `X`) and the response `y` (or `Y`) — are received as function arguments.
2. `cbind` combines them into a two-column matrix (`datmat` or `xy`).
3. The resulting matrix is immediately sorted row-wise by the first column, using `sort.list` to obtain sorted-index permutation, which is applied to reorder both vectors simultaneously.
4. The sorted columns are then extracted back into separate 1D vectors.

The sole purpose of `cbind` in all three cases is to facilitate **synchronized co-sorting** of two equal-length numeric vectors.

---

### 3. Python Conversion Strategy

`numpy` is the correct Python equivalent. The R `cbind` call on two 1D vectors maps directly to `numpy.column_stack`, which stacks 1D arrays as columns into a 2D array of shape `(n, 2)`.

For the specific idiom used here (co-sorting two vectors), **the full R pattern can be replaced more idiomatically in Python using `numpy.argsort`**, which directly yields the sorted-index permutation without the need to construct and decompose a temporary 2D array.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Combining Two Vectors Column-wise for Co-sorting

**Locations:**
- `KernSmooth/R/all.R` — function `blkest`, line 247
- `KernSmooth/R/all.R` — function `cpblock`, line 282
- `KernSmooth/R/all.R` — function `dpill`, line 494

**Original R Context:**

```r
# Generalized R pattern (identical across all three sites)
datmat <- cbind(x, y)                          # n x 2 numeric matrix
datmat <- datmat[sort.list(datmat[, 1L]), ]    # sort rows by first column
x <- datmat[, 1L]                             # extract sorted x
y <- datmat[, 2L]                             # extract sorted y
```

**Python Equivalent (direct structural translation):**

```python
import numpy as np

# x, y are 1D numpy arrays of dtype float64, equal length n
datmat = np.column_stack((x, y))          # shape (n, 2), equivalent to cbind(x, y)
sort_idx = np.argsort(datmat[:, 0])       # equivalent to sort.list(datmat[, 1L])
datmat = datmat[sort_idx, :]              # reorder rows by sorted x
x = datmat[:, 0]                          # extract sorted x (0-based column index)
y = datmat[:, 1]                          # extract sorted y
```

**Python Equivalent (idiomatic — avoids temporary 2D array):**

```python
import numpy as np

# x, y are 1D numpy arrays of dtype float64, equal length n
sort_idx = np.argsort(x)   # indices that would sort x
x = x[sort_idx]            # apply permutation to x
y = y[sort_idx]            # apply same permutation to y
```

**Explanation:**

| R construct | Python equivalent | Notes |
|---|---|---|
| `cbind(x, y)` | `np.column_stack((x, y))` | Stacks two 1D arrays as columns into a `(n, 2)` array. |
| `sort.list(datmat[, 1L])` | `np.argsort(datmat[:, 0])` | Both return integer indices that sort the target array. R uses 1-based column indexing (`1L`); NumPy uses 0-based (`0`). |
| `datmat[sort_indices, ]` | `datmat[sort_idx, :]` | Row-wise fancy indexing works identically in both languages. |
| `datmat[, 1L]` | `datmat[:, 0]` | Column extraction: R is 1-based, Python is 0-based. |

The idiomatic Python version is preferred when the intermediate matrix is never used for any purpose beyond co-sorting, which is the case in all three locations.
