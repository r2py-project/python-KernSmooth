## Conversion Guide: `sort.list` (R) to Python

---

### 1. Overview of `sort.list` in R

`sort.list` is a base R function that returns the **integer indices** that would sort a given vector in ascending (or descending) order. It is the index-returning counterpart to `sort()`.

**Signature:**
```r
sort.list(x, partial = NULL, na.last = TRUE, decreasing = FALSE, method = ...)
```

**Key behaviour:**
- Input: a vector `x` (numeric, character, logical, etc.).
- Output: an integer vector of the same length as `x`, where each element is the position in `x` that belongs at that rank. This is equivalent to the permutation that sorts `x`.
- It is functionally identical to `order(x)` when called with default arguments. In modern R, `order()` is the preferred spelling; `sort.list` is an older alias.

---

### 2. Contextual Usage Analysis

All three occurrences follow the exact same pattern and serve the same purpose: **sorting a two-column matrix by its first column (the x-predictor values)**.

The recurring idiom is:

```r
datmat <- cbind(col1, col2)
datmat <- datmat[sort.list(datmat[, 1L]), ]
```

`sort.list(datmat[, 1L])` returns the row-permutation that puts the first column (`x` values) into ascending order. That permutation is then used as a row-index into `datmat`, reordering both columns simultaneously.

| Location | Function | Line |
|------|----------|------|
| `KernSmooth/R/all.R` | `blkest` | 248 |
| `KernSmooth/R/all.R` | `cpblock` | 283 |
| `KernSmooth/R/all.R` | `dpill` | 495 |

---

### 3. Python Conversion Strategy

**Chosen library: `numpy`**

R matrices are naturally represented as 2-D `numpy` arrays. NumPy's `numpy.argsort()` is the direct vectorised equivalent of `sort.list` / `order`: it accepts an array and returns the integer indices that would sort it, in ascending order by default. Using NumPy for both the matrix construction (`np.column_stack`) and the sort-indexing (`np.argsort`) keeps the entire operation in contiguous array memory.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Sort a 2-column numeric matrix by its first column

**Locations:** `blkest` (line 248), `cpblock` (line 283), `dpill` (line 495)

**Original R Context:**

```r
# Generalized R pattern
datmat <- cbind(col1, col2)
datmat <- datmat[sort.list(datmat[, 1L]), ]
col1 <- datmat[, 1L]
col2 <- datmat[, 2L]
```

**Python Equivalent (direct structural translation):**

```python
import numpy as np

# col1, col2 are 1-D numpy arrays of dtype float64, equal length n
datmat = np.column_stack((col1, col2))          # shape (n, 2)
sort_idx = np.argsort(datmat[:, 0])             # argsort of first column
datmat = datmat[sort_idx, :]                    # reorder rows
col1 = datmat[:, 0]
col2 = datmat[:, 1]
```

**Python Equivalent (idiomatic — avoids temporary 2D array):**

```python
import numpy as np

# The most Pythonic approach: sort indices directly, skip the matrix construction
sort_idx = np.argsort(col1)
col1 = col1[sort_idx]
col2 = col2[sort_idx]
```

**Explanation:**

| R | Python | Note |
|---|--------|-------|
| `cbind(col1, col2)` | `np.column_stack((col1, col2))` | Both produce an `n x 2` numeric matrix / array. |
| `sort.list(datmat[, 1L])` | `np.argsort(datmat[:, 0])` | Both return the integer permutation that sorts the column. `argsort` defaults to ascending order, matching `sort.list`'s default. |
| `datmat[idx, ]` | `datmat[idx, :]` | Row-subsetting with an integer index array. |
| `datmat[, 1L]` / `datmat[, 2L]` | `datmat[:, 0]` / `datmat[:, 1]` | Same 0-based vs 1-based shift applies to column extraction after the sort. |

The only substantive translation nuance is the **1-based to 0-based index shift**: R's `[, 1L]` (first column) becomes Python's `[:, 0]`, and R's `[, 2L]` (second column) becomes `[:, 1]`. Everything else is a direct syntactic mapping between `sort.list` and `np.argsort`.
