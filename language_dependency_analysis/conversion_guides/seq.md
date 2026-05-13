## Conversion Guide: `seq` in R to Python

---

### 1. Overview of `seq` in R

`seq` is a base R function that generates a sequence of numbers. Its most commonly used form is:

```r
seq(from, to, length.out = n)
```

- **`from`**: The starting value of the sequence (numeric scalar).
- **`to`**: The ending value of the sequence (numeric scalar).
- **`length.out`**: The total number of equally spaced values to generate, including both endpoints.

The result is a numeric vector of length `length.out` whose values are uniformly distributed (linearly spaced) between `from` and `to`, inclusive. Both endpoints are always included. This is the only form used throughout this codebase.

---

### 2. Contextual Usage Analysis

Every call to `seq` in `KernSmooth/R/all.R` uses the three-argument `seq(from, to, length.out = n)` form. There are two functionally distinct patterns:

**Pattern A — Linear grid over a data range (`seq(a, b, length.out = M)`).**
This is by far the most common pattern, appearing in 17 of 20 call sites. `a` and `b` are scalar numeric values derived from `range.x`. `M` (or `gridsize`) is a positive integer specifying how many grid points to generate. The returned vector is assigned to `gpoints` and used to define the evaluation grid for binning and kernel density operations.

**Pattern B — Log-scale grid over a bandwidth range (`seq(log(hlow), log(hupp), length.out = Q)`).**
This appears in 3 call sites: `locpoly` line 663, `sdiag` line 768, and `sstdiag` line 845. Here `hlow` and `hupp` are scalars representing the minimum and maximum values of a variable-bandwidth array. The `seq` call generates `Q` equally spaced values on the log scale; the result is immediately passed to `exp(...)` to recover `Q` geometrically spaced bandwidth values `hdisc`.

---

### 3. Python Conversion Strategy

The direct Python equivalent of `seq(from, to, length.out = n)` is `numpy.linspace(start, stop, num)`. The mapping is exact:

| R argument | Python argument |
|---|---|
| `from` | `start` |
| `to` | `stop` |
| `length.out` | `num` |

`numpy.linspace` is the correct choice because:

- It returns a NumPy array, which is the natural vectorized counterpart to an R numeric vector.
- Both `seq` and `numpy.linspace` include both endpoints by default.
- Both produce exactly `num` (`length.out`) evenly spaced points.
- `numpy.arange` with a step argument is an unreliable substitute because floating-point step arithmetic can shift the endpoint; `numpy.linspace` avoids this entirely.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A: Linear evaluation grid

**Locations:** `bkde` (line 48), `bkde2D` (lines 115, 116), `bkfe` (lines 188, 193), `dpih` (lines 324, 344), `dpik` (lines 420, 437), `dpill` (line 512), `locpoly` (lines 637, 644, 652), `sdiag` (lines 751, 756), `sstdiag` (lines 829, 834)

**Original R Context:**

```r
a <- range.x[1L]          # numeric scalar
b <- range.x[2L]          # numeric scalar
M <- gridsize              # positive integer

gpoints <- seq(a, b, length.out = M)

delta <- (b - a) / (M - 1)   # uniform grid spacing
```

For the 2-D case (`bkde2D`):

```r
gpoints1 <- seq(a[1L], b[1L], length.out = M[1L])
gpoints2 <- seq(a[2L], b[2L], length.out = M[2L])
```

**Python Equivalent:**

```python
import numpy as np

# Scalars
a = range_x[0]       # float (R's range_x[1L] → Python range_x[0])
b = range_x[1]       # float (R's range_x[2L] → Python range_x[1])
M = gridsize         # int, e.g. 401

gpoints = np.linspace(a, b, M)
delta = (b - a) / (M - 1)   # identical formula

# 2-D case (bkde2D)
gpoints1 = np.linspace(a[0], b[0], M[0])
gpoints2 = np.linspace(a[1], b[1], M[1])
```

**Explanation:**
- `np.linspace(a, b, M)` maps argument-for-argument onto `seq(a, b, length.out = M)`.
- R uses 1-based indexing so `range.x[1L]` and `range.x[2L]` become `range_x[0]` and `range_x[1]` in Python.

---

#### 4.2 Pattern B: Log-scale bandwidth discretisation

**Locations:** `locpoly` (line 663), `sdiag` (line 768), `sstdiag` (line 845)

**Original R Context:**

```r
hlow <- sort(bandwidth)[1L]    # numeric scalar, minimum bandwidth
hupp <- sort(bandwidth)[M]     # numeric scalar, maximum bandwidth
Q    <- as.integer(bwdisc)     # positive integer

hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
```

**Python Equivalent:**

```python
import numpy as np

hlow = np.sort(bandwidth)[0]          # 0-based: R's [1L] → Python [0]
hupp = np.sort(bandwidth)[M - 1]      # 0-based: R's [M] → Python [M-1]
Q    = int(bwdisc)

hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))
```

Alternatively, NumPy provides `numpy.geomspace` which encapsulates this exact log-linear pattern:

```python
hdisc = np.geomspace(hlow, hupp, Q)
# Equivalent to np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))
```

**Explanation:**
- The R idiom `exp(seq(log(from), log(to), length.out = n))` is a standard geometric sequence: equally spaced on the log scale.
- `np.linspace(np.log(hlow), np.log(hupp), Q)` is the direct structural translation, wrapped in `np.exp(...)` exactly as in R.
- `np.geomspace(hlow, hupp, Q)` is the more idiomatic NumPy shorthand for the same operation.
- R's 1-based `sort(bandwidth)[1L]` and `sort(bandwidth)[M]` translate to `np.sort(bandwidth)[0]` and `np.sort(bandwidth)[M - 1]` in Python's 0-based indexing.
