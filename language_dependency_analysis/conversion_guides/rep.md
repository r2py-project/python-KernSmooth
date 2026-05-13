## Conversion Guide: R `rep` to Python

---

### 1. Overview of `rep` in R

`rep(x, times)` is a base R function that **replicates** its first argument `x` to produce a vector of a specified length. In the two-argument form used consistently throughout this codebase, `rep(value, n)` creates a numeric (or integer) vector of length `n` in which every element equals `value`. The return type mirrors the type of the input `x`: `rep(0L, n)` returns an integer vector, while `rep(0, n)` returns a double vector.

---

### 2. Contextual Usage Analysis

All 49 call sites in the CSV fall into exactly **three distinct semantic patterns**:

**Pattern A — Zero-fill buffer allocation (the dominant pattern)**
`rep(0, n)` and `rep(0L, n)` appear in `bkde`, `bkfe`, `blkest`, `cpblock`, `locpoly`, `sdiag`, and `sstdiag`. In every case the result is a named local variable (e.g., `fkap`, `curvest`, `xj`, `RSS`, `work`, `det`, `ipvt`) that is immediately passed as a pre-allocated output buffer into a Fortran subroutine via `.Fortran()`.

**Pattern B — One-fill index/weight initialization**
`rep(1, M)` appears at lines 640, 673, 675, 677, 778, 780, 782, 855, 857, 859. In the `locpoly` density branch `xcounts <- rep(1, M)` replaces the actual bin counts with a uniform weight vector of length `M`. In `sdiag` and `sstdiag` the same call initializes `indic` to a uniform value of 1 for the scalar-bandwidth code path.

**Pattern C — Scalar broadcast to vector of length Q**
`rep(floor(tau*bandwidth/delta), Q)` and `rep(bandwidth, Q)` broadcast a computed scalar value across a length-`Q` vector. This arises when a single scalar bandwidth is provided: both `Lvec` and `hdisc` must be length-`Q` vectors for the Fortran routine's interface.

---

### 3. Python Conversion Strategy

**`numpy.zeros()` and `numpy.ones()` with an explicit `dtype`** are the canonical Python equivalents:

- `np.zeros(n, dtype=np.float64)` directly translates `rep(0, n)`.
- `np.zeros(n, dtype=np.int32)` directly translates `rep(0L, n)`.
- `np.ones(M, dtype=np.float64)` directly translates `rep(1, M)`.
- `np.full(n, fill_value, dtype)` handles Pattern C (scalar broadcast) in a single, readable call.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A — Zero-fill buffer allocation with double values

**Locations:** `blkest` (lines 255–260), `cpblock` (lines 290–297), `locpoly` (lines 690–697), `sdiag` (lines 790–797), `sstdiag` (lines 867–876)

```r
fkap    <- rep(0, dimfkap)   # double vector
curvest <- rep(0, M)          # double vector
midpts  <- rep(0, Q)          # double vector
Tvec    <- rep(0, pp)         # double vector
```

**Python Equivalent:**

```python
import numpy as np

fkap    = np.zeros(dimfkap, dtype=np.float64)
curvest = np.zeros(M,       dtype=np.float64)
midpts  = np.zeros(Q,       dtype=np.float64)
Tvec    = np.zeros(pp,      dtype=np.float64)
```

---

#### 4.2 Pattern A (variant) — Zero-fill buffer allocation with integer values

**Locations:** `bkde` (line 75 — `rep(0L, P-M)`), `sdiag` (line 795 — `rep(0, 2L)`), `locpoly` (lines 703–707)

```r
gcounts <- c(gcounts, rep(0L, P-M))  # integer zero-padding for FFT
det  <- rep(0, 2L)   # 2-element double workspace
ipvt <- rep(0, pp)   # pivot index workspace
```

**Python Equivalent:**

```python
import numpy as np

# integer zero-padding (rep(0L, P-M))
gcounts = np.concatenate([gcounts, np.zeros(P - M, dtype=np.int32)])

# 2-element double workspace (rep(0, 2L))
det  = np.zeros(2, dtype=np.float64)

# integer pivot storage
ipvt = np.zeros(pp, dtype=np.int32)
```

---

#### 4.3 Pattern B — One-fill initialization of a length-M vector

**Locations:** `locpoly` (lines 640, 673, 675, 677), `sdiag` (lines 778, 780, 782), `sstdiag` (lines 855, 857, 859)

```r
xcounts <- rep(1, M)    # uniform weights
indic   <- rep(1, M)    # bandwidth index, all = 1
```

**Python Equivalent:**

```python
import numpy as np

xcounts = np.ones(M, dtype=np.float64)
indic   = np.ones(M, dtype=np.float64)
```

---

#### 4.4 Pattern C — Scalar broadcast to a uniform vector of length Q

**Locations:** `locpoly` (lines 679–680), `sdiag` (lines 784–785), `sstdiag` (lines 861–862)

```r
Q     <- 1L
Lvec  <- rep(floor(tau * bandwidth / delta), Q)   # integer scalar broadcast
hdisc <- rep(bandwidth, Q)                        # double scalar broadcast
```

**Python Equivalent:**

```python
import numpy as np

Q     = 1
Lvec  = np.full(Q, int(np.floor(tau * bandwidth / delta)), dtype=np.int32)
hdisc = np.full(Q, bandwidth, dtype=np.float64)
```

**Explanation:** `np.full(n, fill_value, dtype)` is the most direct translation of `rep(scalar, n)` when the fill value is a non-trivial expression. It broadcasts `fill_value` across an array of length `n`, exactly matching R's semantics.

---

### Summary Mapping Table

| R call | Pattern | Python equivalent |
|---|---|---|
| `rep(0, n)` | zero-fill double | `np.zeros(n, dtype=np.float64)` |
| `rep(0L, n)` | zero-fill integer | `np.zeros(n, dtype=np.int32)` |
| `rep(0, 2L)` | zero-fill double, literal length | `np.zeros(2, dtype=np.float64)` |
| `rep(1, M)` | one-fill double | `np.ones(M, dtype=np.float64)` |
| `rep(scalar_double, Q)` | scalar broadcast double | `np.full(Q, scalar_double, dtype=np.float64)` |
| `rep(floor(...), Q)` | scalar broadcast integer | `np.full(Q, int(np.floor(...)), dtype=np.int32)` |
