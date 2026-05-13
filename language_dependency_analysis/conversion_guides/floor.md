## Conversion Guide: `floor` in R to Python

---

### 1. Overview of `floor` in R

`floor` is a base R mathematical function that computes the **floor** of a numeric value — the largest integer less than or equal to the input. It is the mathematical floor function, equivalent to rounding towards negative infinity.

- **Signature:** `floor(x)` where `x` is a numeric scalar or vector.
- **Output:** A numeric value or vector of the same length as the input, with each element rounded down to the nearest integer. The return type in R is still numeric (double), not integer, unless the input is already integer-typed.
- **Vectorized:** Yes. Like nearly all base R arithmetic functions, `floor` operates element-wise over vectors without any explicit looping.

---

### 2. Contextual Usage Analysis

All 14 usages in `KernSmooth/R/all.R` share a single overarching pattern: `floor` is used to compute a **kernel support half-width in grid units**. The result, typically stored in a scalar `L` or a vector `Lvec`, is subsequently used as an integer index or count.

The usages split into four functionally distinct patterns:

**Pattern A — Scalar `tau/delta`** (`bkde`, line 54): Result is a single numeric passed to `min(...)` to cap it at `M`.

**Pattern B — Scalar `tau*h/delta`** (`bkfe` line 204; `locpoly`/`sdiag`/`sstdiag` scalar-bandwidth branch): Result is a scalar, then replicated by `rep(...)` into a length-`Q` vector `Lvec`.

**Pattern C — Vector `tau*hdisc/delta`** (`locpoly` lines 666/679, `sdiag` lines 771/784, `sstdiag` lines 848/861, `bkde2D` line 125): `hdisc` is a numeric vector. The result `Lvec` is an integer-valued numeric vector of length `Q`.

**Pattern D — Scalar proportional trimming for index computation** (`dpill` lines 498/499/538/539): `floor(trim*length(x))` and `floor(proptrun*M)` compute integer offsets used to define array slices.

**Pattern E — Block size computation** (`dpill` line 518): `floor(n/divisor)` computes an integer block count.

---

### 3. Python Conversion Strategy

The primary Python equivalent is **`numpy.floor`**.

- R's `floor` is inherently vectorized. `numpy.floor` matches this: it operates element-wise on scalars, 1-D arrays, and N-D arrays without modification.
- In the vector cases (Pattern C), `hdisc` will be a NumPy array. Using `math.floor` would require an explicit loop.
- `numpy.floor` returns a `float64` array by default, matching R's behavior. When an integer array is explicitly required, a subsequent `.astype(int)` or wrapping with `int(...)` for scalars mirrors R's `as.integer(...)` coercions.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Pattern A — Kernel half-width from pure ratio (`bkde`)

**Location:** `KernSmooth/R/all.R`, function `bkde`, line 54.

```r
L <- min(floor(tau / delta), M)
```

**Python Equivalent:**

```python
import numpy as np

L = int(min(np.floor(tau / delta), M))
```

---

#### 4.2 Pattern C — Vector bandwidth produces vector `Lvec` (`locpoly`, `sdiag`, `sstdiag`)

**Locations:** `locpoly` (line 666), `sdiag` (line 771), `sstdiag` (line 848)

```r
# hdisc: numeric vector of length Q — a log-spaced grid of discretized bandwidths
Lvec <- floor(tau * hdisc / delta)
```

**Python Equivalent:**

```python
import numpy as np

hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))
Lvec = np.floor(tau * hdisc / delta).astype(int)
```

**Explanation:** This is the critical vectorized case. `hdisc` is an array, and `np.floor` applied to the product produces an array of the same shape entirely within NumPy. The `.astype(int)` converts the `float64` result to an integer array, which is necessary because `Lvec` is passed as integer to Fortran via `as.integer(Lvec)`.

---

#### 4.3 Pattern D — Proportional trimming index offsets (`dpill`)

**Locations:** `KernSmooth/R/all.R`, function `dpill`, lines 498, 499, 538, 539.

```r
indlow <- floor(trim * length(x)) + 1     # first index to keep (1-based)
indupp <- length(x) - floor(trim * length(x))  # last index to keep (1-based)
llow   <- floor(proptrun * M) + 1
lupp   <- M - floor(proptrun * M)
```

**Python Equivalent:**

```python
import numpy as np

n = len(x)
indlow = int(np.floor(trim * n))          # 0-based: first index to keep
indupp = n - int(np.floor(trim * n))      # 0-based exclusive stop (slice end)

llow = int(np.floor(proptrun * M))        # 0-based: first grid index to use
lupp = M - int(np.floor(proptrun * M))    # 0-based exclusive stop
```

**Explanation:** The most important nuance here is the **index base shift**. In R, `floor(trim*length(x)) + 1` adds 1 to convert from a zero-based count to a one-based index. In Python (0-based), the `+ 1` is dropped: `int(np.floor(trim * n))` already gives the correct 0-based start index for a Python slice.

---

### Summary

| R call | Python equivalent | Notes |
|---|---|---|
| `floor(scalar)` | `int(np.floor(scalar))` | Returns Python int for use as index |
| `floor(tau * hdisc / delta)` (vector) | `np.floor(tau * hdisc / delta).astype(int)` | Element-wise, returns int array |
| `floor(trim * length(x)) + 1` (1-based index) | `int(np.floor(trim * n))` (0-based index) | Drop the `+1` due to 0-based Python indexing |
| `floor(n / divisor)` (block count) | `int(np.floor(n / divisor))` or `n // divisor` | Integer floor division |
