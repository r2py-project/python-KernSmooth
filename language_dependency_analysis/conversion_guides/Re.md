## Conversion Guide: `Re` (R to Python)

---

### 1. Overview of `Re` in R

`Re` is a base R function that extracts the **real part** of a complex number or complex vector. Its signature is:

```r
Re(z)
```

- **Input:** A scalar or vector of complex numbers (type `complex`).
- **Output:** A numeric scalar or numeric vector of the same length, containing only the real components, with the imaginary parts discarded.

In the context of signal processing and density estimation, `Re` is almost always used in combination with R's `fft()` to discard the numerically negligible imaginary residuals that arise from floating-point arithmetic in inverse FFT operations. Mathematically, the output of a correctly computed inverse FFT applied to a real-valued signal should be purely real; `Re` enforces this by stripping the near-zero imaginary noise.

---

### 2. Contextual Usage Analysis

All three usages appear in `KernSmooth/R/all.R` and share a single recurring pattern:

**Pattern:** `Re(fft(<product>, [inverse = TRUE | TRUE]) [/ normalisation])`

| Row | Function | Purpose |
|-----|----------|---------|
| Line 78 | `bkde` | 1-D binned kernel density estimate: inverse FFT of kernel-weights times binned counts, result sliced to grid length `M`. |
| Line 156 | `bkde2D` | 2-D binned kernel density estimate: inverse FFT of 2-D kernel-weights times binned counts, divided by `P1*P2`, result sliced to `[1:M1, 1:M2]`. |
| Line 232 | `bkfe` | Binned kernel functional estimate: same convolution-via-FFT pattern as `bkde`, but the result is summed rather than returned as a grid. |

Key observations:

- All inputs to `Re` are **complex vectors or matrices** produced by `fft(...)`.
- All three sites use R's `fft(..., inverse = TRUE)`. In `bkde` and `bkfe` the argument is passed positionally as `TRUE`; in `bkde2D` it is passed as `inverse = TRUE`.
- The 1-D cases (`bkde`, `bkfe`) pass a **vector** to `fft`; the 2-D case (`bkde2D`) passes a **matrix**.
- All three require normalisation: R's `fft(..., inverse = TRUE)` computes an *unnormalised* inverse DFT. `bkde` and `bkfe` handle this by dividing by scalar `P` after calling `Re`; `bkde2D` divides by `P1*P2` inside the `Re(...)` call.

---

### 3. Python Conversion Strategy

**Chosen library: `numpy`**

`numpy.fft` is the direct structural equivalent of R's `fft` for this codebase:

- `numpy.fft.fft` / `numpy.fft.ifft` operate on 1-D arrays, matching R's 1-D `fft`.
- `numpy.fft.fft2` / `numpy.fft.ifft2` operate on 2-D arrays (matrices), matching R's 2-D `fft` applied to a matrix.
- Both R and NumPy return **complex-valued arrays**; `numpy.real()` (or equivalently `.real`) extracts the real part, directly replacing `Re`.
- NumPy's `ifft` / `ifft2` **normalise by default** (they divide by N), whereas R's `fft(..., inverse = TRUE)` does **not** normalise. The existing R code compensates for this by explicitly dividing by `P` (or `P1*P2`) after taking the real part. In NumPy the normalisation is already baked into `ifft`, so that explicit division must be **removed** when translating.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 `bkde` — 1-D Kernel Density Estimate (Line 78)

**Original R Context**

```r
# kappa : complex vector of length P (FFT of normalised kernel weights)
# gcounts : complex vector of length P (FFT of binned counts)
# P : integer, next power of 2 >= M + L + 1 (padding length)
# M : integer, number of grid points

kappa  <- fft(kappa / tot)        # forward FFT of kernel weights
gcounts <- fft(gcounts)           # forward FFT of binned counts

# Inverse FFT of element-wise product; R does NOT normalise → divide by P
list(x = gpoints,
     y = (Re(fft(kappa * gcounts, TRUE)) / P)[1L:M])
```

**Python Equivalent**

```python
import numpy as np

# kappa_fft : np.ndarray, complex, shape (P,)  — np.fft.fft of normalised kernel weights
# gcounts_fft : np.ndarray, complex, shape (P,) — np.fft.fft of binned counts
# M : int — number of grid points

# np.fft.ifft already normalises by P, so no explicit division needed
y = np.real(np.fft.ifft(kappa_fft * gcounts_fft))[:M]
```

**Explanation**

- `Re(fft(..., TRUE))` → `np.real(np.fft.ifft(...))`
- R's `fft(..., TRUE)` is unnormalised, so the original R code divides by `P` afterwards. NumPy's `ifft` normalises internally, so the `/P` term is dropped.
- `numpy.real()` is equivalent to `Re`: it discards the floating-point imaginary residuals produced by the inverse FFT.
- Python uses 0-based slicing: `[:M]` is equivalent to R's `[1:M]`.

---

#### 4.2 `bkde2D` — 2-D Kernel Density Estimate (Line 156)

**Original R Context**

```r
# rp : complex matrix, shape (P1, P2) — FFT of 2-D kernel weights (wrap-around padded)
# sp : complex matrix, shape (P1, P2) — FFT of zero-padded 2-D binned counts
# P1, P2 : integers, next powers of 2 for each dimension
# M1, M2 : integers, grid sizes in each dimension

rp <- fft(rp)          # forward 2-D FFT
sp <- fft(sp)          # forward 2-D FFT

# Inverse 2-D FFT of element-wise product; R's fft is unnormalised → divide by P1*P2
rp <- Re(fft(rp * sp, inverse = TRUE) / (P1 * P2))[1L:M1, 1L:M2]
```

**Python Equivalent**

```python
import numpy as np

# rp_fft : np.ndarray, complex, shape (P1, P2) — np.fft.fft2 of 2-D kernel weights
# sp_fft : np.ndarray, complex, shape (P1, P2) — np.fft.fft2 of 2-D binned counts
# M1, M2 : int — grid sizes

# np.fft.ifft2 normalises by P1*P2 internally
rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[:M1, :M2]
```

**Explanation**

- R applies `fft` to a matrix and performs a full 2-D DFT; the NumPy equivalent is `numpy.fft.fft2` / `numpy.fft.ifft2`.
- `Re(fft(rp*sp, inverse = TRUE) / (P1*P2))` → `np.real(np.fft.ifft2(rp_fft * sp_fft))`. The `/ (P1*P2)` term is absorbed into `ifft2`'s default normalisation and must be removed.
- The 2-D slice `[1:M1, 1:M2]` becomes `[:M1, :M2]` in Python's 0-based indexing.

---

#### 4.3 `bkfe` — Binned Kernel Functional Estimate (Line 232)

**Original R Context**

```r
# kappam : complex vector of length P (FFT of derivative kernel weights)
# Gcounts : complex vector of length P (FFT of binned counts)
# P : integer, next power of 2 >= M + L + 1
# M : integer, grid size
# n : integer, sample size

kappam  <- fft(kappam)    # forward FFT of kernel weights
Gcounts <- fft(Gcounts)   # forward FFT of binned counts

# Inverse FFT of product, unnormalised → divide by P; slice to M; then sum and normalise
sum(gcounts * (Re(fft(kappam * Gcounts, TRUE)) / P)[1L:M]) / (n^2)
```

**Python Equivalent**

```python
import numpy as np

result = np.sum(gcounts * np.real(np.fft.ifft(kappam_fft * Gcounts_fft))[:M]) / (n ** 2)
```

**Explanation**

- Structurally identical to `bkde`: `Re(fft(..., TRUE)) / P` → `np.real(np.fft.ifft(...))`, dropping the explicit `/P`.
- The R expression `sum(gcounts * (...)[1:M])` maps to `np.sum(gcounts * (...)[:M])`.
- The `/P` normalisation correction is again unnecessary in NumPy because `ifft` normalises internally.

---

### Summary Table

| R expression | Python equivalent | Key difference |
|---|---|---|
| `Re(fft(v, TRUE)) / P` | `np.real(np.fft.ifft(v))` | Drop `/P`; NumPy normalises internally |
| `Re(fft(M, inverse=TRUE)) / (P1*P2)` | `np.real(np.fft.ifft2(M))` | Drop `/(P1*P2)`; use `ifft2` for 2-D |
| `Re(z)` (generic) | `np.real(z)` or `z.real` | Direct 1-to-1 replacement for any complex array |
