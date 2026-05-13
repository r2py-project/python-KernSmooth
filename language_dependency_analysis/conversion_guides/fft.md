## Conversion Guide: R `fft` to Python

---

### 1. Overview of `fft` in R

`fft` is R's built-in Fast Fourier Transform function, provided by the base package. Its signature is:

```
fft(z, inverse = FALSE)
```

- `z`: A real or complex numeric vector or matrix of values to transform.
- `inverse`: A logical flag. When `FALSE` (the default), a forward DFT is computed. When `TRUE`, an inverse DFT is computed.

The inverse FFT is **unnormalized** by default: R returns the sum divided by nothing, so the caller must divide by `N` (or `P1*P2` for 2D arrays) to recover the original signal. This normalization responsibility is explicit and visible in the KernSmooth source code.

---

### 2. Contextual Usage Analysis

All nine usages appear in three functions in `KernSmooth/R/all.R`, following a consistent three-step convolution pattern:

1. **Forward FFT of the kernel weight array**
2. **Forward FFT of the binned counts array**
3. **Inverse FFT of their element-wise product** — then take the real part and divide by the total number of FFT points to normalize

| Function | Dimensionality | Array sizes | Normalization divisor |
|----------|---------------|-------------|----------------------|
| `bkde` | 1D (vector) | length `P` (power of 2) | `P` (line 78) |
| `bkde2D` | 2D (matrix) | `P1 x P2` (powers of 2) | `P1*P2` (line 156) |
| `bkfe` | 1D (vector) | length `P` (power of 2) | `P` (line 232) |

---

### 3. Python Conversion Strategy

The correct Python equivalent is `numpy.fft`. Specifically:

- For 1D arrays: `numpy.fft.fft` (forward) and `numpy.fft.ifft` (inverse).
- For 2D arrays: `numpy.fft.fft2` (forward) and `numpy.fft.ifft2` (inverse).

**Critical normalization difference:** NumPy's `ifft` / `ifft2` normalizes by `1/N` by default, while R's `fft(..., inverse=TRUE)` does **not** normalize. The existing R code compensates by explicitly dividing by `P` (or `P1*P2`) after taking the real part. In NumPy the normalisation is already baked into `ifft`, so that explicit division must be **removed** when translating.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 1D Forward FFT of a Real-Valued Vector

**Locations:** `bkde` (lines 76-77), `bkfe` (lines 229-230)

```r
kappa   <- fft(kappa / tot)
gcounts <- fft(gcounts)
```

**Python Equivalent:**

```python
import numpy as np

kappa   = np.fft.fft(kappa / tot)
gcounts = np.fft.fft(gcounts)
```

---

#### 4.2 1D Inverse FFT With Manual Normalization

**Locations:** `bkde` (line 78), `bkfe` (line 232)

```r
# kappa, gcounts: complex vectors length P
# P: integer scalar (power of 2), M: integer scalar (grid size)
(Re(fft(kappa * gcounts, TRUE)) / P)[1L:M]
```

**Python Equivalent:**

```python
import numpy as np

result = np.fft.ifft(kappa * gcounts).real[:M]
```

**Explanation:**
- R: `Re(fft(kappa * gcounts, inverse=TRUE)) / P`
- NumPy: `np.fft.ifft(kappa * gcounts).real`
- NumPy's `ifft` internally divides by `N` (i.e., `P`), so the division by `P` that appears in R is absorbed automatically.
- The `.real` attribute is the Python equivalent of R's `Re()`.
- Python's 0-based slicing `[:M]` replaces R's 1-based `[1L:M]`.

---

#### 4.3 2D Forward FFT of a Real-Valued Matrix

**Locations:** `bkde2D` (lines 154-155)

```r
rp <- fft(rp)
sp <- fft(sp)
```

**Python Equivalent:**

```python
import numpy as np

rp = np.fft.fft2(rp)
sp = np.fft.fft2(sp)
```

**Explanation:** R's `fft` on a matrix performs a multidimensional DFT. The direct Python equivalent is `np.fft.fft2`, which applies a 2D DFT over both axes. Using `np.fft.fft` on a 2D array would apply 1D FFTs along only the last axis and would produce incorrect results.

---

#### 4.4 2D Inverse FFT With Manual Normalization

**Locations:** `bkde2D` (line 156)

```r
Re(fft(rp * sp, inverse = TRUE) / (P1 * P2))[1L:M1, 1L:M2]
```

**Python Equivalent:**

```python
import numpy as np

result = np.fft.ifft2(rp * sp).real[:M1, :M2]
```

**Explanation:** `np.fft.ifft2` performs the 2D inverse FFT and normalizes by `P1*P2` internally, so no explicit division is needed. The 2D slice `[:M1, :M2]` is the Python equivalent of R's `[1L:M1, 1L:M2]`.

---

### Summary of R-to-Python Mapping

| R call | Python equivalent | Notes |
|---|---|---|
| `fft(v)` | `np.fft.fft(v)` | 1D forward FFT |
| `fft(v, TRUE)` / `fft(v, inverse=TRUE)` | `np.fft.ifft(v)` | 1D inverse FFT; NumPy normalizes by `1/N` automatically |
| `Re(fft(v, TRUE)) / P` | `np.fft.ifft(v).real` | Normalized inverse with real part extraction |
| `fft(M)` on a matrix | `np.fft.fft2(M)` | 2D forward FFT |
| `Re(fft(M, TRUE)) / (P1*P2)` | `np.fft.ifft2(M).real` | 2D normalized inverse with real part extraction |
