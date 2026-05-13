## Conversion Guide: `gamma` (R) to Python

---

### 1. Overview of `gamma` in R

`gamma` is a base R function that computes the mathematical Gamma function, denoted Γ(x). It is defined by the integral Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt for positive real x, with the key property that Γ(n) = (n-1)! for positive integers n.

- **Input:** A numeric scalar or numeric vector.
- **Output:** A numeric scalar or vector of the same length, containing the Gamma function evaluated at each element.
- **Key identity used here:** `gamma(n + 1) == n!` for non-negative integers `n`.

---

### 2. Contextual Usage Analysis

There is exactly one usage of `gamma` in the codebase, located at `KernSmooth/R/all.R`, line 709, inside the `locpoly` function:

```r
curvest <- gamma(drv+1) * out[[19L]]
```

**Context:**
- `drv` is a non-negative integer (derivative order; default `0L`), so `drv + 1` is a small positive integer — most practically: 1, 2, 3, or 4.
- `out[[19L]]` is the 19th output from a FORTRAN routine (`locpol`), a numeric vector of length `M` (the grid size, e.g. 401 by default).
- `gamma(drv+1)` evaluates to a **scalar**: for `drv=0` it gives `1.0`, for `drv=1` it gives `1.0`, for `drv=2` it gives `2.0`, for `drv=3` it gives `6.0`, and so on (`gamma(n+1) = n!`).
- This scalar is multiplied element-wise across the entire `out[[19L]]` vector, scaling all estimated derivative values uniformly before returning them.

---

### 3. Python Conversion Strategy

Two Python equivalents are applicable:

- **`math.gamma(x)`** (standard library): Operates on a single scalar float. Appropriate when the input is provably a scalar, as is the case here (`drv + 1` is always a single integer).
- **`scipy.special.gamma(x)`**: Handles both scalars and NumPy arrays. Preferred if SciPy is already a project dependency.
- **`math.factorial(n)`**: Since the argument is always a positive integer and `gamma(n+1) = n!`, the factorial function is semantically equivalent and more explicit for integer inputs.

---

### 4. Step-by-Step Conversion Examples

#### Usage 1: Scalar Gamma as a Factorial Scaling Factor

**Locations:** `KernSmooth/R/all.R`, function `locpoly`, line 709.

**Original R Context:**

```r
# drv: non-negative integer scalar
# out[[19L]]: numeric vector of length M (FORTRAN output)
curvest <- gamma(drv + 1) * out[[19L]]
# e.g. for drv=2: gamma(3) == 2.0, so this scales each element by 2.0
```

**Python Equivalent:**

```python
import math
import numpy as np

# drv is a non-negative Python int
# out_19 is a 1-D numpy array (equivalent of R's out[[19L]])
# Note: R index 19 → Python index 18 (0-based)
out_19 = out[18]   # shape: (M,), float64

curvest = math.gamma(drv + 1) * out_19
```

Or using `scipy.special.gamma`:

```python
from scipy.special import gamma
import numpy as np

curvest = gamma(drv + 1) * out[18]
```

Or using the integer identity `gamma(n+1) == factorial(n)`:

```python
import math
import numpy as np

curvest = math.factorial(drv) * out[18]
```

**Explanation:**

| Aspect | R | Python |
|---|---|---|
| Function | `gamma(drv + 1)` | `math.gamma(drv + 1)` or `scipy.special.gamma(drv + 1)` |
| Return type | Numeric scalar | `float` |
| Vector multiplication | `gamma(drv+1) * out[[19L]]` — R recycles the scalar | `math.gamma(drv + 1) * out_19` — NumPy broadcasts the scalar automatically |
| Integer shortcut | `gamma(n+1) = n!` | `math.factorial(drv)` is an exact equivalent for non-negative integer `drv` |
| 1-based to 0-based index | `out[[19L]]` | `out[18]` |

Python's scalar-array multiplication (`scalar * ndarray`) broadcasts the scalar over the array automatically, exactly like R's recycling.
