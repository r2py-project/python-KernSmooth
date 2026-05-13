## Conversion Guide: `match.arg` (R to Python)

---

### 1. Overview of `match.arg` in R

`match.arg(arg, choices, several.ok = FALSE)` is a base R utility for validating and normalizing string arguments passed to a function. It performs two operations in one call:

1. **Partial matching:** If the caller supplies an unambiguous prefix of a valid choice (e.g., `"norm"` instead of `"normal"`), `match.arg` resolves it to the full canonical string.
2. **Validation:** If the value cannot be unambiguously matched to any element of `choices`, the call raises an error.

**Inputs:**
- `arg` — a character string (length 1), typically a function parameter with a default value drawn from the `choices` vector.
- `choices` — a character vector of valid values.

**Output:** A single character string that is the fully-resolved element from `choices`.

---

### 2. Contextual Usage Analysis

`match.arg` appears in four locations across two functions in `KernSmooth/R/all.R`. There are two structurally distinct usage patterns:

**Pattern A — kernel string validation** (`bkde` line 13, `dpik` line 402)

The parameter `kernel` has a default of `"normal"` declared in the function signature. `match.arg` normalizes and validates it against the five permitted kernel names.

**Pattern B — scale estimator string validation** (`dpih` line 329, `dpik` line 424)

The parameter `scalest` has a default of `"minim"` in the function signature. `match.arg` normalizes and validates it against three permitted names.

---

### 3. Python Conversion Strategy

Because `match.arg` operates purely on scalar strings and serves as a guard/normalizer at the top of a function, no NumPy or pandas vectorization is appropriate or necessary. The correct Python equivalent is:

- **A short helper function** replicating the partial-match-and-validate semantics, or simply **strict exact validation with a `ValueError`** when partial matching is not needed.

In the KernSmooth codebase, callers consistently pass full strings (the default values are full strings), so **strict exact validation** is the safe and idiomatic Python translation.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Kernel Name Validation (`bkde` line 13, `dpik` line 402)

```r
kernel <- match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
```

**Python Equivalent:**

```python
_KERNEL_CHOICES = ("normal", "box", "epanech", "biweight", "triweight")

def _match_arg(value: str, choices: tuple, arg_name: str = "argument") -> str:
    """Replicate R's match.arg: validate and return the matched choice."""
    if value in choices:
        return value
    # Support unambiguous prefix matching
    matches = [c for c in choices if c.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"'{arg_name}' = '{value}' matches multiple choices: {matches}. "
            f"Must be one of {choices}."
        )
    raise ValueError(
        f"'{arg_name}' = '{value}' is not a valid choice. "
        f"Must be one of {choices}."
    )

# Usage
kernel = _match_arg(kernel, _KERNEL_CHOICES, arg_name="kernel")
```

---

#### 4.2 Scale Estimator Validation (`dpih` line 329, `dpik` line 424)

```r
scalest <- match.arg(scalest, c("minim", "stdev", "iqr"))

scalest <- switch(scalest,
                  "stdev" = sqrt(var(x)),
                  "iqr"   = (quantile(x, 3/4)-quantile(x, 1/4))/1.349,
                  "minim" = min((quantile(x, 3/4)-quantile(x, 1/4))/1.349, sqrt(var(x))))
```

**Python Equivalent:**

```python
import numpy as np

_SCALEST_CHOICES = ("minim", "stdev", "iqr")

def dpih(x, scalest="minim", ...):
    scalest = _match_arg(scalest, _SCALEST_CHOICES, arg_name="scalest")

    x = np.asarray(x, dtype=float)
    std_val = np.std(x, ddof=1)                          # R: sqrt(var(x)) uses ddof=1
    iqr_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349

    scalest_value_map = {
        "stdev": std_val,
        "iqr":   iqr_val,
        "minim": min(iqr_val, std_val),
    }
    scalest_val = scalest_value_map[scalest]

    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")
```

**Explanation:**
- The `_match_arg` helper defined in Section 4.1 is reused directly.
- `np.std(x, ddof=1)` is the correct translation of R's `sqrt(var(x))`: R's `var()` uses `n-1` (Bessel's correction), which corresponds to `ddof=1` in NumPy.
- `np.quantile(x, 0.75)` maps directly to R's `quantile(x, 3/4)`. R's default quantile type is 7; `numpy.quantile` with default `method='linear'` is equivalent.
- The dict-based dispatch replaces R's `switch()`.

---

### Summary: The `_match_arg` Helper

The single helper below covers all four usages in the codebase:

```python
def _match_arg(value: str, choices: tuple, arg_name: str = "argument") -> str:
    """
    Replicates R's match.arg(arg, choices) behaviour:
      - Returns value if it is already an exact member of choices.
      - Returns the unique match if value is an unambiguous prefix of exactly one choice.
      - Raises ValueError otherwise.
    """
    if value in choices:
        return value
    matches = [c for c in choices if c.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"'{arg_name}' = '{value}' matches multiple choices: {matches}; "
            f"must be one of {choices}."
        )
    raise ValueError(
        f"'{arg_name}' = '{value}' is not a valid choice; "
        f"must be one of {choices}."
    )
```

This helper requires no external dependencies and can be placed at the top of the translated module.
