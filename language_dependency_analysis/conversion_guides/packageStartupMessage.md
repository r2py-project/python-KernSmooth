## Conversion Guide: `packageStartupMessage` (R to Python)

---

### 1. Overview of `packageStartupMessage` in R

`packageStartupMessage` is a base R function used exclusively inside a package's `.onAttach` hook — a special function that R calls automatically when a package is attached via `library()` or `require()`. Its purpose is to emit an informational startup message to the user's console (via `stderr`) announcing the package version, copyright, or other load-time notices.

Key characteristics:

- Output is directed to `stderr`, not `stdout`.
- The message can be suppressed by wrapping the `library()` call in `suppressPackageStartupMessages()`, making it opt-out friendly for scripted or non-interactive use.
- It is distinct from `message()` in that it is specifically designed for the package-attach lifecycle event.
- Inputs: one or more character strings (concatenated), with `\n` interpreted as a newline.
- Return value: `NULL` (called for its side effect only).

---

### 2. Contextual Usage Analysis

There is exactly one usage of `packageStartupMessage` in the codebase, located in `KernSmooth/R/all.R` at line 890.

It appears inside the `.onAttach` hook:

```r
.onAttach <- function(libname, pkgname)
   packageStartupMessage("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")
```

The function produces two lines of output on the console when the `KernSmooth` package is loaded:

```
KernSmooth 2.23 loaded
Copyright M. P. Wand 1997-2009
```

---

### 3. Python Conversion Strategy

The Python equivalent depends on context:

- **For a pure Python package (installed via pip/conda):** The closest idiomatic equivalent is `print(..., file=sys.stderr)` placed in the package's `__init__.py`. Since the message is informational (not a warning), `print` to `stderr` is the most direct translation.
- **For suppression support:** `warnings.warn()` is useful when the consuming environment already has warning-filter infrastructure in place.

No third-party library is required. The standard library (`sys`, optionally `warnings`) is sufficient.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 Startup Announcement Message

**Location:** `KernSmooth/R/all.R`, function `.onAttach` (line 890)

**Original R Context:**

```r
.onAttach <- function(libname, pkgname)
   packageStartupMessage("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")
```

**Python Equivalent — Primary (`sys.stderr`):**

```python
# In the package's __init__.py
import sys

def _on_attach() -> None:
    """Emit a package startup message to stderr, mirroring R's .onAttach hook."""
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)

_on_attach()
```

**Python Equivalent — Secondary (`warnings`, suppression-friendly):**

```python
# In the package's __init__.py
import warnings

warnings.warn(
    "KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009",
    category=UserWarning,
    stacklevel=1,
)
```

To suppress it at the call site (analogous to `suppressPackageStartupMessages`):

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import py_kernsmooth
```

**Explanation:**

| R concept | Python translation |
|---|---|
| `.onAttach` hook | Module-level code in `__init__.py`, executed at `import` time |
| `packageStartupMessage(...)` | `print(..., file=sys.stderr)` — writes to stderr at import time |
| `\n` in string literal | Identical `\n` in Python string literal |
| `suppressPackageStartupMessages()` | `warnings.catch_warnings()` + `simplefilter("ignore")`, or an environment variable guard |
| Returns `NULL` | Function returns `None`; or omit the wrapper and write the `print` call directly at module scope |

The primary `sys.stderr` approach is the most faithful translation because `packageStartupMessage` writes to `stderr` and is purely informational.
