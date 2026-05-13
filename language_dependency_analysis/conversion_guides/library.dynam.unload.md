## Conversion Guide: `library.dynam.unload` (R → Python)

---

### 1. Overview of `library.dynam.unload` in R

`library.dynam.unload` is a base R function defined in the `base` package. Its sole responsibility is to **unload a compiled shared library** (a `.so` / `.dll` file) that was previously loaded into R's native/C runtime via `library.dynam()`.

**Signature:**
```r
library.dynam.unload(chname, libpath, verbose = getOption("verbose"),
                     file.ext = .Platform$dynlib.ext)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `chname` | `character(1)` | The base name of the shared library (e.g., `"KernSmooth"` maps to `KernSmooth.so` on Linux). |
| `libpath` | `character(1)` | The full filesystem path to the installed package directory, supplied automatically by R's package loader. |

**Return value:** Called for its side effect only.

**Typical context:** `library.dynam.unload` is only ever called inside a package's `.onUnload` hook — a special function that R invokes automatically when a package is detached. It is the proper counterpart to calling `library.dynam()` inside `.onLoad`.

---

### 2. Contextual Usage Analysis

**Source file:** `KernSmooth/R/all.R`, lines 892–893.

```r
.onUnload <- function(libpath)
    library.dynam.unload("KernSmooth",  libpath)
```

Key observations:
- `chname` is always the string literal `"KernSmooth"` — this matches the package name and the compiled shared library name exactly.
- `libpath` is a scalar `character` path string injected by R's namespace machinery.
- There is exactly one usage in the entire codebase.

---

### 3. Python Conversion Strategy

Python extension modules use a fundamentally different loading mechanism. A compiled C extension (e.g., `_KernSmooth.cpython-311-x86_64-linux-gnu.so`) is loaded by the Python import system at `import` time and is managed by `sys.modules` as a regular module object.

**There is no direct Python equivalent of `library.dynam.unload` because Python's import system does not expose a stable, safe API for unloading native extension modules.** This is a well-known CPython limitation: `importlib` explicitly does not support unloading extension modules because the C runtime may hold global state that cannot be safely torn down.

The correct Python conversion strategy is therefore:

- **Do not translate `library.dynam.unload` at all.** The unload hook has no meaningful Python counterpart.
- If cleanup of native resources is genuinely required, use Python's standard module-level teardown mechanisms: `atexit` callbacks or `__del__` on a module-level sentinel object.

---

### 4. Step-by-Step Conversion Examples

#### 4.1 The `.onUnload` / `library.dynam.unload` Pattern

**Original R Context:**

```r
.onUnload <- function(libpath)
    library.dynam.unload("KernSmooth", libpath)
```

**Python Equivalent (compiled C extension — standard case):**

```python
# For a standard Python C extension (the normal build outcome of a package
# that uses a C/Fortran backend), NO explicit unload code is needed.
# The import system manages the shared library lifetime automatically.

# The R .onUnload hook simply disappears in the Python translation.
# There is nothing to write here.
```

**Python Equivalent (atexit-based cleanup, if custom teardown is required):**

```python
import atexit

def _on_unload():
    """
    Perform any necessary cleanup when the Python interpreter exits.
    This is the closest analog to R's .onUnload hook.
    For a pure C extension module, this body is typically empty.
    """
    pass

atexit.register(_on_unload)
```

**Explanation:**

| R concept | Python equivalent | Notes |
|---|---|---|
| `.onUnload` hook | `atexit.register(fn)` | `atexit` callbacks fire at interpreter shutdown, mirroring R's namespace unload event. |
| `library.dynam.unload("KernSmooth", libpath)` | _(omitted for C extensions)_ | CPython does not safely unload extension modules; omitting the call is correct behavior, not a gap. |
| `chname` argument (`"KernSmooth"`) | Module name in `sys.modules` | Identified as `"KernSmooth"` or the C extension's `__name__`; used only for lookup, not unloading. |
| `libpath` argument | Not needed | Python's `importlib` resolves library paths internally; no caller-supplied path is required at unload time. |

**Key nuance — why Python cannot directly replicate this:**
CPython's C API documentation explicitly states that extension module unloading is unsupported because `Py_Finalize()` does not call module `__del__` methods reliably, and `dlclose()` on a loaded extension can cause segmentation faults if any Python object still holds a reference to a symbol in that library. R's `library.dynam.unload` is safe in R because R's garbage collector and namespace system guarantee no live references remain before calling `dyn.unload()`. Python provides no equivalent guarantee at the C level, hence the omission is intentional and correct.
