## Conversion Guide: General R-to-Python Translation Notes

---

### 1. General Instruction

Translate R idioms to idiomatic NumPy and Python. Be careful with indexing, types, and normalisation conventions. The goal is a Python function that is numerically equivalent to the original R implementation while reading as natural, idiomatic Python rather than a transliteration.

There is no construct-specific reference material below; apply general R-to-Python translation judgement to whatever function or expression you are converting.

---

### 2. Indexing and Ranges

R uses 1-based indexing; Python and NumPy use 0-based indexing. Any loop bound, index variable, or subsetting expression carried over from R must be adjusted by one where appropriate. R's `a:b` range is inclusive of both endpoints; the corresponding NumPy or Python range must be constructed so that it includes the same elements (for example, `range(a, b + 1)` rather than `range(a, b)`, depending on context). Take care with negative indices and with any code that mixes index arithmetic and value arithmetic in the same expression.

---

### 3. Types and Coercion

R is loosely typed and coerces automatically between integer, double, logical, and character values in many contexts. Python and NumPy are stricter. Check whether a value flowing into arithmetic, comparison, or an array constructor needs an explicit cast (`int`, `float`, `np.asarray(..., dtype=...)`) to avoid unintended integer division, type errors, or silently wrong dtypes. Pay particular attention to functions whose R behavior depends on the type of the input (numeric vs. character vs. logical), since the Python translation must replicate that branching explicitly rather than relying on implicit coercion.

---

### 4. Vectorization and Shape

Many R base functions are implicitly vectorized over vectors, matrices, and arrays, and silently handle scalars as length-one vectors. When translating to NumPy, prefer the vectorized NumPy equivalent over a Python-level loop, and confirm that the shape and dimensionality of the output match the R behavior (including edge cases such as scalar input, empty input, or a matrix collapsing to a vector). Where R silently drops a dimension (for example, when subsetting a matrix down to a single row or column), decide explicitly whether the Python translation should preserve or drop that dimension, and keep the choice consistent with how the calling code uses the result.

---

### 5. Missing Values and Edge Cases

R's `NA` and Python's `NaN`/`None` are not interchangeable in general; check how missing or undefined values are represented and propagated in the original code, and choose the Python representation (`np.nan`, `None`, or a masked value) that preserves the same downstream behavior. Consider zero-length input, single-element input, and boundary values explicitly rather than assuming the general-case translation covers them.

---

### 6. Numerical Conventions

Keep normalisation conventions consistent with the original R implementation: R and NumPy may differ in details such as denominator conventions for variance-like quantities, rounding-to-even versus rounding-away-from-zero, or the base and argument order of transcendental functions. When in doubt, prefer the NumPy function whose documented behavior matches R's documented behavior for the same named operation, rather than the first similarly-named function found.

---

### 7. Style and Naming

Favor standard NumPy/SciPy idioms over hand-written replacements of built-in R functionality. Keep variable names and overall control flow close to the original where that does not conflict with idiomatic Python, so that the translation remains easy to audit line by line against the R source. Avoid introducing helper abstractions, extra parameters, or behavior not present in the original function.

---

### 8. Verification

Before finalizing a translation, mentally trace at least one representative call through both the R and Python versions and confirm the intermediate values agree at each step, paying special attention to indexing offsets, type coercion, and shape handling, which are the most common sources of silent disagreement between an R implementation and its Python translation.
