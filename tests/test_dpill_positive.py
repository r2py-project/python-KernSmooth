# Positive test cases for r2py_kernsmooth.dpill
#
# Each test calls R's KernSmooth::dpill via rpy2 to obtain the reference
# result, then calls the Python port and asserts the two scalar outputs agree
# to within floating-point tolerance.
#
# R signature:
#   dpill(x, y, blockmax = 5, divisor = 20, trim = 0.01, proptrun = 0.05,
#         gridsize = 401L, range.x, truncate = TRUE)
#
# Python signature:
#   dpill(x, y, blockmax=5, divisor=20, trim=0.01, proptrun=0.05,
#         gridsize=401, range_x=None, truncate=True) -> np.float64
#
# The function selects a bandwidth for local linear regression using the
# direct plug-in methodology of Ruppert, Sheather and Wand (1995).
#
# KNOWN ISSUE: The Python port currently fails for all valid (non-degenerate)
# inputs because the blkest Fortran/f2py wrapper does not write back the
# scalar outputs sigsqe/th22e/th24e (they are declared as 'input float' in
# the f2py interface instead of 'intent(inout)'). As a result, blkest always
# returns zeros, causing gamseh to become NaN and dpill to raise a ValueError.
# All positive tests document this divergence from R; they will pass once the
# blkest interface is fixed.

import math
import warnings

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

_ks = importr("KernSmooth")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r_dpill(x_py, y_py, **kwargs):
    """Call R's dpill and return the bandwidth scalar as a Python float."""
    x_r = ro.FloatVector(np.asarray(x_py, dtype=float).tolist())
    y_r = ro.FloatVector(np.asarray(y_py, dtype=float).tolist())
    r_kwargs = {}
    if "blockmax" in kwargs:
        r_kwargs["blockmax"] = int(kwargs["blockmax"])
    if "divisor" in kwargs:
        r_kwargs["divisor"] = int(kwargs["divisor"])
    if "trim" in kwargs:
        r_kwargs["trim"] = float(kwargs["trim"])
    if "proptrun" in kwargs:
        r_kwargs["proptrun"] = float(kwargs["proptrun"])
    if "gridsize" in kwargs:
        r_kwargs["gridsize"] = int(kwargs["gridsize"])
    if "range_x" in kwargs:
        r_kwargs["range.x"] = ro.FloatVector(list(kwargs["range_x"]))
    if "truncate" in kwargs:
        r_kwargs["truncate"] = bool(kwargs["truncate"])
    result = _ks.dpill(x_r, y_r, **r_kwargs)
    return float(result[0])


def _assert_close(py_val, r_val, *, rtol=1e-5, label=""):
    assert np.isfinite(py_val), f"{label}: Python returned non-finite {py_val!r}"
    assert np.isfinite(r_val), f"{label}: R returned non-finite {r_val!r}"
    np.testing.assert_allclose(
        py_val,
        r_val,
        rtol=rtol,
        err_msg=f"{label}: Python={py_val!r} vs R={r_val!r}",
    )


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------


def test_dpill_default_parameters_sine_regression():
    """dpill with all defaults on sin(2x) + noise data matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    _assert_close(py_val, r_val, label="default params sine n=200")


def test_dpill_returns_np_float64():
    """dpill always returns an np.float64 scalar."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, 300)
    y = np.cos(x) + rng.normal(0, 0.2, 300)
    result = r2py_kernsmooth.dpill(x, y)
    assert isinstance(result, np.float64), (
        f"Expected np.float64, got {type(result)}"
    )
    assert result > 0, f"Bandwidth must be positive, got {result}"


def test_dpill_returns_positive_scalar():
    """dpill returns a strictly positive bandwidth."""
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 200)
    y = x ** 2 + rng.normal(0, 0.2, 200)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    assert py_val > 0, f"Expected positive bandwidth, got {py_val}"
    _assert_close(py_val, r_val, label="quadratic regression n=200")


def test_dpill_blockmax_1():
    """dpill with blockmax=1 (minimum block count) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, blockmax=1)
    py_val = r2py_kernsmooth.dpill(x, y, blockmax=1)
    _assert_close(py_val, r_val, label="blockmax=1")


def test_dpill_blockmax_3():
    """dpill with blockmax=3 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, blockmax=3)
    py_val = r2py_kernsmooth.dpill(x, y, blockmax=3)
    _assert_close(py_val, r_val, label="blockmax=3")


def test_dpill_blockmax_10():
    """dpill with blockmax=10 (above default) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, blockmax=10)
    py_val = r2py_kernsmooth.dpill(x, y, blockmax=10)
    _assert_close(py_val, r_val, label="blockmax=10")


def test_dpill_divisor_5():
    """dpill with divisor=5 (smaller blocks) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, divisor=5)
    py_val = r2py_kernsmooth.dpill(x, y, divisor=5)
    _assert_close(py_val, r_val, label="divisor=5")


def test_dpill_divisor_40():
    """dpill with divisor=40 (larger blocks) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, divisor=40)
    py_val = r2py_kernsmooth.dpill(x, y, divisor=40)
    _assert_close(py_val, r_val, label="divisor=40")


def test_dpill_trim_0():
    """dpill with trim=0.0 (no trimming): Python and R both return NaN for this dataset."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, trim=0.0)
    py_val = r2py_kernsmooth.dpill(x, y, trim=0.0)
    if not np.isfinite(r_val):
        assert not np.isfinite(float(py_val)), \
            f"R returned NaN but Python returned finite {py_val}"
    else:
        _assert_close(py_val, r_val, label="trim=0.0")


def test_dpill_trim_0_02():
    """dpill with trim=0.02 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, trim=0.02)
    py_val = r2py_kernsmooth.dpill(x, y, trim=0.02)
    _assert_close(py_val, r_val, label="trim=0.02")


def test_dpill_trim_0_1():
    """dpill with trim=0.1 (trim 10% from each end) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, trim=0.1)
    py_val = r2py_kernsmooth.dpill(x, y, trim=0.1)
    _assert_close(py_val, r_val, label="trim=0.1")


def test_dpill_proptrun_0():
    """dpill with proptrun=0.0 (no truncation of functional estimates) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, proptrun=0.0)
    py_val = r2py_kernsmooth.dpill(x, y, proptrun=0.0)
    _assert_close(py_val, r_val, label="proptrun=0.0")


def test_dpill_proptrun_0_02():
    """dpill with proptrun=0.02 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, proptrun=0.02)
    py_val = r2py_kernsmooth.dpill(x, y, proptrun=0.02)
    _assert_close(py_val, r_val, label="proptrun=0.02")


def test_dpill_proptrun_0_1():
    """dpill with proptrun=0.1 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, proptrun=0.1)
    py_val = r2py_kernsmooth.dpill(x, y, proptrun=0.1)
    _assert_close(py_val, r_val, label="proptrun=0.1")


def test_dpill_gridsize_201():
    """dpill with gridsize=201 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, gridsize=201)
    py_val = r2py_kernsmooth.dpill(x, y, gridsize=201)
    _assert_close(py_val, r_val, label="gridsize=201")


def test_dpill_gridsize_801():
    """dpill with gridsize=801 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, gridsize=801)
    py_val = r2py_kernsmooth.dpill(x, y, gridsize=801)
    _assert_close(py_val, r_val, label="gridsize=801")


def test_dpill_custom_range_x():
    """dpill with an explicit range_x: Python and R both return NaN for this input."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    range_x = (float(x.min()), float(x.max()))
    r_val = _r_dpill(x, y, range_x=range_x)
    py_val = r2py_kernsmooth.dpill(x, y, range_x=range_x)
    if not np.isfinite(r_val):
        assert not np.isfinite(float(py_val)), \
            f"R returned NaN but Python returned finite {py_val}"
    else:
        _assert_close(py_val, r_val, label="custom range_x=data range")


def test_dpill_truncate_false():
    """dpill with truncate=False matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    range_x = (-2.0, 2.0)
    r_val = _r_dpill(x, y, range_x=range_x, truncate=False)
    py_val = r2py_kernsmooth.dpill(x, y, range_x=range_x, truncate=False)
    _assert_close(py_val, r_val, label="truncate=False")


def test_dpill_truncate_true():
    """dpill with truncate=True (default) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    range_x = (-2.0, 2.0)
    r_val = _r_dpill(x, y, range_x=range_x, truncate=True)
    py_val = r2py_kernsmooth.dpill(x, y, range_x=range_x, truncate=True)
    _assert_close(py_val, r_val, label="truncate=True")


def test_dpill_truncate_true_vs_false_differ():
    """truncate=True and truncate=False produce different results when data extend beyond range_x."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    range_x = (-2.0, 2.0)
    py_true = r2py_kernsmooth.dpill(x, y, range_x=range_x, truncate=True)
    py_false = r2py_kernsmooth.dpill(x, y, range_x=range_x, truncate=False)
    assert py_true != pytest.approx(py_false, rel=1e-4), (
        "truncate=True and truncate=False should produce different bandwidths"
    )


def test_dpill_large_sample_n1000():
    """dpill on n=1000 observations matches R."""
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 1000)
    y = np.cos(x) + rng.normal(0, 0.1, 1000)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    _assert_close(py_val, r_val, label="n=1000")


def test_dpill_large_sample_n2000():
    """dpill on n=2000 observations matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 2000)
    y = x ** 2 + rng.normal(0, 0.1, 2000)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    _assert_close(py_val, r_val, label="n=2000")


def test_dpill_n50_quadratic():
    """dpill on n=50 observations with quadratic response matches R."""
    rng = np.random.default_rng(99)
    x = rng.normal(0, 1, 50)
    y = x ** 3 + rng.normal(0, 0.3, 50)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    _assert_close(py_val, r_val, label="n=50 cubic")


def test_dpill_n100_quadratic():
    """dpill on n=100 observations matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = x ** 2 + rng.normal(0, 0.2, 100)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    _assert_close(py_val, r_val, label="n=100 quadratic")


def test_dpill_uniform_x_cosine_y():
    """dpill with uniform x and cosine y matches R."""
    rng = np.random.default_rng(77)
    x = rng.uniform(-3, 3, 300)
    y = np.cos(x) + rng.normal(0, 0.2, 300)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    _assert_close(py_val, r_val, label="uniform x cosine y n=300")


def test_dpill_large_magnitude_data():
    """dpill on data with mean=1e6, std=1e4 matches R."""
    rng = np.random.default_rng(55)
    x = rng.normal(1e6, 1e4, 200)
    y = 2 * x + rng.normal(0, 1e3, 200)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    _assert_close(py_val, r_val, label="large-magnitude mean=1e6")


def test_dpill_small_magnitude_data():
    """dpill on data with std=1e-4 matches R."""
    rng = np.random.default_rng(55)
    x = rng.normal(0, 1e-4, 200)
    y = 2 * x + rng.normal(0, 1e-5, 200)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    _assert_close(py_val, r_val, label="small magnitude sd=1e-4")


def test_dpill_negative_x_values():
    """dpill on data with all-negative x matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(-50, 5, 200)
    y = x ** 2 + rng.normal(0, 10, 200)
    r_val = _r_dpill(x, y)
    py_val = r2py_kernsmooth.dpill(x, y)
    _assert_close(py_val, r_val, label="all-negative x")


def test_dpill_integer_input_coerced_to_float():
    """dpill accepts an integer-typed array and matches the float version."""
    rng = np.random.default_rng(42)
    x_int = np.arange(1, 51, dtype=np.int32)
    y_int = (2 * x_int + rng.integers(0, 5, 50)).astype(np.int32)
    x_float = x_int.astype(np.float64)
    y_float = y_int.astype(np.float64)
    py_int = r2py_kernsmooth.dpill(x_int, y_int)
    py_float = r2py_kernsmooth.dpill(x_float, y_float)
    assert py_int == pytest.approx(py_float, rel=1e-10), (
        "Integer input should give the same result as float input"
    )


def test_dpill_reproducibility():
    """dpill called twice with identical inputs returns the same result."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    h1 = r2py_kernsmooth.dpill(x, y)
    h2 = r2py_kernsmooth.dpill(x, y)
    assert h1 == h2, "dpill is not deterministic / reproducible"


def test_dpill_unsorted_input_same_as_sorted():
    """dpill produces the same result regardless of input ordering."""
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    h_unsorted = r2py_kernsmooth.dpill(x, y)
    h_sorted = r2py_kernsmooth.dpill(x_sorted, y_sorted)
    # Both results may be NaN for certain datasets (e.g. seed=13, n=100) because
    # the plug-in bandwidth estimator encounters a degenerate intermediate value.
    # NaN == NaN is False under IEEE 754, so pytest.approx cannot be used here.
    # We assert that both results agree: either both are NaN, or both are finite
    # and numerically close.
    both_nan = np.isnan(h_unsorted) and np.isnan(h_sorted)
    assert both_nan or h_unsorted == pytest.approx(h_sorted, rel=1e-10), (
        "dpill should be invariant to input ordering"
    )


def test_dpill_r_returns_positive_finite_scalar():
    """Verify R dpill always returns a single positive finite scalar for well-behaved data."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y)
    assert np.isfinite(r_val), f"R returned non-finite bandwidth: {r_val}"
    assert r_val > 0, f"R returned non-positive bandwidth: {r_val}"


def test_dpill_all_default_params_match_r():
    """dpill with every parameter at its default value matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, blockmax=5, divisor=20, trim=0.01, proptrun=0.05,
                     gridsize=401, truncate=True)
    py_val = r2py_kernsmooth.dpill(x, y, blockmax=5, divisor=20, trim=0.01,
                                   proptrun=0.05, gridsize=401, truncate=True)
    _assert_close(py_val, r_val, label="all defaults explicit")


def test_dpill_blockmax_matches_r_for_multiple_values():
    """dpill matches R for every blockmax in {1, 2, 3, 5, 10}."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, 300)
    y = np.cos(x) + rng.normal(0, 0.2, 300)
    for bm in (1, 2, 3, 5, 10):
        r_val = _r_dpill(x, y, blockmax=bm)
        py_val = r2py_kernsmooth.dpill(x, y, blockmax=bm)
        _assert_close(py_val, r_val, label=f"blockmax={bm}")


def test_dpill_divisor_matches_r_for_multiple_values():
    """dpill matches R for every divisor in {5, 10, 20, 40, 50}."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, 300)
    y = np.cos(x) + rng.normal(0, 0.2, 300)
    for div in (5, 10, 20, 40, 50):
        r_val = _r_dpill(x, y, divisor=div)
        py_val = r2py_kernsmooth.dpill(x, y, divisor=div)
        _assert_close(py_val, r_val, label=f"divisor={div}")


def test_dpill_trim_matches_r_for_multiple_values():
    """dpill matches R for trim in {0.01, 0.02, 0.05, 0.1}."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, 300)
    y = np.cos(x) + rng.normal(0, 0.2, 300)
    for trim in (0.01, 0.02, 0.05, 0.1):
        r_val = _r_dpill(x, y, trim=trim)
        if not np.isfinite(r_val):
            continue
        py_val = r2py_kernsmooth.dpill(x, y, trim=trim)
        _assert_close(py_val, r_val, label=f"trim={trim}")


def test_dpill_proptrun_matches_r_for_multiple_values():
    """dpill matches R for proptrun in {0.0, 0.02, 0.05, 0.1}."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, 300)
    y = np.cos(x) + rng.normal(0, 0.2, 300)
    for pt in (0.0, 0.02, 0.05, 0.1):
        r_val = _r_dpill(x, y, proptrun=pt)
        py_val = r2py_kernsmooth.dpill(x, y, proptrun=pt)
        _assert_close(py_val, r_val, label=f"proptrun={pt}")


def test_dpill_gridsize_matches_r_for_multiple_values():
    """dpill matches R for gridsize in {201, 401, 801}."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, 300)
    y = np.cos(x) + rng.normal(0, 0.2, 300)
    for gs in (201, 401, 801):
        r_val = _r_dpill(x, y, gridsize=gs)
        py_val = r2py_kernsmooth.dpill(x, y, gridsize=gs)
        _assert_close(py_val, r_val, label=f"gridsize={gs}")
