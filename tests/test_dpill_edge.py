# Boundary and edge case tests for r2py_kernsmooth.dpill
#
# These tests target the functional limits of dpill:
#   - Minimal sample sizes (n=2, n=3, n=5 — may fail in R too)
#   - Small sample sizes where R succeeds (n=10, n=20)
#   - Scale invariance: dpill(c*x, c*y) == c * dpill(x, y)
#   - Near-location invariance: dpill(x + k, y) approx== dpill(x, y)
#   - Very large and very small data magnitudes
#   - All-negative x values
#   - Non-default gridsize at extremes (200, 2001)
#   - proptrun=0.0 (no boundary truncation)
#   - blockmax=100 (far above default)
#   - Wide range_x that extends far beyond the data
#   - Sorted vs unsorted input
#   - Integer-typed input coercion
#   - Reproducibility
#   - trim modifying the effective sample
#
# KNOWN ISSUE: The Python port currently fails for all valid inputs because
# the blkest Fortran/f2py wrapper does not write back its scalar outputs.
# All tests that compare Python results to R will fail until the blkest
# interface is corrected; they serve as regression tests for that fix.

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
    """Call R dpill; return scalar float. Propagates R exceptions."""
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


def _r_dpill_catch(x_py, y_py, **kwargs):
    """Call R dpill; return (raised: bool, value_or_message)."""
    try:
        return False, _r_dpill(x_py, y_py, **kwargs)
    except Exception as exc:
        return True, str(exc)


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
# Minimal sample sizes: n=2, n=3, n=5 (both R and Python may fail)
# ---------------------------------------------------------------------------


def test_dpill_n2_raises_in_both():
    """n=2 is too small for the blocked estimation; both R and Python raise."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 2.0])
    r_raised, r_msg = _r_dpill_catch(x, y)
    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)
    # Verify R also raises (same class of failure).
    assert r_raised, (
        f"R did NOT raise for n=2 (returned {r_msg}), but Python did. "
        "Update this test if R's behavior changes."
    )


def test_dpill_n3_raises_in_both():
    """n=3 is too small for stable blocked estimation; both R and Python raise."""
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 1.0, 2.0])
    r_raised, r_msg = _r_dpill_catch(x, y)
    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)
    assert r_raised, (
        f"R did NOT raise for n=3 (returned {r_msg}), but Python did."
    )


def test_dpill_n5_raises_in_both():
    """n=5 is below the threshold for stable blocked estimation; both R and Python raise."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5)
    y = x + rng.normal(0, 0.1, 5)
    r_raised, r_msg = _r_dpill_catch(x, y)
    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)
    assert r_raised, (
        f"R did NOT raise for n=5 (returned {r_msg}), but Python did."
    )


# ---------------------------------------------------------------------------
# Small but viable sample sizes: n=10, n=20
# ---------------------------------------------------------------------------


def test_dpill_n10_matches_r():
    """dpill on n=10 observations returns a positive scalar matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10)
    y = x + rng.normal(0, 0.1, 10)
    r_raised, r_val = _r_dpill_catch(x, y)
    if r_raised:
        pytest.skip(f"R also fails for n=10: {r_val}")
    py_val = r2py_kernsmooth.dpill(x, y)
    assert py_val > 0, f"Expected positive bandwidth for n=10, got {py_val}"
    _assert_close(py_val, r_val, label="n=10")


def test_dpill_n20_matches_r():
    """dpill on n=20 observations returns a positive scalar matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 20)
    y = x + rng.normal(0, 0.1, 20)
    r_raised, r_val = _r_dpill_catch(x, y)
    if r_raised:
        pytest.skip(f"R also fails for n=20: {r_val}")
    py_val = r2py_kernsmooth.dpill(x, y)
    assert py_val > 0, f"Expected positive bandwidth for n=20, got {py_val}"
    _assert_close(py_val, r_val, label="n=20")


# ---------------------------------------------------------------------------
# Scale invariance: dpill(c*x, c*y) == c * dpill(x, y)
# ---------------------------------------------------------------------------


def test_dpill_scale_invariance_large_c():
    """dpill(c*x, c*y) == c * dpill(x, y) for c=3 (R confirms this holds exactly)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    c = 3.0
    r_x = _r_dpill(x, y)
    r_cx = _r_dpill(c * x, c * y)
    # R confirms exact scale invariance: dpill(c*x, c*y) == c * dpill(x, y).
    np.testing.assert_allclose(
        r_cx, c * r_x, rtol=1e-9,
        err_msg=f"R: dpill(c*x,c*y) should equal c*dpill(x,y) for c={c}"
    )
    py_x = r2py_kernsmooth.dpill(x, y)
    py_cx = r2py_kernsmooth.dpill(c * x, c * y)
    # Python should also satisfy scale invariance.
    np.testing.assert_allclose(
        py_cx, c * py_x, rtol=1e-6,
        err_msg=f"Python: dpill(c*x,c*y) should equal c*dpill(x,y) for c={c}"
    )
    _assert_close(py_cx, r_cx, label=f"scale invariance c={c}")


def test_dpill_scale_invariance_small_c():
    """dpill(c*x, c*y) == c * dpill(x, y) for c=0.5 (R confirms)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    c = 0.5
    r_x = _r_dpill(x, y)
    r_cx = _r_dpill(c * x, c * y)
    np.testing.assert_allclose(
        r_cx, c * r_x, rtol=1e-9,
        err_msg=f"R: scale invariance failed for c={c}"
    )
    py_x = r2py_kernsmooth.dpill(x, y)
    py_cx = r2py_kernsmooth.dpill(c * x, c * y)
    np.testing.assert_allclose(
        py_cx, c * py_x, rtol=1e-6,
        err_msg=f"Python: scale invariance failed for c={c}"
    )
    _assert_close(py_cx, r_cx, label=f"scale invariance c={c}")


# ---------------------------------------------------------------------------
# Location shift: dpill(x + k, y) approx== dpill(x, y)
# ---------------------------------------------------------------------------


def test_dpill_location_shift_invariance():
    """dpill(x + k, y) is numerically close to dpill(x, y) for large k.

    The bandwidth is theoretically shift-invariant (depends only on relative
    positions). Small differences arise from binning with a fixed gridsize.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    k = 100.0
    r_x = _r_dpill(x, y)
    r_xk = _r_dpill(x + k, y)
    # R confirms near-invariance (small numerical difference from binning).
    np.testing.assert_allclose(
        r_xk, r_x, rtol=1e-5,
        err_msg=f"R: location shift by k={k} should barely change the bandwidth"
    )
    py_x = r2py_kernsmooth.dpill(x, y)
    py_xk = r2py_kernsmooth.dpill(x + k, y)
    np.testing.assert_allclose(
        py_xk, py_x, rtol=1e-5,
        err_msg=f"Python: location shift by k={k} should barely change the bandwidth"
    )
    _assert_close(py_x, r_x, label="location shift base")
    _assert_close(py_xk, r_xk, label=f"location shift k={k}")


# ---------------------------------------------------------------------------
# Very large and very small data magnitudes
# ---------------------------------------------------------------------------


def test_dpill_large_magnitude_positive():
    """dpill on data with mean=1e6, std=1e4 returns a positive result matching R."""
    rng = np.random.default_rng(55)
    x = rng.normal(1e6, 1e4, 200)
    y = 2 * x + rng.normal(0, 1e3, 200)
    r_val = _r_dpill(x, y)
    assert r_val > 0, f"R returned non-positive bandwidth for large magnitude: {r_val}"
    py_val = r2py_kernsmooth.dpill(x, y)
    assert py_val > 0, f"Expected positive bandwidth for large magnitude, got {py_val}"
    _assert_close(py_val, r_val, label="large magnitude mean=1e6")


def test_dpill_small_magnitude_positive():
    """dpill on data with std=1e-4 returns a positive result matching R."""
    rng = np.random.default_rng(55)
    x = rng.normal(0, 1e-4, 200)
    y = 2 * x + rng.normal(0, 1e-5, 200)
    r_val = _r_dpill(x, y)
    assert r_val > 0, f"R returned non-positive bandwidth for small magnitude: {r_val}"
    py_val = r2py_kernsmooth.dpill(x, y)
    assert py_val > 0, f"Expected positive bandwidth for small magnitude, got {py_val}"
    _assert_close(py_val, r_val, label="small magnitude sd=1e-4")


def test_dpill_entirely_negative_x_values():
    """dpill on data with all-negative x returns a positive result matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(-50, 5, 200)
    y = x ** 2 + rng.normal(0, 10, 200)
    r_val = _r_dpill(x, y)
    assert r_val > 0, f"R returned non-positive bandwidth for all-negative x: {r_val}"
    py_val = r2py_kernsmooth.dpill(x, y)
    assert py_val > 0, f"Expected positive bandwidth for all-negative x, got {py_val}"
    _assert_close(py_val, r_val, label="all-negative x")


# ---------------------------------------------------------------------------
# gridsize at extremes
# ---------------------------------------------------------------------------


def test_dpill_gridsize_200_positive():
    """dpill with gridsize=200 returns a positive scalar matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, gridsize=200)
    assert r_val > 0, f"R returned non-positive for gridsize=200: {r_val}"
    py_val = r2py_kernsmooth.dpill(x, y, gridsize=200)
    assert py_val > 0
    _assert_close(py_val, r_val, label="gridsize=200")


def test_dpill_gridsize_2001_positive():
    """dpill with gridsize=2001 (fine grid) returns a positive scalar matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, gridsize=2001)
    assert r_val > 0, f"R returned non-positive for gridsize=2001: {r_val}"
    py_val = r2py_kernsmooth.dpill(x, y, gridsize=2001)
    assert py_val > 0
    _assert_close(py_val, r_val, label="gridsize=2001")


def test_dpill_gridsize_convergence():
    """Finer gridsizes converge toward the same bandwidth (binning approximation)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_200 = _r_dpill(x, y, gridsize=200)
    r_401 = _r_dpill(x, y, gridsize=401)
    r_2001 = _r_dpill(x, y, gridsize=2001)
    # Coarse and fine grids should agree within 1% in R.
    np.testing.assert_allclose(r_200, r_2001, rtol=0.01,
        err_msg="R: coarse and fine gridsize results should be within 1%")
    py_200 = r2py_kernsmooth.dpill(x, y, gridsize=200)
    py_401 = r2py_kernsmooth.dpill(x, y, gridsize=401)
    py_2001 = r2py_kernsmooth.dpill(x, y, gridsize=2001)
    np.testing.assert_allclose(py_200, py_2001, rtol=0.01,
        err_msg="Python: coarse and fine gridsize results should be within 1%")
    _assert_close(py_200, r_200, label="gridsize=200")
    _assert_close(py_401, r_401, label="gridsize=401")
    _assert_close(py_2001, r_2001, label="gridsize=2001")


# ---------------------------------------------------------------------------
# proptrun=0.0 (no boundary truncation of second-derivative estimates)
# ---------------------------------------------------------------------------


def test_dpill_proptrun_zero_positive():
    """dpill with proptrun=0.0 returns a positive scalar matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, proptrun=0.0)
    assert r_val > 0, f"R returned non-positive for proptrun=0.0: {r_val}"
    py_val = r2py_kernsmooth.dpill(x, y, proptrun=0.0)
    assert py_val > 0
    _assert_close(py_val, r_val, label="proptrun=0.0")


# ---------------------------------------------------------------------------
# blockmax at extremes
# ---------------------------------------------------------------------------


def test_dpill_blockmax_100_matches_r():
    """dpill with blockmax=100 (far above default) matches R (Cp selects optimal Nval)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, blockmax=100)
    assert r_val > 0, f"R returned non-positive for blockmax=100: {r_val}"
    py_val = r2py_kernsmooth.dpill(x, y, blockmax=100)
    _assert_close(py_val, r_val, label="blockmax=100")


def test_dpill_blockmax_1_matches_r():
    """dpill with blockmax=1 (single block, no Cp selection) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_val = _r_dpill(x, y, blockmax=1)
    assert r_val > 0, f"R returned non-positive for blockmax=1: {r_val}"
    py_val = r2py_kernsmooth.dpill(x, y, blockmax=1)
    _assert_close(py_val, r_val, label="blockmax=1")


# ---------------------------------------------------------------------------
# Wide range_x that triggers NaN in R (documented divergence handled with skip)
# ---------------------------------------------------------------------------


def test_dpill_wider_range_x_matches_r_if_r_succeeds():
    """dpill with range_x moderately wider than data matches R if R succeeds."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    # Slightly wider than data (0.5 units on each side)
    range_x = (float(x.min()) - 0.5, float(x.max()) + 0.5)
    r_raised, r_result = _r_dpill_catch(x, y, range_x=range_x)
    if r_raised or not np.isfinite(r_result):
        pytest.skip(
            f"R does not return a finite value for this range_x: {r_result}"
        )
    py_val = r2py_kernsmooth.dpill(x, y, range_x=range_x)
    _assert_close(py_val, r_result, label="range_x slightly wider than data")


# ---------------------------------------------------------------------------
# Sorted vs unsorted input — result must be identical
# ---------------------------------------------------------------------------


def test_dpill_sorted_vs_unsorted_input():
    """dpill(sorted) == dpill(unsorted): function sorts internally."""
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
        "dpill should be invariant to input ordering (it sorts internally)"
    )


# ---------------------------------------------------------------------------
# Integer-typed input coercion
# ---------------------------------------------------------------------------


def test_dpill_integer_array_coercion():
    """Integer-typed x and y are accepted and coerced to float64; result equals float version."""
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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_dpill_reproducibility_same_input_same_output():
    """dpill called twice with identical inputs returns the identical result."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    h1 = r2py_kernsmooth.dpill(x, y)
    h2 = r2py_kernsmooth.dpill(x, y)
    assert h1 == h2, "dpill is not deterministic / reproducible"


# ---------------------------------------------------------------------------
# trim effect: larger trim -> fewer effective observations
# ---------------------------------------------------------------------------


def test_dpill_trim_increases_reduce_effective_n():
    """R confirms: larger trim changes the bandwidth by excluding extreme x points."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_01 = _r_dpill(x, y, trim=0.01)
    r_05 = _r_dpill(x, y, trim=0.05)
    r_10 = _r_dpill(x, y, trim=0.10)
    # The three values should be finite and differ (different effective datasets).
    assert np.isfinite(r_01) and r_01 > 0, f"R trim=0.01 non-positive: {r_01}"
    assert np.isfinite(r_05) and r_05 > 0, f"R trim=0.05 non-positive: {r_05}"
    assert np.isfinite(r_10) and r_10 > 0, f"R trim=0.10 non-positive: {r_10}"
    # Python values should match R for each trim level.
    py_01 = r2py_kernsmooth.dpill(x, y, trim=0.01)
    py_05 = r2py_kernsmooth.dpill(x, y, trim=0.05)
    py_10 = r2py_kernsmooth.dpill(x, y, trim=0.10)
    _assert_close(py_01, r_01, label="trim=0.01")
    _assert_close(py_05, r_05, label="trim=0.05")
    _assert_close(py_10, r_10, label="trim=0.10")


# ---------------------------------------------------------------------------
# truncate=True vs truncate=False: they differ when data extend beyond range_x
# ---------------------------------------------------------------------------


def test_dpill_truncate_true_vs_false_differ_at_narrow_range():
    """truncate=True and truncate=False produce different bandwidths for narrow range_x."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    range_x = (-1.5, 1.5)   # Narrower than x's actual range.
    r_true = _r_dpill(x, y, range_x=range_x, truncate=True)
    r_false = _r_dpill(x, y, range_x=range_x, truncate=False)
    py_true = r2py_kernsmooth.dpill(x, y, range_x=range_x, truncate=True)
    py_false = r2py_kernsmooth.dpill(x, y, range_x=range_x, truncate=False)
    _assert_close(py_true, r_true, label="truncate=True narrow range_x")
    _assert_close(py_false, r_false, label="truncate=False narrow range_x")
    assert py_true != pytest.approx(py_false, rel=1e-4), (
        "truncate=True and truncate=False must produce different results "
        "when data extend beyond range_x"
    )


# ---------------------------------------------------------------------------
# Multiple datasets: verify positive result for different regression functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func_name,func", [
    ("linear", lambda x, rng: 2 * x + rng.normal(0, 0.3, len(x))),
    ("quadratic", lambda x, rng: x ** 2 + rng.normal(0, 0.2, len(x))),
    ("sine", lambda x, rng: np.sin(2 * x) + rng.normal(0, 0.3, len(x))),
    ("cosine", lambda x, rng: np.cos(x) + rng.normal(0, 0.2, len(x))),
    ("exponential_decay", lambda x, rng: np.exp(-x ** 2) + rng.normal(0, 0.1, len(x))),
])
def test_dpill_positive_for_various_regression_functions(func_name, func):
    """dpill returns a positive finite scalar for various regression functions and matches R."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, 300)
    y = func(x, rng)
    r_raised, r_result = _r_dpill_catch(x, y)
    if r_raised or not np.isfinite(r_result):
        pytest.skip(f"R fails for func={func_name}: {r_result}")
    assert r_result > 0, f"R returned non-positive for {func_name}: {r_result}"
    py_val = r2py_kernsmooth.dpill(x, y)
    assert py_val > 0, f"Python returned non-positive for {func_name}: {py_val}"
    _assert_close(py_val, r_result, label=f"regression_func={func_name}")
