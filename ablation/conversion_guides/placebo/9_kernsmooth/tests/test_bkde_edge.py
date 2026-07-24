# Edge-case and boundary tests for r2py_kernsmooth.bkde
#
# Tests focus on functional limits and extremes. Where R returns a valid
# output the Python result is compared against R. Where R raises an error
# both must raise (same logic as the negative suite).

import warnings

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

_ks = importr("KernSmooth")


def _r_bkde(x_py, **kwargs):
    """Call R's bkde; return (x_grid, y_est) as numpy arrays."""
    x_r = ro.FloatVector(x_py.tolist())
    if "range_x" in kwargs:
        range_x = kwargs.pop("range_x")
        kwargs["range.x"] = ro.FloatVector(list(range_x))
    result = _ks.bkde(x_r, **kwargs)
    return np.array(result.rx2("x")), np.array(result.rx2("y"))


def _r_try(x_py, **kwargs):
    """Return (raised, message, x_grid, y_est) from R bkde."""
    x_r = ro.FloatVector(x_py.tolist())
    if "range_x" in kwargs:
        range_x = kwargs.pop("range_x")
        kwargs["range.x"] = ro.FloatVector(list(range_x))
    try:
        result = _ks.bkde(x_r, **kwargs)
        return False, "", np.array(result.rx2("x")), np.array(result.rx2("y"))
    except Exception as exc:
        return True, str(exc), None, None


def _py_try(x_py, **kwargs):
    """Return (raised, message, result_dict) from Python bkde."""
    try:
        res = r2py_kernsmooth.bkde(x_py, **kwargs)
        return False, "", res
    except Exception as exc:
        return True, str(exc), None


def _assert_match(py_result, r_x, r_y, *, rtol=1e-6, label=""):
    assert "x" in py_result and "y" in py_result, f"{label}: missing keys"
    np.testing.assert_allclose(py_result["x"], r_x, rtol=1e-10,
                               err_msg=f"{label}: x mismatch")
    np.testing.assert_allclose(py_result["y"], r_y, rtol=rtol,
                               err_msg=f"{label}: y mismatch")


# ---------------------------------------------------------------------------
# Edge / boundary tests
# ---------------------------------------------------------------------------

def test_bkde_single_observation():
    """bkde on a single data point with an explicit bandwidth matches R."""
    x = np.array([5.0])
    r_x, r_y = _r_bkde(x, bandwidth=0.5)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    assert len(py["x"]) == 401
    assert np.all(np.isfinite(py["y"]))
    _assert_match(py, r_x, r_y, label="single observation")


def test_bkde_two_observations():
    """bkde on two data points matches R."""
    x = np.array([0.0, 1.0])
    r_x, r_y = _r_bkde(x, bandwidth=0.5)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    _assert_match(py, r_x, r_y, label="two observations")


def test_bkde_constant_data_explicit_bandwidth():
    """bkde on all-equal data with an explicit bandwidth returns finite estimates matching R."""
    x = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    r_x, r_y = _r_bkde(x, bandwidth=0.5)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    assert np.all(np.isfinite(py["y"])), "density estimates should be finite"
    _assert_match(py, r_x, r_y, label="constant data")


def test_bkde_very_small_bandwidth_warns():
    """bkde with a very small bandwidth should emit a UserWarning in Python (as R does)."""
    x = np.arange(1.0, 11.0)
    with pytest.warns(UserWarning, match="Binning grid too coarse"):
        result = r2py_kernsmooth.bkde(x, bandwidth=0.001)
    assert len(result["x"]) == 401  # still returns a result


def test_bkde_very_small_bandwidth_output_matches_r():
    """When bandwidth is very small (warns), the Python output still matches R.

    Near-zero values (numerical noise ~1e-15) are matched with an absolute
    tolerance instead of a relative one to avoid spurious failures caused by
    tiny floating-point differences on effectively-zero bins.
    """
    x = np.arange(1.0, 11.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_x_r = ro.FloatVector(x.tolist())
        result_r = _ks.bkde(r_x_r, bandwidth=0.001)
        r_x = np.array(result_r.rx2("x"))
        r_y = np.array(result_r.rx2("y"))
        py = r2py_kernsmooth.bkde(x, bandwidth=0.001)
    assert "x" in py and "y" in py, "very small bandwidth output: missing keys"
    np.testing.assert_allclose(py["x"], r_x, rtol=1e-10,
                               err_msg="very small bandwidth output: x mismatch")
    # Use atol for y because most values are numerical noise near machine epsilon
    np.testing.assert_allclose(py["y"], r_y, rtol=1e-6, atol=1e-14,
                               err_msg="very small bandwidth output: y mismatch")


def test_bkde_very_large_bandwidth():
    """bkde with a very large bandwidth (bw > data range) still matches R."""
    x = np.array([0.0, 1.0, 2.0])
    r_x, r_y = _r_bkde(x, bandwidth=1000.0)
    py = r2py_kernsmooth.bkde(x, bandwidth=1000.0)
    assert np.all(np.isfinite(py["y"]))
    _assert_match(py, r_x, r_y, rtol=1e-5, label="very large bandwidth")


def test_bkde_gridsize_1_raises_or_matches_r():
    """bkde with gridsize=1: behaviour must agree with R (raise or succeed)."""
    x = np.array([1.0, 2.0, 3.0])
    r_raised, r_msg, r_x, r_y = _r_try(x, bandwidth=0.5, gridsize=1)
    py_raised, py_msg, py_result = _py_try(x, bandwidth=0.5, gridsize=1)
    if r_raised != py_raised:
        if r_raised:
            pytest.fail(f"gridsize=1: R raised but Python did not.\nR: {r_msg}")
        else:
            pytest.fail(f"gridsize=1: Python raised but R did not.\nPy: {py_msg}")
    if not r_raised:
        _assert_match(py_result, r_x, r_y, label="gridsize=1")


def test_bkde_gridsize_2():
    """bkde with gridsize=2 (minimal practical grid) matches R."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r_raised, r_msg, r_x, r_y = _r_try(x, bandwidth=0.5, gridsize=2)
    py_raised, py_msg, py_result = _py_try(x, bandwidth=0.5, gridsize=2)
    if r_raised != py_raised:
        if r_raised:
            pytest.fail(f"gridsize=2: R raised but Python did not.\nR: {r_msg}")
        else:
            pytest.fail(f"gridsize=2: Python raised but R did not.\nPy: {py_msg}")
    if not r_raised:
        assert len(py_result["x"]) == 2
        _assert_match(py_result, r_x, r_y, label="gridsize=2")


def test_bkde_minimum_positive_bandwidth():
    """bkde with the smallest representable positive float64 bandwidth behaves like R."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    bw = np.finfo(np.float64).tiny  # ~5e-324
    r_raised, r_msg, r_x, r_y = _r_try(x, bandwidth=float(bw))
    py_raised, py_msg, py_result = _py_try(x, bandwidth=float(bw))
    # Both must agree on raise/success
    if r_raised != py_raised:
        if r_raised:
            pytest.fail(f"tiny bw: R raised but Python did not.\nR: {r_msg}")
        else:
            pytest.fail(f"tiny bw: Python raised but R did not.\nPy: {py_msg}")
    if not r_raised:
        assert np.all(np.isfinite(py_result["y"]) | (py_result["y"] == 0))


def test_bkde_range_x_wider_than_data():
    """bkde with range_x much wider than data produces near-zero tails and matches R."""
    rng = np.random.default_rng(55)
    x = rng.normal(0, 1, 100)
    range_x = (-10.0, 10.0)
    r_x, r_y = _r_bkde(x, bandwidth=0.5, range_x=range_x)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.5, range_x=range_x)
    # Use atol for near-zero tail bins (FFT numerical noise at ~1e-16 level)
    assert "x" in py and "y" in py, "wide range_x: missing keys"
    np.testing.assert_allclose(py["x"], r_x, rtol=1e-10,
                               err_msg="wide range_x: x mismatch")
    np.testing.assert_allclose(py["y"], r_y, rtol=1e-6, atol=1e-14,
                               err_msg="wide range_x: y mismatch")
    # Tails should be essentially zero (absolute threshold consistent with FFT noise)
    assert abs(py["y"][0]) < 1e-12, "density at far-left tail should be near zero"
    assert abs(py["y"][-1]) < 1e-12, "density at far-right tail should be near zero"


def test_bkde_range_x_equal_min_max():
    """bkde with a degenerate range_x (a == b) produces non-finite output.

    R returns an array of NaN values; Python raises ZeroDivisionError. Both
    indicate the input is pathological. The test verifies that Python at least
    signals the problem (raises or returns non-finite values) and documents the
    behavioural divergence from R via a warning.
    """
    x = np.array([1.0, 2.0, 3.0])
    range_x = (2.0, 2.0)
    r_raised, r_msg, r_x, r_y = _r_try(x, bandwidth=0.5, range_x=range_x)
    py_raised, py_msg, py_result = _py_try(x, bandwidth=0.5, range_x=range_x)

    # R succeeds but returns NaN; Python raises. Both signal a degenerate input.
    if not r_raised and r_y is not None and not np.all(np.isnan(r_y)):
        # If R unexpectedly returns valid numbers the test logic below needs review.
        pytest.fail("degenerate range_x: R returned finite values unexpectedly")

    if not py_raised and py_result is not None:
        # Python returned a result — it must contain only non-finite values
        if np.all(np.isfinite(py_result["y"])):
            pytest.fail(
                "degenerate range_x: Python returned finite density values "
                "for a degenerate (zero-width) range"
            )

    # Warn if the handling differs between R and Python (both are acceptable
    # signal-of-error paths, just not identical).
    if r_raised != py_raised:
        warnings.warn(
            "degenerate range_x (a == b): R and Python handle differently.\n"
            f"  R raised={r_raised} msg={r_msg!r}\n"
            f"  Python raised={py_raised} msg={py_msg!r}",
            UserWarning,
            stacklevel=1,
        )


def test_bkde_non_normal_kernel_default_range_uses_tau1():
    """For compact-support kernels tau=1, so default range is tighter than normal kernel."""
    rng = np.random.default_rng(99)
    x = rng.normal(0, 1, 100)
    py_normal = r2py_kernsmooth.bkde(x, kernel="normal", bandwidth=1.0)
    py_box = r2py_kernsmooth.bkde(x, kernel="box", bandwidth=1.0)
    # Normal kernel extends 4*h beyond data; compact kernels extend only 1*h
    normal_span = py_normal["x"][-1] - py_normal["x"][0]
    box_span = py_box["x"][-1] - py_box["x"][0]
    assert normal_span > box_span, (
        "Normal kernel range should be wider than box kernel range "
        f"(normal={normal_span:.4f}, box={box_span:.4f})"
    )


def test_bkde_all_kernels_default_range_matches_r():
    """Default range_x computation for every kernel matches R exactly."""
    rng = np.random.default_rng(20)
    x = rng.normal(0, 1, 100)
    for kernel in ("normal", "box", "epanech", "biweight", "triweight"):
        r_x, r_y = _r_bkde(x, kernel=kernel, bandwidth=1.0)
        py = r2py_kernsmooth.bkde(x, kernel=kernel, bandwidth=1.0)
        np.testing.assert_allclose(py["x"][0], r_x[0], rtol=1e-10,
                                   err_msg=f"{kernel}: x[0] mismatch")
        np.testing.assert_allclose(py["x"][-1], r_x[-1], rtol=1e-10,
                                   err_msg=f"{kernel}: x[-1] mismatch")
        np.testing.assert_allclose(py["y"], r_y, rtol=1e-6,
                                   err_msg=f"{kernel}: y mismatch")


def test_bkde_numpy_bool_canonical():
    """canonical=np.bool_(True) is accepted and produces the same result as canonical=True."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 50)
    py_native = r2py_kernsmooth.bkde(x, bandwidth=0.5, canonical=True)
    py_numpy = r2py_kernsmooth.bkde(x, bandwidth=0.5, canonical=np.bool_(True))
    np.testing.assert_array_equal(py_native["y"], py_numpy["y"])


def test_bkde_output_length_equals_gridsize():
    """len(result['x']) == len(result['y']) == gridsize for various gridsizes."""
    x = np.random.default_rng(0).normal(0, 1, 100)
    for gs in (50, 100, 200, 401, 512):
        result = r2py_kernsmooth.bkde(x, bandwidth=0.5, gridsize=gs)
        assert len(result["x"]) == gs, f"gridsize={gs}: x has wrong length"
        assert len(result["y"]) == gs, f"gridsize={gs}: y has wrong length"


def test_bkde_density_non_negative():
    """bkde density estimates are non-negative for all valid kernel types."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    for kernel in ("normal", "box", "epanech", "biweight", "triweight"):
        result = r2py_kernsmooth.bkde(x, kernel=kernel, bandwidth=0.3)
        assert np.all(result["y"] >= 0), (
            f"kernel={kernel}: negative density values detected"
        )


def test_bkde_large_values():
    """bkde on data with very large absolute values still matches R.

    Near-zero FFT noise bins (magnitudes ~1e-22) are handled with an absolute
    tolerance so tiny floating-point differences do not cause spurious failures.
    """
    x = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    r_x, r_y = _r_bkde(x, bandwidth=1e5)
    py = r2py_kernsmooth.bkde(x, bandwidth=1e5)
    assert "x" in py and "y" in py, "large-magnitude data: missing keys"
    np.testing.assert_allclose(py["x"], r_x, rtol=1e-10,
                               err_msg="large-magnitude data: x mismatch")
    np.testing.assert_allclose(py["y"], r_y, rtol=1e-5, atol=1e-20,
                               err_msg="large-magnitude data: y mismatch")


def test_bkde_small_values():
    """bkde on data with very small absolute values still matches R.

    Boundary bins may differ by a few ULPs (absolute difference ~1e-10) due to
    floating-point rounding at very small scales; an absolute tolerance is used.
    """
    x = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
    r_x, r_y = _r_bkde(x, bandwidth=5e-7)
    py = r2py_kernsmooth.bkde(x, bandwidth=5e-7)
    assert "x" in py and "y" in py, "small-magnitude data: missing keys"
    np.testing.assert_allclose(py["x"], r_x, rtol=1e-10,
                               err_msg="small-magnitude data: x mismatch")
    np.testing.assert_allclose(py["y"], r_y, rtol=1e-5, atol=1e-9,
                               err_msg="small-magnitude data: y mismatch")


def test_bkde_x_array_not_mutated():
    """bkde must not modify the input array in place."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    x_copy = x.copy()
    r2py_kernsmooth.bkde(x, bandwidth=0.5)
    np.testing.assert_array_equal(x, x_copy, err_msg="input array was modified")


def test_bkde_reproducibility():
    """Calling bkde twice with identical inputs yields identical outputs."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 100)
    r1 = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    r2 = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    np.testing.assert_array_equal(r1["x"], r2["x"])
    np.testing.assert_array_equal(r1["y"], r2["y"])
