# Edge-case and boundary tests for r2py_kernsmooth.locpoly
#
# Tests focus on functional limits and extremes. Where R returns a valid
# output the Python result is compared against R. Where R raises an error
# both must raise (using the same logic as the negative suite).
#
# Covered scenarios:
#   - Minimum practical data sizes (n=2 observations).
#   - Single-observation inputs (expected to raise in both R and Python).
#   - Very large / very small bandwidth values.
#   - Very large / very small x and y magnitudes.
#   - Constant bandwidth vector (same as scalar).
#   - Maximum and minimum drv values exercised.
#   - Density mode on extreme inputs.
#   - Degenerate range_x (a == b).
#   - Gridsize extremes.
#   - Output finiteness checks.
#   - x grid is equispaced and sorted.

import warnings

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

_ks = importr("KernSmooth")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _r_locpoly(x_py, y_py=None, **kwargs):
    """Call R's locpoly; return (x_grid, y_est) as numpy arrays."""
    x_r = ro.FloatVector(x_py.tolist())
    args = [x_r]
    if y_py is not None:
        args.append(ro.FloatVector(y_py.tolist()))
    if "range_x" in kwargs:
        range_x = kwargs.pop("range_x")
        kwargs["range.x"] = ro.FloatVector(list(range_x))
    if "bandwidth" in kwargs and isinstance(kwargs["bandwidth"], np.ndarray):
        kwargs["bandwidth"] = ro.FloatVector(kwargs["bandwidth"].tolist())
    result = _ks.locpoly(*args, **kwargs)
    return np.array(result.rx2("x")), np.array(result.rx2("y"))


def _r_try(x_py, y_py=None, **kwargs):
    """Return (raised, message, x_grid, y_est) from R locpoly."""
    x_r = ro.FloatVector(x_py.tolist())
    args = [x_r]
    if y_py is not None:
        args.append(ro.FloatVector(y_py.tolist()))
    if "range_x" in kwargs:
        range_x = kwargs.pop("range_x")
        kwargs["range.x"] = ro.FloatVector(list(range_x))
    if "bandwidth" in kwargs and isinstance(kwargs["bandwidth"], np.ndarray):
        kwargs["bandwidth"] = ro.FloatVector(kwargs["bandwidth"].tolist())
    try:
        result = _ks.locpoly(*args, **kwargs)
        return False, "", np.array(result.rx2("x")), np.array(result.rx2("y"))
    except Exception as exc:
        return True, str(exc), None, None


def _py_try(x_py, y_py=None, **kwargs):
    """Return (raised, message, result_dict) from Python locpoly."""
    try:
        res = r2py_kernsmooth.locpoly(x_py, y_py, **kwargs)
        return False, "", res
    except Exception as exc:
        return True, str(exc), None


def _assert_match(py_result, r_x, r_y, *, rtol=1e-6, atol=0.0, label=""):
    assert "x" in py_result and "y" in py_result, f"{label}: missing keys"
    np.testing.assert_allclose(
        py_result["x"], r_x, rtol=1e-10, atol=0.0, err_msg=f"{label}: x mismatch"
    )
    np.testing.assert_allclose(
        py_result["y"], r_y, rtol=rtol, atol=atol, err_msg=f"{label}: y mismatch"
    )


def _check_both_raise(x_py, y_py=None, *, r_kwargs=None, py_kwargs=None, label):
    """Assert both R and Python raise; warn when messages differ."""
    r_kwargs = r_kwargs or {}
    py_kwargs = py_kwargs or {}
    r_raised, r_msg, _, _ = _r_try(x_py, y_py, **r_kwargs)
    py_raised, py_msg, _ = _py_try(x_py, y_py, **py_kwargs)
    if not r_raised and not py_raised:
        pytest.fail(f"{label}: neither raised")
    if r_raised and not py_raised:
        pytest.fail(f"{label}: R raised but Python did not.\nR: {r_msg}")
    if not r_raised and py_raised:
        pytest.fail(f"{label}: Python raised but R did not.\nPy: {py_msg}")
    if r_msg.strip() != py_msg.strip():
        warnings.warn(
            f"{label}: both raised, messages differ.\n"
            f"  R:      {r_msg!r}\n  Python: {py_msg!r}",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Edge tests — minimum data sizes
# ---------------------------------------------------------------------------

def test_locpoly_regression_single_observation_raises():
    """locpoly regression on a single observation raises in both R and Python."""
    x = np.array([3.0])
    y = np.array([7.0])
    _check_both_raise(
        x, y,
        r_kwargs={"bandwidth": 0.5},
        py_kwargs={"bandwidth": 0.5},
        label="single observation regression",
    )


def test_locpoly_density_single_observation_raises():
    """locpoly density estimation on a single observation raises in both."""
    x = np.array([3.0])
    _check_both_raise(
        x,
        r_kwargs={"bandwidth": 0.5},
        py_kwargs={"bandwidth": 0.5},
        label="single observation density",
    )


def test_locpoly_regression_two_observations():
    """locpoly regression on two observations matches R."""
    x = np.array([1.0, 5.0])
    y = np.array([2.0, 10.0])
    r_x, r_y = _r_locpoly(x, y, bandwidth=2.0)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=2.0)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="two observations regression")


def test_locpoly_density_two_observations():
    """locpoly density on two observations matches R."""
    x = np.array([1.0, 5.0])
    r_x, r_y = _r_locpoly(x, bandwidth=1.0)
    py = r2py_kernsmooth.locpoly(x, bandwidth=1.0)
    _assert_match(py, r_x, r_y, label="two observations density")


# ---------------------------------------------------------------------------
# Edge tests — bandwidth extremes
# ---------------------------------------------------------------------------

def test_locpoly_very_large_bandwidth_regression():
    """locpoly regression with a bandwidth larger than the data range matches R."""
    x = np.linspace(0, 5, 100)
    y = x ** 2
    r_x, r_y = _r_locpoly(x, y, bandwidth=100.0)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=100.0)
    assert np.all(np.isfinite(py["y"])), "y should be finite for very large bw"
    _assert_match(py, r_x, r_y, rtol=1e-5, label="very large bandwidth regression")


def test_locpoly_very_large_bandwidth_density():
    """locpoly density with a bandwidth much larger than data spread matches R."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r_x, r_y = _r_locpoly(x, bandwidth=100.0)
    py = r2py_kernsmooth.locpoly(x, bandwidth=100.0)
    assert np.all(np.isfinite(py["y"])), "density y should be finite for large bw"
    _assert_match(py, r_x, r_y, rtol=1e-5, label="very large bandwidth density")


def test_locpoly_minimum_bandwidth_raises():
    """A bandwidth so small that Lvec==0 raises identically in both."""
    # With gridsize=401 and range [0,5], delta=5/400=0.0125.
    # Lvec = floor(4 * bw / delta). For bw=1e-4: floor(0.032) = 0 -> error.
    x = np.linspace(0, 5, 100)
    y = x ** 2
    _check_both_raise(
        x, y,
        r_kwargs={"bandwidth": 1e-4},
        py_kwargs={"bandwidth": 1e-4},
        label="minimum bandwidth Lvec==0",
    )


def test_locpoly_constant_bandwidth_vector_equals_scalar():
    """A constant bandwidth vector of length gridsize gives the same result as a scalar."""
    x = np.linspace(0, 5, 200)
    y = x ** 2
    bw_scalar = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    bw_vec = np.full(401, 0.5)
    bw_vector_result = r2py_kernsmooth.locpoly(x, y, bandwidth=bw_vec)
    # Both should also match R's scalar bandwidth output
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5)
    _assert_match(bw_scalar, r_x, r_y, label="scalar bw baseline")
    _assert_match(bw_vector_result, r_x, r_y, rtol=1e-5, label="constant bw vector")


# ---------------------------------------------------------------------------
# Edge tests — x and y magnitude extremes
# ---------------------------------------------------------------------------

def test_locpoly_large_x_values():
    """locpoly regression on very large x values (1e6 scale) matches R."""
    x = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    y = x ** 2 / 1e12
    r_x, r_y = _r_locpoly(x, y, bandwidth=5e5)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=5e5)
    assert np.all(np.isfinite(py["y"])), "y should be finite for large x scale"
    np.testing.assert_allclose(
        py["x"], r_x, rtol=1e-10, err_msg="large x: x mismatch"
    )
    np.testing.assert_allclose(
        py["y"], r_y, rtol=1e-5, atol=1e-20, err_msg="large x: y mismatch"
    )


def test_locpoly_small_x_values():
    """locpoly regression on very small x values (1e-6 scale) matches R."""
    x = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
    y = x * 1e6
    r_x, r_y = _r_locpoly(x, y, bandwidth=5e-7)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=5e-7)
    assert np.all(np.isfinite(py["y"])), "y should be finite for small x scale"
    np.testing.assert_allclose(
        py["x"], r_x, rtol=1e-10, err_msg="small x: x mismatch"
    )
    np.testing.assert_allclose(
        py["y"], r_y, rtol=1e-5, atol=1e-9, err_msg="small x: y mismatch"
    )


def test_locpoly_large_y_values():
    """locpoly regression with very large y magnitudes (1e8 scale) matches R."""
    x = np.linspace(0, 5, 100)
    y = x * 1e8
    r_x, r_y = _r_locpoly(x, y, bandwidth=1.0)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=1.0)
    assert np.all(np.isfinite(py["y"])), "y should be finite for large y scale"
    np.testing.assert_allclose(
        py["y"], r_y, rtol=1e-5, atol=1.0, err_msg="large y: y mismatch"
    )


# ---------------------------------------------------------------------------
# Edge tests — degenerate range_x
# ---------------------------------------------------------------------------

def test_locpoly_degenerate_range_x_zero_width():
    """locpoly with range_x[0] == range_x[1] signals a degenerate input.

    R and Python may differ in how they signal this (error vs NaN output).
    The test checks that Python at least signals the problem (raises or
    returns non-finite values), and documents any divergence via a warning.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x ** 2
    range_x = (3.0, 3.0)
    r_raised, r_msg, r_x, r_y = _r_try(x, y, bandwidth=0.5, range_x=range_x)
    py_raised, py_msg, py_result = _py_try(x, y, bandwidth=0.5, range_x=range_x)

    # If R returns finite numbers for a zero-width range that would be unexpected.
    if not r_raised and r_y is not None and np.all(np.isfinite(r_y)):
        pytest.fail("degenerate range_x: R returned all-finite values unexpectedly")

    if not py_raised and py_result is not None:
        if np.all(np.isfinite(py_result["y"])):
            pytest.fail(
                "degenerate range_x: Python returned all-finite values "
                "for a zero-width range"
            )

    if r_raised != py_raised:
        warnings.warn(
            "degenerate range_x (a==b): R and Python handle differently.\n"
            f"  R raised={r_raised} msg={r_msg!r}\n"
            f"  Python raised={py_raised} msg={py_msg!r}",
            UserWarning,
            stacklevel=1,
        )


# ---------------------------------------------------------------------------
# Edge tests — gridsize extremes
# ---------------------------------------------------------------------------

def test_locpoly_gridsize_2():
    """locpoly with gridsize=2 (minimal practical grid) matches R."""
    x = np.linspace(0, 5, 50)
    y = x ** 2
    r_raised, r_msg, r_x, r_y = _r_try(x, y, bandwidth=1.0, gridsize=2)
    py_raised, py_msg, py_result = _py_try(x, y, bandwidth=1.0, gridsize=2)
    if r_raised != py_raised:
        if r_raised:
            pytest.fail(f"gridsize=2: R raised but Python did not.\nR: {r_msg}")
        else:
            pytest.fail(f"gridsize=2: Python raised but R did not.\nPy: {py_msg}")
    if not r_raised:
        assert len(py_result["x"]) == 2, "gridsize=2: x should have 2 points"
        _assert_match(py_result, r_x, r_y, label="gridsize=2")


def test_locpoly_gridsize_large():
    """locpoly with gridsize=801 returns 801 points and matches R."""
    x = np.linspace(0, 5, 200)
    y = x ** 2
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, gridsize=801)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, gridsize=801)
    assert len(py["x"]) == 801
    assert len(py["y"]) == 801
    _assert_match(py, r_x, r_y, label="gridsize=801")


# ---------------------------------------------------------------------------
# Edge tests — drv extremes
# ---------------------------------------------------------------------------

def test_locpoly_drv0_density_output_non_negative():
    """locpoly density drv=0 output should be non-negative (density values)."""
    x = np.random.default_rng(42).normal(0, 1, 500)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.3)
    # Interior estimates should be non-negative; boundary effects may dip slightly.
    interior = py["y"][20:-20]
    assert np.all(interior >= -1e-10), (
        "density estimates in the interior should be non-negative"
    )


def test_locpoly_regression_all_drv_values_finite():
    """locpoly regression with drv=0,1,2,3 all produce finite output matching R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    for drv in [0, 1, 2, 3]:
        r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, drv=drv)
        py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, drv=drv)
        assert np.all(np.isfinite(py["y"])), f"drv={drv}: non-finite values in output"
        # atol=1e-12 guards against ~1e-17 floating-point noise where sin(x)~0.
        _assert_match(py, r_x, r_y, rtol=1e-5, atol=1e-12, label=f"drv={drv}")


# ---------------------------------------------------------------------------
# Edge tests — density at boundaries
# ---------------------------------------------------------------------------

def test_locpoly_density_wide_range_near_zero_tails():
    """locpoly density with range_x much wider than data has near-zero tails matching R."""
    x = np.random.default_rng(55).normal(0, 1, 300)
    range_x = (-10.0, 10.0)
    r_x, r_y = _r_locpoly(x, bandwidth=0.5, range_x=range_x)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.5, range_x=range_x)
    np.testing.assert_allclose(
        py["x"], r_x, rtol=1e-10, err_msg="wide density range: x mismatch"
    )
    np.testing.assert_allclose(
        py["y"], r_y, rtol=1e-6, atol=1e-14, err_msg="wide density range: y mismatch"
    )
    assert abs(py["y"][0]) < 1e-10, "density far-left tail should be ~0"
    assert abs(py["y"][-1]) < 1e-10, "density far-right tail should be ~0"


# ---------------------------------------------------------------------------
# Edge tests — binned mode boundaries
# ---------------------------------------------------------------------------

def test_locpoly_binned_drv1_matches_raw():
    """locpoly binned=True with drv=1 gives a result matching R (binned reference)."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    gpoints = np.linspace(float(x.min()), float(x.max()), 401)
    out_bin = r2py_kernsmooth.rlbin(x, y, gpoints)
    xcounts = out_bin["xcounts"]
    ycounts = out_bin["ycounts"]
    range_x = (float(x.min()), float(x.max()))
    r_x, r_y = _r_locpoly(
        xcounts, ycounts, bandwidth=0.5, range_x=range_x, binned=True, drv=1
    )
    py = r2py_kernsmooth.locpoly(
        xcounts, ycounts, bandwidth=0.5, range_x=range_x, binned=True, drv=1
    )
    _assert_match(py, r_x, r_y, rtol=1e-5, label="binned drv=1")


# ---------------------------------------------------------------------------
# Edge tests — output grid properties
# ---------------------------------------------------------------------------

def test_locpoly_x_grid_equispaced_various_gridsizes():
    """For various gridsize values, the returned x grid is uniformly spaced."""
    x = np.random.default_rng(0).normal(0, 1, 100)
    y = x + np.random.default_rng(0).normal(0, 0.1, 100)
    for gs in [50, 100, 200, 401]:
        result = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, gridsize=gs)
        diffs = np.diff(result["x"])
        assert np.all(diffs > 0), f"gridsize={gs}: x grid not strictly increasing"
        np.testing.assert_allclose(
            diffs, diffs[0], rtol=1e-10,
            err_msg=f"gridsize={gs}: x grid not uniformly spaced",
        )


def test_locpoly_output_y_all_finite_typical_inputs():
    """locpoly output y values are all finite for typical regression inputs."""
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 10, 500)
    y = np.sin(x) + rng.normal(0, 0.1, 500)
    result = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    assert np.all(np.isfinite(result["y"])), "y values should all be finite"


def test_locpoly_regression_negative_x_data():
    """locpoly regression on all-negative x data matches R."""
    x = np.linspace(-5, -1, 100)
    y = x ** 2
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="all-negative x regression")


def test_locpoly_density_large_n_integral():
    """locpoly density on n=1000 points integrates to approximately 1."""
    x = np.random.default_rng(99).normal(0, 1, 1000)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.3)
    integral = np.trapezoid(py["y"], py["x"])
    assert abs(integral - 1.0) < 0.05, (
        f"Density integral for large n should be ~1 (got {integral:.4f})"
    )


def test_locpoly_density_and_regression_same_x_different_y():
    """locpoly density mode and regression mode on the same x produce different y."""
    rng = np.random.default_rng(33)
    x = rng.normal(0, 1, 300)
    y = x ** 2 + rng.normal(0, 0.1, 300)
    py_density = r2py_kernsmooth.locpoly(x, bandwidth=0.3)
    py_regression = r2py_kernsmooth.locpoly(x, y, bandwidth=0.3)
    assert not np.allclose(py_density["y"], py_regression["y"]), (
        "density and regression on the same x should yield different y"
    )
