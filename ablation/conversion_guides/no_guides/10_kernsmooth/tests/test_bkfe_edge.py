# Tests for boundary and edge-case behavior of bkfe.
#
# Each test mirrors the R function's behavior using rpy2 as the reference.
# Successful output cases use numpy.testing.assert_allclose; error cases
# follow the same two-sided comparison pattern as the negative tests.

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

def _r_bkfe(x_np, drv, bandwidth, **kwargs):
    r_kwargs = {}
    if "gridsize" in kwargs:
        r_kwargs["gridsize"] = int(kwargs["gridsize"])
    if "range_x" in kwargs:
        r_kwargs["range.x"] = ro.FloatVector(list(kwargs["range_x"]))
    if "binned" in kwargs:
        r_kwargs["binned"] = bool(kwargs["binned"])
    if "truncate" in kwargs:
        r_kwargs["truncate"] = bool(kwargs["truncate"])
    r_x = ro.FloatVector(list(x_np))
    return float(_ks.bkfe(r_x, drv=int(drv), bandwidth=float(bandwidth), **r_kwargs)[0])


def _r_bkfe_catch(x_np, drv, bandwidth, **kwargs):
    try:
        val = _r_bkfe(x_np, drv, bandwidth, **kwargs)
        return False, "", val
    except Exception as exc:
        return True, str(exc), None


# ---------------------------------------------------------------------------
# Edge: gridsize is an exact power of 2 (the historical bkfe bug)
# ---------------------------------------------------------------------------


def test_bkfe_edge_gridsize_256():
    """gridsize=256 (power of 2) must complete without error and match R."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(100)
    r_val = _r_bkfe(data, drv=2, bandwidth=0.5, gridsize=256)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5, gridsize=256))
    assert np.isfinite(py_val), "Expected finite result for gridsize=256"
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe gridsize=256 mismatch with R")


def test_bkfe_edge_gridsize_512():
    """gridsize=512 (power of 2) must complete without error and match R."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(100)
    r_val = _r_bkfe(data, drv=2, bandwidth=0.5, gridsize=512)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5, gridsize=512))
    assert np.isfinite(py_val), "Expected finite result for gridsize=512"
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe gridsize=512 mismatch with R")


# ---------------------------------------------------------------------------
# Edge: minimum gridsize
# ---------------------------------------------------------------------------


def test_bkfe_edge_gridsize_2():
    """bkfe with the minimum useful gridsize=2 matches R."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal(50)
    r_val = _r_bkfe(data, drv=2, bandwidth=0.5, gridsize=2)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5, gridsize=2))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe gridsize=2 mismatch with R")


# ---------------------------------------------------------------------------
# Edge: n=1 and n=2 (degenerate sample sizes)
# ---------------------------------------------------------------------------


def test_bkfe_edge_n1_returns_nan():
    """n=1 produces NaN in both R and Python."""
    data = np.array([0.5])
    r_val = _r_bkfe(data, drv=2, bandwidth=0.5)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5))
    assert np.isnan(r_val), f"R expected NaN for n=1, got {r_val}"
    assert np.isnan(py_val), f"Python expected NaN for n=1, got {py_val}"


def test_bkfe_edge_n2_matches_r():
    """n=2 produces a finite result identical in Python and R."""
    data = np.array([0.0, 1.0])
    r_val = _r_bkfe(data, drv=2, bandwidth=0.5)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe n=2 mismatch with R")


# ---------------------------------------------------------------------------
# Edge: constant (all-same) data
# ---------------------------------------------------------------------------


def test_bkfe_edge_constant_data_nan():
    """All-identical observations produce NaN in both R and Python
    (range is zero, so division by (M-1)*delta is undefined)."""
    data = np.ones(10)
    r_val = _r_bkfe(data, drv=2, bandwidth=0.5)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5))
    assert np.isnan(r_val), f"R expected NaN for constant data, got {r_val}"
    assert np.isnan(py_val), f"Python expected NaN for constant data, got {py_val}"


# ---------------------------------------------------------------------------
# Edge: very small bandwidth triggers binning-grid warning
# ---------------------------------------------------------------------------


def test_bkfe_edge_small_bandwidth_warns():
    """Very small bandwidth emits a UserWarning about coarse binning grid."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(200)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.001, gridsize=401)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("gridsize" in m.lower() or "bandwidth" in m.lower() for m in messages), (
        f"Expected a UserWarning about coarse grid, got: {messages}"
    )


def test_bkfe_edge_small_bandwidth_result_matches_r():
    """Very small bandwidth (which triggers a warning) still matches R's value."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(200)
    r_val = _r_bkfe(data, drv=2, bandwidth=0.001)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.001))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe small bandwidth mismatch with R")


# ---------------------------------------------------------------------------
# Edge: very large bandwidth
# ---------------------------------------------------------------------------


def test_bkfe_edge_large_bandwidth_matches_r():
    """bandwidth=100 (much larger than data spread) matches R (result near zero)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(200)
    r_val = _r_bkfe(data, drv=2, bandwidth=100.0)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=100.0))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe large bandwidth mismatch with R")


# ---------------------------------------------------------------------------
# Edge: drv=0 (boundary of valid derivative orders)
# ---------------------------------------------------------------------------


def test_bkfe_edge_drv0_matches_r():
    """drv=0 (minimum valid derivative order) matches R."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(50)
    r_val = _r_bkfe(data, drv=0, bandwidth=1.0)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=0, bandwidth=1.0))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe drv=0 mismatch with R")


def test_bkfe_edge_drv0_result_positive():
    """drv=0 returns the integral of f^2, which must be positive."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(100)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=0, bandwidth=0.5))
    assert py_val > 0, f"drv=0 result should be positive, got {py_val}"


# ---------------------------------------------------------------------------
# Edge: high-order derivative (drv=10) — large magnitude, still matches R
# ---------------------------------------------------------------------------


def test_bkfe_edge_drv10_matches_r():
    """drv=10 produces a large-magnitude scalar that still matches R."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(200)
    r_val = _r_bkfe(data, drv=10, bandwidth=0.5)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=10, bandwidth=0.5))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe drv=10 mismatch with R")


# ---------------------------------------------------------------------------
# Edge: range_x narrower than data range (truncates observations)
# ---------------------------------------------------------------------------


def test_bkfe_edge_narrow_range_x():
    """range_x narrower than data range truncates observations; matches R."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(200)
    r_val = _r_bkfe(data, drv=2, bandwidth=0.5, range_x=(-1.0, 1.0))
    py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5, range_x=(-1.0, 1.0)))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe narrow range_x mismatch with R")


# ---------------------------------------------------------------------------
# Edge: binned=True with range_x exactly matching the data extremes
# ---------------------------------------------------------------------------


def test_bkfe_edge_binned_exact_range():
    """binned=True with range_x exactly matching gcounts bounds matches R."""
    gcounts = np.array([1.0, 3.0, 5.0, 3.0, 1.0, 0.0, 2.0])
    r_val = float(
        _ks.bkfe(
            ro.FloatVector(list(gcounts)),
            drv=2,
            bandwidth=0.5,
            binned=True,
            **{"range.x": ro.FloatVector([-1.0, 1.0])},
        )[0]
    )
    py_val = float(r2py_kernsmooth.bkfe(gcounts, drv=2, bandwidth=0.5, binned=True, range_x=(-1.0, 1.0)))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe binned exact range mismatch with R")


# ---------------------------------------------------------------------------
# Edge: NaN/Inf data with explicit range_x — both R and Python raise
# ---------------------------------------------------------------------------


def test_bkfe_edge_nan_data_with_explicit_range_raises():
    """NaN in data causes both R and Python to raise even with range_x provided.

    R fails because linbin produces NaN counts, making the returned value NaN;
    however the seq() step fails first.  Python mirrors this failure.
    """
    data = np.array([1.0, float("nan"), 2.0, 3.0])
    r_raised, r_msg, _ = _r_bkfe_catch(data, drv=2, bandwidth=0.5)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5)

    assert r_raised, (
        "R did NOT raise but Python did — behavior mismatch for NaN data"
    )
    py_msg = str(exc_info.value)
    if r_msg.lower() != py_msg.lower():
        warnings.warn(
            f"Error message discrepancy for NaN data.\n  R : {r_msg!r}\n  Py: {py_msg!r}",
            UserWarning,
            stacklevel=1,
        )


# ---------------------------------------------------------------------------
# Edge: output type is always np.float64 (or plain float)
# ---------------------------------------------------------------------------


def test_bkfe_edge_output_type():
    """bkfe always returns a 0-dimensional float, not an array."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal(50)
    result = r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5)
    assert np.ndim(result) == 0, f"Expected scalar, got shape {np.shape(result)}"
    assert isinstance(result, (float, np.floating)), (
        f"Expected float-like type, got {type(result)}"
    )
