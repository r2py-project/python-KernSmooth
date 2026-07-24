# Tests for the positive (valid input) behavior of bkfe.
#
# Each test calls KernSmooth::bkfe in R via rpy2 to obtain a reference scalar,
# calls the Python bkfe with the same arguments, and asserts that the two
# results agree to within a small relative tolerance.

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

_ks = importr("KernSmooth")

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_DATA_100 = _RNG.standard_normal(100)
_DATA_200 = _RNG.standard_normal(200)
_DATA_1000 = _RNG.standard_normal(1000)


def _r_bkfe(x_np, drv, bandwidth, **kwargs):
    """Call R KernSmooth::bkfe and return a Python float."""
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


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------


def test_bkfe_drv0_standard_normal():
    """drv=0 returns the integral of f^2 (should be positive for any density)."""
    r_val = _r_bkfe(_DATA_100, drv=0, bandwidth=0.5)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_100, drv=0, bandwidth=0.5))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe drv=0 mismatch with R")


def test_bkfe_drv2_standard_normal():
    """drv=2 with standard normal data and default gridsize matches R."""
    r_val = _r_bkfe(_DATA_200, drv=2, bandwidth=0.3)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=2, bandwidth=0.3))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe drv=2 mismatch with R")


def test_bkfe_drv4_standard_normal():
    """drv=4 with standard normal data matches R."""
    r_val = _r_bkfe(_DATA_200, drv=4, bandwidth=0.3)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=4, bandwidth=0.3))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe drv=4 mismatch with R")


def test_bkfe_drv6_standard_normal():
    """drv=6 with standard normal data matches R."""
    r_val = _r_bkfe(_DATA_200, drv=6, bandwidth=0.3)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=6, bandwidth=0.3))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe drv=6 mismatch with R")


def test_bkfe_drv8_standard_normal():
    """drv=8 with standard normal data matches R."""
    r_val = _r_bkfe(_DATA_200, drv=8, bandwidth=0.3)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=8, bandwidth=0.3))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe drv=8 mismatch with R")


def test_bkfe_drv10_standard_normal():
    """drv=10 (high-order derivative) with standard normal data matches R."""
    r_val = _r_bkfe(_DATA_200, drv=10, bandwidth=0.5)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=10, bandwidth=0.5))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe drv=10 mismatch with R")


def test_bkfe_large_n():
    """bkfe with n=1000 observations matches R."""
    r_val = _r_bkfe(_DATA_1000, drv=2, bandwidth=0.5)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_1000, drv=2, bandwidth=0.5))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe large n mismatch with R")


def test_bkfe_uniform_distribution_drv2():
    """bkfe on uniform data with drv=2 matches R."""
    data_unif = _RNG.uniform(0, 1, 200)
    r_val = _r_bkfe(data_unif, drv=2, bandwidth=0.3)
    py_val = float(r2py_kernsmooth.bkfe(data_unif, drv=2, bandwidth=0.3))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe uniform drv=2 mismatch with R")


def test_bkfe_uniform_distribution_drv4():
    """bkfe on uniform data with drv=4 matches R."""
    data_unif = _RNG.uniform(0, 1, 200)
    r_val = _r_bkfe(data_unif, drv=4, bandwidth=0.3)
    py_val = float(r2py_kernsmooth.bkfe(data_unif, drv=4, bandwidth=0.3))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe uniform drv=4 mismatch with R")


def test_bkfe_exponential_distribution():
    """bkfe on exponential data matches R."""
    data_exp = _RNG.exponential(scale=1.0, size=100)
    r_val = _r_bkfe(data_exp, drv=2, bandwidth=0.5)
    py_val = float(r2py_kernsmooth.bkfe(data_exp, drv=2, bandwidth=0.5))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe exponential mismatch with R")


def test_bkfe_integer_array_input():
    """bkfe accepts integer-typed numpy arrays and matches R."""
    int_data = np.arange(1, 21, dtype=np.int64)
    r_val = float(
        _ks.bkfe(ro.IntVector(list(int_data)), drv=2, bandwidth=2.0)[0]
    )
    py_val = float(r2py_kernsmooth.bkfe(int_data, drv=2, bandwidth=2.0))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe integer input mismatch with R")


def test_bkfe_custom_gridsize():
    """bkfe with a non-default gridsize=200 matches R."""
    r_val = _r_bkfe(_DATA_200, drv=2, bandwidth=0.3, gridsize=200)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=2, bandwidth=0.3, gridsize=200))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe custom gridsize mismatch with R")


def test_bkfe_custom_range_x():
    """bkfe with an explicit range_x matches R."""
    r_val = _r_bkfe(_DATA_200, drv=2, bandwidth=0.5, range_x=(-2.0, 2.0))
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=2, bandwidth=0.5, range_x=(-2.0, 2.0)))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe custom range_x mismatch with R")


def test_bkfe_wide_range_x():
    """bkfe with range_x wider than data range matches R."""
    r_val = _r_bkfe(_DATA_200, drv=2, bandwidth=0.5, range_x=(-5.0, 5.0))
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=2, bandwidth=0.5, range_x=(-5.0, 5.0)))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe wide range_x mismatch with R")


def test_bkfe_truncate_true():
    """bkfe with truncate=True matches R (default behaviour)."""
    r_val = _r_bkfe(_DATA_200, drv=2, bandwidth=0.5, truncate=True)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=2, bandwidth=0.5, truncate=True))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe truncate=True mismatch with R")


def test_bkfe_truncate_false():
    """bkfe with truncate=False matches R and differs from truncate=True."""
    r_val = _r_bkfe(_DATA_200, drv=2, bandwidth=0.5, truncate=False)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=2, bandwidth=0.5, truncate=False))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe truncate=False mismatch with R")
    # Truncate=True and truncate=False should give different results for this data
    py_val_t = float(r2py_kernsmooth.bkfe(_DATA_200, drv=2, bandwidth=0.5, truncate=True))
    assert py_val != py_val_t, "truncate=True and truncate=False should produce different results"


def test_bkfe_binned_symmetric_counts_drv2():
    """bkfe with pre-binned symmetric counts and drv=2 matches R."""
    gcounts = np.array([0.0, 2.0, 5.0, 8.0, 10.0, 8.0, 5.0, 2.0, 0.0])
    r_val = float(
        _ks.bkfe(
            ro.FloatVector(list(gcounts)),
            drv=2,
            bandwidth=0.5,
            binned=True,
            **{"range.x": ro.FloatVector([-2.0, 2.0])},
        )[0]
    )
    py_val = float(r2py_kernsmooth.bkfe(gcounts, drv=2, bandwidth=0.5, binned=True, range_x=(-2.0, 2.0)))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe binned symmetric drv=2 mismatch with R")


def test_bkfe_binned_symmetric_counts_drv4():
    """bkfe with pre-binned symmetric counts and drv=4 matches R."""
    gcounts = np.array([0.0, 2.0, 5.0, 8.0, 10.0, 8.0, 5.0, 2.0, 0.0])
    r_val = float(
        _ks.bkfe(
            ro.FloatVector(list(gcounts)),
            drv=4,
            bandwidth=0.5,
            binned=True,
            **{"range.x": ro.FloatVector([-2.0, 2.0])},
        )[0]
    )
    py_val = float(r2py_kernsmooth.bkfe(gcounts, drv=4, bandwidth=0.5, binned=True, range_x=(-2.0, 2.0)))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe binned symmetric drv=4 mismatch with R")


def test_bkfe_binned_symmetric_counts_drv0():
    """bkfe with pre-binned counts and drv=0 matches R."""
    gcounts = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 8.0, 4.0, 2.0, 1.0])
    r_val = float(
        _ks.bkfe(
            ro.FloatVector(list(gcounts)),
            drv=0,
            bandwidth=0.5,
            binned=True,
            **{"range.x": ro.FloatVector([-2.0, 2.0])},
        )[0]
    )
    py_val = float(r2py_kernsmooth.bkfe(gcounts, drv=0, bandwidth=0.5, binned=True, range_x=(-2.0, 2.0)))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe binned sym drv=0 mismatch with R")


def test_bkfe_binned_large_gcounts():
    """bkfe with 50-element binned counts array matches R."""
    gcounts = _RNG.integers(0, 20, size=50).astype(float)
    r_val = float(
        _ks.bkfe(
            ro.FloatVector(list(gcounts)),
            drv=2,
            bandwidth=0.5,
            binned=True,
            **{"range.x": ro.FloatVector([0.0, 5.0])},
        )[0]
    )
    py_val = float(r2py_kernsmooth.bkfe(gcounts, drv=2, bandwidth=0.5, binned=True, range_x=(0.0, 5.0)))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe binned large gcounts mismatch with R")


def test_bkfe_returns_scalar():
    """bkfe returns a numpy scalar (np.float64), not an array."""
    result = r2py_kernsmooth.bkfe(_DATA_100, drv=2, bandwidth=0.5)
    assert np.ndim(result) == 0, f"Expected scalar, got ndim={np.ndim(result)}"
    assert isinstance(result, (float, np.floating)), (
        f"Expected float-like, got {type(result)}"
    )


def test_bkfe_result_is_finite_for_normal_data():
    """bkfe returns a finite value for well-behaved normal data."""
    result = float(r2py_kernsmooth.bkfe(_DATA_100, drv=2, bandwidth=0.5))
    assert np.isfinite(result), f"Expected finite result, got {result}"


def test_bkfe_large_bandwidth():
    """bkfe with a very large bandwidth matches R (result approaches zero)."""
    r_val = _r_bkfe(_DATA_200, drv=2, bandwidth=100.0)
    py_val = float(r2py_kernsmooth.bkfe(_DATA_200, drv=2, bandwidth=100.0))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe large bandwidth mismatch with R")


def test_bkfe_gridsize_1001():
    """bkfe with gridsize=1001 matches R."""
    rng = np.random.default_rng(99)
    data = rng.standard_normal(100)
    r_val = _r_bkfe(data, drv=2, bandwidth=0.5, gridsize=1001)
    py_val = float(r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5, gridsize=1001))
    np.testing.assert_allclose(py_val, r_val, rtol=1e-9,
                               err_msg="bkfe gridsize=1001 mismatch with R")
