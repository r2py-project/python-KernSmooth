# Translated from KernSmooth/tests/bkfe.R
#
# Original comment: failed in bkfe with exact powers of 2 prior to 2.23-5
#
# Tests that dpik and bkde complete without error when gridsize is an exact
# power of 2 (256), which previously triggered a bug in the underlying bkfe
# routine.
#
# Each test calls the R function via rpy2 to obtain the reference result, calls
# the equivalent Python function, and asserts that the two results agree.

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

# Load the KernSmooth R package once at module level.
_ks = importr("KernSmooth")


def test_dpik_gridsize_power_of_2():
    """dpik with gridsize=256 (exact power of 2) must not raise and must match R."""
    # R: x <- 1:100; dpik(x, gridsize = 256)
    x_py = np.arange(1, 101, dtype=float)
    r_x = ro.IntVector(list(range(1, 101)))

    # Obtain the R reference result via rpy2
    r_result = _ks.dpik(r_x, gridsize=256)
    r_value = float(r_result[0])

    # Obtain the Python result
    py_result = r2py_kernsmooth.dpik(x_py, gridsize=256)
    py_value = float(py_result)

    # Both results must be finite
    assert np.isfinite(r_value), f"R dpik returned non-finite value: {r_value}"
    assert np.isfinite(py_value), f"Python dpik returned non-finite value: {py_value}"

    # Python result must agree with R result to within floating-point tolerance
    assert abs(py_value - r_value) < 1e-6, (
        f"dpik mismatch: Python={py_value}, R={r_value}"
    )


def test_bkde_gridsize_power_of_2():
    """bkde with gridsize=256 (exact power of 2) must not raise and must match R."""
    # R: x <- c(0.036, 0.042, 0.052, 0.216, 0.368, 0.511, 0.705, 0.753, 0.776, 0.84)
    #    bkde(x, gridsize = 256, range.x = range(x))
    x_vals = [0.036, 0.042, 0.052, 0.216, 0.368, 0.511, 0.705, 0.753, 0.776, 0.84]
    x_py = np.array(x_vals)
    r_x = ro.FloatVector(x_vals)

    # Obtain the R reference result via rpy2
    r_result = _ks.bkde(r_x, gridsize=256, **{"range.x": ro.r("range")(r_x)})
    r_x_grid = np.array(r_result.rx2("x"))
    r_y_est = np.array(r_result.rx2("y"))

    # Obtain the Python result
    py_result = r2py_kernsmooth.bkde(
        x_py, gridsize=256, range_x=(np.min(x_py), np.max(x_py))
    )

    assert "x" in py_result and "y" in py_result

    py_x_grid = py_result["x"]
    py_y_est = py_result["y"]

    # Grid lengths must match
    assert len(py_x_grid) == len(r_x_grid), (
        f"Grid length mismatch: Python={len(py_x_grid)}, R={len(r_x_grid)}"
    )
    assert len(py_y_est) == len(r_y_est), (
        f"Estimate length mismatch: Python={len(py_y_est)}, R={len(r_y_est)}"
    )

    # All density estimates must be finite
    assert np.all(np.isfinite(py_y_est)), "Python bkde returned non-finite density estimates"
    assert np.all(np.isfinite(r_y_est)), "R bkde returned non-finite density estimates"

    # Grid points must agree element-wise
    np.testing.assert_allclose(
        py_x_grid, r_x_grid, rtol=1e-10,
        err_msg="bkde grid points differ between Python and R"
    )

    # Density estimates must agree element-wise
    np.testing.assert_allclose(
        py_y_est, r_y_est, rtol=1e-6,
        err_msg="bkde density estimates differ between Python and R"
    )
