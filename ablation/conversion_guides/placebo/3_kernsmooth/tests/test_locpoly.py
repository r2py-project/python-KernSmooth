# Translated from KernSmooth/tests/locpoly.R
#
# Original comment (Peter Dalgaard, 2020-11-24):
# '(without the truncate=FALSE, curves pretty much go through
#   the penultimate point, whatever reasonable bandwidth is chosen.)'
#
# The original R test uses the 'carData' package's Prestige dataset and calls
# locpoly twice on (income, prestige) with bandwidth=5000:
#   once with the default truncate=TRUE and once with truncate=FALSE.
#
# The original R script only plots the results (no numeric assertions). Here we
# call the R function via rpy2 to obtain reference results and assert that the
# Python implementation agrees with R to within floating-point tolerance.
# Plotting calls are omitted as they are not part of the library under test.

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

# Load the KernSmooth and carData R packages once at module level.
_ks = importr("KernSmooth")
_cardata = importr("carData")

# Extract the Prestige dataset columns via rpy2.
_income = np.array(ro.r("Prestige$income"), dtype=float)
_prestige = np.array(ro.r("Prestige$prestige"), dtype=float)
_r_income = ro.FloatVector(_income.tolist())
_r_prestige = ro.FloatVector(_prestige.tolist())

_BANDWIDTH = 5000.0


def test_locpoly_truncate_true():
    """locpoly on (income, prestige) with bandwidth=5000 and truncate=True must match R."""
    # R: with(Prestige, lines(locpoly(income, prestige, bandwidth = 5000)))
    r_result = _ks.locpoly(_r_income, _r_prestige, bandwidth=_BANDWIDTH)
    r_x = np.array(r_result.rx2("x"))
    r_y = np.array(r_result.rx2("y"))

    py_result = r2py_kernsmooth.locpoly(_income, _prestige, bandwidth=_BANDWIDTH)
    assert "x" in py_result and "y" in py_result

    py_x = py_result["x"]
    py_y = py_result["y"]

    assert len(py_x) == len(r_x), (
        f"Grid length mismatch (truncate=True): Python={len(py_x)}, R={len(r_x)}"
    )
    assert len(py_y) == len(r_y), (
        f"Estimate length mismatch (truncate=True): Python={len(py_y)}, R={len(r_y)}"
    )

    assert np.all(np.isfinite(py_x)), "Python locpoly returned non-finite x values (truncate=True)"
    assert np.all(np.isfinite(py_y)), "Python locpoly returned non-finite y values (truncate=True)"

    np.testing.assert_allclose(
        py_x, r_x, rtol=1e-10,
        err_msg="locpoly grid points differ between Python and R (truncate=True)",
    )
    np.testing.assert_allclose(
        py_y, r_y, rtol=1e-6,
        err_msg="locpoly estimates differ between Python and R (truncate=True)",
    )


def test_locpoly_truncate_false():
    """locpoly on (income, prestige) with bandwidth=5000 and truncate=False must match R."""
    # R: with(Prestige, lines(locpoly(income, prestige, bandwidth = 5000, truncate = FALSE)))
    r_result = _ks.locpoly(_r_income, _r_prestige, bandwidth=_BANDWIDTH, truncate=False)
    r_x = np.array(r_result.rx2("x"))
    r_y = np.array(r_result.rx2("y"))

    py_result = r2py_kernsmooth.locpoly(_income, _prestige, bandwidth=_BANDWIDTH, truncate=False)
    assert "x" in py_result and "y" in py_result

    py_x = py_result["x"]
    py_y = py_result["y"]

    assert len(py_x) == len(r_x), (
        f"Grid length mismatch (truncate=False): Python={len(py_x)}, R={len(r_x)}"
    )
    assert len(py_y) == len(r_y), (
        f"Estimate length mismatch (truncate=False): Python={len(py_y)}, R={len(r_y)}"
    )

    assert np.all(np.isfinite(py_x)), "Python locpoly returned non-finite x values (truncate=False)"
    assert np.all(np.isfinite(py_y)), "Python locpoly returned non-finite y values (truncate=False)"

    np.testing.assert_allclose(
        py_x, r_x, rtol=1e-10,
        err_msg="locpoly grid points differ between Python and R (truncate=False)",
    )
    np.testing.assert_allclose(
        py_y, r_y, rtol=1e-6,
        err_msg="locpoly estimates differ between Python and R (truncate=False)",
    )
