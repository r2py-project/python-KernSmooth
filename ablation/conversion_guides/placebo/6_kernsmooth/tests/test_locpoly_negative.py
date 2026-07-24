# Negative test cases for r2py_kernsmooth.locpoly
#
# Each test exercises an invalid input scenario. The test passes when BOTH the
# Python function and R's KernSmooth::locpoly raise an exception. If one raises
# and the other does not, the test fails. When both raise but the error messages
# differ, a warning is emitted (messages need not be identical across languages).
#
# locpoly has two modes (regression and density). Negative cases for both are
# exercised here.

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
    """Call R's locpoly; return (raised: bool, message: str)."""
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
        _ks.locpoly(*args, **kwargs)
        return False, ""
    except Exception as exc:
        return True, str(exc)


def _py_locpoly(x_py, y_py=None, **kwargs):
    """Call Python's locpoly; return (raised: bool, message: str)."""
    try:
        r2py_kernsmooth.locpoly(x_py, y_py, **kwargs)
        return False, ""
    except Exception as exc:
        return True, str(exc)


def _check(x_py, y_py=None, *, r_kwargs=None, py_kwargs=None, test_name):
    """Assert both R and Python raise. Warn if messages differ."""
    r_kwargs = r_kwargs or {}
    py_kwargs = py_kwargs or {}
    r_raised, r_msg = _r_locpoly(x_py, y_py, **r_kwargs)
    py_raised, py_msg = _py_locpoly(x_py, y_py, **py_kwargs)

    if not r_raised and not py_raised:
        pytest.fail(f"{test_name}: neither R nor Python raised an error")
    if r_raised and not py_raised:
        pytest.fail(
            f"{test_name}: R raised but Python did not.\nR error: {r_msg}"
        )
    if not r_raised and py_raised:
        pytest.fail(
            f"{test_name}: Python raised but R did not.\nPython error: {py_msg}"
        )
    # Both raised — compare messages
    if r_msg.strip() != py_msg.strip():
        warnings.warn(
            f"{test_name}: both raised, but messages differ.\n"
            f"  R:      {r_msg!r}\n"
            f"  Python: {py_msg!r}",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Negative tests — bandwidth validity
# ---------------------------------------------------------------------------

def test_locpoly_negative_bandwidth_regression():
    """Negative scalar bandwidth must raise in both R and Python (regression mode)."""
    x = np.linspace(0, 5, 50)
    y = x ** 2
    _check(
        x, y,
        r_kwargs={"bandwidth": -0.5},
        py_kwargs={"bandwidth": -0.5},
        test_name="negative bandwidth regression",
    )


def test_locpoly_zero_bandwidth_regression():
    """Zero bandwidth must raise in both R and Python (regression mode)."""
    x = np.linspace(0, 5, 50)
    y = x ** 2
    _check(
        x, y,
        r_kwargs={"bandwidth": 0.0},
        py_kwargs={"bandwidth": 0.0},
        test_name="zero bandwidth regression",
    )


def test_locpoly_very_negative_bandwidth():
    """A very large negative bandwidth must raise in both."""
    x = np.linspace(0, 5, 50)
    y = x ** 2
    _check(
        x, y,
        r_kwargs={"bandwidth": -1e10},
        py_kwargs={"bandwidth": -1e10},
        test_name="very negative bandwidth",
    )


def test_locpoly_negative_bandwidth_density():
    """Negative scalar bandwidth must raise in both R and Python (density mode)."""
    x = np.linspace(0, 5, 50)
    _check(
        x,
        r_kwargs={"bandwidth": -0.5},
        py_kwargs={"bandwidth": -0.5},
        test_name="negative bandwidth density",
    )


def test_locpoly_zero_bandwidth_density():
    """Zero bandwidth must raise in both R and Python (density mode)."""
    x = np.linspace(0, 5, 50)
    _check(
        x,
        r_kwargs={"bandwidth": 0.0},
        py_kwargs={"bandwidth": 0.0},
        test_name="zero bandwidth density",
    )


def test_locpoly_bandwidth_vector_with_any_non_positive():
    """A bandwidth vector containing a non-positive value must raise in both."""
    x = np.linspace(0, 5, 50)
    y = x ** 2
    bw = np.full(401, 0.5)
    bw[200] = -0.1
    _check(
        x, y,
        r_kwargs={"bandwidth": bw},
        py_kwargs={"bandwidth": bw},
        test_name="bandwidth vector with non-positive entry",
    )


# ---------------------------------------------------------------------------
# Negative tests — bandwidth vector length
# ---------------------------------------------------------------------------

def test_locpoly_bandwidth_wrong_length_vector():
    """A bandwidth vector whose length is neither 1 nor gridsize must raise in both."""
    x = np.linspace(0, 5, 50)
    y = x ** 2
    bw_wrong = np.array([0.3, 0.4, 0.5])   # length 3, gridsize defaults to 401
    _check(
        x, y,
        r_kwargs={"bandwidth": bw_wrong},
        py_kwargs={"bandwidth": bw_wrong},
        test_name="bandwidth wrong-length vector",
    )


def test_locpoly_bandwidth_wrong_length_vector_density():
    """A bandwidth vector of wrong length must raise in density mode too."""
    x = np.linspace(0, 5, 50)
    bw_wrong = np.array([0.3, 0.4, 0.5])
    _check(
        x,
        r_kwargs={"bandwidth": bw_wrong},
        py_kwargs={"bandwidth": bw_wrong},
        test_name="bandwidth wrong-length vector density",
    )


# ---------------------------------------------------------------------------
# Negative tests — grid too coarse (bandwidth too small)
# ---------------------------------------------------------------------------

def test_locpoly_bandwidth_too_small_grid_coarse():
    """A bandwidth so small that Lvec==0 must raise in both R and Python."""
    x = np.linspace(0, 5, 50)
    y = x ** 2
    # With gridsize=401 and range [0,5], delta=5/400=0.0125; tau=4.
    # Lvec = floor(4 * bw / 0.0125).  For bw=1e-4: floor(4*1e-4/0.0125)=0.
    _check(
        x, y,
        r_kwargs={"bandwidth": 1e-4},
        py_kwargs={"bandwidth": 1e-4},
        test_name="bandwidth too small (Lvec==0)",
    )


# ---------------------------------------------------------------------------
# Negative tests — missing required bandwidth
# ---------------------------------------------------------------------------

def test_locpoly_missing_bandwidth_regression():
    """Calling locpoly without bandwidth must raise in both R and Python (regression)."""
    x = np.linspace(0, 5, 50)
    y = x ** 2
    # bandwidth is required in R; Python uses None which propagates to NaN
    r_raised, r_msg = _r_locpoly(x, y)
    py_raised, py_msg = _py_locpoly(x, y)
    if not r_raised and not py_raised:
        pytest.fail("missing bandwidth regression: neither R nor Python raised")
    if r_raised and not py_raised:
        pytest.fail(
            f"missing bandwidth regression: R raised but Python did not.\nR: {r_msg}"
        )
    if not r_raised and py_raised:
        pytest.fail(
            f"missing bandwidth regression: Python raised but R did not.\nPy: {py_msg}"
        )
    if r_msg.strip() != py_msg.strip():
        warnings.warn(
            "missing bandwidth regression: both raised, messages differ.\n"
            f"  R:      {r_msg!r}\n  Python: {py_msg!r}",
            UserWarning,
            stacklevel=1,
        )


def test_locpoly_missing_bandwidth_density():
    """Calling locpoly without bandwidth in density mode must raise in both."""
    x = np.linspace(0, 5, 50)
    r_raised, r_msg = _r_locpoly(x)
    py_raised, py_msg = _py_locpoly(x)
    if not r_raised and not py_raised:
        pytest.fail("missing bandwidth density: neither raised")
    if r_raised and not py_raised:
        pytest.fail(
            f"missing bandwidth density: R raised but Python did not.\nR: {r_msg}"
        )
    if not r_raised and py_raised:
        pytest.fail(
            f"missing bandwidth density: Python raised but R did not.\nPy: {py_msg}"
        )
    if r_msg.strip() != py_msg.strip():
        warnings.warn(
            "missing bandwidth density: both raised, messages differ.\n"
            f"  R:      {r_msg!r}\n  Python: {py_msg!r}",
            UserWarning,
            stacklevel=1,
        )
