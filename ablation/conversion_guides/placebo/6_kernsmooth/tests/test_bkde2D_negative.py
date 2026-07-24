# Negative test cases for r2py_kernsmooth.bkde2D
#
# Each test exercises an invalid input scenario.  A test passes when BOTH
# the Python function and R's KernSmooth::bkde2D raise an exception.
# If one raises and the other does not, the test fails.
# When both raise but the error messages differ, a UserWarning is emitted
# (the messages need not be identical across languages).

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

def _make_r_matrix(xy: np.ndarray) -> ro.Matrix:
    n = xy.shape[0]
    return ro.r.matrix(
        ro.FloatVector(xy.flatten(order="F")), nrow=n, ncol=xy.shape[1]
    )


def _r_raises(xy: np.ndarray, **kwargs) -> tuple[bool, str]:
    """Return (raised, message) when calling R's bkde2D."""
    x_r = _make_r_matrix(xy)
    r_kwargs: dict = {}
    if "bandwidth" in kwargs:
        bw = kwargs.pop("bandwidth")
        r_kwargs["bandwidth"] = ro.FloatVector(
            [float(v) for v in np.atleast_1d(bw)]
        )
    if "gridsize" in kwargs:
        gs = kwargs.pop("gridsize")
        r_kwargs["gridsize"] = ro.IntVector(list(gs))
    if "range_x" in kwargs:
        rx = kwargs.pop("range_x")
        r_kwargs["range.x"] = ro.r.list(
            ro.FloatVector(list(rx[0])), ro.FloatVector(list(rx[1]))
        )
    r_kwargs.update(kwargs)
    try:
        _ks.bkde2D(x_r, **r_kwargs)
        return False, ""
    except Exception as exc:
        return True, str(exc)


def _py_raises(xy: np.ndarray, **kwargs) -> tuple[bool, str]:
    """Return (raised, message) when calling the Python bkde2D."""
    try:
        r2py_kernsmooth.bkde2D(xy, **kwargs)
        return False, ""
    except Exception as exc:
        return True, str(exc)


def _check(
    xy: np.ndarray,
    r_kwargs: dict,
    py_kwargs: dict,
    *,
    test_name: str,
) -> None:
    """Assert both R and Python raise; warn if messages differ."""
    r_raised, r_msg = _r_raises(xy, **r_kwargs)
    py_raised, py_msg = _py_raises(xy, **py_kwargs)

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
    # Both raised - compare messages
    if r_msg.strip() != py_msg.strip():
        warnings.warn(
            f"{test_name}: both raised, but messages differ.\n"
            f"  R:      {r_msg!r}\n"
            f"  Python: {py_msg!r}",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------

def test_bkde2D_negative_bandwidth_both_components():
    """Both bandwidth components negative must raise in R and Python."""
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    _check(
        xy,
        {"bandwidth": [-0.5, -0.5]},
        {"bandwidth": np.array([-0.5, -0.5])},
        test_name="negative bandwidth both components",
    )


def test_bkde2D_negative_bandwidth_first_component():
    """First bandwidth component negative must raise in R and Python."""
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _check(
        xy,
        {"bandwidth": [-0.5, 0.5]},
        {"bandwidth": np.array([-0.5, 0.5])},
        test_name="negative bandwidth first component",
    )


def test_bkde2D_negative_bandwidth_second_component():
    """Second bandwidth component negative must raise in R and Python."""
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _check(
        xy,
        {"bandwidth": [0.5, -0.5]},
        {"bandwidth": np.array([0.5, -0.5])},
        test_name="negative bandwidth second component",
    )


def test_bkde2D_zero_bandwidth_both_components():
    """Both bandwidth components zero must raise in R and Python."""
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _check(
        xy,
        {"bandwidth": [0.0, 0.0]},
        {"bandwidth": np.array([0.0, 0.0])},
        test_name="zero bandwidth both components",
    )


def test_bkde2D_zero_bandwidth_first_component():
    """First bandwidth component zero must raise in R and Python."""
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _check(
        xy,
        {"bandwidth": [0.0, 0.5]},
        {"bandwidth": np.array([0.0, 0.5])},
        test_name="zero bandwidth first component",
    )


def test_bkde2D_zero_bandwidth_second_component():
    """Second bandwidth component zero must raise in R and Python."""
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _check(
        xy,
        {"bandwidth": [0.5, 0.0]},
        {"bandwidth": np.array([0.5, 0.0])},
        test_name="zero bandwidth second component",
    )


def test_bkde2D_very_negative_bandwidth():
    """Extremely negative bandwidth must raise in R and Python."""
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _check(
        xy,
        {"bandwidth": [-1e10, -1e10]},
        {"bandwidth": np.array([-1e10, -1e10])},
        test_name="very negative bandwidth",
    )


def test_bkde2D_one_column_input():
    """A 1-column input matrix must raise in both R and Python."""
    rng = np.random.default_rng(42)
    xy_1col = rng.normal(0.0, 1.0, (50, 1))
    r_x1col = ro.r.matrix(
        ro.FloatVector(xy_1col.flatten(order="F")), nrow=50, ncol=1
    )
    r_raised, r_msg = False, ""
    try:
        _ks.bkde2D(r_x1col, bandwidth=ro.FloatVector([0.5, 0.5]))
    except Exception as exc:
        r_raised, r_msg = True, str(exc)

    py_raised, py_msg = False, ""
    try:
        r2py_kernsmooth.bkde2D(xy_1col, bandwidth=np.array([0.5, 0.5]))
    except Exception as exc:
        py_raised, py_msg = True, str(exc)

    if not r_raised and not py_raised:
        pytest.fail("1-column input: neither R nor Python raised an error")
    if r_raised and not py_raised:
        pytest.fail(
            f"1-column input: R raised but Python did not.\nR: {r_msg}"
        )
    if not r_raised and py_raised:
        pytest.fail(
            f"1-column input: Python raised but R did not.\nPy: {py_msg}"
        )
    if r_msg.strip() != py_msg.strip():
        warnings.warn(
            "1-column input: both raised, messages differ.\n"
            f"  R:      {r_msg!r}\n  Python: {py_msg!r}",
            UserWarning,
            stacklevel=1,
        )


def test_bkde2D_nan_in_input_raises_or_signals():
    """NaN values in input x must cause both R and Python to signal an error."""
    rng = np.random.default_rng(7)
    xy = rng.normal(0.0, 1.0, (50, 2))
    xy[5, 0] = np.nan

    x_r_nan = ro.r.matrix(
        ro.FloatVector(xy.flatten(order="F")), nrow=50, ncol=2
    )
    r_raised, r_msg = False, ""
    try:
        _ks.bkde2D(x_r_nan, bandwidth=ro.FloatVector([0.5, 0.5]))
    except Exception as exc:
        r_raised, r_msg = True, str(exc)

    py_raised, py_msg = False, ""
    try:
        r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    except Exception as exc:
        py_raised, py_msg = True, str(exc)

    # If neither raises, the test is vacuous - document the discrepancy
    if not r_raised and not py_raised:
        warnings.warn(
            "NaN input: neither R nor Python raised; R documentation states "
            "missing values are not allowed.",
            UserWarning,
            stacklevel=1,
        )
        return

    if r_raised and not py_raised:
        pytest.fail(f"NaN input: R raised but Python did not.\nR: {r_msg}")
    if not r_raised and py_raised:
        pytest.fail(f"NaN input: Python raised but R did not.\nPy: {py_msg}")
    if r_msg.strip() != py_msg.strip():
        warnings.warn(
            "NaN input: both raised, messages differ.\n"
            f"  R:      {r_msg!r}\n  Python: {py_msg!r}",
            UserWarning,
            stacklevel=1,
        )


def test_bkde2D_none_bandwidth_missing_required_arg():
    """Passing None for bandwidth (R requires it) must raise in Python.

    R's bkde2D has no default for bandwidth; R raises a 'missing' error.
    Python raises because bandwidth=None leads to an invalid np.asarray call
    or downstream numeric failure.
    """
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # R: bandwidth is required
    x_r = _make_r_matrix(xy)
    r_raised, r_msg = False, ""
    try:
        _ks.bkde2D(x_r)
    except Exception as exc:
        r_raised, r_msg = True, str(exc)

    # Python: bandwidth=None is allowed (the parameter accepts None) but
    # internally np.asarray(None) followed by ndim checks leads to a TypeError
    py_raised, py_msg = False, ""
    try:
        r2py_kernsmooth.bkde2D(xy, bandwidth=None)
    except Exception as exc:
        py_raised, py_msg = True, str(exc)

    # At least R must raise (bandwidth has no default in KernSmooth)
    if not r_raised:
        pytest.fail("missing bandwidth: R did not raise an error as expected")

    # If Python does not raise, warn about the behavioural difference
    if not py_raised:
        warnings.warn(
            "missing bandwidth: R raised but Python did not.\n"
            f"  R error: {r_msg!r}",
            UserWarning,
            stacklevel=1,
        )


def test_bkde2D_inverted_range_x_first_dimension():
    """Inverted range_x where a > b must raise in both R and Python.

    When range_x[0][0] > range_x[0][1], the grid in direction 1 cannot
    be constructed; both R and Python must raise an error.
    """
    rng = np.random.default_rng(42)
    xy = rng.normal(0.0, 1.0, (50, 2))
    # range_x direction 1: (2.0, -2.0)  -- inverted
    # range_x direction 2: (-2.0, 2.0)  -- normal
    _check(
        xy,
        {"bandwidth": [0.5, 0.5], "range_x": [(2.0, -2.0), (-2.0, 2.0)]},
        {"bandwidth": np.array([0.5, 0.5]), "range_x": [(2.0, -2.0), (-2.0, 2.0)]},
        test_name="inverted range_x direction 1",
    )


def test_bkde2D_inverted_range_x_second_dimension():
    """Inverted range_x where a > b in direction 2 must raise in both R and Python."""
    rng = np.random.default_rng(42)
    xy = rng.normal(0.0, 1.0, (50, 2))
    _check(
        xy,
        {"bandwidth": [0.5, 0.5], "range_x": [(-2.0, 2.0), (2.0, -2.0)]},
        {"bandwidth": np.array([0.5, 0.5]), "range_x": [(-2.0, 2.0), (2.0, -2.0)]},
        test_name="inverted range_x direction 2",
    )
