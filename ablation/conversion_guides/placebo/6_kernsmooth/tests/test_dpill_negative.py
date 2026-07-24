# Negative test cases for r2py_kernsmooth.dpill
#
# For each invalid-input scenario:
#   1. The R function is called via rpy2; any resulting R error is captured.
#   2. The Python function is called; a Python exception is expected.
#   3. If both raise exceptions, the test passes.
#   4. If the error message texts differ in phrasing, a UserWarning is emitted
#      (not a test failure) to flag the discrepancy for review.
#
# R signature:
#   dpill(x, y, blockmax = 5, divisor = 20, trim = 0.01, proptrun = 0.05,
#         gridsize = 401L, range.x, truncate = TRUE)
#
# Python signature:
#   dpill(x, y, blockmax=5, divisor=20, trim=0.01, proptrun=0.05,
#         gridsize=401, range_x=None, truncate=True) -> np.float64

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

def _r_dpill_catch(x_py, y_py, **kwargs):
    """
    Call R dpill and return (raised: bool, message: str).
    If R raises an error, raised=True and message contains the R error text.
    """
    try:
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
        _ks.dpill(x_r, y_r, **r_kwargs)
        return False, ""
    except Exception as exc:
        return True, str(exc)


def _check_both_raise(r_raised, r_msg, py_exc_info, label=""):
    """Assert R raised; warn if error messages differ in phrasing."""
    assert r_raised, (
        f"{label}: R did NOT raise an error but the Python function did — "
        "behavior mismatch"
    )
    py_msg = str(py_exc_info.value)
    if r_msg.lower().split() != py_msg.lower().split():
        warnings.warn(
            f"{label}: Error message discrepancy.\n  R : {r_msg!r}\n  Py: {py_msg!r}",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Single-element input (n=1)
# ---------------------------------------------------------------------------


def test_dpill_n1_raises():
    """n=1 observation raises an exception in both R and Python."""
    x = np.array([1.0])
    y = np.array([2.0])
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="n=1")


# ---------------------------------------------------------------------------
# Empty arrays
# ---------------------------------------------------------------------------


def test_dpill_empty_arrays_raise():
    """Empty x and y must raise an exception in both R and Python."""
    x = np.array([], dtype=float)
    y = np.array([], dtype=float)
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="empty arrays")


# ---------------------------------------------------------------------------
# Mismatched lengths
# ---------------------------------------------------------------------------


def test_dpill_mismatched_lengths_raise():
    """len(x) != len(y) must raise an exception in both R and Python."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0])
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="mismatched lengths")


def test_dpill_mismatched_lengths_y_longer_raises():
    """len(y) > len(x) must raise an exception in both R and Python."""
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="y longer than x")


# ---------------------------------------------------------------------------
# Constant x (degenerate range)
# ---------------------------------------------------------------------------


def test_dpill_constant_x_raises():
    """Constant x (zero range) causes a degenerate bandwidth; both R and Python raise."""
    rng = np.random.default_rng(42)
    x = np.ones(30)
    y = rng.normal(0, 1, 30)
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="constant x")


# ---------------------------------------------------------------------------
# Non-finite values in x
# ---------------------------------------------------------------------------


def test_dpill_inf_in_x_raises():
    """Inf in x: both R and Python should raise."""
    rng = np.random.default_rng(42)
    x = np.append(rng.normal(0, 1, 20), float("inf"))
    y = np.append(rng.normal(0, 1, 20), 1.0)
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="inf in x")


def test_dpill_neg_inf_in_x_raises():
    """-Inf in x: both R and Python should raise."""
    rng = np.random.default_rng(42)
    x = np.append(rng.normal(0, 1, 20), float("-inf"))
    y = np.append(rng.normal(0, 1, 20), 1.0)
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="-inf in x")


def test_dpill_nan_in_x_raises():
    """NaN in x: both R and Python should raise."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 21).copy()
    x[5] = float("nan")
    y = rng.normal(0, 1, 21)
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="nan in x")


# ---------------------------------------------------------------------------
# Non-finite values in y
# ---------------------------------------------------------------------------


def test_dpill_inf_in_y_raises():
    """Inf in y: both R and Python should raise."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 21)
    y = rng.normal(0, 1, 21).copy()
    y[3] = float("inf")
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="inf in y")


def test_dpill_nan_in_y_raises():
    """NaN in y: both R and Python should raise."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 21)
    y = rng.normal(0, 1, 21).copy()
    y[0] = float("nan")
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="nan in y")


# ---------------------------------------------------------------------------
# Degenerate range_x (a == b)
# ---------------------------------------------------------------------------


def test_dpill_range_x_equal_endpoints_raises():
    """range_x with a==b creates a zero-width grid; both R and Python should raise."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 50)
    y = x + rng.normal(0, 0.1, 50)
    r_raised, r_msg = _r_dpill_catch(x, y, range_x=(1.0, 1.0))

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y, range_x=(1.0, 1.0))

    _check_both_raise(r_raised, r_msg, exc_info, label="range_x a==b")


# ---------------------------------------------------------------------------
# trim stripping all observations
# ---------------------------------------------------------------------------


def test_dpill_trim_0_5_raises():
    """trim=0.5 removes all observations from each end; both R and Python should raise."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_raised, r_msg = _r_dpill_catch(x, y, trim=0.5)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y, trim=0.5)

    _check_both_raise(r_raised, r_msg, exc_info, label="trim=0.5")


# ---------------------------------------------------------------------------
# gridsize too small
# ---------------------------------------------------------------------------


def test_dpill_gridsize_0_raises():
    """gridsize=0 produces an empty grid; both R and Python should raise."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    r_raised, r_msg = _r_dpill_catch(x, y, gridsize=0)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y, gridsize=0)

    _check_both_raise(r_raised, r_msg, exc_info, label="gridsize=0")


def test_dpill_gridsize_1_raises():
    """gridsize=1 (single grid point) causes degenerate computations; both R and Python raise."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    r_raised, r_msg = _r_dpill_catch(x, y, gridsize=1)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y, gridsize=1)

    _check_both_raise(r_raised, r_msg, exc_info, label="gridsize=1")


# ---------------------------------------------------------------------------
# divisor=0 (Python-only check: behavioural divergence from R)
# ---------------------------------------------------------------------------


def test_dpill_divisor_0_behavioral_divergence():
    """divisor=0 causes ZeroDivisionError in Python; R computes Nmax using integer division.

    This is a known behavioral divergence. In R, n %/% 0 does not error (it
    returns Inf), so dpill proceeds. In Python, int(n/0) raises ZeroDivisionError
    immediately. The test documents the divergence and emits a warning.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    y = np.sin(2 * x) + rng.normal(0, 0.3, 200)
    r_raised, r_msg = _r_dpill_catch(x, y, divisor=0)

    with pytest.raises(Exception):
        r2py_kernsmooth.dpill(x, y, divisor=0)

    if not r_raised:
        warnings.warn(
            "Behavioral divergence for divisor=0: R returns a valid bandwidth "
            "(R uses integer division where n%%0 is Inf), but the Python port "
            "raises ZeroDivisionError because int(n/0) fails immediately.",
            UserWarning,
            stacklevel=1,
        )
    else:
        # Both raised — fine, just warn about message differences
        pass


# ---------------------------------------------------------------------------
# All x identical but y varies: R raises, Python also raises
# ---------------------------------------------------------------------------


def test_dpill_all_x_equal_to_zero_raises():
    """All x equal to 0.0: zero range triggers an error in both R and Python."""
    x = np.zeros(20)
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 20)
    r_raised, r_msg = _r_dpill_catch(x, y)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y)

    _check_both_raise(r_raised, r_msg, exc_info, label="all x==0")


# ---------------------------------------------------------------------------
# range_x reversed (b < a)
# ---------------------------------------------------------------------------


def test_dpill_range_x_reversed_raises():
    """range_x=(b, a) with b > a creates a backwards grid; both R and Python raise."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    # b < a means linspace produces a descending grid; R and Python should both fail.
    r_raised, r_msg = _r_dpill_catch(x, y, range_x=(2.0, -2.0))

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpill(x, y, range_x=(2.0, -2.0))

    _check_both_raise(r_raised, r_msg, exc_info, label="range_x reversed")
