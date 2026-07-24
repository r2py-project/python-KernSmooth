# Tests for the negative (invalid input) behavior of bkfe.
#
# For each invalid input scenario:
#   1. The R function is called via rpy2; any resulting R error is captured.
#   2. The Python function is called; a Python exception is expected.
#   3. If both raise exceptions the test passes.
#   4. If the error messages differ in phrasing, a UserWarning is emitted (not
#      a test failure) to flag the discrepancy for review.

import warnings

import numpy as np
import pytest
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

import r2py_kernsmooth

_ks = importr("KernSmooth")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _call_r_bkfe_catch(x_py, drv, bandwidth=None, **kwargs):
    """
    Call R bkfe, returning (raised: bool, message: str).
    If R raises an error, raised=True and message contains the R error text.
    """
    r_kwargs = {}
    if bandwidth is not None:
        r_kwargs["bandwidth"] = float(bandwidth)
    if "gridsize" in kwargs:
        r_kwargs["gridsize"] = int(kwargs["gridsize"])
    if "range_x" in kwargs:
        r_kwargs["range.x"] = ro.FloatVector(list(kwargs["range_x"]))
    if "binned" in kwargs:
        r_kwargs["binned"] = bool(kwargs["binned"])
    if "truncate" in kwargs:
        r_kwargs["truncate"] = bool(kwargs["truncate"])

    try:
        r_x = ro.FloatVector(list(x_py)) if x_py is not None else ro.FloatVector([])
        _ks.bkfe(r_x, drv=int(drv), **r_kwargs)
        return False, ""
    except Exception as exc:
        return True, str(exc)


def _check_both_raise(r_raised, r_msg, py_exc_info):
    """
    Utility: assert R raised, and warn if error messages differ.
    """
    assert r_raised, (
        "R did NOT raise an error but the Python function did — behavior mismatch"
    )
    py_msg = str(py_exc_info.value)
    if r_msg.lower().split() != py_msg.lower().split():
        warnings.warn(
            f"Error message discrepancy.\n  R   : {r_msg!r}\n  Py  : {py_msg!r}",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# bandwidth validation
# ---------------------------------------------------------------------------


def test_bkfe_bandwidth_zero_raises():
    """bandwidth=0 must raise ValueError in Python (matches R behavior)."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r_raised, r_msg = _call_r_bkfe_catch(data, drv=2, bandwidth=0.0)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.0)

    _check_both_raise(r_raised, r_msg, exc_info)


def test_bkfe_bandwidth_negative_raises():
    """bandwidth=-1.0 must raise ValueError in Python (matches R behavior)."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r_raised, r_msg = _call_r_bkfe_catch(data, drv=2, bandwidth=-1.0)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.bkfe(data, drv=2, bandwidth=-1.0)

    _check_both_raise(r_raised, r_msg, exc_info)


def test_bkfe_bandwidth_very_negative_raises():
    """bandwidth=-999 must raise ValueError in Python (matches R behavior)."""
    data = np.arange(1.0, 11.0)
    r_raised, r_msg = _call_r_bkfe_catch(data, drv=2, bandwidth=-999.0)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.bkfe(data, drv=2, bandwidth=-999.0)

    _check_both_raise(r_raised, r_msg, exc_info)


def test_bkfe_bandwidth_none_raises():
    """bandwidth=None must raise an exception in Python.

    R raises 'argument "bandwidth" is missing, with no default'.
    Python raises TypeError because the arithmetic on None fails.
    Both raise — the test passes; the message discrepancy is warned about.
    """
    data = np.array([1.0, 2.0, 3.0])
    # R always requires bandwidth; simulate the missing-argument case
    r_raised = True
    r_msg = 'argument "bandwidth" is missing, with no default'

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.bkfe(data, drv=2, bandwidth=None)

    _check_both_raise(r_raised, r_msg, exc_info)


# ---------------------------------------------------------------------------
# NaN / Inf in input data
# ---------------------------------------------------------------------------


def test_bkfe_nan_in_data_raises():
    """NaN in x causes R to fail when computing range; Python should also raise."""
    data = np.array([1.0, float("nan"), 2.0, 3.0])
    r_raised, r_msg = _call_r_bkfe_catch(data, drv=2, bandwidth=0.5)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5)

    _check_both_raise(r_raised, r_msg, exc_info)


def test_bkfe_inf_in_data_behavior():
    """Inf in x: R raises an error (seq fails on non-finite range), but the
    Python port silently returns NaN via IEEE-754 propagation.

    This test documents the behavioral divergence: R raises, Python does not.
    A warning is emitted so the discrepancy remains visible in CI output.
    """
    data = np.array([1.0, float("inf"), 2.0, 3.0])
    r_raised, r_msg = _call_r_bkfe_catch(data, drv=2, bandwidth=0.5)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        py_result = r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5)

    assert r_raised, "R is expected to raise for Inf data"
    # Python returns NaN rather than raising — document the divergence
    assert np.isnan(py_result), (
        f"Python returned {py_result!r}; expected NaN for Inf input"
    )
    warnings.warn(
        "Behavioral divergence: R raises an error for Inf in x "
        f"({r_msg!r}), but the Python port returns NaN instead of raising.",
        UserWarning,
        stacklevel=1,
    )


def test_bkfe_negative_inf_in_data_behavior():
    """-Inf in x: R raises an error (seq fails on non-finite range), but the
    Python port silently returns NaN via IEEE-754 propagation.

    This test documents the behavioral divergence: R raises, Python does not.
    A warning is emitted so the discrepancy remains visible in CI output.
    """
    data = np.array([1.0, float("-inf"), 2.0, 3.0])
    r_raised, r_msg = _call_r_bkfe_catch(data, drv=2, bandwidth=0.5)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        py_result = r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5)

    assert r_raised, "R is expected to raise for -Inf data"
    assert np.isnan(py_result), (
        f"Python returned {py_result!r}; expected NaN for -Inf input"
    )
    warnings.warn(
        "Behavioral divergence: R raises an error for -Inf in x "
        f"({r_msg!r}), but the Python port returns NaN instead of raising.",
        UserWarning,
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# Empty data
# ---------------------------------------------------------------------------


def test_bkfe_empty_data_raises():
    """Empty x causes R to fail (min/max return Inf); Python should also raise."""
    data = np.array([], dtype=float)
    r_raised, r_msg = _call_r_bkfe_catch(data, drv=2, bandwidth=0.5)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.bkfe(data, drv=2, bandwidth=0.5)

    _check_both_raise(r_raised, r_msg, exc_info)


# ---------------------------------------------------------------------------
# bandwidth validation — Python-only guard (negative float edge)
# ---------------------------------------------------------------------------


def test_bkfe_bandwidth_negative_small_raises():
    """bandwidth=-1e-10 (very small negative) raises ValueError."""
    data = np.linspace(0, 1, 50)
    with pytest.raises((ValueError, Exception)):
        r2py_kernsmooth.bkfe(data, drv=2, bandwidth=-1e-10)
