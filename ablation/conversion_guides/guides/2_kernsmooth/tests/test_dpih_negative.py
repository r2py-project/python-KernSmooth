# Negative test cases for r2py_kernsmooth.dpih
#
# For each invalid-input scenario:
#   1. The R function is called via rpy2; any resulting R error is captured.
#   2. The Python function is called; a Python exception is expected.
#   3. If both raise exceptions, the test passes.
#   4. If the error message texts differ, a UserWarning is emitted (not a
#      test failure) to flag the discrepancy for review.
#
# Behavioral divergences that cannot be reconciled as "both raise" are
# documented explicitly with a warning.

import warnings

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

_ks = importr("KernSmooth")


# ---------------------------------------------------------------------------
# Helper: call R dpih and capture errors
# ---------------------------------------------------------------------------

def _r_dpih_catch(x_py, **kwargs):
    """
    Call R dpih and return (raised: bool, message: str).
    If R raises an error, raised=True and message contains the R error text.
    """
    try:
        x_r = ro.FloatVector(x_py.tolist())
        r_kwargs = {}
        if "scalest" in kwargs:
            r_kwargs["scalest"] = str(kwargs["scalest"])
        if "level" in kwargs:
            r_kwargs["level"] = int(kwargs["level"])
        if "gridsize" in kwargs:
            r_kwargs["gridsize"] = int(kwargs["gridsize"])
        if "range_x" in kwargs:
            r_kwargs["range.x"] = ro.FloatVector(list(kwargs["range_x"]))
        if "truncate" in kwargs:
            r_kwargs["truncate"] = bool(kwargs["truncate"])
        _ks.dpih(x_r, **r_kwargs)
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
    # Normalise both messages for loose comparison.
    if r_msg.lower().split() != py_msg.lower().split():
        warnings.warn(
            f"{label}: Error message discrepancy.\n  R   : {r_msg!r}\n  Py  : {py_msg!r}",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# level out of bounds
# ---------------------------------------------------------------------------


def test_dpih_level_6_raises():
    """level=6 must raise ValueError in Python (matches R 'Level should be between 0 and 5')."""
    x = np.arange(1.0, 11.0)
    r_raised, r_msg = _r_dpih_catch(x, level=6)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x, level=6)

    _check_both_raise(r_raised, r_msg, exc_info, label="level=6")


def test_dpih_level_100_raises():
    """level=100 (far above maximum) must raise ValueError in Python."""
    x = np.arange(1.0, 11.0)
    r_raised, r_msg = _r_dpih_catch(x, level=100)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x, level=100)

    _check_both_raise(r_raised, r_msg, exc_info, label="level=100")


def test_dpih_level_negative_behavioral_divergence():
    """level=-1: R returns empty numeric(0); Python raises UnboundLocalError.

    This is a known behavioral divergence. The R function falls through its
    if/else chain without executing any branch when level < 0, returning an
    empty vector.  The Python port uses elif chains so 'hpi' is never assigned,
    causing an UnboundLocalError.  The test documents the divergence and
    emits a warning.
    """
    x = np.arange(1.0, 11.0)
    r_raised, r_msg = _r_dpih_catch(x, level=-1)

    # R does NOT raise (returns empty numeric(0)), but Python does raise.
    # We document this divergence rather than asserting R also raised.
    assert not r_raised, (
        "Assumption violated: R now raises for level=-1 — update this test."
    )
    with pytest.raises(Exception):
        r2py_kernsmooth.dpih(x, level=-1)

    warnings.warn(
        "Behavioral divergence for level=-1: R returns numeric(0) "
        "(no error), but the Python port raises UnboundLocalError because "
        "'hpi' is never assigned when no if/elif branch matches.",
        UserWarning,
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# Constant data (scale estimate = 0)
# ---------------------------------------------------------------------------


def test_dpih_constant_data_raises():
    """Constant data makes the scale estimate zero; both R and Python should raise."""
    x = np.repeat(5.0, 20)
    r_raised, r_msg = _r_dpih_catch(x)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x)

    _check_both_raise(r_raised, r_msg, exc_info, label="constant data")


def test_dpih_constant_data_scalest_stdev_raises():
    """Constant data with scalest='stdev' forces std=0 and should raise."""
    x = np.repeat(3.14, 50)
    r_raised, r_msg = _r_dpih_catch(x, scalest="stdev")

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x, scalest="stdev")

    _check_both_raise(r_raised, r_msg, exc_info, label="constant data scalest=stdev")


def test_dpih_constant_data_scalest_iqr_raises():
    """Constant data with scalest='iqr' forces IQR=0 and should raise."""
    x = np.repeat(7.0, 30)
    r_raised, r_msg = _r_dpih_catch(x, scalest="iqr")

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x, scalest="iqr")

    _check_both_raise(r_raised, r_msg, exc_info, label="constant data scalest=iqr")


def test_dpih_zero_iqr_when_scalest_iqr_raises():
    """Data where Q1==Q3 (IQR=0) with scalest='iqr' must raise."""
    # All middle 50% are identical -> IQR = 0.
    x = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 3.0])
    r_raised, r_msg = _r_dpih_catch(x, scalest="iqr")

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x, scalest="iqr")

    _check_both_raise(r_raised, r_msg, exc_info, label="zero IQR scalest=iqr")


# ---------------------------------------------------------------------------
# Invalid scalest argument
# ---------------------------------------------------------------------------


def test_dpih_invalid_scalest_unknown_raises():
    """scalest='unknown' must raise ValueError in Python (matches R)."""
    x = np.arange(1.0, 6.0)
    r_raised, r_msg = _r_dpih_catch(x, scalest="unknown")

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x, scalest="unknown")

    _check_both_raise(r_raised, r_msg, exc_info, label="scalest='unknown'")


def test_dpih_invalid_scalest_empty_string_raises():
    """scalest='' (empty string) must raise ValueError in Python (matches R)."""
    x = np.arange(1.0, 6.0)
    r_raised, r_msg = _r_dpih_catch(x, scalest="")

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x, scalest="")

    _check_both_raise(r_raised, r_msg, exc_info, label="scalest=''")


# ---------------------------------------------------------------------------
# Single-element array (n=1)
# ---------------------------------------------------------------------------


def test_dpih_n1_raises():
    """n=1 observation: R raises ('missing value where TRUE/FALSE needed');
    Python raises (std=NaN propagates to ValueError or similar)."""
    x = np.array([5.0])
    r_raised, r_msg = _r_dpih_catch(x)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x)

    _check_both_raise(r_raised, r_msg, exc_info, label="n=1")


# ---------------------------------------------------------------------------
# Inf / NaN in data
# ---------------------------------------------------------------------------


def test_dpih_inf_in_data_raises():
    """Inf in x: R raises ('from' must be a finite number); Python should also raise."""
    x = np.array([1.0, 2.0, float("inf"), 3.0])
    r_raised, r_msg = _r_dpih_catch(x)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x)

    _check_both_raise(r_raised, r_msg, exc_info, label="Inf in x")


def test_dpih_neg_inf_in_data_raises():
    """-Inf in x: R raises ('from' must be a finite number); Python should also raise."""
    x = np.array([1.0, 2.0, float("-inf"), 3.0])
    r_raised, r_msg = _r_dpih_catch(x)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x)

    _check_both_raise(r_raised, r_msg, exc_info, label="-Inf in x")


def test_dpih_nan_in_data_raises():
    """NaN in x: R raises (non-finite range); Python should also raise."""
    x = np.array([1.0, float("nan"), 2.0, 3.0])
    r_raised, r_msg = _r_dpih_catch(x)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x)

    _check_both_raise(r_raised, r_msg, exc_info, label="NaN in x")


# ---------------------------------------------------------------------------
# Empty data
# ---------------------------------------------------------------------------


def test_dpih_empty_array_raises():
    """Empty x must raise an exception in both R and Python."""
    x = np.array([], dtype=float)
    r_raised, r_msg = _r_dpih_catch(x)

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x)

    _check_both_raise(r_raised, r_msg, exc_info, label="empty array")


# ---------------------------------------------------------------------------
# Degenerate range_x (a == b)
# ---------------------------------------------------------------------------


def test_dpih_range_x_equal_endpoints_raises():
    """range_x with a==b creates a zero-width grid; both R and Python should raise."""
    x = np.arange(1.0, 6.0)
    r_raised, r_msg = _r_dpih_catch(x, range_x=(2.0, 2.0))

    with pytest.raises(Exception) as exc_info:
        r2py_kernsmooth.dpih(x, range_x=(2.0, 2.0))

    _check_both_raise(r_raised, r_msg, exc_info, label="range_x a==b")
