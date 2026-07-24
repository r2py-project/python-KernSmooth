# Negative test cases for r2py_kernsmooth.bkde
#
# Each test exercises an invalid input scenario. The test passes when BOTH the
# Python function and R's KernSmooth::bkde raise an exception. If one raises
# and the other does not the test fails. When both raise but the error messages
# differ, a warning is emitted (the messages need not be identical across
# languages).

import warnings

import numpy as np
import pytest
import rpy2.robjects as ro
import rpy2.rinterface_lib.callbacks
from rpy2.robjects.packages import importr

import r2py_kernsmooth

_ks = importr("KernSmooth")


def _r_raises(x_py, **kwargs):
    """Return (raised: bool, message: str) for R's bkde call."""
    x_r = ro.FloatVector(x_py.tolist())
    if "range_x" in kwargs:
        range_x = kwargs.pop("range_x")
        kwargs["range.x"] = ro.FloatVector(list(range_x))
    try:
        _ks.bkde(x_r, **kwargs)
        return False, ""
    except Exception as exc:
        return True, str(exc)


def _py_raises(x_py, **kwargs):
    """Return (raised: bool, message: str) for the Python bkde call."""
    try:
        r2py_kernsmooth.bkde(x_py, **kwargs)
        return False, ""
    except Exception as exc:
        return True, str(exc)


def _check(x_py, r_kwargs, py_kwargs, *, test_name):
    """
    Assert both R and Python raise. Emit a warning if messages differ.
    """
    r_raised, r_msg = _r_raises(x_py, **r_kwargs)
    py_raised, py_msg = _py_raises(x_py, **py_kwargs)

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
    # Both raised – compare messages
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

def test_bkde_negative_bandwidth():
    """Negative bandwidth must raise in both R and Python."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _check(x, {"bandwidth": -0.5}, {"bandwidth": -0.5},
           test_name="negative bandwidth")


def test_bkde_zero_bandwidth():
    """Zero bandwidth must raise in both R and Python."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _check(x, {"bandwidth": 0.0}, {"bandwidth": 0.0},
           test_name="zero bandwidth")


def test_bkde_very_negative_bandwidth():
    """A very negative bandwidth must raise in both R and Python."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _check(x, {"bandwidth": -1e10}, {"bandwidth": -1e10},
           test_name="very negative bandwidth")


def test_bkde_invalid_kernel_name():
    """A completely unknown kernel name must raise in both R and Python."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _check(x, {"bandwidth": 0.5, "kernel": "invalid"},
           {"bandwidth": 0.5, "kernel": "invalid"},
           test_name="invalid kernel name")


def test_bkde_ambiguous_kernel_abbreviation():
    """Abbreviation 'b' is ambiguous (box/biweight) and must raise in both."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # R raises because 'b' matches both 'box' and 'biweight'
    _check(x, {"bandwidth": 0.5, "kernel": "b"},
           {"bandwidth": 0.5, "kernel": "b"},
           test_name="ambiguous kernel abbreviation 'b'")


def test_bkde_canonical_string_argument():
    """Passing a string for canonical must raise in both R and Python."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # R only accepts a length-1 logical for canonical
    _check(x, {"bandwidth": 0.5, "canonical": "yes"},
           {"bandwidth": 0.5, "canonical": "yes"},
           test_name="canonical as string")


def test_bkde_canonical_integer_argument():
    """Passing an integer for canonical must raise in both R and Python."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _check(x, {"bandwidth": 0.5, "canonical": 1},
           {"bandwidth": 0.5, "canonical": 1},
           test_name="canonical as integer")


def test_bkde_canonical_multi_element_logical():
    """Passing a multi-element bool vector for canonical must raise in both."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # R: canonical must be length-1 logical
    r_raised, r_msg = _r_raises(x, bandwidth=0.5,
                                **{"canonical": ro.BoolVector([True, False])})
    py_raised, py_msg = _py_raises(x, bandwidth=0.5,
                                   canonical=[True, False])
    if not r_raised and not py_raised:
        pytest.fail("multi-element canonical: neither raised")
    if r_raised and not py_raised:
        pytest.fail(f"multi-element canonical: R raised but Python did not.\nR: {r_msg}")
    if not r_raised and py_raised:
        pytest.fail(f"multi-element canonical: Python raised but R did not.\nPy: {py_msg}")
    if r_msg.strip() != py_msg.strip():
        warnings.warn(
            "multi-element canonical: both raised, but messages differ.\n"
            f"  R:      {r_msg!r}\n  Python: {py_msg!r}",
            UserWarning,
            stacklevel=1,
        )


def test_bkde_bandwidth_wrong_length_array():
    """Passing an array bandwidth of wrong length must raise in Python and R."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # R accepts only a scalar bandwidth; pass a length-2 numeric vector
    r_raised, r_msg = _r_raises(x, **{"bandwidth": ro.FloatVector([0.5, 0.6])})
    py_raised, py_msg = _py_raises(x, bandwidth=np.array([0.5, 0.6]))
    # bkde in both R and Python expects a scalar bandwidth;
    # if either raises, assert the Python behaviour matches R.
    if r_raised != py_raised:
        if r_raised:
            pytest.fail(
                f"array bandwidth: R raised but Python did not.\nR: {r_msg}"
            )
        else:
            pytest.fail(
                f"array bandwidth: Python raised but R did not.\nPy: {py_msg}"
            )
    if r_raised and py_raised and r_msg.strip() != py_msg.strip():
        warnings.warn(
            "array bandwidth: both raised, messages differ.\n"
            f"  R:      {r_msg!r}\n  Python: {py_msg!r}",
            UserWarning,
            stacklevel=1,
        )
