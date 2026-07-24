# Edge-case and boundary tests for r2py_kernsmooth.bkde2D
#
# Tests focus on functional limits and extremes. Where R returns a valid
# output the Python result is compared against R. Where R raises an error
# both must raise (same logic as the negative suite). Where behaviour
# differs between R and Python, a UserWarning is emitted and the test
# documents the discrepancy without failing.

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


def _r_bkde2D(xy: np.ndarray, **kwargs):
    """Call R's bkde2D; return (x1, x2, fhat) as numpy arrays."""
    x_r = _make_r_matrix(xy)
    r_kwargs: dict = {}
    if "bandwidth" in kwargs:
        bw = np.atleast_1d(np.asarray(kwargs.pop("bandwidth"), dtype=np.float64))
        r_kwargs["bandwidth"] = ro.FloatVector(bw.tolist())
    if "gridsize" in kwargs:
        gs = kwargs.pop("gridsize")
        r_kwargs["gridsize"] = ro.IntVector(list(gs))
    if "range_x" in kwargs:
        rx = kwargs.pop("range_x")
        r_kwargs["range.x"] = ro.r.list(
            ro.FloatVector(list(rx[0])),
            ro.FloatVector(list(rx[1])),
        )
    r_kwargs.update(kwargs)
    result = _ks.bkde2D(x_r, **r_kwargs)
    return (
        np.array(result.rx2("x1")),
        np.array(result.rx2("x2")),
        np.array(result.rx2("fhat")),
    )


def _r_try(xy: np.ndarray, **kwargs):
    """Return (raised, msg, x1, x2, fhat) for an R bkde2D call."""
    try:
        x1, x2, fhat = _r_bkde2D(xy, **kwargs)
        return False, "", x1, x2, fhat
    except Exception as exc:
        return True, str(exc), None, None, None


def _py_try(xy: np.ndarray, **kwargs):
    """Return (raised, msg, result_dict) for a Python bkde2D call."""
    try:
        res = r2py_kernsmooth.bkde2D(xy, **kwargs)
        return False, "", res
    except Exception as exc:
        return True, str(exc), None


def _assert_match(
    py: dict,
    r_x1: np.ndarray,
    r_x2: np.ndarray,
    r_fhat: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-15,
    label: str = "",
) -> None:
    """Assert Python result matches R reference.

    The default atol=1e-15 absorbs the sign-of-zero difference that arises
    because R's ``rp > 0`` produces -0.0 for near-zero FFT artefacts while
    Python's ``np.maximum(rp, 0)`` produces +0.0.  The absolute residual
    in those bins is at most ~2e-18.
    """
    assert set(py.keys()) >= {"x1", "x2", "fhat"}, f"{label}: missing keys"
    np.testing.assert_allclose(
        py["x1"], r_x1, rtol=1e-10, atol=0.0,
        err_msg=f"{label}: x1 mismatch",
    )
    np.testing.assert_allclose(
        py["x2"], r_x2, rtol=1e-10, atol=0.0,
        err_msg=f"{label}: x2 mismatch",
    )
    np.testing.assert_allclose(
        py["fhat"], r_fhat, rtol=rtol, atol=atol,
        err_msg=f"{label}: fhat mismatch",
    )


# ---------------------------------------------------------------------------
# Edge / boundary tests
# ---------------------------------------------------------------------------

def test_bkde2D_single_observation():
    """bkde2D on a single data point with explicit bandwidth matches R."""
    xy = np.array([[3.0, 4.0]])
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    assert py["x1"].shape == (51,)
    assert py["x2"].shape == (51,)
    assert py["fhat"].shape == (51, 51)
    assert np.all(np.isfinite(py["fhat"]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="single observation")


def test_bkde2D_two_observations():
    """bkde2D on two data points matches R."""
    xy = np.array([[0.0, 0.0], [1.0, 1.0]])
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    assert np.all(np.isfinite(py["fhat"]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="two observations")


def test_bkde2D_constant_data():
    """bkde2D on all-identical rows with explicit bandwidth returns finite fhat matching R."""
    xy = np.full((20, 2), 2.0)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    assert np.all(np.isfinite(py["fhat"])), "fhat must be finite for constant data"
    # Use relative tolerance plus a small absolute tolerance for near-zero values
    _assert_match(py, r_x1, r_x2, r_fhat, rtol=1e-6, atol=1e-15,
                  label="constant data")


def test_bkde2D_small_bandwidth_emits_warning():
    """bkde2D with a very small bandwidth emits a UserWarning about the grid being too coarse."""
    rng = np.random.default_rng(1)
    xy = rng.normal(0.0, 1.0, (50, 2))
    with pytest.warns(UserWarning, match="Binning grid too coarse"):
        result = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.001, 0.001]))
    # Still returns a result of the correct shape
    assert result["fhat"].shape == (51, 51)


def test_bkde2D_small_bandwidth_output_matches_r():
    """When bandwidth is very small (warns), Python fhat still matches R."""
    rng = np.random.default_rng(1)
    xy = rng.normal(0.0, 1.0, (50, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.001, 0.001])
        py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.001, 0.001]))
    # Most bins contain numerical noise at ~machine-epsilon level
    _assert_match(py, r_x1, r_x2, r_fhat, rtol=1e-5, atol=1e-14,
                  label="small bandwidth output matches R")


def test_bkde2D_very_large_bandwidth():
    """bkde2D with bandwidth >> data range produces near-uniform surface and matches R."""
    rng = np.random.default_rng(2)
    xy = rng.normal(0.0, 1.0, (50, 2))
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[1000.0, 1000.0])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([1000.0, 1000.0]))
    assert np.all(np.isfinite(py["fhat"]))
    _assert_match(py, r_x1, r_x2, r_fhat, rtol=1e-5, atol=1e-22,
                  label="very large bandwidth")


def test_bkde2D_gridsize_2x2_minimum_grid():
    """bkde2D with the smallest useful gridsize (2,2) matches R."""
    rng = np.random.default_rng(42)
    xy = rng.normal(0.0, 1.0, (20, 2))
    r_raised, r_msg, r_x1, r_x2, r_fhat = _r_try(
        xy, bandwidth=[0.5, 0.5], gridsize=(2, 2)
    )
    py_raised, py_msg, py_result = _py_try(
        xy, bandwidth=np.array([0.5, 0.5]), gridsize=(2, 2)
    )
    if r_raised != py_raised:
        if r_raised:
            pytest.fail(f"gridsize (2,2): R raised but Python did not.\nR: {r_msg}")
        else:
            pytest.fail(f"gridsize (2,2): Python raised but R did not.\nPy: {py_msg}")
    if not r_raised:
        assert py_result["fhat"].shape == (2, 2)
        _assert_match(py_result, r_x1, r_x2, r_fhat, label="gridsize (2,2)")


def test_bkde2D_fhat_non_negative():
    """fhat must be non-negative everywhere (enforced by np.maximum(rp, 0))."""
    rng = np.random.default_rng(88)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 200)
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.3, 0.3]))
    assert np.all(py["fhat"] >= 0.0), "fhat must be non-negative everywhere"


def test_bkde2D_range_x_wider_than_data():
    """bkde2D with range_x much wider than data produces near-zero tail densities and matches R."""
    rng = np.random.default_rng(55)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    rx = [(-10.0, 10.0), (-10.0, 10.0)]
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5], range_x=rx)
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]), range_x=rx)
    _assert_match(py, r_x1, r_x2, r_fhat, rtol=1e-6, atol=1e-14,
                  label="wide range_x")
    # Corners of the grid (far from data) should be essentially zero
    assert abs(py["fhat"][0, 0]) < 1e-12, "corner (0,0) should be near zero"
    assert abs(py["fhat"][-1, -1]) < 1e-12, "corner (-1,-1) should be near zero"


def test_bkde2D_range_x_tight_at_data_boundary():
    """bkde2D with range_x exactly at data min/max matches R."""
    rng = np.random.default_rng(33)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    tight_rx = [
        (float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))),
        (float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))),
    ]
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5], range_x=tight_rx)
    py = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.5, 0.5]), range_x=tight_rx
    )
    _assert_match(py, r_x1, r_x2, r_fhat, label="tight range_x at data boundary")


def test_bkde2D_large_scale_data():
    """bkde2D on data with very large magnitude values matches R."""
    rng = np.random.default_rng(10)
    xy = rng.normal(1e6, 1e4, (100, 2))
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[500.0, 500.0])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([500.0, 500.0]))
    assert np.all(np.isfinite(py["fhat"]))
    _assert_match(py, r_x1, r_x2, r_fhat, rtol=1e-5, atol=1e-22,
                  label="large scale data ~1e6")


def test_bkde2D_small_scale_data():
    """bkde2D on data with very small magnitude values matches R.

    At very small scales numerical precision decreases slightly; an absolute
    tolerance accounts for the residual floating-point difference.
    """
    rng = np.random.default_rng(10)
    xy = rng.normal(0.0, 1e-5, (50, 2))
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[5e-6, 5e-6])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([5e-6, 5e-6]))
    assert np.all(np.isfinite(py["fhat"]))
    _assert_match(py, r_x1, r_x2, r_fhat, rtol=1e-3, atol=1e-3,
                  label="small scale data ~1e-5")


def test_bkde2D_output_shape_equals_gridsize_various_sizes():
    """fhat.shape == (M1, M2) and len(x1)==M1, len(x2)==M2 for various gridsizes."""
    rng = np.random.default_rng(0)
    xy = rng.normal(0.0, 1.0, (100, 2))
    for gs in [(10, 10), (51, 51), (100, 200), (201, 101)]:
        py = r2py_kernsmooth.bkde2D(
            xy, bandwidth=np.array([0.5, 0.5]), gridsize=gs
        )
        assert py["x1"].shape == (gs[0],), f"gridsize {gs}: x1 wrong length"
        assert py["x2"].shape == (gs[1],), f"gridsize {gs}: x2 wrong length"
        assert py["fhat"].shape == gs, f"gridsize {gs}: fhat wrong shape"


def test_bkde2D_inf_in_input_signals_problem():
    """Inf values in input x cause both R and Python to signal an error or
    produce non-finite output.

    R raises an error; Python may raise or produce non-finite fhat values.
    Either outcome is acceptable as long as Python does not silently return
    a finite density estimate for infinite-valued data.
    """
    rng = np.random.default_rng(42)
    xy = rng.normal(0.0, 1.0, (50, 2))
    xy[0, 0] = np.inf

    x_r_inf = _make_r_matrix(xy)
    r_raised, r_msg = False, ""
    try:
        _ks.bkde2D(x_r_inf, bandwidth=ro.FloatVector([0.5, 0.5]))
    except Exception as exc:
        r_raised, r_msg = True, str(exc)

    py_raised, py_msg, py_result = _py_try(xy, bandwidth=np.array([0.5, 0.5]))

    if not r_raised:
        # Unexpected: document but do not fail
        warnings.warn(
            "Inf input: R did not raise; test expectation may need updating",
            UserWarning,
            stacklevel=1,
        )

    if not py_raised:
        # Python returned a result; it must contain at least some non-finite value
        if py_result is not None and np.all(np.isfinite(py_result["fhat"])):
            pytest.fail(
                "Inf input: Python returned entirely finite fhat for data "
                "containing Inf, which is not meaningful"
            )
        # Warn about behavioural difference from R
        warnings.warn(
            "Inf input: R raised but Python returned (partially non-finite) result.\n"
            f"  R error: {r_msg!r}",
            UserWarning,
            stacklevel=1,
        )


def test_bkde2D_scalar_bandwidth_equals_length2_bandwidth():
    """A scalar bandwidth and a length-2 vector with equal components give identical results."""
    rng = np.random.default_rng(7)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    py_scalar = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5]))
    py_vec = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    np.testing.assert_array_equal(py_scalar["x1"], py_vec["x1"])
    np.testing.assert_array_equal(py_scalar["x2"], py_vec["x2"])
    np.testing.assert_array_equal(py_scalar["fhat"], py_vec["fhat"])


def test_bkde2D_nan_in_input_signals_problem():
    """NaN values in input x cause both R and Python to signal an error.

    R raises because seq() cannot form a grid from NaN bounds.
    Python raises because NaN propagates into integer conversion.
    Both must raise; if messages differ a warning is emitted.
    """
    rng = np.random.default_rng(7)
    xy = rng.normal(0.0, 1.0, (50, 2))
    xy[5, 0] = np.nan

    x_r_nan = _make_r_matrix(xy)
    r_raised, r_msg = False, ""
    try:
        _ks.bkde2D(x_r_nan, bandwidth=ro.FloatVector([0.5, 0.5]))
    except Exception as exc:
        r_raised, r_msg = True, str(exc)

    py_raised, py_msg, _ = _py_try(xy, bandwidth=np.array([0.5, 0.5]))

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


def test_bkde2D_data_far_from_origin():
    """bkde2D on data centred far from the origin (mean ~[100, 200]) matches R."""
    rng = np.random.default_rng(10)
    xy = rng.multivariate_normal([100.0, 200.0], [[4.0, 0.0], [0.0, 9.0]], 100)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.8])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.8]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="data far from origin")


def test_bkde2D_truncate_parameter_has_no_effect():
    """Both truncate=True and truncate=False produce identical results for bkde2D.

    The underlying Fortran routine ``lbtwod`` (called via ``linbin2D``) always
    truncates observations outside the grid range regardless of the ``truncate``
    flag.  This matches R's behaviour: R's bkde2D also returns identical results
    for truncate=TRUE and truncate=FALSE.  The parameter is accepted for API
    compatibility but is effectively a no-op in both implementations.
    """
    rng = np.random.default_rng(11)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    # Use a narrow range_x so that many observations lie outside the grid
    rx = [(-1.0, 1.0), (-1.0, 1.0)]
    py_t = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.5, 0.5]), range_x=rx, truncate=True
    )
    py_f = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.5, 0.5]), range_x=rx, truncate=False
    )
    # Python: identical regardless of truncate
    np.testing.assert_array_equal(
        py_t["fhat"], py_f["fhat"],
        err_msg="truncate=True and truncate=False must produce identical fhat",
    )
    # Verify this also matches R's behaviour (R returns same for both flags)
    x_r = _make_r_matrix(xy)
    range_x_r = ro.r.list(ro.FloatVector([-1.0, 1.0]), ro.FloatVector([-1.0, 1.0]))
    bw_r = ro.FloatVector([0.5, 0.5])
    r_true = _ks.bkde2D(x_r, bandwidth=bw_r, **{"range.x": range_x_r}, truncate=True)
    r_false = _ks.bkde2D(x_r, bandwidth=bw_r, **{"range.x": range_x_r}, truncate=False)
    np.testing.assert_array_equal(
        np.array(r_true.rx2("fhat")), np.array(r_false.rx2("fhat")),
        err_msg="R truncate=TRUE and truncate=FALSE must produce identical fhat",
    )


def test_bkde2D_gridsize_non_square_101x51():
    """bkde2D with non-square gridsize (101, 51) produces the correct shape and matches R."""
    rng = np.random.default_rng(20)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5], gridsize=(101, 51))
    py = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.5, 0.5]), gridsize=(101, 51)
    )
    assert py["fhat"].shape == (101, 51)
    _assert_match(py, r_x1, r_x2, r_fhat, label="non-square gridsize (101,51)")


def test_bkde2D_output_dtype_is_float64():
    """x1, x2, and fhat must have dtype float64."""
    rng = np.random.default_rng(0)
    xy = rng.normal(0.0, 1.0, (50, 2))
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    assert py["x1"].dtype == np.float64, "x1 must be float64"
    assert py["x2"].dtype == np.float64, "x2 must be float64"
    assert py["fhat"].dtype == np.float64, "fhat must be float64"


def test_bkde2D_asymmetric_bandwidth_small_and_large():
    """Asymmetric bandwidth with one very small and one large component warns and matches R.

    The small component triggers the coarse-grid warning; the result should
    still match R within relaxed tolerance.
    """
    rng = np.random.default_rng(5)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.001, 2.0])
        py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.001, 2.0]))
    _assert_match(py, r_x1, r_x2, r_fhat, rtol=1e-4, atol=1e-13,
                  label="asymmetric bw [0.001, 2.0]")


def test_bkde2D_minimum_three_observations():
    """bkde2D on the minimum practical number of observations (n=3) matches R."""
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    assert np.all(np.isfinite(py["fhat"]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="minimum n=3 observations")


def test_bkde2D_bandwidth_tau_boundary():
    """bkde2D with bandwidth exactly at the tau=3.4 smoothing boundary matches R.

    The kernel support extends tau*h in each direction from the data range.
    Verify Python's result is correct when bandwidth equals exactly the grid
    spacing (``h == (b-a)/(M-1)``), which is the tightest non-zero bandwidth
    that still places kernel weights in more than one bin.
    """
    rng = np.random.default_rng(8)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 80)
    # Compute a bandwidth equal to roughly one grid spacing for default 51-point grid
    data_range = np.max(xy, axis=0) - np.min(xy, axis=0) + 3.0  # with 1.5*h margin
    h_boundary = data_range / (51 - 1)
    bw = h_boundary  # shape (2,)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=bw.tolist())
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=bw)
    assert np.all(np.isfinite(py["fhat"]))
    _assert_match(py, r_x1, r_x2, r_fhat, rtol=1e-6, atol=1e-15,
                  label="bandwidth at tau boundary")
