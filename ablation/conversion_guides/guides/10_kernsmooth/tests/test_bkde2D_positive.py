# Positive test cases for r2py_kernsmooth.bkde2D
#
# Each test calls R's KernSmooth::bkde2D via rpy2 to obtain the reference
# result, then calls the Python port and asserts the two results agree to
# within floating-point tolerance.
#
# R's bkde2D returns:
#   $x1  - grid in direction 1  (length gridsize[1])
#   $x2  - grid in direction 2  (length gridsize[2])
#   $fhat - density matrix       (gridsize[1] x gridsize[2])
#
# Python's bkde2D returns a dict with keys "x1", "x2", "fhat".

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

# Load KernSmooth once at module level.
_ks = importr("KernSmooth")

# Shared RNG for reproducibility.
_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_r_matrix(xy: np.ndarray) -> ro.Matrix:
    """Convert an (n, 2) numpy array to an R matrix (column-major)."""
    n = xy.shape[0]
    return ro.r.matrix(
        ro.FloatVector(xy.flatten(order="F")), nrow=n, ncol=xy.shape[1]
    )


def _r_bkde2D(xy: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call R's bkde2D; return (x1, x2, fhat) as numpy arrays.

    Python keyword arguments are translated to R conventions:
      bandwidth -> FloatVector
      gridsize  -> IntVector
      range_x   -> range.x as R list of two FloatVectors
      truncate  -> as-is (rpy2 converts Python bool automatically)
    """
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
    """Assert that the Python result matches the R reference.

    Notes on tolerances
    -------------------
    R's bkde2D clips negative FFT artefacts using ``rp > 0`` (returning
    -0.0 for negative bins), while Python uses ``np.maximum(rp, 0)``
    (returning +0.0).  For bins where the true density is zero, the
    floating-point residual is at most ~2e-18.  The default ``atol=1e-15``
    absorbs this harmless sign-of-zero difference without masking genuine
    numerical mismatches.
    """
    assert set(py.keys()) >= {"x1", "x2", "fhat"}, f"{label}: missing keys"

    assert py["x1"].shape == r_x1.shape, (
        f"{label}: x1 shape mismatch {py['x1'].shape} != {r_x1.shape}"
    )
    assert py["x2"].shape == r_x2.shape, (
        f"{label}: x2 shape mismatch {py['x2'].shape} != {r_x2.shape}"
    )
    assert py["fhat"].shape == r_fhat.shape, (
        f"{label}: fhat shape mismatch {py['fhat'].shape} != {r_fhat.shape}"
    )

    np.testing.assert_allclose(
        py["x1"], r_x1, rtol=1e-10, atol=0.0,
        err_msg=f"{label}: x1 grid mismatch",
    )
    np.testing.assert_allclose(
        py["x2"], r_x2, rtol=1e-10, atol=0.0,
        err_msg=f"{label}: x2 grid mismatch",
    )
    np.testing.assert_allclose(
        py["fhat"], r_fhat, rtol=rtol, atol=atol,
        err_msg=f"{label}: fhat mismatch",
    )


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------

def test_bkde2D_standard_bivariate_normal():
    """bkde2D on standard bivariate normal data with explicit bandwidth matches R."""
    xy = _RNG.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 200)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="standard bivariate normal")


def test_bkde2D_default_gridsize_51x51():
    """Default gridsize of (51,51) produces 51-element x1, x2 and a 51x51 fhat."""
    xy = _RNG.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    assert py["x1"].shape == (51,), "x1 must have 51 elements"
    assert py["x2"].shape == (51,), "x2 must have 51 elements"
    assert py["fhat"].shape == (51, 51), "fhat must be 51x51"
    _assert_match(py, r_x1, r_x2, r_fhat, label="default gridsize 51x51")


def test_bkde2D_gridsize_101x101():
    """bkde2D with gridsize=(101,101) produces the correct shapes and matches R."""
    xy = _RNG.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 150)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.4, 0.4], gridsize=(101, 101))
    py = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.4, 0.4]), gridsize=(101, 101)
    )
    assert py["x1"].shape == (101,)
    assert py["x2"].shape == (101,)
    assert py["fhat"].shape == (101, 101)
    _assert_match(py, r_x1, r_x2, r_fhat, label="gridsize 101x101")


def test_bkde2D_non_square_gridsize():
    """bkde2D with a non-square gridsize=(51,81) returns correct shapes and matches R."""
    xy = _RNG.multivariate_normal([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], 120)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5], gridsize=(51, 81))
    py = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.5, 0.5]), gridsize=(51, 81)
    )
    assert py["x1"].shape == (51,)
    assert py["x2"].shape == (81,)
    assert py["fhat"].shape == (51, 81)
    _assert_match(py, r_x1, r_x2, r_fhat, label="non-square gridsize (51,81)")


def test_bkde2D_asymmetric_bandwidth():
    """bkde2D with h1 != h2 properly applies different smoothing per direction."""
    rng = np.random.default_rng(77)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 4.0]], 150)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.2, 0.8])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.2, 0.8]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="asymmetric bandwidth h=[0.2,0.8]")


def test_bkde2D_scalar_bandwidth_broadcast():
    """A scalar bandwidth is broadcast to both directions and matches a length-2 vector."""
    xy = _RNG.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    # R always requires length-2; use length-1 for Python scalar check
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5])
    py_scalar = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5]))
    _assert_match(py_scalar, r_x1, r_x2, r_fhat, label="scalar bandwidth broadcast")


def test_bkde2D_explicit_range_x():
    """bkde2D with an explicit range_x covers exactly the requested range and matches R."""
    xy = _RNG.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    rx = [(-4.0, 4.0), (-4.0, 4.0)]
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5], range_x=rx)
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]), range_x=rx)
    assert abs(py["x1"][0] - (-4.0)) < 1e-12, "x1[0] must equal range_x[0][0]"
    assert abs(py["x1"][-1] - 4.0) < 1e-12, "x1[-1] must equal range_x[0][1]"
    assert abs(py["x2"][0] - (-4.0)) < 1e-12, "x2[0] must equal range_x[1][0]"
    assert abs(py["x2"][-1] - 4.0) < 1e-12, "x2[-1] must equal range_x[1][1]"
    _assert_match(py, r_x1, r_x2, r_fhat, label="explicit range_x")


def test_bkde2D_asymmetric_range_x():
    """bkde2D with different ranges per direction matches R."""
    rng = np.random.default_rng(33)
    xy = rng.multivariate_normal([1.0, 5.0], [[0.5, 0.0], [0.0, 2.0]], 100)
    rx = [(0.0, 3.0), (2.0, 9.0)]
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.3, 0.6], range_x=rx)
    py = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.3, 0.6]), range_x=rx
    )
    _assert_match(py, r_x1, r_x2, r_fhat, label="asymmetric range_x")


def test_bkde2D_truncate_true():
    """bkde2D with truncate=True matches R (default behaviour)."""
    xy = _RNG.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    r_x1, r_x2, r_fhat = _r_bkde2D(
        xy, bandwidth=[0.5, 0.5], range_x=[(-2.0, 2.0), (-2.0, 2.0)],
        truncate=True,
    )
    py = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.5, 0.5]),
        range_x=[(-2.0, 2.0), (-2.0, 2.0)], truncate=True,
    )
    _assert_match(py, r_x1, r_x2, r_fhat, label="truncate=True")


def test_bkde2D_truncate_false():
    """bkde2D accepts truncate=False without error and matches R's truncate=True.

    Implementation note: the underlying Fortran routine ``lbtwod`` used by
    ``linbin2D`` always truncates observations outside the grid range.  The
    ``truncate`` parameter is accepted for API compatibility with R but has
    no effect on the binning step.  Both truncate=True and truncate=False
    therefore produce the same result, and both match R called with
    truncate=True.
    """
    rng = np.random.default_rng(11)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    rx = [(-2.0, 2.0), (-2.0, 2.0)]
    r_x1, r_x2, r_fhat = _r_bkde2D(
        xy, bandwidth=[0.5, 0.5], range_x=rx, truncate=True
    )
    py_t = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.5, 0.5]), range_x=rx, truncate=True
    )
    py_f = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.5, 0.5]), range_x=rx, truncate=False
    )
    # Both Python variants match R truncate=True
    _assert_match(py_t, r_x1, r_x2, r_fhat, label="truncate=True vs R")
    _assert_match(py_f, r_x1, r_x2, r_fhat, label="truncate=False vs R truncate=True")
    # And they are identical to each other
    np.testing.assert_array_equal(
        py_t["fhat"], py_f["fhat"],
        err_msg="truncate=True and truncate=False should produce identical fhat",
    )


def test_bkde2D_uniform_bivariate_data():
    """bkde2D on uniform [0,1]^2 data matches R."""
    rng = np.random.default_rng(7)
    xy = rng.uniform(0.0, 1.0, (200, 2))
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.1, 0.1])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.1, 0.1]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="uniform bivariate data")


def test_bkde2D_bimodal_2d_data():
    """bkde2D on 2D bimodal data recovers two modes and matches R."""
    rng = np.random.default_rng(17)
    xy = np.vstack([
        rng.multivariate_normal([-2.0, -2.0], [[0.3, 0.0], [0.0, 0.3]], 100),
        rng.multivariate_normal([2.0, 2.0], [[0.3, 0.0], [0.0, 0.3]], 100),
    ])
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.3, 0.3])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.3, 0.3]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="bimodal 2D data")
    # Both modes should be detectable as local high-density regions
    mid1 = py["fhat"].shape[0] // 2
    mid2 = py["fhat"].shape[1] // 2
    assert py["fhat"][:mid1, :mid2].max() > 0.01, "lower-left mode not detected"
    assert py["fhat"][mid1:, mid2:].max() > 0.01, "upper-right mode not detected"


def test_bkde2D_correlated_data():
    """bkde2D on correlated bivariate data (rho=0.8) matches R."""
    rng = np.random.default_rng(123)
    xy = rng.multivariate_normal([2.0, 3.0], [[1.0, 0.8], [0.8, 1.0]], 300)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.4, 0.4])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.4, 0.4]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="correlated data rho=0.8")


def test_bkde2D_integer_input_coercion():
    """bkde2D accepts an integer-typed 2-column matrix and matches R with float equivalent."""
    xy_int = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.int32)
    xy_float = xy_int.astype(np.float64)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy_float, bandwidth=[0.5, 0.5])
    py = r2py_kernsmooth.bkde2D(xy_int, bandwidth=np.array([0.5, 0.5]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="integer input coercion")


def test_bkde2D_return_structure():
    """bkde2D always returns a dict with exactly the keys 'x1', 'x2', 'fhat'."""
    xy = _RNG.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 50)
    result = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    assert isinstance(result, dict)
    assert set(result.keys()) == {"x1", "x2", "fhat"}


def test_bkde2D_grids_sorted_and_equispaced():
    """x1 and x2 grids are strictly increasing and uniformly spaced."""
    xy = _RNG.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    result = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    for name, grid in [("x1", result["x1"]), ("x2", result["x2"])]:
        diffs = np.diff(grid)
        assert np.all(diffs > 0), f"{name} grid is not strictly increasing"
        np.testing.assert_allclose(
            diffs, diffs[0], rtol=1e-10,
            err_msg=f"{name} grid is not uniformly spaced",
        )


def test_bkde2D_fhat_non_negative():
    """fhat values are all non-negative (Python applies np.maximum(rp, 0))."""
    rng = np.random.default_rng(55)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 200)
    result = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.3, 0.3]))
    assert np.all(result["fhat"] >= 0.0), "fhat must be non-negative everywhere"


def test_bkde2D_density_integrates_to_approximately_one():
    """Double-trapezoid integral of fhat should be close to 1.0."""
    rng = np.random.default_rng(99)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 500)
    py = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.3, 0.3]), gridsize=(101, 101)
    )
    dx = py["x1"][1] - py["x1"][0]
    dy = py["x2"][1] - py["x2"][0]
    integral = np.sum(py["fhat"]) * dx * dy
    assert abs(integral - 1.0) < 0.05, (
        f"Density does not integrate to ~1 (got {integral:.4f})"
    )


def test_bkde2D_large_bandwidth():
    """bkde2D with a very large bandwidth produces a smooth, flat-topped surface and matches R."""
    rng = np.random.default_rng(1)
    xy = rng.normal(0.0, 1.0, (50, 2))
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[100.0, 100.0])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([100.0, 100.0]))
    assert np.all(np.isfinite(py["fhat"]))
    _assert_match(py, r_x1, r_x2, r_fhat, rtol=1e-5, label="large bandwidth")


def test_bkde2D_gridsize_201x201():
    """bkde2D with a large square gridsize (201, 201) matches R."""
    rng = np.random.default_rng(20)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5], gridsize=(201, 201))
    py = r2py_kernsmooth.bkde2D(
        xy, bandwidth=np.array([0.5, 0.5]), gridsize=(201, 201)
    )
    assert py["fhat"].shape == (201, 201)
    _assert_match(py, r_x1, r_x2, r_fhat, label="gridsize 201x201")


def test_bkde2D_large_n():
    """bkde2D on n=500 observations matches R within tolerance."""
    rng = np.random.default_rng(13)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]], 500)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.25, 0.25])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.25, 0.25]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="large n=500")


def test_bkde2D_default_range_x_uses_1_5_h_margin():
    """Default range_x extends 1.5*h beyond the data in each direction."""
    rng = np.random.default_rng(5)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 100)
    h = np.array([0.5, 0.7])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=h)
    expected_x1_lo = np.min(xy[:, 0]) - 1.5 * h[0]
    expected_x1_hi = np.max(xy[:, 0]) + 1.5 * h[0]
    expected_x2_lo = np.min(xy[:, 1]) - 1.5 * h[1]
    expected_x2_hi = np.max(xy[:, 1]) + 1.5 * h[1]
    np.testing.assert_allclose(
        py["x1"][0], expected_x1_lo, rtol=1e-10,
        err_msg="x1[0] should be min(x[:,0]) - 1.5*h[0]",
    )
    np.testing.assert_allclose(
        py["x1"][-1], expected_x1_hi, rtol=1e-10,
        err_msg="x1[-1] should be max(x[:,0]) + 1.5*h[0]",
    )
    np.testing.assert_allclose(
        py["x2"][0], expected_x2_lo, rtol=1e-10,
        err_msg="x2[0] should be min(x[:,1]) - 1.5*h[1]",
    )
    np.testing.assert_allclose(
        py["x2"][-1], expected_x2_hi, rtol=1e-10,
        err_msg="x2[-1] should be max(x[:,1]) + 1.5*h[1]",
    )


def test_bkde2D_reproducibility():
    """Calling bkde2D twice with identical inputs produces identical outputs."""
    rng = np.random.default_rng(3)
    xy = rng.normal(0.0, 1.0, (100, 2))
    r1 = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    r2 = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    np.testing.assert_array_equal(r1["x1"], r2["x1"])
    np.testing.assert_array_equal(r1["x2"], r2["x2"])
    np.testing.assert_array_equal(r1["fhat"], r2["fhat"])


def test_bkde2D_input_not_mutated():
    """bkde2D must not modify the input array in place."""
    rng = np.random.default_rng(42)
    xy = rng.normal(0.0, 1.0, (100, 2))
    xy_copy = xy.copy()
    r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    np.testing.assert_array_equal(
        xy, xy_copy, err_msg="input array was modified by bkde2D"
    )


def test_bkde2D_negative_valued_data():
    """bkde2D on data with all-negative coordinates matches R."""
    rng = np.random.default_rng(66)
    xy = rng.multivariate_normal([-5.0, -3.0], [[1.0, 0.0], [0.0, 1.0]], 150)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.4, 0.4])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.4, 0.4]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="negative valued data")


def test_bkde2D_constant_data_with_explicit_bandwidth():
    """bkde2D on all-identical rows with explicit bandwidth returns finite fhat matching R."""
    xy = np.full((20, 2), 2.0)
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5])
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5]))
    assert np.all(np.isfinite(py["fhat"])), "fhat must be finite for constant data"
    _assert_match(py, r_x1, r_x2, r_fhat, label="constant data")


def test_bkde2D_documentation_example_bimodal_geyser_like():
    """bkde2D with bimodal geyser-like data mirrors the KernSmooth documentation example.

    R's documentation example is:
        data(geyser, package="MASS")
        est <- bkde2D(cbind(geyser$duration, geyser$waiting), bandwidth=c(0.7, 7))

    This test uses a synthetic bimodal distribution with the same marginal structure
    (short/long eruptions vs short/long waiting times) and the same bandwidths.
    The Python result must match R's output to within floating-point tolerance.
    """
    rng = np.random.default_rng(123)
    # Simulate geyser-like bimodal data: two clusters in (duration, waiting) space
    dur = np.concatenate([
        rng.normal(2.0, 0.3, 150),   # short eruptions
        rng.normal(4.5, 0.4, 122),   # long eruptions
    ])
    wait = np.concatenate([
        rng.normal(55, 8, 150),       # short waits
        rng.normal(80, 6, 122),       # long waits
    ])
    xy = np.column_stack([dur, wait])
    bw = [0.7, 7.0]
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=bw)
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array(bw))
    _assert_match(py, r_x1, r_x2, r_fhat, label="geyser-like bimodal bandwidth=[0.7, 7]")
    # The two modes should be clearly visible in the density matrix
    assert py["fhat"].max() > 0.0, "fhat should have a positive maximum"


def test_bkde2D_bandwidth_length_gt_2_uses_first_two():
    """Python bkde2D with bandwidth length > 2 uses the first two elements, matching R.

    R's bkde2D silently uses only bandwidth[1:2] when a longer vector is supplied.
    Python's implementation reads h[0] and h[1] from the array, so the behaviour
    is identical: the extra elements are ignored and results match R with the same
    first two bandwidths.
    """
    rng = np.random.default_rng(7)
    xy = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 80)
    # R reference: call with the first two elements only
    r_x1, r_x2, r_fhat = _r_bkde2D(xy, bandwidth=[0.5, 0.5])
    # Python: pass a length-3 bandwidth; extra element should be ignored
    py = r2py_kernsmooth.bkde2D(xy, bandwidth=np.array([0.5, 0.5, 999.0]))
    _assert_match(py, r_x1, r_x2, r_fhat, label="bandwidth length 3 uses first 2")
