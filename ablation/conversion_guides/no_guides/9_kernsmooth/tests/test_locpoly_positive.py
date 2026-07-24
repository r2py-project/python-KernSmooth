# Positive test cases for r2py_kernsmooth.locpoly
#
# Each test calls the R function KernSmooth::locpoly via rpy2 to obtain the
# reference result, then calls the Python port and asserts the two results
# agree to within floating-point tolerance.
#
# locpoly supports two modes:
#   - Regression:    x and y are raw data vectors.
#   - Density:       y is omitted (None); locpoly estimates the density of x.
# Both modes are exercised here, along with all major parameters.

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

# Load the KernSmooth R package once at module level.
_ks = importr("KernSmooth")

# Shared RNG for reproducibility.
_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _r_locpoly(x_py, y_py=None, **kwargs):
    """Call R's locpoly and return (x_grid, y_est) as numpy arrays.

    Maps Python keyword names to R names where they differ
    (range_x -> range.x, bwdisc -> bwdisc).
    """
    x_r = ro.FloatVector(x_py.tolist())
    args = [x_r]
    if y_py is not None:
        args.append(ro.FloatVector(y_py.tolist()))
    if "range_x" in kwargs:
        range_x = kwargs.pop("range_x")
        kwargs["range.x"] = ro.FloatVector(list(range_x))
    if "bandwidth" in kwargs and isinstance(kwargs["bandwidth"], np.ndarray):
        kwargs["bandwidth"] = ro.FloatVector(kwargs["bandwidth"].tolist())
    result = _ks.locpoly(*args, **kwargs)
    return np.array(result.rx2("x")), np.array(result.rx2("y"))


def _assert_match(py_result, r_x, r_y, *, rtol=1e-6, atol=0.0, label=""):
    assert isinstance(py_result, dict), f"{label}: result must be a dict"
    assert "x" in py_result and "y" in py_result, f"{label}: missing keys 'x' or 'y'"
    py_x = py_result["x"]
    py_y = py_result["y"]
    assert len(py_x) == len(r_x), (
        f"{label}: x length mismatch {len(py_x)} != {len(r_x)}"
    )
    assert len(py_y) == len(r_y), (
        f"{label}: y length mismatch {len(py_y)} != {len(r_y)}"
    )
    np.testing.assert_allclose(
        py_x, r_x, rtol=1e-10, atol=0.0, err_msg=f"{label}: x grid mismatch"
    )
    np.testing.assert_allclose(
        py_y, r_y, rtol=rtol, atol=atol, err_msg=f"{label}: y estimate mismatch"
    )


# ---------------------------------------------------------------------------
# Positive tests — regression mode (y provided)
# ---------------------------------------------------------------------------

def test_locpoly_regression_standard_data():
    """locpoly regression on a smooth curve with default drv=0 matches R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x) + _RNG.normal(0, 0.1, 200)
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    _assert_match(py, r_x, r_y, label="regression standard")


def test_locpoly_regression_drv0_default_degree():
    """locpoly regression drv=0 uses default degree=1 and matches R."""
    x = np.linspace(0, 5, 200)
    y = x ** 2
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    _assert_match(py, r_x, r_y, label="drv=0 default degree")


def test_locpoly_regression_drv1_first_derivative():
    """locpoly regression drv=1 estimates the first derivative and matches R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, drv=1)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, drv=1)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="drv=1 first derivative")


def test_locpoly_regression_drv2_second_derivative():
    """locpoly regression drv=2 estimates the second derivative and matches R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, drv=2)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, drv=2)
    # atol=1e-12 guards against near-zero floating-point noise at sin(pi)~0.
    _assert_match(py, r_x, r_y, rtol=1e-5, atol=1e-12, label="drv=2 second derivative")


def test_locpoly_regression_drv3_third_derivative():
    """locpoly regression drv=3 estimates the third derivative and matches R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, drv=3)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, drv=3)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="drv=3 third derivative")


def test_locpoly_regression_explicit_degree_higher_than_drv():
    """Explicit degree > drv+1 gives a valid estimate matching R."""
    x = np.linspace(0, 5, 200)
    y = x ** 3
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, drv=0, degree=4)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, drv=0, degree=4)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="degree=4, drv=0")


def test_locpoly_regression_drv1_explicit_degree3():
    """Explicit degree=3, drv=1 (over-determined polynomial) matches R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, drv=1, degree=3)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, drv=1, degree=3)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="drv=1 degree=3")


def test_locpoly_regression_scalar_bandwidth():
    """locpoly regression with a scalar bandwidth matches R."""
    x = np.linspace(0, 10, 300)
    y = np.cos(x) + _RNG.normal(0, 0.05, 300)
    r_x, r_y = _r_locpoly(x, y, bandwidth=1.0)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=1.0)
    _assert_match(py, r_x, r_y, label="scalar bandwidth regression")


def test_locpoly_regression_vector_bandwidth():
    """locpoly regression with a bandwidth vector of length gridsize matches R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    bw = np.linspace(0.3, 0.7, 401)
    r_x, r_y = _r_locpoly(x, y, bandwidth=bw)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=bw)
    # atol guards against floating-point noise where sin(x)~0 at midpoint.
    _assert_match(py, r_x, r_y, rtol=1e-5, atol=1e-12, label="vector bandwidth")


def test_locpoly_regression_custom_gridsize():
    """locpoly regression returns exactly the requested number of grid points."""
    x = np.linspace(0, 5, 200)
    y = x ** 2
    for gs in [101, 201, 801]:
        r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, gridsize=gs)
        py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, gridsize=gs)
        assert len(py["x"]) == gs, f"gridsize={gs}: wrong number of x points"
        assert len(py["y"]) == gs, f"gridsize={gs}: wrong number of y points"
        _assert_match(py, r_x, r_y, label=f"gridsize={gs}")


def test_locpoly_regression_custom_range_x():
    """locpoly regression with an explicit range_x matches R exactly."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    range_x = (-0.5, 7.0)
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, range_x=range_x)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, range_x=range_x)
    assert abs(py["x"][0] - range_x[0]) < 1e-12, "x[0] must equal range_x[0]"
    assert abs(py["x"][-1] - range_x[1]) < 1e-12, "x[-1] must equal range_x[1]"
    _assert_match(py, r_x, r_y, label="custom range_x regression")


def test_locpoly_regression_truncate_true():
    """locpoly regression with truncate=True (default) matches R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, truncate=True)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, truncate=True)
    # atol guards against ~1e-17 floating-point noise where sin(x)~0.
    _assert_match(py, r_x, r_y, atol=1e-12, label="truncate=True")


def test_locpoly_regression_truncate_false():
    """locpoly regression with truncate=False matches R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5, truncate=False)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, truncate=False)
    # atol guards against ~1e-17 floating-point noise where sin(x)~0.
    _assert_match(py, r_x, r_y, atol=1e-12, label="truncate=False")


def test_locpoly_regression_truncate_changes_estimate():
    """truncate=True and truncate=False produce different estimates for out-of-range data."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 300)
    y = x ** 2 + rng.normal(0, 0.1, 300)
    range_x = (-1.5, 1.5)
    py_true = r2py_kernsmooth.locpoly(x, y, bandwidth=0.3, range_x=range_x, truncate=True)
    py_false = r2py_kernsmooth.locpoly(x, y, bandwidth=0.3, range_x=range_x, truncate=False)
    assert not np.allclose(py_true["y"], py_false["y"]), (
        "truncate=True and truncate=False should yield different estimates when data fall outside range_x"
    )


def test_locpoly_regression_binned_mode():
    """locpoly with binned=True (pre-binned counts) matches R's binned output."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    gpoints = np.linspace(float(x.min()), float(x.max()), 401)
    out_bin = r2py_kernsmooth.rlbin(x, y, gpoints)
    xcounts = out_bin["xcounts"]
    ycounts = out_bin["ycounts"]
    range_x = (float(x.min()), float(x.max()))
    r_x, r_y = _r_locpoly(
        xcounts, ycounts, bandwidth=0.5, range_x=range_x, binned=True
    )
    py = r2py_kernsmooth.locpoly(xcounts, ycounts, bandwidth=0.5, range_x=range_x, binned=True)
    # atol guards against ~1e-17 noise where sin(x)~0 at midpoint.
    _assert_match(py, r_x, r_y, atol=1e-12, label="binned=True")


def test_locpoly_regression_bwdisc_varied():
    """Different bwdisc values with a vector bandwidth both match R."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    bw = np.linspace(0.3, 0.7, 401)
    for bwdisc in [10, 50]:
        r_x, r_y = _r_locpoly(x, y, bandwidth=bw, bwdisc=bwdisc)
        py = r2py_kernsmooth.locpoly(x, y, bandwidth=bw, bwdisc=bwdisc)
        # atol guards against ~1e-17 noise where sin(x)~0 at midpoint.
        _assert_match(py, r_x, r_y, rtol=1e-5, atol=1e-12, label=f"bwdisc={bwdisc}")


def test_locpoly_regression_negative_x():
    """locpoly regression on negative-valued x matches R."""
    x = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    y = x ** 2
    r_x, r_y = _r_locpoly(x, y, bandwidth=1.0)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=1.0)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="negative x")


def test_locpoly_regression_constant_y():
    """locpoly regression on constant y recovers the constant and matches R."""
    x = np.linspace(1, 5, 100)
    y = np.full(100, 3.0)
    r_x, r_y = _r_locpoly(x, y, bandwidth=1.0)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=1.0)
    _assert_match(py, r_x, r_y, label="constant y")
    # The interior estimates should be close to 3.0
    interior = py["y"][50:350]
    np.testing.assert_allclose(interior, 3.0, atol=0.01,
                               err_msg="constant y: interior estimate should be ~3.0")


def test_locpoly_regression_linear_y():
    """locpoly regression on linear y recovers the line and matches R."""
    x = np.linspace(1, 5, 100)
    y = 2.0 * x + 1.0
    r_x, r_y = _r_locpoly(x, y, bandwidth=1.0)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=1.0)
    _assert_match(py, r_x, r_y, label="linear y")


def test_locpoly_regression_large_n():
    """locpoly regression on n=1000 observations matches R."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 1000)
    y = np.sin(x) + rng.normal(0, 0.1, 1000)
    r_x, r_y = _r_locpoly(x, y, bandwidth=0.5)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="n=1000")


def test_locpoly_regression_integer_input_coercion():
    """locpoly accepts integer-typed x and y arrays (coerces to float64) and matches R."""
    x_int = np.arange(1, 51, dtype=np.int32)
    y_int = (x_int ** 2).astype(np.int32)
    x_float = x_int.astype(np.float64)
    y_float = y_int.astype(np.float64)
    r_x, r_y = _r_locpoly(x_float, y_float, bandwidth=5.0)
    py = r2py_kernsmooth.locpoly(x_int, y_int, bandwidth=5.0)
    _assert_match(py, r_x, r_y, label="integer input coercion")


def test_locpoly_regression_large_bandwidth():
    """locpoly regression with a very large bandwidth produces a smooth estimate matching R."""
    x = np.linspace(0, 5, 200)
    y = x ** 2
    r_x, r_y = _r_locpoly(x, y, bandwidth=100.0)
    py = r2py_kernsmooth.locpoly(x, y, bandwidth=100.0)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="large bandwidth regression")


def test_locpoly_regression_prestige_dataset():
    """locpoly on the carData::Prestige dataset (income, prestige) matches R.

    This replicates the canonical test from KernSmooth/tests/locpoly.R.
    """
    _cardata = importr("carData")
    income = np.array(ro.r("Prestige$income"), dtype=float)
    prestige = np.array(ro.r("Prestige$prestige"), dtype=float)
    r_x, r_y = _r_locpoly(income, prestige, bandwidth=5000.0)
    py = r2py_kernsmooth.locpoly(income, prestige, bandwidth=5000.0)
    _assert_match(py, r_x, r_y, rtol=1e-6, label="Prestige dataset")


# ---------------------------------------------------------------------------
# Positive tests — density mode (y omitted)
# ---------------------------------------------------------------------------

def test_locpoly_density_standard_normal():
    """locpoly density estimation on standard-normal data matches R."""
    x = _RNG.normal(0, 1, 300)
    r_x, r_y = _r_locpoly(x, bandwidth=0.3)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.3)
    _assert_match(py, r_x, r_y, label="density standard normal")


def test_locpoly_density_uniform_data():
    """locpoly density on uniform [0,1] data matches R."""
    x = np.random.default_rng(1).uniform(0, 1, 200)
    r_x, r_y = _r_locpoly(x, bandwidth=0.1)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.1)
    _assert_match(py, r_x, r_y, label="density uniform")


def test_locpoly_density_range_extends_by_5_percent():
    """locpoly density default range_x extends 5% of data range beyond min/max."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r_x, r_y = _r_locpoly(x, bandwidth=0.5)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.5)
    extra = 0.05 * (x.max() - x.min())
    assert abs(py["x"][0] - (x.min() - extra)) < 1e-12, "density x[0] mismatch"
    assert abs(py["x"][-1] - (x.max() + extra)) < 1e-12, "density x[-1] mismatch"
    _assert_match(py, r_x, r_y, label="density range extension")


def test_locpoly_density_drv1():
    """locpoly density with drv=1 estimates the first derivative of density and matches R."""
    x = np.random.default_rng(13).normal(0, 1, 500)
    r_x, r_y = _r_locpoly(x, bandwidth=0.4, drv=1)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.4, drv=1)
    _assert_match(py, r_x, r_y, rtol=1e-5, label="density drv=1")


def test_locpoly_density_gridsize_101():
    """locpoly density with gridsize=101 returns exactly 101 points and matches R."""
    x = _RNG.normal(0, 1, 200)
    r_x, r_y = _r_locpoly(x, bandwidth=0.3, gridsize=101)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.3, gridsize=101)
    assert len(py["x"]) == 101
    assert len(py["y"]) == 101
    _assert_match(py, r_x, r_y, label="density gridsize=101")


def test_locpoly_density_custom_range_x():
    """locpoly density with explicit range_x matches R."""
    x = _RNG.normal(0, 1, 200)
    range_x = (-3.0, 3.0)
    r_x, r_y = _r_locpoly(x, bandwidth=0.3, range_x=range_x)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.3, range_x=range_x)
    assert abs(py["x"][0] - range_x[0]) < 1e-12
    assert abs(py["x"][-1] - range_x[1]) < 1e-12
    _assert_match(py, r_x, r_y, label="density custom range_x")


def test_locpoly_density_integrates_to_approximately_one():
    """Trapezoid-rule integral of the locpoly density estimate should be close to 1."""
    x = np.random.default_rng(99).normal(0, 1, 1000)
    py = r2py_kernsmooth.locpoly(x, bandwidth=0.3)
    integral = np.trapezoid(py["y"], py["x"])
    assert abs(integral - 1.0) < 0.05, (
        f"Density does not integrate to ~1 (got {integral:.4f})"
    )


# ---------------------------------------------------------------------------
# Structural / contract tests (apply to both modes)
# ---------------------------------------------------------------------------

def test_locpoly_return_structure_regression():
    """locpoly always returns a dict with exactly the keys 'x' and 'y'."""
    x = np.linspace(0, 5, 100)
    y = x ** 2
    result = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"x", "y"}


def test_locpoly_return_structure_density():
    """locpoly density mode also returns a dict with exactly the keys 'x' and 'y'."""
    x = _RNG.normal(0, 1, 100)
    result = r2py_kernsmooth.locpoly(x, bandwidth=0.3)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"x", "y"}


def test_locpoly_output_dtype_float64():
    """locpoly output arrays must have dtype float64."""
    x = np.linspace(0, 5, 100)
    y = x ** 2
    result = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    assert result["x"].dtype == np.float64, "x array should be float64"
    assert result["y"].dtype == np.float64, "y array should be float64"


def test_locpoly_x_grid_is_sorted_and_equispaced():
    """locpoly x grid must be strictly ascending and uniformly spaced."""
    x = _RNG.normal(0, 1, 200)
    y = x + _RNG.normal(0, 0.1, 200)
    result = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    x_grid = result["x"]
    diffs = np.diff(x_grid)
    assert np.all(diffs > 0), "x grid is not strictly increasing"
    np.testing.assert_allclose(
        diffs, diffs[0], rtol=1e-10, err_msg="x grid is not uniformly spaced"
    )


def test_locpoly_output_length_equals_gridsize():
    """len(result['x']) == len(result['y']) == gridsize for various gridsizes."""
    x = _RNG.normal(0, 1, 100)
    y = x + _RNG.normal(0, 0.1, 100)
    for gs in [50, 100, 200, 401, 512]:
        result = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5, gridsize=gs)
        assert len(result["x"]) == gs, f"gridsize={gs}: x has wrong length"
        assert len(result["y"]) == gs, f"gridsize={gs}: y has wrong length"


def test_locpoly_regression_range_x_equals_data_range():
    """For regression the default range_x equals [min(x), max(x)] exactly."""
    x = np.linspace(1.5, 8.3, 200)
    y = x ** 2
    result = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    assert abs(result["x"][0] - x.min()) < 1e-12, "x[0] must equal min(x)"
    assert abs(result["x"][-1] - x.max()) < 1e-12, "x[-1] must equal max(x)"


def test_locpoly_reproducibility():
    """Calling locpoly twice with identical inputs yields identical outputs."""
    x = _RNG.normal(0, 1, 100)
    y = x + _RNG.normal(0, 0.1, 100)
    r1 = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    r2 = r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    np.testing.assert_array_equal(r1["x"], r2["x"])
    np.testing.assert_array_equal(r1["y"], r2["y"])


def test_locpoly_input_not_mutated():
    """locpoly must not modify the input arrays in place."""
    x = np.linspace(0, 5, 100)
    y = x ** 2
    x_copy = x.copy()
    y_copy = y.copy()
    r2py_kernsmooth.locpoly(x, y, bandwidth=0.5)
    np.testing.assert_array_equal(x, x_copy, err_msg="x input array was modified")
    np.testing.assert_array_equal(y, y_copy, err_msg="y input array was modified")
