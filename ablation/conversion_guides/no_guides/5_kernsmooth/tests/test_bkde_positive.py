# Positive test cases for r2py_kernsmooth.bkde
#
# Each test calls the R function KernSmooth::bkde via rpy2 to obtain the
# reference result, then calls the Python port and asserts the two results
# agree to within floating-point tolerance.

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

# Load the KernSmooth R package once at module level.
_ks = importr("KernSmooth")

# Shared RNG seed for reproducibility.
_RNG = np.random.default_rng(42)


def _r_bkde(x_py, **kwargs):
    """Call R's bkde and return (x_grid, y_est) as numpy arrays."""
    x_r = ro.FloatVector(x_py.tolist())
    # Map Python snake_case range_x -> R range.x
    if "range_x" in kwargs:
        range_x = kwargs.pop("range_x")
        kwargs["range.x"] = ro.FloatVector(list(range_x))
    result = _ks.bkde(x_r, **kwargs)
    return np.array(result.rx2("x")), np.array(result.rx2("y"))


# ---------------------------------------------------------------------------
# Helper assertion
# ---------------------------------------------------------------------------

def _assert_match(py_result, r_x, r_y, *, rtol=1e-6, atol=0.0, label=""):
    assert "x" in py_result and "y" in py_result, f"{label}: missing keys"
    py_x = py_result["x"]
    py_y = py_result["y"]
    assert len(py_x) == len(r_x), f"{label}: x length mismatch {len(py_x)} != {len(r_x)}"
    assert len(py_y) == len(r_y), f"{label}: y length mismatch {len(py_y)} != {len(r_y)}"
    np.testing.assert_allclose(py_x, r_x, rtol=1e-10, atol=0.0,
                               err_msg=f"{label}: x grid mismatch")
    np.testing.assert_allclose(py_y, r_y, rtol=rtol, atol=atol,
                               err_msg=f"{label}: y density mismatch")


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------

def test_bkde_default_parameters_standard_normal():
    """bkde with all defaults on standard-normal data matches R."""
    x = _RNG.normal(0, 1, 100)
    r_x, r_y = _r_bkde(x)
    py = r2py_kernsmooth.bkde(x)
    _assert_match(py, r_x, r_y, label="default params / normal data")


def test_bkde_explicit_bandwidth_normal_kernel():
    """bkde with an explicit bandwidth and default normal kernel matches R."""
    x = _RNG.normal(0, 1, 100)
    r_x, r_y = _r_bkde(x, bandwidth=0.5)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    _assert_match(py, r_x, r_y, label="explicit bw=0.5 normal kernel")


def test_bkde_kernel_box():
    """bkde with box kernel matches R."""
    x = _RNG.normal(0, 1, 100)
    r_x, r_y = _r_bkde(x, bandwidth=0.5, kernel="box")
    py = r2py_kernsmooth.bkde(x, kernel="box", bandwidth=0.5)
    _assert_match(py, r_x, r_y, label="kernel=box")


def test_bkde_kernel_epanech():
    """bkde with Epanechnikov kernel matches R."""
    x = _RNG.normal(0, 1, 100)
    r_x, r_y = _r_bkde(x, bandwidth=0.5, kernel="epanech")
    py = r2py_kernsmooth.bkde(x, kernel="epanech", bandwidth=0.5)
    _assert_match(py, r_x, r_y, label="kernel=epanech")


def test_bkde_kernel_biweight():
    """bkde with biweight kernel matches R."""
    x = _RNG.normal(0, 1, 100)
    r_x, r_y = _r_bkde(x, bandwidth=0.5, kernel="biweight")
    py = r2py_kernsmooth.bkde(x, kernel="biweight", bandwidth=0.5)
    _assert_match(py, r_x, r_y, label="kernel=biweight")


def test_bkde_kernel_triweight():
    """bkde with triweight kernel matches R."""
    x = _RNG.normal(0, 1, 100)
    r_x, r_y = _r_bkde(x, bandwidth=0.5, kernel="triweight")
    py = r2py_kernsmooth.bkde(x, kernel="triweight", bandwidth=0.5)
    _assert_match(py, r_x, r_y, label="kernel=triweight")


def test_bkde_canonical_true_normal_kernel():
    """bkde with canonical=True scales the bandwidth by del0 and matches R."""
    x = _RNG.normal(0, 1, 100)
    r_x, r_y = _r_bkde(x, bandwidth=1.0, canonical=True)
    py = r2py_kernsmooth.bkde(x, bandwidth=1.0, canonical=True)
    _assert_match(py, r_x, r_y, label="canonical=True normal")


def test_bkde_canonical_true_all_kernels():
    """canonical=True with every kernel type matches R."""
    x = _RNG.normal(0, 1, 100)
    for kernel in ("normal", "box", "epanech", "biweight", "triweight"):
        r_x, r_y = _r_bkde(x, bandwidth=1.0, canonical=True, kernel=kernel)
        py = r2py_kernsmooth.bkde(x, kernel=kernel, bandwidth=1.0, canonical=True)
        _assert_match(py, r_x, r_y, label=f"canonical=True kernel={kernel}")


def test_bkde_custom_range_x():
    """bkde with an explicit range_x covers exactly that range and matches R."""
    x = _RNG.normal(0, 1, 100)
    range_x = (-3.0, 3.0)
    r_x, r_y = _r_bkde(x, bandwidth=0.5, range_x=range_x)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.5, range_x=range_x)
    assert abs(py["x"][0] - range_x[0]) < 1e-12, "x[0] must equal range_x[0]"
    assert abs(py["x"][-1] - range_x[1]) < 1e-12, "x[-1] must equal range_x[1]"
    _assert_match(py, r_x, r_y, label="custom range_x")


def test_bkde_gridsize_101():
    """bkde returns exactly 101 grid points when gridsize=101."""
    x = _RNG.normal(0, 1, 100)
    r_x, r_y = _r_bkde(x, bandwidth=0.5, gridsize=101)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.5, gridsize=101)
    assert len(py["x"]) == 101
    assert len(py["y"]) == 101
    _assert_match(py, r_x, r_y, label="gridsize=101")


def test_bkde_gridsize_256_power_of_two():
    """bkde with gridsize=256 (exact power of 2) matches R (regression for old bkfe bug)."""
    x_vals = [0.036, 0.042, 0.052, 0.216, 0.368, 0.511, 0.705, 0.753, 0.776, 0.84]
    x = np.array(x_vals)
    r_x, r_y = _r_bkde(x, gridsize=256, range_x=(x.min(), x.max()))
    py = r2py_kernsmooth.bkde(x, gridsize=256, range_x=(x.min(), x.max()))
    _assert_match(py, r_x, r_y, label="gridsize=256 (power of 2)")


def test_bkde_gridsize_512():
    """bkde with gridsize=512 returns 512 grid points and matches R."""
    x = _RNG.normal(0, 1, 200)
    r_x, r_y = _r_bkde(x, bandwidth=0.3, gridsize=512)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.3, gridsize=512)
    assert len(py["x"]) == 512
    _assert_match(py, r_x, r_y, label="gridsize=512")


def test_bkde_truncate_false():
    """bkde with truncate=False produces a different estimate from truncate=True and matches R."""
    x = _RNG.normal(0, 1, 100)
    range_x = (-2.0, 2.0)
    r_x_t, r_y_t = _r_bkde(x, bandwidth=0.5, truncate=True, range_x=range_x)
    r_x_f, r_y_f = _r_bkde(x, bandwidth=0.5, truncate=False, range_x=range_x)
    py_t = r2py_kernsmooth.bkde(x, bandwidth=0.5, truncate=True, range_x=range_x)
    py_f = r2py_kernsmooth.bkde(x, bandwidth=0.5, truncate=False, range_x=range_x)
    _assert_match(py_t, r_x_t, r_y_t, label="truncate=True")
    _assert_match(py_f, r_x_f, r_y_f, label="truncate=False")
    # Truncation must change the estimate
    assert not np.allclose(py_t["y"], py_f["y"]), (
        "truncate=True and truncate=False should yield different densities"
    )


def test_bkde_uniform_data():
    """bkde on uniform [0,1] data with a given bandwidth matches R."""
    x = np.random.default_rng(0).uniform(0, 1, 200)
    r_x, r_y = _r_bkde(x, bandwidth=0.1)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.1)
    _assert_match(py, r_x, r_y, label="uniform data")


def test_bkde_bimodal_data():
    """bkde on bimodal data recovers two modes and matches R."""
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(-2, 0.5, 150), rng.normal(2, 0.5, 150)])
    r_x, r_y = _r_bkde(x, bandwidth=0.3)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.3)
    _assert_match(py, r_x, r_y, label="bimodal data")
    # Both modes should be visible as local maxima
    py_y = py["y"]
    midpoint_idx = len(py_y) // 2
    assert py_y[:midpoint_idx].max() > 0.1, "left mode not detected"
    assert py_y[midpoint_idx:].max() > 0.1, "right mode not detected"


def test_bkde_large_n():
    """bkde on n=500 observations with all kernels matches R."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal(500)
    for kernel in ("normal", "box", "epanech", "biweight", "triweight"):
        r_x, r_y = _r_bkde(x, kernel=kernel, bandwidth=0.3, gridsize=401)
        py = r2py_kernsmooth.bkde(x, kernel=kernel, bandwidth=0.3, gridsize=401)
        _assert_match(py, r_x, r_y, label=f"n=500 kernel={kernel}")


def test_bkde_integer_input_coercion():
    """bkde accepts integer-typed arrays (coercing to float) and matches R."""
    x_int = np.arange(1, 101, dtype=np.int32)
    x_float = x_int.astype(np.float64)
    r_x, r_y = _r_bkde(x_float, bandwidth=5.0)
    py = r2py_kernsmooth.bkde(x_int, bandwidth=5.0)
    _assert_match(py, r_x, r_y, label="integer input coercion")


def test_bkde_kernel_partial_name_normal():
    """bkde accepts the unambiguous abbreviation 'nor' for 'normal'."""
    x = _RNG.normal(0, 1, 50)
    py_full = r2py_kernsmooth.bkde(x, kernel="normal", bandwidth=0.5)
    py_abbr = r2py_kernsmooth.bkde(x, kernel="nor", bandwidth=0.5)
    np.testing.assert_array_equal(py_full["x"], py_abbr["x"])
    np.testing.assert_array_equal(py_full["y"], py_abbr["y"])


def test_bkde_kernel_partial_name_epanech():
    """bkde accepts 'ep' as unambiguous abbreviation of 'epanech'."""
    x = _RNG.normal(0, 1, 50)
    py_full = r2py_kernsmooth.bkde(x, kernel="epanech", bandwidth=0.5)
    py_abbr = r2py_kernsmooth.bkde(x, kernel="ep", bandwidth=0.5)
    np.testing.assert_array_equal(py_full["y"], py_abbr["y"])


def test_bkde_return_structure():
    """bkde always returns a dict with exactly the keys 'x' and 'y'."""
    x = _RNG.normal(0, 1, 50)
    result = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"x", "y"}


def test_bkde_x_grid_is_sorted_and_equispaced():
    """bkde x grid must be sorted in ascending order and uniformly spaced."""
    x = _RNG.normal(0, 1, 100)
    result = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    x_grid = result["x"]
    diffs = np.diff(x_grid)
    assert np.all(diffs > 0), "x grid is not strictly increasing"
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10,
                               err_msg="x grid is not uniformly spaced")


def test_bkde_density_sums_to_approximately_one():
    """Trapezoid-rule integral of the density estimate should be close to 1."""
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 1000)
    result = r2py_kernsmooth.bkde(x, bandwidth=0.3)
    integral = np.trapezoid(result["y"], result["x"])
    assert abs(integral - 1.0) < 0.02, (
        f"Density does not integrate to ~1 (got {integral:.4f})"
    )


def test_bkde_default_bandwidth_auto_computed():
    """When no bandwidth is given, bkde auto-computes one and matches R."""
    rng = np.random.default_rng(99)
    x = rng.normal(0, 1, 100)
    r_x, r_y = _r_bkde(x)
    py = r2py_kernsmooth.bkde(x)
    _assert_match(py, r_x, r_y, label="auto bandwidth")


def test_bkde_exponential_data():
    """bkde on right-skewed exponential data matches R."""
    x = np.random.default_rng(33).exponential(1, 200)
    r_x, r_y = _r_bkde(x, bandwidth=0.3)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.3)
    _assert_match(py, r_x, r_y, label="exponential data")


def test_bkde_negative_data():
    """bkde on negative-valued data matches R."""
    x = -np.abs(np.random.default_rng(5).normal(0, 1, 100))
    r_x, r_y = _r_bkde(x, bandwidth=0.3)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.3)
    _assert_match(py, r_x, r_y, label="negative data")


def test_bkde_large_bandwidth():
    """bkde with a very large bandwidth produces a smooth estimate matching R."""
    x = np.random.default_rng(5).normal(0, 1, 100)
    r_x, r_y = _r_bkde(x, bandwidth=5.0)
    py = r2py_kernsmooth.bkde(x, bandwidth=5.0)
    _assert_match(py, r_x, r_y, label="large bandwidth=5.0")


def test_bkde_small_n():
    """bkde on n=3 observations with an explicit bandwidth matches R."""
    x = np.array([1.0, 2.0, 3.0])
    r_x, r_y = _r_bkde(x, bandwidth=0.5)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    _assert_match(py, r_x, r_y, label="small n=3")


def test_bkde_constant_data_explicit_bandwidth():
    """bkde on constant data with explicit bandwidth returns finite estimates matching R."""
    x = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    r_x, r_y = _r_bkde(x, bandwidth=0.5)
    py = r2py_kernsmooth.bkde(x, bandwidth=0.5)
    assert np.all(np.isfinite(py["y"])), "y should be finite for constant data"
    _assert_match(py, r_x, r_y, label="constant data")
