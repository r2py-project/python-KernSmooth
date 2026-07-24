# Boundary and edge case tests for r2py_kernsmooth.dpik
#
# These tests target the functional limits of dpik:
#   - Minimal sample sizes (n=2, n=3, n=5)
#   - Very large and very small data magnitudes (scale invariance)
#   - Data with extreme outliers (scalest='minim' picks IQR)
#   - All five kernel types across all six plug-in levels
#   - canonical=True uniformity across all kernels
#   - gridsize at boundary values (small: 10, large: 2001)
#   - truncate=True vs truncate=False when range_x is narrower than data
#   - All scalest options when stdev and IQR/1.349 are nearly equal
#   - Reproducibility: calling dpik twice on the same data returns same result
#   - Exact scale-invariance: dpik(c*x) == c * dpik(x)
#   - Location shift: dpik(x + k) == dpik(x) for any constant k
#   - Two-valued data (minimum distinct observations for non-zero IQR)
#   - Integer-typed input coercion

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

def _r_dpik(x_py, **kwargs):
    """Call R dpik; return scalar float.  Propagates R exceptions."""
    x_r = ro.FloatVector(x_py.tolist())
    r_kwargs = {}
    if "scalest" in kwargs:
        r_kwargs["scalest"] = str(kwargs["scalest"])
    if "level" in kwargs:
        r_kwargs["level"] = int(kwargs["level"])
    if "kernel" in kwargs:
        r_kwargs["kernel"] = str(kwargs["kernel"])
    if "canonical" in kwargs:
        r_kwargs["canonical"] = bool(kwargs["canonical"])
    if "gridsize" in kwargs:
        r_kwargs["gridsize"] = int(kwargs["gridsize"])
    if "range_x" in kwargs:
        r_kwargs["range.x"] = ro.FloatVector(list(kwargs["range_x"]))
    if "truncate" in kwargs:
        r_kwargs["truncate"] = bool(kwargs["truncate"])
    result = _ks.dpik(x_r, **r_kwargs)
    return float(result[0])


def _r_dpik_catch(x_py, **kwargs):
    """Call R dpik; return (raised: bool, message: str)."""
    try:
        val = _r_dpik(x_py, **kwargs)
        return False, ""
    except Exception as exc:
        return True, str(exc)


def _assert_close(py_val, r_val, *, rtol=1e-5, label=""):
    assert np.isfinite(py_val), f"{label}: Python returned non-finite {py_val!r}"
    assert np.isfinite(r_val), f"{label}: R returned non-finite {r_val!r}"
    np.testing.assert_allclose(
        py_val,
        r_val,
        rtol=rtol,
        err_msg=f"{label}: Python={py_val!r} vs R={r_val!r}",
    )


# ---------------------------------------------------------------------------
# Minimal sample sizes
# ---------------------------------------------------------------------------


def test_dpik_n2_minimal_sample():
    """n=2 (minimum non-trivial sample) returns a positive scalar matching R."""
    x = np.array([1.0, 2.0])
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    assert py_val > 0, f"n=2: expected positive result, got {py_val}"
    _assert_close(py_val, r_val, label="n=2")


def test_dpik_n3_sample():
    """n=3 returns a positive scalar matching R."""
    x = np.array([1.0, 2.0, 3.0])
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="n=3")


def test_dpik_n5_sample():
    """n=5 returns a positive scalar matching R."""
    x = np.arange(1.0, 6.0)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="n=5")


# ---------------------------------------------------------------------------
# Scale invariance: dpik(c*x) == c * dpik(x) exactly for normal kernel
# ---------------------------------------------------------------------------


def test_dpik_exact_scale_invariance():
    """dpik(c*x) == c * dpik(x) exactly (verified against R as well)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    c = 100.0
    r_x = _r_dpik(x)
    r_cx = _r_dpik(c * x)
    py_x = float(r2py_kernsmooth.dpik(x))
    py_cx = float(r2py_kernsmooth.dpik(c * x))
    # R confirms exact scale invariance.
    np.testing.assert_allclose(
        r_cx, c * r_x, rtol=1e-12,
        err_msg="R: dpik(c*x) should equal c*dpik(x)"
    )
    # Python should match.
    np.testing.assert_allclose(
        py_cx, c * py_x, rtol=1e-6,
        err_msg="Python: dpik(c*x) should equal c*dpik(x)"
    )
    _assert_close(py_cx, r_cx, label="scale invariance c=100")


def test_dpik_scale_invariance_small_c():
    """dpik(c*x) == c * dpik(x) for c=0.001 (small scale factor)."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    c = 0.001
    py_x = float(r2py_kernsmooth.dpik(x))
    py_cx = float(r2py_kernsmooth.dpik(c * x))
    np.testing.assert_allclose(
        py_cx, c * py_x, rtol=1e-5,
        err_msg="dpik should be scale-invariant for c=0.001"
    )


# ---------------------------------------------------------------------------
# Location shift: dpik(x + k) == dpik(x) exactly
# ---------------------------------------------------------------------------


def test_dpik_location_shift_invariance():
    """dpik(x + k) == dpik(x) for any constant k (bandwidth is shift-invariant)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    k = 1000.0
    py_x = float(r2py_kernsmooth.dpik(x))
    py_xk = float(r2py_kernsmooth.dpik(x + k))
    np.testing.assert_allclose(
        py_xk, py_x, rtol=1e-10,
        err_msg="dpik should be invariant to location shifts"
    )


# ---------------------------------------------------------------------------
# Very large and very small magnitudes
# ---------------------------------------------------------------------------


def test_dpik_large_magnitude_matches_r():
    """dpik on data with mean=1e6 and std=1e4 returns a positive result matching R."""
    rng = np.random.default_rng(55)
    x = rng.normal(1e6, 1e4, 100)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="large magnitude mean=1e6")


def test_dpik_small_magnitude_matches_r():
    """dpik on data with std=1e-4 returns a positive result matching R."""
    rng = np.random.default_rng(55)
    x = rng.normal(0.0, 1e-4, 100)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="small magnitude sd=1e-4")


def test_dpik_entirely_negative_values_matches_r():
    """dpik on entirely negative data (location shift) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(-50.0, 3.0, 100)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="all negative data")


# ---------------------------------------------------------------------------
# Outlier-heavy data: minim picks IQR over stdev
# ---------------------------------------------------------------------------


def test_dpik_outliers_minim_uses_iqr():
    """When outliers inflate stdev far above IQR/1.349, scalest='minim' uses IQR."""
    rng = np.random.default_rng(99)
    core = rng.normal(0, 1, 95)
    outliers = np.array([100.0, 200.0, 300.0, -100.0, -200.0])
    x = np.concatenate([core, outliers])
    std_val = float(np.std(x, ddof=1))
    iqr_val = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    assert iqr_val < std_val, "Setup failure: expected iqr < stdev for this data"
    r_val = _r_dpik(x, scalest="minim")
    py_val = r2py_kernsmooth.dpik(x, scalest="minim")
    _assert_close(py_val, r_val, label="outliers minim=iqr")
    # 'minim' should agree with 'iqr' result for this dataset.
    py_iqr = r2py_kernsmooth.dpik(x, scalest="iqr")
    assert py_val == pytest.approx(py_iqr, rel=1e-10), (
        "scalest='minim' should equal scalest='iqr' when iqr < stdev"
    )


# ---------------------------------------------------------------------------
# All plug-in levels (0-5) on the same dataset, each matching R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 5])
def test_dpik_all_levels_match_r(level):
    """dpik with each level (0-5) matches R on a fixed dataset."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, level=level)
    py_val = r2py_kernsmooth.dpik(x, level=level)
    _assert_close(py_val, r_val, label=f"level={level}")


# ---------------------------------------------------------------------------
# All five kernels on the same dataset, matching R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["normal", "box", "epanech", "biweight", "triweight"])
def test_dpik_all_kernels_match_r(kernel):
    """dpik with each kernel matches R on a fixed dataset."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, kernel=kernel)
    py_val = r2py_kernsmooth.dpik(x, kernel=kernel)
    _assert_close(py_val, r_val, label=f"kernel={kernel}")


# ---------------------------------------------------------------------------
# canonical=True: all kernels produce the same bandwidth
# ---------------------------------------------------------------------------


def test_dpik_canonical_true_all_kernels_match_r_and_each_other():
    """canonical=True makes del0=1 for all kernels; result equals R and is kernel-independent."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_canonical_normal = _r_dpik(x, canonical=True, kernel="normal")
    kernels = ("normal", "box", "epanech", "biweight", "triweight")
    for kernel in kernels:
        r_val = _r_dpik(x, canonical=True, kernel=kernel)
        py_val = r2py_kernsmooth.dpik(x, canonical=True, kernel=kernel)
        # R confirms canonical results are kernel-independent.
        np.testing.assert_allclose(
            r_val, r_canonical_normal, rtol=1e-12,
            err_msg=f"R canonical results should be equal: kernel={kernel}"
        )
        _assert_close(py_val, r_val, label=f"canonical=True kernel={kernel}")


# ---------------------------------------------------------------------------
# Gridsize edge values
# ---------------------------------------------------------------------------


def test_dpik_small_gridsize_10():
    """dpik with gridsize=10 returns a positive scalar matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, gridsize=10)
    py_val = r2py_kernsmooth.dpik(x, gridsize=10)
    assert py_val > 0
    _assert_close(py_val, r_val, label="gridsize=10")


def test_dpik_large_gridsize_2001():
    """dpik with gridsize=2001 returns a positive scalar matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, gridsize=2001)
    py_val = r2py_kernsmooth.dpik(x, gridsize=2001)
    assert py_val > 0
    _assert_close(py_val, r_val, label="gridsize=2001")


def test_dpik_gridsize_convergence():
    """Finer gridsizes converge toward the same bandwidth (binning approximation)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    h_coarse = r2py_kernsmooth.dpik(x, gridsize=50)
    h_fine = r2py_kernsmooth.dpik(x, gridsize=2001)
    assert h_coarse > 0 and h_fine > 0
    # Coarse and fine grids should agree within 5%.
    np.testing.assert_allclose(
        h_coarse, h_fine, rtol=0.05,
        err_msg="Coarse and fine gridsize results should be within 5%",
    )


# ---------------------------------------------------------------------------
# truncate=True vs truncate=False with a narrow range_x
# ---------------------------------------------------------------------------


def test_dpik_truncate_true_vs_false_narrow_range():
    """When range_x is narrower than the data, truncate affects bin counts and result."""
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(0, 1, 90), rng.uniform(5, 10, 10)])
    range_x = (-3.0, 3.0)   # Excludes the 10 uniform points.
    r_true = _r_dpik(x, range_x=range_x, truncate=True)
    r_false = _r_dpik(x, range_x=range_x, truncate=False)
    py_true = r2py_kernsmooth.dpik(x, range_x=range_x, truncate=True)
    py_false = r2py_kernsmooth.dpik(x, range_x=range_x, truncate=False)
    _assert_close(py_true, r_true, label="truncate=True narrow range")
    _assert_close(py_false, r_false, label="truncate=False narrow range")
    assert py_true != pytest.approx(py_false, rel=1e-4), (
        "truncate=True and truncate=False should differ when data extend beyond range_x"
    )


# ---------------------------------------------------------------------------
# All scalest options when stdev ≈ IQR/1.349 (near-normal data at large n)
# ---------------------------------------------------------------------------


def test_dpik_scalest_all_options_near_equal_scales():
    """When stdev ≈ IQR/1.349 (large normal sample), all scalest options are close."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    results = {s: r2py_kernsmooth.dpik(x, scalest=s) for s in ("minim", "stdev", "iqr")}
    for s, v in results.items():
        assert v > 0 and np.isfinite(v), f"scalest={s} returned {v}"
    # For large normal data, all three should agree within 10%.
    np.testing.assert_allclose(
        list(results.values()),
        results["minim"],
        rtol=0.1,
        err_msg="scalest options should agree within 10% for large normal sample",
    )


def test_dpik_scalest_minim_bounded_by_stdev_and_iqr():
    """scalest='minim' result is always <= min(scalest='stdev', scalest='iqr')."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    py_minim = r2py_kernsmooth.dpik(x, scalest="minim")
    py_stdev = r2py_kernsmooth.dpik(x, scalest="stdev")
    py_iqr = r2py_kernsmooth.dpik(x, scalest="iqr")
    assert py_minim <= py_stdev + 1e-10, (
        "minim result should be <= stdev result"
    )
    assert py_minim <= py_iqr + 1e-10, (
        "minim result should be <= iqr result"
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_dpik_reproducibility():
    """dpik called twice with the same input returns the identical result."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    h1 = r2py_kernsmooth.dpik(x)
    h2 = r2py_kernsmooth.dpik(x)
    assert h1 == h2, "dpik is not deterministic / reproducible"


# ---------------------------------------------------------------------------
# range_x wider than the data span
# ---------------------------------------------------------------------------


def test_dpik_range_x_wider_than_data_matches_r():
    """range_x extending beyond the data bounds gives a valid result matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    range_x = (float(x.min()) - 5.0, float(x.max()) + 5.0)
    r_val = _r_dpik(x, range_x=range_x)
    py_val = r2py_kernsmooth.dpik(x, range_x=range_x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="range_x wider than data")


# ---------------------------------------------------------------------------
# Two-valued data (minimum distinct observations for non-zero IQR)
# ---------------------------------------------------------------------------


def test_dpik_two_valued_data_iqr_nonzero():
    """Data alternating between two values: IQR > 0 so dpik should succeed."""
    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="two-valued data")


# ---------------------------------------------------------------------------
# Integer-typed input coercion
# ---------------------------------------------------------------------------


def test_dpik_integer_array_coercion_matches_r():
    """Integer-typed x is accepted and coerced to float; result matches R."""
    x_int = np.arange(1, 51, dtype=np.int32)
    x_float = x_int.astype(np.float64)
    r_val = _r_dpik(x_float)
    py_val = r2py_kernsmooth.dpik(x_int)
    assert py_val > 0
    _assert_close(py_val, r_val, label="integer array coercion")


# ---------------------------------------------------------------------------
# Kernel-ordering: non-canonical bandwidths are ordered by kernel smoothness
# ---------------------------------------------------------------------------


def test_dpik_kernel_bandwidth_ordering():
    """Non-canonical bandwidths increase with kernel order (normal < box < ... < triweight)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    bw = {
        k: float(r2py_kernsmooth.dpik(x, kernel=k, canonical=False))
        for k in ("normal", "box", "epanech", "biweight", "triweight")
    }
    # Each kernel's del0 increases: normal < box < epanech < biweight < triweight.
    assert bw["normal"] < bw["box"], "normal bw should be < box bw"
    assert bw["box"] < bw["epanech"], "box bw should be < epanech bw"
    assert bw["epanech"] < bw["biweight"], "epanech bw should be < biweight bw"
    assert bw["biweight"] < bw["triweight"], "biweight bw should be < triweight bw"


# ---------------------------------------------------------------------------
# All (kernel, level, scalest) combinations matching R — parametrised spot-checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel,level", [
    ("normal", 0), ("box", 1), ("epanech", 2),
    ("biweight", 3), ("triweight", 4), ("normal", 5),
])
def test_dpik_kernel_level_combinations_match_r(kernel, level):
    """dpik matches R for selected (kernel, level) pairs."""
    rng = np.random.default_rng(99)
    x = rng.normal(0, 1, 150)
    r_val = _r_dpik(x, kernel=kernel, level=level)
    py_val = r2py_kernsmooth.dpik(x, kernel=kernel, level=level)
    _assert_close(py_val, r_val, label=f"kernel={kernel} level={level}")


@pytest.mark.parametrize("kernel,canonical", [
    ("normal", True), ("box", True), ("epanech", False),
    ("biweight", False), ("triweight", True),
])
def test_dpik_kernel_canonical_combinations_match_r(kernel, canonical):
    """dpik matches R for selected (kernel, canonical) pairs."""
    rng = np.random.default_rng(77)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, kernel=kernel, canonical=canonical)
    py_val = r2py_kernsmooth.dpik(x, kernel=kernel, canonical=canonical)
    _assert_close(py_val, r_val, label=f"kernel={kernel} canonical={canonical}")
