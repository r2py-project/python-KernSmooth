# Positive test cases for r2py_kernsmooth.dpik
#
# Each test calls the R function KernSmooth::dpik via rpy2 to obtain the
# reference result, then calls the Python port and asserts the two scalar
# outputs agree to within floating-point tolerance.
#
# R signature:
#   dpik(x, scalest = "minim", level = 2L, kernel = "normal",
#        canonical = FALSE, gridsize = 401L,
#        range.x = range(x), truncate = TRUE)
#
# Python signature:
#   dpik(x, scalest="minim", level=2, kernel="normal",
#        canonical=False, gridsize=401,
#        range_x=None, truncate=True) -> np.float64
#
# The function selects a bandwidth for kernel density estimation using
# the direct plug-in method of Sheather and Jones (1991).

import math

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

# Load KernSmooth once at module level.
_ks = importr("KernSmooth")


# ---------------------------------------------------------------------------
# Helper: call R dpik and return the scalar as a Python float
# ---------------------------------------------------------------------------

def _r_dpik(x_py, **kwargs):
    """Call R's dpik and return the bandwidth scalar as a Python float."""
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


# ---------------------------------------------------------------------------
# Helper assertion: compare Python scalar to R scalar
# ---------------------------------------------------------------------------

def _assert_close(py_val, r_val, *, rtol=1e-5, label=""):
    assert np.isfinite(py_val), f"{label}: Python returned non-finite value {py_val!r}"
    assert np.isfinite(r_val), f"{label}: R returned non-finite value {r_val!r}"
    np.testing.assert_allclose(
        py_val,
        r_val,
        rtol=rtol,
        err_msg=f"{label}: Python={py_val!r} vs R={r_val!r}",
    )


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------


def test_dpik_default_parameters_standard_normal():
    """dpik with all defaults on standard-normal data matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="default params / normal n=100")


def test_dpik_returns_np_float64():
    """dpik always returns an np.float64 scalar."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    result = r2py_kernsmooth.dpik(x)
    assert isinstance(result, np.float64), (
        f"Expected np.float64, got {type(result)}"
    )
    assert result > 0, f"Bandwidth must be positive, got {result}"


def test_dpik_kernel_normal():
    """dpik with kernel='normal' (default) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, kernel="normal")
    py_val = r2py_kernsmooth.dpik(x, kernel="normal")
    _assert_close(py_val, r_val, label="kernel=normal")


def test_dpik_kernel_box():
    """dpik with kernel='box' matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, kernel="box")
    py_val = r2py_kernsmooth.dpik(x, kernel="box")
    _assert_close(py_val, r_val, label="kernel=box")


def test_dpik_kernel_epanech():
    """dpik with kernel='epanech' matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, kernel="epanech")
    py_val = r2py_kernsmooth.dpik(x, kernel="epanech")
    _assert_close(py_val, r_val, label="kernel=epanech")


def test_dpik_kernel_biweight():
    """dpik with kernel='biweight' matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, kernel="biweight")
    py_val = r2py_kernsmooth.dpik(x, kernel="biweight")
    _assert_close(py_val, r_val, label="kernel=biweight")


def test_dpik_kernel_triweight():
    """dpik with kernel='triweight' matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, kernel="triweight")
    py_val = r2py_kernsmooth.dpik(x, kernel="triweight")
    _assert_close(py_val, r_val, label="kernel=triweight")


def test_dpik_all_kernels_return_positive():
    """dpik returns a positive scalar for every supported kernel."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    for kernel in ("normal", "box", "epanech", "biweight", "triweight"):
        py_val = r2py_kernsmooth.dpik(x, kernel=kernel)
        assert py_val > 0, f"kernel={kernel} returned non-positive {py_val}"


def test_dpik_canonical_false_default():
    """dpik with canonical=False (default) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, canonical=False)
    py_val = r2py_kernsmooth.dpik(x, canonical=False)
    _assert_close(py_val, r_val, label="canonical=False")


def test_dpik_canonical_true():
    """dpik with canonical=True matches R and ignores kernel-specific del0."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, canonical=True)
    py_val = r2py_kernsmooth.dpik(x, canonical=True)
    _assert_close(py_val, r_val, label="canonical=True")


def test_dpik_canonical_true_same_for_all_kernels():
    """When canonical=True, all kernels produce the same bandwidth (del0=1)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    results = {
        kernel: r2py_kernsmooth.dpik(x, kernel=kernel, canonical=True)
        for kernel in ("normal", "box", "epanech", "biweight", "triweight")
    }
    values = list(results.values())
    # All canonical results should be identical because del0=1 for all kernels.
    for k, v in results.items():
        np.testing.assert_allclose(
            v,
            values[0],
            rtol=1e-12,
            err_msg=f"canonical=True results differ: kernel=normal vs kernel={k}",
        )


def test_dpik_canonical_true_vs_false_differ():
    """canonical=True and canonical=False produce different bandwidths for non-normal kernels."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    # For box kernel the del0 != 1, so canonical changes the result.
    py_can_true = r2py_kernsmooth.dpik(x, kernel="box", canonical=True)
    py_can_false = r2py_kernsmooth.dpik(x, kernel="box", canonical=False)
    assert py_can_true != pytest.approx(py_can_false, rel=1e-6), (
        "canonical=True and canonical=False should differ for kernel='box'"
    )


def test_dpik_scalest_minim_default():
    """dpik with scalest='minim' (default) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, scalest="minim")
    py_val = r2py_kernsmooth.dpik(x, scalest="minim")
    _assert_close(py_val, r_val, label="scalest=minim")


def test_dpik_scalest_stdev():
    """dpik with scalest='stdev' uses standard deviation as scale estimate."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, scalest="stdev")
    py_val = r2py_kernsmooth.dpik(x, scalest="stdev")
    _assert_close(py_val, r_val, label="scalest=stdev")


def test_dpik_scalest_iqr():
    """dpik with scalest='iqr' uses IQR/1.349 as scale estimate."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, scalest="iqr")
    py_val = r2py_kernsmooth.dpik(x, scalest="iqr")
    _assert_close(py_val, r_val, label="scalest=iqr")


def test_dpik_scalest_minim_picks_iqr_for_heavy_tailed():
    """scalest='minim' picks IQR over stdev when outliers inflate stdev."""
    rng = np.random.default_rng(42)
    # Heavy-tailed data: outliers inflate stdev but IQR stays small.
    x = np.concatenate([rng.normal(0, 1, 90), rng.normal(0, 5, 10)])
    r_minim = _r_dpik(x, scalest="minim")
    r_stdev = _r_dpik(x, scalest="stdev")
    r_iqr = _r_dpik(x, scalest="iqr")
    py_val = r2py_kernsmooth.dpik(x, scalest="minim")
    # 'minim' result should equal the smaller of stdev / iqr results from R.
    assert r_minim == pytest.approx(min(r_stdev, r_iqr), rel=1e-6)
    _assert_close(py_val, r_minim, label="scalest=minim heavy-tailed")


def test_dpik_level_0():
    """dpik with level=0 (normal-scale rule) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, level=0)
    py_val = r2py_kernsmooth.dpik(x, level=0)
    _assert_close(py_val, r_val, label="level=0")


def test_dpik_level_1():
    """dpik with level=1 (one plug-in step) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, level=1)
    py_val = r2py_kernsmooth.dpik(x, level=1)
    _assert_close(py_val, r_val, label="level=1")


def test_dpik_level_2_default():
    """dpik with level=2 (default, two-step plug-in) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, level=2)
    py_val = r2py_kernsmooth.dpik(x, level=2)
    _assert_close(py_val, r_val, label="level=2 (default)")


def test_dpik_level_3():
    """dpik with level=3 (three plug-in steps) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, level=3)
    py_val = r2py_kernsmooth.dpik(x, level=3)
    _assert_close(py_val, r_val, label="level=3")


def test_dpik_level_4():
    """dpik with level=4 (four plug-in steps) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, level=4)
    py_val = r2py_kernsmooth.dpik(x, level=4)
    _assert_close(py_val, r_val, label="level=4")


def test_dpik_level_5():
    """dpik with level=5 (maximum plug-in depth) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, level=5)
    py_val = r2py_kernsmooth.dpik(x, level=5)
    _assert_close(py_val, r_val, label="level=5")


def test_dpik_large_sample_n1000():
    """dpik on n=1000 standard-normal observations matches R."""
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 1000)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="n=1000 normal")


def test_dpik_large_sample_n5000():
    """dpik on n=5000 observations matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5000)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="n=5000 normal")


def test_dpik_small_sample_n10():
    """dpik on n=10 observations matches R."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 10)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="n=10 normal")


def test_dpik_exponential_data():
    """dpik on right-skewed exponential data matches R."""
    rng = np.random.default_rng(123)
    x = rng.exponential(1.0, 200)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="exponential n=200")


def test_dpik_uniform_data():
    """dpik on uniform data matches R."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-5.0, 5.0, 100)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="uniform[-5,5] n=100")


def test_dpik_bimodal_data():
    """dpik on bimodal data matches R."""
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(-2, 0.5, 50), rng.normal(2, 0.5, 50)])
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="bimodal n=100")


def test_dpik_custom_range_x():
    """dpik with an explicit range_x matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    range_x = (-3.0, 3.0)
    r_val = _r_dpik(x, range_x=range_x)
    py_val = r2py_kernsmooth.dpik(x, range_x=range_x)
    _assert_close(py_val, r_val, label="custom range_x=(-3,3)")


def test_dpik_truncate_false():
    """dpik with truncate=False matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    range_x = (-2.0, 2.0)
    r_val = _r_dpik(x, range_x=range_x, truncate=False)
    py_val = r2py_kernsmooth.dpik(x, range_x=range_x, truncate=False)
    _assert_close(py_val, r_val, label="truncate=False")


def test_dpik_truncate_true_vs_false_differ():
    """truncate=True and truncate=False produce different bandwidths when data exceed range_x."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    range_x = (-2.0, 2.0)
    r_true = _r_dpik(x, range_x=range_x, truncate=True)
    r_false = _r_dpik(x, range_x=range_x, truncate=False)
    py_true = r2py_kernsmooth.dpik(x, range_x=range_x, truncate=True)
    py_false = r2py_kernsmooth.dpik(x, range_x=range_x, truncate=False)
    _assert_close(py_true, r_true, label="truncate=True")
    _assert_close(py_false, r_false, label="truncate=False")
    assert py_true != pytest.approx(py_false, rel=1e-6), (
        "truncate=True and truncate=False should produce different bandwidths"
    )


def test_dpik_gridsize_100():
    """dpik with gridsize=100 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, gridsize=100)
    py_val = r2py_kernsmooth.dpik(x, gridsize=100)
    _assert_close(py_val, r_val, label="gridsize=100")


def test_dpik_gridsize_801():
    """dpik with gridsize=801 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpik(x, gridsize=801)
    py_val = r2py_kernsmooth.dpik(x, gridsize=801)
    _assert_close(py_val, r_val, label="gridsize=801")


def test_dpik_large_valued_data():
    """dpik on data centered at 1e6 with std=1e4 matches R (scale translation)."""
    rng = np.random.default_rng(42)
    x = rng.normal(1000.0, 100.0, 100)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="large-valued data mean=1000 sd=100")


def test_dpik_small_valued_data():
    """dpik on data with very small std=0.001 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 0.001, 100)
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="small-valued data sd=0.001")


def test_dpik_data_with_outliers_matches_r():
    """dpik on data containing extreme outliers matches R."""
    rng = np.random.default_rng(99)
    x = np.concatenate([
        rng.normal(0, 1, 95),
        np.array([100.0, 200.0, 300.0, -100.0, -200.0])
    ])
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="data with outliers")


def test_dpik_all_levels_all_scalest_match_r():
    """dpik matches R for every (level, scalest) combination in a 6x3 grid."""
    rng = np.random.default_rng(77)
    x = rng.normal(0, 1, 200)
    for level in range(6):
        for scalest in ("minim", "stdev", "iqr"):
            r_val = _r_dpik(x, level=level, scalest=scalest)
            py_val = r2py_kernsmooth.dpik(x, level=level, scalest=scalest)
            _assert_close(
                py_val,
                r_val,
                label=f"level={level} scalest={scalest}",
            )


def test_dpik_all_kernels_all_levels_match_r():
    """dpik matches R for every (kernel, level) combination in a 5x6 grid."""
    rng = np.random.default_rng(77)
    x = rng.normal(0, 1, 200)
    for kernel in ("normal", "box", "epanech", "biweight", "triweight"):
        for level in range(6):
            r_val = _r_dpik(x, kernel=kernel, level=level)
            py_val = r2py_kernsmooth.dpik(x, kernel=kernel, level=level)
            _assert_close(
                py_val,
                r_val,
                label=f"kernel={kernel} level={level}",
            )


def test_dpik_canonical_all_kernels_match_r():
    """dpik with canonical=True matches R for every kernel."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    for kernel in ("normal", "box", "epanech", "biweight", "triweight"):
        r_val = _r_dpik(x, kernel=kernel, canonical=True)
        py_val = r2py_kernsmooth.dpik(x, kernel=kernel, canonical=True)
        _assert_close(py_val, r_val, label=f"canonical=True kernel={kernel}")


def test_dpik_partial_name_kernel_n():
    """dpik accepts 'n' as an unambiguous abbreviation of 'normal'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpik(x, kernel="normal")
    py_abbr = r2py_kernsmooth.dpik(x, kernel="n")
    assert py_full == pytest.approx(py_abbr, rel=1e-12), (
        "kernel='n' should resolve to 'normal'"
    )


def test_dpik_partial_name_kernel_bo():
    """dpik accepts 'bo' as an unambiguous abbreviation of 'box'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpik(x, kernel="box")
    py_abbr = r2py_kernsmooth.dpik(x, kernel="bo")
    assert py_full == pytest.approx(py_abbr, rel=1e-12), (
        "kernel='bo' should resolve to 'box'"
    )


def test_dpik_partial_name_kernel_ep():
    """dpik accepts 'ep' as an unambiguous abbreviation of 'epanech'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpik(x, kernel="epanech")
    py_abbr = r2py_kernsmooth.dpik(x, kernel="ep")
    assert py_full == pytest.approx(py_abbr, rel=1e-12), (
        "kernel='ep' should resolve to 'epanech'"
    )


def test_dpik_partial_name_kernel_bi():
    """dpik accepts 'bi' as an unambiguous abbreviation of 'biweight'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpik(x, kernel="biweight")
    py_abbr = r2py_kernsmooth.dpik(x, kernel="bi")
    assert py_full == pytest.approx(py_abbr, rel=1e-12), (
        "kernel='bi' should resolve to 'biweight'"
    )


def test_dpik_partial_name_kernel_tr():
    """dpik accepts 'tr' as an unambiguous abbreviation of 'triweight'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpik(x, kernel="triweight")
    py_abbr = r2py_kernsmooth.dpik(x, kernel="tr")
    assert py_full == pytest.approx(py_abbr, rel=1e-12), (
        "kernel='tr' should resolve to 'triweight'"
    )


def test_dpik_partial_name_scalest_m():
    """dpik accepts 'm' as an unambiguous abbreviation of 'minim'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpik(x, scalest="minim")
    py_abbr = r2py_kernsmooth.dpik(x, scalest="m")
    assert py_full == pytest.approx(py_abbr, rel=1e-12), (
        "scalest='m' should resolve to 'minim'"
    )


def test_dpik_partial_name_scalest_st():
    """dpik accepts 'st' as an unambiguous abbreviation of 'stdev'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpik(x, scalest="stdev")
    py_abbr = r2py_kernsmooth.dpik(x, scalest="st")
    assert py_full == pytest.approx(py_abbr, rel=1e-12), (
        "scalest='st' should resolve to 'stdev'"
    )


def test_dpik_partial_name_scalest_i():
    """dpik accepts 'i' as an unambiguous abbreviation of 'iqr'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpik(x, scalest="iqr")
    py_abbr = r2py_kernsmooth.dpik(x, scalest="i")
    assert py_full == pytest.approx(py_abbr, rel=1e-12), (
        "scalest='i' should resolve to 'iqr'"
    )


def test_dpik_n2_matches_r():
    """dpik with n=2 observations returns a positive scalar matching R."""
    x = np.array([1.0, 2.0])
    r_val = _r_dpik(x)
    py_val = r2py_kernsmooth.dpik(x)
    _assert_close(py_val, r_val, label="n=2")


def test_dpik_integer_input_coerced_to_float():
    """dpik accepts an integer-typed array and matches the float version."""
    rng = np.random.default_rng(42)
    x_int = np.arange(1, 101, dtype=np.int32)
    x_float = x_int.astype(np.float64)
    py_int = r2py_kernsmooth.dpik(x_int)
    py_float = r2py_kernsmooth.dpik(x_float)
    assert py_int == pytest.approx(py_float, rel=1e-10), (
        "Integer input should give the same result as float input"
    )


def test_dpik_level_0_matches_analytical_formula():
    """level=0 result equals scalest_val * del0 * (1/(psi4_normal*n))^(1/5)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_val = r2py_kernsmooth.dpik(x, level=0)
    # Compute expected value analytically.
    std_val = float(np.std(x, ddof=1))
    iqr_val = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    scalest_val = min(std_val, iqr_val)
    del0_normal = 1.0 / ((4.0 * math.pi) ** (1.0 / 10.0))
    psi4_normal = 3.0 / (8.0 * math.sqrt(math.pi))
    n = len(x)
    expected = scalest_val * del0_normal * (1.0 / (psi4_normal * n)) ** (1.0 / 5.0)
    np.testing.assert_allclose(
        py_val,
        expected,
        rtol=1e-10,
        err_msg="level=0 does not match the closed-form normal-scale formula",
    )


def test_dpik_reproducibility():
    """dpik called twice with identical input returns the same result."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    h1 = r2py_kernsmooth.dpik(x)
    h2 = r2py_kernsmooth.dpik(x)
    assert h1 == h2, "dpik is not deterministic / reproducible"


def test_dpik_all_scalest_options_positive():
    """dpik returns a positive scalar for all three scalest options."""
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, 100)
    for scalest in ("minim", "stdev", "iqr"):
        py_val = r2py_kernsmooth.dpik(x, scalest=scalest)
        assert py_val > 0, f"scalest={scalest} returned non-positive {py_val}"
