# Positive test cases for r2py_kernsmooth.dpih
#
# Each test calls the R function KernSmooth::dpih via rpy2 to obtain the
# reference result, then calls the Python port and asserts the two scalar
# outputs agree to within floating-point tolerance.
#
# R signature:
#   dpih(x, scalest = "minim", level = 2L, gridsize = 401L,
#        range.x = range(x), truncate = TRUE)
#
# Python signature:
#   dpih(x, scalest="minim", level=2, gridsize=401,
#        range_x=None, truncate=True) -> np.float64

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

# Load KernSmooth once at module level.
_ks = importr("KernSmooth")

# Shared RNGs for reproducibility.
_RNG42 = np.random.default_rng(42)
_RNG123 = np.random.default_rng(123)


# ---------------------------------------------------------------------------
# Helper: call R dpih and return the scalar as a Python float
# ---------------------------------------------------------------------------

def _r_dpih(x_py, **kwargs):
    """Call R's dpih and return the scalar bin-width as a Python float."""
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
    result = _ks.dpih(x_r, **r_kwargs)
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


def test_dpih_default_parameters_standard_normal():
    """dpih with all defaults on standard-normal data matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="default params / normal n=100")


def test_dpih_scalest_stdev():
    """dpih with scalest='stdev' uses standard deviation as scale estimate."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, scalest="stdev")
    py_val = r2py_kernsmooth.dpih(x, scalest="stdev")
    _assert_close(py_val, r_val, label="scalest=stdev")


def test_dpih_scalest_iqr():
    """dpih with scalest='iqr' uses IQR/1.349 as scale estimate."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, scalest="iqr")
    py_val = r2py_kernsmooth.dpih(x, scalest="iqr")
    _assert_close(py_val, r_val, label="scalest=iqr")


def test_dpih_scalest_minim_selects_smaller_of_stdev_and_iqr():
    """dpih scalest='minim' returns min(stdev, iqr/1.349)-scaled result."""
    rng = np.random.default_rng(42)
    # Use heavy-tailed data so that iqr < stdev, confirming 'minim' picks IQR.
    x = np.concatenate([rng.normal(0, 1, 90), rng.normal(0, 5, 10)])
    r_minim = _r_dpih(x, scalest="minim")
    r_stdev = _r_dpih(x, scalest="stdev")
    r_iqr = _r_dpih(x, scalest="iqr")
    py_val = r2py_kernsmooth.dpih(x, scalest="minim")
    # 'minim' result should equal the smaller of stdev / iqr results.
    assert r_minim == pytest.approx(min(r_stdev, r_iqr), rel=1e-6)
    _assert_close(py_val, r_minim, label="scalest=minim heavy-tailed")


def test_dpih_level_0():
    """dpih with level=0 (normal-scale rule) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, level=0)
    py_val = r2py_kernsmooth.dpih(x, level=0)
    _assert_close(py_val, r_val, label="level=0")


def test_dpih_level_1():
    """dpih with level=1 (one plug-in step) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, level=1)
    py_val = r2py_kernsmooth.dpih(x, level=1)
    _assert_close(py_val, r_val, label="level=1")


def test_dpih_level_2_default():
    """dpih with level=2 (default two-step plug-in) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, level=2)
    py_val = r2py_kernsmooth.dpih(x, level=2)
    _assert_close(py_val, r_val, label="level=2 (default)")


def test_dpih_level_3():
    """dpih with level=3 (three plug-in steps) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, level=3)
    py_val = r2py_kernsmooth.dpih(x, level=3)
    _assert_close(py_val, r_val, label="level=3")


def test_dpih_level_4():
    """dpih with level=4 (four plug-in steps) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, level=4)
    py_val = r2py_kernsmooth.dpih(x, level=4)
    _assert_close(py_val, r_val, label="level=4")


def test_dpih_level_5():
    """dpih with level=5 (maximum plug-in depth) matches R."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, level=5)
    py_val = r2py_kernsmooth.dpih(x, level=5)
    _assert_close(py_val, r_val, label="level=5")


def test_dpih_large_sample_n1000():
    """dpih on n=1000 standard-normal observations matches R."""
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 1000)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="n=1000 normal")


def test_dpih_small_sample_n10():
    """dpih on n=10 observations matches R."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 10)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="n=10 normal")


def test_dpih_exponential_data():
    """dpih on right-skewed exponential data matches R."""
    rng = np.random.default_rng(123)
    x = rng.exponential(1.0, 200)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="exponential n=200")


def test_dpih_uniform_data():
    """dpih on uniform data matches R."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-5.0, 5.0, 100)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="uniform[-5,5] n=100")


def test_dpih_bimodal_data():
    """dpih on bimodal data matches R."""
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(-2, 0.5, 50), rng.normal(2, 0.5, 50)])
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="bimodal n=100")


def test_dpih_custom_range_x():
    """dpih with an explicit range_x matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    range_x = (-3.0, 3.0)
    r_val = _r_dpih(x, range_x=range_x)
    py_val = r2py_kernsmooth.dpih(x, range_x=range_x)
    _assert_close(py_val, r_val, label="custom range_x=(-3,3)")


def test_dpih_truncate_false():
    """dpih with truncate=False matches R and differs from truncate=True."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    range_x = (-2.0, 2.0)
    r_true = _r_dpih(x, range_x=range_x, truncate=True)
    r_false = _r_dpih(x, range_x=range_x, truncate=False)
    py_true = r2py_kernsmooth.dpih(x, range_x=range_x, truncate=True)
    py_false = r2py_kernsmooth.dpih(x, range_x=range_x, truncate=False)
    _assert_close(py_true, r_true, label="truncate=True")
    _assert_close(py_false, r_false, label="truncate=False")
    # The two estimates must differ because outlier weighting changes counts.
    assert py_true != pytest.approx(py_false, rel=1e-6), (
        "truncate=True and truncate=False should produce different bin widths"
    )


def test_dpih_gridsize_100():
    """dpih with gridsize=100 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, gridsize=100)
    py_val = r2py_kernsmooth.dpih(x, gridsize=100)
    _assert_close(py_val, r_val, label="gridsize=100")


def test_dpih_gridsize_801():
    """dpih with gridsize=801 matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, gridsize=801)
    py_val = r2py_kernsmooth.dpih(x, gridsize=801)
    _assert_close(py_val, r_val, label="gridsize=801")


def test_dpih_large_valued_data():
    """dpih on data centered at 1000 with std=100 matches R (scale-invariance)."""
    rng = np.random.default_rng(42)
    x = rng.normal(1000.0, 100.0, 100)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="large-valued data mean=1000 sd=100")


def test_dpih_small_valued_data():
    """dpih on data with very small std=0.001 matches R (scale-invariance)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 0.001, 100)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="small-valued data sd=0.001")


def test_dpih_returns_positive_scalar():
    """dpih always returns a positive np.float64 scalar."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    result = r2py_kernsmooth.dpih(x)
    assert isinstance(result, np.float64), (
        f"Expected np.float64, got {type(result)}"
    )
    assert result > 0, f"Bin width must be positive, got {result}"


def test_dpih_all_scalest_options_all_positive():
    """dpih returns a positive scalar for all three scalest options."""
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, 100)
    for scalest in ("minim", "stdev", "iqr"):
        py_val = r2py_kernsmooth.dpih(x, scalest=scalest)
        assert py_val > 0, f"scalest={scalest} returned non-positive {py_val}"


def test_dpih_level_increases_bin_width_monotonically():
    """Higher plug-in levels converge; bin width should be broadly similar."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    # Collect bin widths for levels 1–5.
    widths = [r2py_kernsmooth.dpih(x, level=lv) for lv in range(1, 6)]
    # All must be finite and positive.
    for i, w in enumerate(widths, start=1):
        assert w > 0 and np.isfinite(w), f"level={i} returned {w}"
    # Levels 2–5 are well-known to converge; differences should be small.
    np.testing.assert_allclose(
        widths[1:],
        widths[1],
        rtol=0.15,
        err_msg="Levels 2-5 should be within 15% of each other for normal data",
    )


def test_dpih_partial_name_scalest_m():
    """dpih accepts 'm' as an unambiguous abbreviation of 'minim'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpih(x, scalest="minim")
    py_abbr = r2py_kernsmooth.dpih(x, scalest="m")
    assert py_full == pytest.approx(py_abbr, rel=1e-10), (
        "scalest='m' should resolve to 'minim'"
    )


def test_dpih_partial_name_scalest_st():
    """dpih accepts 'st' as an unambiguous abbreviation of 'stdev'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpih(x, scalest="stdev")
    py_abbr = r2py_kernsmooth.dpih(x, scalest="st")
    assert py_full == pytest.approx(py_abbr, rel=1e-10), (
        "scalest='st' should resolve to 'stdev'"
    )


def test_dpih_partial_name_scalest_i():
    """dpih accepts 'i' as an unambiguous abbreviation of 'iqr'."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_full = r2py_kernsmooth.dpih(x, scalest="iqr")
    py_abbr = r2py_kernsmooth.dpih(x, scalest="i")
    assert py_full == pytest.approx(py_abbr, rel=1e-10), (
        "scalest='i' should resolve to 'iqr'"
    )


def test_dpih_n2_matches_r():
    """dpih with n=2 observations returns a positive scalar matching R."""
    x = np.array([1.0, 2.0])
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="n=2")


def test_dpih_integer_input_coerced_to_float():
    """dpih accepts an integer-typed array and matches the float version."""
    rng = np.random.default_rng(42)
    x_int = np.arange(1, 101, dtype=np.int32)
    x_float = x_int.astype(np.float64)
    py_int = r2py_kernsmooth.dpih(x_int)
    py_float = r2py_kernsmooth.dpih(x_float)
    assert py_int == pytest.approx(py_float, rel=1e-10), (
        "Integer input should give the same result as float input"
    )


def test_dpih_level_0_matches_analytical_formula():
    """level=0 result equals scalest_val * (24*sqrt(pi)/n)^(1/3) analytically."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    py_val = r2py_kernsmooth.dpih(x, level=0)
    std_val = np.std(x, ddof=1)
    iqr_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    scale_val = min(std_val, iqr_val)
    n = len(x)
    expected = scale_val * (24 * np.sqrt(np.pi) / n) ** (1.0 / 3.0)
    np.testing.assert_allclose(
        py_val,
        expected,
        rtol=1e-10,
        err_msg="level=0 does not match the normal-scale formula",
    )


def test_dpih_data_with_outliers_matches_r():
    """dpih on data containing outliers matches R."""
    rng = np.random.default_rng(99)
    x = np.concatenate([rng.normal(0, 1, 95), np.array([100.0, 200.0, 300.0, -100.0, -200.0])])
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    _assert_close(py_val, r_val, label="data with outliers")


def test_dpih_all_levels_all_scalest_match_r():
    """dpih matches R for every (level, scalest) combination in a 5x3 grid."""
    rng = np.random.default_rng(77)
    x = rng.normal(0, 1, 200)
    for level in range(6):
        for scalest in ("minim", "stdev", "iqr"):
            r_val = _r_dpih(x, level=level, scalest=scalest)
            py_val = r2py_kernsmooth.dpih(x, level=level, scalest=scalest)
            _assert_close(
                py_val,
                r_val,
                label=f"level={level} scalest={scalest}",
            )
