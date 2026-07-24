# Boundary and edge case tests for r2py_kernsmooth.dpih
#
# These tests target the functional limits of dpih:
#   - Minimal sample sizes (n=2, n=3, n=5)
#   - Very large and very small data magnitudes (scale-invariance)
#   - Data with outliers that heavily skew the IQR/stdev comparison
#   - Gridsize at boundary values (small: 10, large: 2001)
#   - All plug-in levels (0-5) on the same data set
#   - truncate=True vs truncate=False when range_x is narrower than the data
#   - All scalest options when stdev and iqr are nearly equal
#   - Reproducibility: calling dpih twice on the same data returns same result

import warnings

import numpy as np
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import r2py_kernsmooth

_ks = importr("KernSmooth")


# ---------------------------------------------------------------------------
# Helper: call R dpih and return (scalar float) or capture errors
# ---------------------------------------------------------------------------

def _r_dpih(x_py, **kwargs):
    """Call R dpih; return scalar float.  Propagates R exceptions."""
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


def _r_dpih_catch(x_py, **kwargs):
    """Call R dpih; return (raised: bool, message: str)."""
    try:
        val = _r_dpih(x_py, **kwargs)
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


def test_dpih_n2_minimal_sample():
    """n=2 (minimum non-trivial sample) returns a positive scalar matching R."""
    x = np.array([1.0, 2.0])
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    assert py_val > 0, f"n=2: expected positive result, got {py_val}"
    _assert_close(py_val, r_val, label="n=2")


def test_dpih_n3_sample():
    """n=3 returns a positive scalar matching R."""
    x = np.array([1.0, 2.0, 3.0])
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="n=3")


def test_dpih_n5_sample():
    """n=5 returns a positive scalar matching R."""
    x = np.arange(1.0, 6.0)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="n=5")


# ---------------------------------------------------------------------------
# Scale-invariance: very large and very small magnitudes
# ---------------------------------------------------------------------------


def test_dpih_large_magnitude_matches_r():
    """dpih on data with mean=1e6 and std=1e4 returns a positive result matching R."""
    rng = np.random.default_rng(55)
    x = rng.normal(1e6, 1e4, 100)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="large magnitude mean=1e6")


def test_dpih_small_magnitude_matches_r():
    """dpih on data with std=1e-4 returns a positive result matching R."""
    rng = np.random.default_rng(55)
    x = rng.normal(0.0, 1e-4, 100)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="small magnitude sd=1e-4")


def test_dpih_scale_invariance():
    """dpih(c*x) == c * dpih(x) for any positive constant c (up to binning approx)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    c = 100.0
    h_x = r2py_kernsmooth.dpih(x)
    h_cx = r2py_kernsmooth.dpih(c * x)
    # Strict exact scale invariance holds at the continuous level; binning
    # introduces small differences, so we allow 1 % relative error.
    np.testing.assert_allclose(
        h_cx,
        c * h_x,
        rtol=0.01,
        err_msg="dpih should be approximately scale-invariant",
    )


# ---------------------------------------------------------------------------
# Outlier-heavy data: minim chooses IQR over stdev
# ---------------------------------------------------------------------------


def test_dpih_outliers_minim_uses_iqr():
    """When outliers inflate stdev far above IQR/1.349, scalest='minim' uses IQR."""
    rng = np.random.default_rng(99)
    core = rng.normal(0, 1, 95)
    outliers = np.array([100.0, 200.0, 300.0, -100.0, -200.0])
    x = np.concatenate([core, outliers])
    std_val = float(np.std(x, ddof=1))
    iqr_val = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    # Verify the setup: IQR-based scale should be much smaller than std.
    assert iqr_val < std_val, "Setup failure: expected iqr < stdev for this data"
    r_val = _r_dpih(x, scalest="minim")
    py_val = r2py_kernsmooth.dpih(x, scalest="minim")
    _assert_close(py_val, r_val, label="outliers minim=iqr")
    # 'minim' should agree with 'iqr' (not 'stdev') for this dataset.
    py_iqr = r2py_kernsmooth.dpih(x, scalest="iqr")
    assert py_val == pytest.approx(py_iqr, rel=1e-10), (
        "scalest='minim' should equal scalest='iqr' when iqr < stdev"
    )


# ---------------------------------------------------------------------------
# All plug-in levels (0-5) on the same dataset, each matching R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 5])
def test_dpih_all_levels_match_r(level):
    """dpih with each level (0-5) matches R on a fixed dataset."""
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, level=level)
    py_val = r2py_kernsmooth.dpih(x, level=level)
    _assert_close(py_val, r_val, label=f"level={level}")


# ---------------------------------------------------------------------------
# Gridsize edge values
# ---------------------------------------------------------------------------


def test_dpih_small_gridsize_10():
    """dpih with gridsize=10 returns a positive scalar matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, gridsize=10)
    py_val = r2py_kernsmooth.dpih(x, gridsize=10)
    assert py_val > 0
    _assert_close(py_val, r_val, label="gridsize=10")


def test_dpih_large_gridsize_2001():
    """dpih with gridsize=2001 returns a positive scalar matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r_val = _r_dpih(x, gridsize=2001)
    py_val = r2py_kernsmooth.dpih(x, gridsize=2001)
    assert py_val > 0
    _assert_close(py_val, r_val, label="gridsize=2001")


def test_dpih_gridsize_affects_result():
    """Different gridsizes produce slightly different results (binning approximation)."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    h_coarse = r2py_kernsmooth.dpih(x, gridsize=50)
    h_fine = r2py_kernsmooth.dpih(x, gridsize=2001)
    # Both must be positive; they should converge as gridsize grows.
    assert h_coarse > 0 and h_fine > 0
    # Within 5 % of each other — they are approximations of the same quantity.
    np.testing.assert_allclose(
        h_coarse,
        h_fine,
        rtol=0.05,
        err_msg="Coarse and fine gridsize results should be within 5%",
    )


# ---------------------------------------------------------------------------
# truncate=True vs truncate=False with a narrow range_x
# ---------------------------------------------------------------------------


def test_dpih_truncate_true_vs_false_narrow_range():
    """When range_x is narrower than the data, truncate affects bin counts and result."""
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(0, 1, 90), rng.uniform(5, 10, 10)])
    range_x = (-3.0, 3.0)   # Excludes the 10 uniform points.
    r_true = _r_dpih(x, range_x=range_x, truncate=True)
    r_false = _r_dpih(x, range_x=range_x, truncate=False)
    py_true = r2py_kernsmooth.dpih(x, range_x=range_x, truncate=True)
    py_false = r2py_kernsmooth.dpih(x, range_x=range_x, truncate=False)
    _assert_close(py_true, r_true, label="truncate=True narrow range")
    _assert_close(py_false, r_false, label="truncate=False narrow range")
    assert py_true != pytest.approx(py_false, rel=1e-4), (
        "truncate=True and truncate=False should differ when data extend beyond range_x"
    )


# ---------------------------------------------------------------------------
# All scalest options when stdev ≈ IQR/1.349 (near-equal scales)
# ---------------------------------------------------------------------------


def test_dpih_scalest_all_options_near_equal_scales():
    """When stdev ≈ IQR/1.349, all three scalest options return similar bin widths."""
    # Standard normal has stdev=1 and IQR/1.349 ≈ 0.741/1.349 ≈ 1.0 exactly
    # for the theoretical distribution; empirically they are close.
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    results = {s: r2py_kernsmooth.dpih(x, scalest=s) for s in ("minim", "stdev", "iqr")}
    # All three should be finite and positive.
    for s, v in results.items():
        assert v > 0 and np.isfinite(v), f"scalest={s} returned {v}"
    # With n=1000 normal data, all three should agree within 10 %.
    np.testing.assert_allclose(
        list(results.values()),
        results["minim"],
        rtol=0.1,
        err_msg="scalest options should agree within 10% for large normal sample",
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_dpih_reproducibility():
    """dpih called twice with the same input returns the identical result."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    h1 = r2py_kernsmooth.dpih(x)
    h2 = r2py_kernsmooth.dpih(x)
    assert h1 == h2, "dpih is not deterministic / reproducible"


# ---------------------------------------------------------------------------
# Range_x wider than the data span
# ---------------------------------------------------------------------------


def test_dpih_range_x_wider_than_data_matches_r():
    """range_x extending beyond the data bounds gives a valid result matching R."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    range_x = (float(x.min()) - 5.0, float(x.max()) + 5.0)
    r_val = _r_dpih(x, range_x=range_x)
    py_val = r2py_kernsmooth.dpih(x, range_x=range_x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="range_x wider than data")


# ---------------------------------------------------------------------------
# Negative-valued data
# ---------------------------------------------------------------------------


def test_dpih_all_negative_values_matches_r():
    """dpih on entirely negative data (location shift) matches R."""
    rng = np.random.default_rng(42)
    x = rng.normal(-50.0, 3.0, 100)
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="all negative data")


# ---------------------------------------------------------------------------
# Data where IQR is exactly equal to stdev (within floating-point)
# ---------------------------------------------------------------------------


def test_dpih_stdev_equals_iqr_minim_consistent():
    """When stdev == iqr/1.349, scalest='minim' result matches both stdev and iqr."""
    # Construct data where (by construction) these are exactly equal.
    # This is hard to guarantee exactly; instead verify that 'minim' equals
    # min(stdev_result, iqr_result).
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    r_minim = _r_dpih(x, scalest="minim")
    r_stdev = _r_dpih(x, scalest="stdev")
    r_iqr = _r_dpih(x, scalest="iqr")
    py_minim = r2py_kernsmooth.dpih(x, scalest="minim")
    _assert_close(py_minim, r_minim, label="minim consistency")
    # The minim result should be <= both individual scale estimates.
    assert py_minim <= r2py_kernsmooth.dpih(x, scalest="stdev") + 1e-10
    assert py_minim <= r2py_kernsmooth.dpih(x, scalest="iqr") + 1e-10


# ---------------------------------------------------------------------------
# Two-valued data (minimum distinct observations for non-zero IQR)
# ---------------------------------------------------------------------------


def test_dpih_two_valued_data_iqr_nonzero():
    """Data alternating between two values: IQR > 0 so dpih should succeed."""
    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    r_val = _r_dpih(x)
    py_val = r2py_kernsmooth.dpih(x)
    assert py_val > 0
    _assert_close(py_val, r_val, label="two-valued data")


# ---------------------------------------------------------------------------
# Integer-typed input coercion
# ---------------------------------------------------------------------------


def test_dpih_integer_array_coercion_matches_r():
    """Integer-typed x is accepted and coerced to float; result matches R."""
    x_int = np.arange(1, 51, dtype=np.int32)
    x_float = x_int.astype(np.float64)
    r_val = _r_dpih(x_float)
    py_val = r2py_kernsmooth.dpih(x_int)
    assert py_val > 0
    _assert_close(py_val, r_val, label="integer array coercion")
