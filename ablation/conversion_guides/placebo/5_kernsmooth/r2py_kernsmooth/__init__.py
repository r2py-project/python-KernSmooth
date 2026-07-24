import math
import warnings
from typing import Any

import numpy as np

__all__ = [
    "bkde",
    "bkde2D",
    "bkfe",
    "blkest",
    "cpblock",
    "dpih",
    "dpik",
    "dpill",
    "linbin",
    "linbin2D",
    "locpoly",
    "rlbin",
    "sdiag",
    "sstdiag",
]


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Equivalent of .Fortran(F_linbin, as.double(X), as.integer(n),
    #                        as.double(a), as.double(b), as.integer(M),
    #                        as.integer(trun), double(M))[[7]]
    gcnts = np.zeros(M, dtype=np.float64)

    if n == 0:
        return gcnts

    delta = (b - a) / (M - 1)
    lxi = ((X - a) / delta) + 1.0

    # Fortran's int() truncates toward zero, matching np.trunc here.
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    in_range = (li >= 1) & (li < M)
    li_in = li[in_range]
    rem_in = rem[in_range]

    # Convert 1-based Fortran indices to 0-based Python indices.
    idx_lower = li_in - 1
    idx_upper = li_in

    np.add.at(gcnts, idx_lower, 1.0 - rem_in)
    np.add.at(gcnts, idx_upper, rem_in)

    if trun == 0:
        n_low = int(np.sum(li < 1))
        if n_low > 0:
            gcnts[0] += n_low

        n_high = int(np.sum(li >= M))
        if n_high > 0:
            gcnts[M - 1] += n_high

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)

    n = X.shape[0]
    x1 = X[:, 0]
    x2 = X[:, 1]

    M1 = gpoints1.shape[0]
    M2 = gpoints2.shape[0]
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Equivalent of .Fortran(F_lbtwod, as.double(X), as.integer(n),
    #                        as.double(a1), as.double(a2), as.double(b1), as.double(b2),
    #                        as.integer(M1), as.integer(M2), double(M1*M2))[[9L]]
    # reshaped as matrix(out[[9L]], M1, M2). Since we build the (M1, M2)
    # array directly with the same (row=grid1 index, col=grid2 index)
    # layout that the Fortran routine's Cnts(M1,M2) array uses, no extra
    # Fortran-order reshape is needed here.
    cnts = np.zeros((M1, M2), dtype=np.float64)

    if n == 0:
        return cnts

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    lxi = ((x1 - a1) / delta1) + 1.0
    lyi = ((x2 - a2) / delta2) + 1.0

    # Fortran's int() truncates toward zero, matching np.trunc here.
    li = np.trunc(lxi).astype(np.int64)
    lj = np.trunc(lyi).astype(np.int64)
    remx = lxi - li
    remy = lyi - lj

    in_range = (li >= 1) & (li < M1) & (lj >= 1) & (lj < M2)
    li_in = li[in_range]
    lj_in = lj[in_range]
    remx_in = remx[in_range]
    remy_in = remy[in_range]

    # Convert 1-based Fortran indices to 0-based Python indices.
    idx_li = li_in - 1
    idx_lj = lj_in - 1

    np.add.at(cnts, (idx_li, idx_lj), (1.0 - remx_in) * (1.0 - remy_in))
    np.add.at(cnts, (idx_li + 1, idx_lj), remx_in * (1.0 - remy_in))
    np.add.at(cnts, (idx_li, idx_lj + 1), (1.0 - remx_in) * remy_in)
    np.add.at(cnts, (idx_li + 1, idx_lj + 1), remx_in * remy_in)

    return cnts


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Equivalent of .Fortran(F_rlbin, as.double(X), as.double(Y), as.integer(n),
    #                        as.double(a), as.double(b), as.integer(M), as.integer(trun),
    #                        double(M), double(M))[[8L]] and [[9L]]
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)

    if n == 0:
        return {"xcounts": xcnts, "ycounts": ycnts}

    delta = (b - a) / (M - 1)
    lxi = ((X - a) / delta) + 1.0

    # Fortran's int() truncates toward zero, matching np.trunc here.
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Correction for right endpoint (not included if li == M in 1-based terms).
    at_b = X == b
    li = np.where(at_b, M - 1, li)
    rem = np.where(at_b, 1.0, rem)

    in_range = (li >= 1) & (li < M)
    li_in = li[in_range]
    rem_in = rem[in_range]
    y_in = Y[in_range]

    # Convert 1-based Fortran indices to 0-based Python indices.
    idx_lower = li_in - 1
    idx_upper = li_in

    np.add.at(xcnts, idx_lower, 1.0 - rem_in)
    np.add.at(xcnts, idx_upper, rem_in)
    np.add.at(ycnts, idx_lower, (1.0 - rem_in) * y_in)
    np.add.at(ycnts, idx_upper, rem_in * y_in)

    if trun == 0:
        low_mask = li < 1
        n_low = int(np.sum(low_mask))
        if n_low > 0:
            xcnts[0] += n_low
            ycnts[0] += np.sum(Y[low_mask])

        high_mask = li >= M
        n_high = int(np.sum(high_mask))
        if n_high > 0:
            xcnts[M - 1] += n_high
            ycnts[M - 1] += np.sum(Y[high_mask])

    return {"xcounts": xcnts, "ycounts": ycnts}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    # (R's `missing(bandwidth)` is modeled here by the `bandwidth is None`
    # sentinel; a truly required bandwidth that is never supplied will
    # simply fail later, exactly as in R, when it is first used as `h`.)
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename variables
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])
    h = bandwidth

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, gridsize)
        gcounts = linbin(x_arr, gpoints, truncate)
    else:
        gcounts = x_arr
        M = gcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width
    n = np.sum(gcounts)
    delta = (b - a) / (M - 1)

    # Obtain kernel weights
    tau = 4 + drv
    L = min(int(np.floor(tau * h / delta)), M)

    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    # dnorm(arg): standard normal density, closed form.
    kappam = (np.exp(-arg ** 2 / 2.0) / np.sqrt(2.0 * np.pi)) / (h ** (drv + 1))
    hmold0 = 1.0
    hmold1 = arg
    hmnew = 1.0
    # NOTE: the `for (i in (2L:drv))` loop in R is only ever entered when
    # `drv >= 2L` (it is wrapped in `if (drv >= 2L)`), so the descending-range
    # quirk of R's `:` operator for `drv < 2` never actually triggers: for
    # drv == 0 and drv == 1 the loop body never runs and `hmnew` simply keeps
    # its initial value of 1. Replicate that faithfully here (i.e. do NOT
    # execute the recurrence at all when drv < 2).
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1  # Compute mth degree Hermite polynomial
            hmold1 = hmnew   # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    # R's `fft(x, inverse = TRUE)` is the *unnormalized* inverse FFT, i.e.
    # `P * np.fft.ifft(x)`; dividing by `P` below then reproduces exactly
    # what `Re(fft(kappam*Gcounts, TRUE))/P` computes in R.
    conv = np.fft.ifft(kappam * Gcounts) * P
    estimate = (np.real(conv) / P)[0:M]

    return float(np.sum(gcounts * estimate) / (n ** 2))


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    # (R's `missing(bandwidth)` is modeled here by the `bandwidth is None`
    # sentinel.)
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'arg' should be one of " + ", ".join(repr(k) for k in valid_kernels))

    x_arr = np.asarray(x, dtype=np.float64)

    # Rename common variables
    n = x_arr.shape[0]
    M = gridsize

    # Set canonical scaling factors
    if kernel == "normal":
        del0 = (1.0 / (4.0 * np.pi)) ** (1.0 / 10.0)
    elif kernel == "box":
        del0 = (9.0 / 2.0) ** (1.0 / 5.0)
    elif kernel == "epanech":
        del0 = 15.0 ** (1.0 / 5.0)
    elif kernel == "biweight":
        del0 = 35.0 ** (1.0 / 5.0)
    else:  # triweight
        del0 = (9450.0 / 143.0) ** (1.0 / 5.0)

    if not isinstance(canonical, bool):
        raise TypeError("'canonical' must be a length-1 logical vector")

    # Set default bandwidth
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * math.sqrt(float(np.var(x_arr, ddof=1)))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values
    tau = 4 if kernel == "normal" else 1

    if range_x is None:
        range_x = (float(np.min(x_arr)) - tau * h, float(np.max(x_arr)) + tau * h)
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    # Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)

    def _dbeta(y: np.ndarray[Any, np.dtype[np.float64]], shape1: int, shape2: int) -> np.ndarray[Any, np.dtype[np.float64]]:
        # Closed-form Beta(shape1, shape2) density for the small positive
        # integer shape parameters used by this function (avoids a scipy
        # dependency, matching bkfe's use of a closed-form dnorm).
        beta_const = (
            math.factorial(shape1 - 1)
            * math.factorial(shape2 - 1)
            / math.factorial(shape1 + shape2 - 1)
        )
        return (y ** (shape1 - 1)) * ((1.0 - y) ** (shape2 - 1)) / beta_const

    if kernel == "normal":
        arg = lvec * delta
        # dnorm(arg): standard normal density, closed form.
        kappa = (np.exp(-arg ** 2 / 2.0) / np.sqrt(2.0 * np.pi)) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * _dbeta(0.5 * (lvec * delta + 1.0), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * _dbeta(0.5 * (lvec * delta + 1.0), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * _dbeta(0.5 * (lvec * delta + 1.0), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * _dbeta(0.5 * (lvec * delta + 1.0), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    # R's `fft(x, inverse = TRUE)` is the *unnormalized* inverse FFT, i.e.
    # `P * np.fft.ifft(x)`; dividing by `P` below then reproduces exactly
    # what `Re(fft(kappa*gcounts, TRUE))/P` computes in R.
    conv = np.fft.ifft(kappa * gcounts) * P
    estimate = (np.real(conv) / P)[0:M]

    return {"x": gpoints, "y": estimate}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: tuple[tuple[float, float], tuple[float, float]] | list[tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    # (R's `missing(bandwidth)` is modeled here by the `bandwidth is None`
    # sentinel; `bandwidth` has no default in R either, so a call that never
    # supplies it will simply fail later, exactly as in R, the first time it
    # is coerced to an array below.)
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x_arr = np.asarray(x, dtype=np.float64)

    # Rename common variables
    n = x_arr.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([float(h[0]), float(h[0])], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x_list: list[tuple[float, float]] = [(0.0, 0.0), (0.0, 0.0)]
        for idx in range(2):
            range_x_list[idx] = (
                float(np.min(x_arr[:, idx])) - 1.5 * h[idx],
                float(np.max(x_arr[:, idx])) + 1.5 * h[idx],
            )
        range_x = range_x_list

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x_arr, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.empty(0), np.empty(0)]
    for idx in range(2):
        L[idx] = min(
            int(np.floor(tau * h[idx] * (int(M[idx]) - 1) / (b[idx] - a[idx]))),
            int(M[idx]) - 1,
        )
        lvecid = np.arange(0, int(L[idx]) + 1, dtype=np.float64)
        facid = (b[idx] - a[idx]) / (h[idx] * (int(M[idx]) - 1))
        # dnorm(lvecid*facid): standard normal density, closed form.
        z = (np.exp(-((lvecid * facid) ** 2) / 2.0) / np.sqrt(2.0 * np.pi)) / h[idx]
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * facid * h[idx]
        kapid[idx] = z / tot

    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    # smallest powers of 2 >= M+L
    P = (2.0 ** np.ceil(np.log(M.astype(np.float64) + L.astype(np.float64)) / np.log(2.0))).astype(np.int64)
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0:L1 + 1, 0:L2 + 1] = kapp
    if L1:
        rp[P1 - L1:P1, 0:L2 + 1] = kapp[1:L1 + 1, 0:L2 + 1][::-1, :]
    if L2:
        rp[:, P2 - L2:P2] = rp[:, 1:L2 + 1][:, ::-1]
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)

    # R's `fft(z, inverse = TRUE)` is the *unnormalized* inverse FFT, i.e.
    # `(P1*P2) * np.fft.ifft2(z)`; dividing by `(P1*P2)` below then reproduces
    # exactly what `Re(fft(rp*sp, inverse = TRUE)/(P1*P2))` computes in R.
    conv = np.fft.ifft2(rp_fft * sp_fft) * (P1 * P2)
    rp_out = (np.real(conv) / (P1 * P2))[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp_out = np.maximum(rp_out, 0.0)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp_out}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # Chooses the number of blocks for the preliminary step of a plug-in
    # rule using Mallows' C_p.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]

    # Sort the (X, Y) data with respect to the X's.
    order_idx = np.argsort(X, kind="stable")
    X = X[order_idx]
    Y = Y[order_idx]

    qq = q + 1

    # Equivalent of .Fortran(F_cp, as.double(X), as.double(Y), as.integer(n),
    #                        as.integer(qq), as.integer(Nmax), as.double(RSS),
    #                        as.double(Xj), as.double(Yj), as.double(coef),
    #                        as.double(Xmat), as.double(wk), as.double(qraux),
    #                        Cpvals = as.double(Cpvals))
    # reimplemented directly in NumPy since the compiled Fortran routine is
    # unavailable; it is assumed that (X, Y) are sorted with respect to X,
    # as in cp.f. For each candidate number of blocks Nval = 1..Nmax, the
    # data are partitioned into Nval contiguous blocks (the last block
    # absorbing any remainder), a q'th degree polynomial is fit within
    # each block by least squares, and the residual sums of squares over
    # all blocks are accumulated into RSS[Nval].
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        idiv = n // Nval
        for j in range(1, Nval + 1):
            # For each member of the partition (1-based Fortran block bounds).
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n

            # Convert to 0-based slice of the current block.
            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]

            # Obtain a q'th degree fit over the current member of the partition.
            # Set up the design ("X") matrix: Xmat[i, k] = Xj[i] ** k, k = 0..q.
            Xmat = np.vander(Xj, N=qq, increasing=True)

            # QR-based least squares fit (equivalent to dqrdc/dqrsl).
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fiti = Xmat @ coef
            RSSj = float(np.sum((Yj - fiti) ** 2))
            RSS[Nval - 1] += RSSj

    # Now compute array of Mallow's C_p values.
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = (
            (n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1] + 2 * qq * i - n
        )

    # order(Cpvec)[1L] in R returns the 1-based index of the (first)
    # minimum Cp value; np.argmin similarly returns the first occurrence
    # of the minimum, so adding 1 preserves R's 1-based return semantics.
    # NOTE: this function returns a 1-BASED index (a valid block count
    # from 1..Nmax), matching R's contract so callers such as dpill can
    # pass the result directly as blkest's Nval parameter.
    return int(np.argmin(Cpvals)) + 1


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]

    # Sort the (x, y) data with respect to the x's.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    qq = q + 1

    # Equivalent of .Fortran(F_blkest, as.double(x), as.double(y),
    #                        as.integer(n), as.integer(q), as.integer(qq),
    #                        as.integer(Nval), as.double(xj), as.double(yj),
    #                        as.double(coef), as.double(Xmat), as.double(wk),
    #                        as.double(qraux), as.double(sigsqe),
    #                        as.double(th22e), as.double(th24e))[13:15]
    # reimplemented directly in NumPy since the compiled Fortran routine
    # is unavailable; it is assumed that (x, y) are sorted with respect
    # to x, as in blkest.f.
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    idiv = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition (1-based Fortran block bounds).
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        nj = iupp - ilow + 1

        # Convert to 0-based slice of the current block.
        Xj = x[ilow - 1:iupp]
        Yj = y[ilow - 1:iupp]

        # Obtain a q'th degree fit over the current member of the partition.
        # Set up the design ("X") matrix: Xmat[i, k] = Xj[i] ** k, k = 0..q.
        Xmat = np.vander(Xj, N=qq, increasing=True)

        # QR-based least squares fit (equivalent to dqrdc/dqrsl).
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        fiti = Xmat @ coef
        RSS += float(np.sum((Yj - fiti) ** 2))

        ddm = np.full(nj, 2.0 * coef[2], dtype=np.float64)
        ddddm = np.full(nj, 24.0 * coef[4], dtype=np.float64)
        for k in range(2, qq + 1):
            if k <= q - 1:
                ddm = ddm + k * (k + 1) * coef[k + 1] * Xj ** (k - 1)
            if k <= q - 3:
                ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef[k + 3] * Xj ** (k - 1)

        th22e += float(np.sum(ddm ** 2))
        th24e += float(np.sum(ddm * ddddm))

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": sigsqe, "th22e": th22e, "th24e": th24e}


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    # (R's `missing(bandwidth)` is modeled here by the `bandwidth is None`
    # sentinel; `bandwidth` has no default in R either, so a call that never
    # supplies it will simply fail later, exactly as in R, the first time it
    # is coerced to an array below.)
    if bandwidth is not None:
        bandwidth_check = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
        if np.any(bandwidth_check <= 0):
            raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = None if y is None else np.asarray(y, dtype=np.float64)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x_arr) - np.min(x_arr))
            range_x = (float(np.min(x_arr) - extra), float(np.max(x_arr) + extra))
        else:
            range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Decide whether a density estimate or regression estimate is required.
    if y is None:  # obtain density estimate
        n = x_arr.shape[0]
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        # Bin the data if not already binned
        if not binned:
            gpoints = np.linspace(a, b, M)
            out = rlbin(x_arr, y_arr, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = x_arr
            ycounts = y_arr
            M = xcounts.shape[0]
            gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth" (1-based, matching R's "indic").
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * float(bandwidth_arr[0]) / delta), dtype=np.int64)
        hdisc = np.full(Q, float(bandwidth_arr[0]), dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Convert "indic" to 0-based bandwidth-level indices for use below.
    indic0 = indic - 1

    # Reimplementation of Fortran routine "locpol": for every discretised
    # bandwidth level i, accumulate weighted moment sums "ss" (from the
    # xcounts weights, ippp columns) and "tt" (from the ycounts weights,
    # pp columns) at each grid point j that is assigned to level i and
    # falls within that level's window [j - Lvec[i], j + Lvec[i]].
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)

    for k in range(M):
        if xcounts[k] == 0:
            continue
        for i in range(Q):
            L = int(Lvec[i])
            h = hdisc[i]
            lo = max(0, k - L)
            hi = min(M - 1, k + L)
            for j in range(lo, hi + 1):
                if indic0[j] != i:
                    continue
                offset = k - j
                w = math.exp(-((delta * offset / h) ** 2) / 2.0)
                fac = 1.0
                ss[j, 0] += xcounts[k] * w
                tt[j, 0] += ycounts[k] * w
                for ii in range(2, ppp + 1):
                    fac *= delta * offset
                    ss[j, ii - 1] += xcounts[k] * w * fac
                    if ii <= pp:
                        tt[j, ii - 1] += ycounts[k] * w * fac

    # For every grid point, build the pp x pp moment matrix "Smat" and the
    # moment vector "Tvec" from "ss"/"tt", then solve Smat @ coef = Tvec
    # (originally via LINPACK's dgefa/dgesl; np.linalg.solve is equivalent).
    curvest = np.zeros(M, dtype=np.float64)
    row_idx = np.arange(pp)
    col_idx = row_idx[:, None] + row_idx[None, :]
    for k in range(M):
        Smat = ss[k][col_idx]
        Tvec = tt[k, :pp]
        coef = np.linalg.solve(Smat, Tvec)
        curvest[k] = coef[drv]

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Note: `drv` is accepted for interface parity with the R function, but
    # (exactly as in the original R/Fortran source) it is never referenced in
    # the computation below -- "sdiag" always extracts the leading (0, 0)
    # entry of the inverted moment matrix at each grid point.
    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    degree = int(degree)
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
    else:
        xcounts = x_arr
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth" (1-based, matching R's "indic").
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * float(bandwidth_arr[0]) / delta), dtype=np.int64)
        hdisc = np.full(Q, float(bandwidth_arr[0]), dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Convert "indic" to 0-based bandwidth-level indices for use below.
    indic0 = indic - 1

    # Reimplementation of Fortran routine "sdiag": for every discretised
    # bandwidth level i, accumulate weighted moment sums "ss" (from the
    # xcounts weights, ippp columns) at each grid point j that is assigned
    # to level i and falls within that level's window
    # [j - Lvec[i], j + Lvec[i]]. Unlike "locpoly", there is no ycounts /
    # "tt" accumulation here since sdiag only needs the moment matrix built
    # from the bin counts.
    ss = np.zeros((M, ppp), dtype=np.float64)

    for k in range(M):
        if xcounts[k] == 0:
            continue
        for i in range(Q):
            L = int(Lvec[i])
            h = hdisc[i]
            lo = max(0, k - L)
            hi = min(M - 1, k + L)
            for j in range(lo, hi + 1):
                if indic0[j] != i:
                    continue
                offset = k - j
                w = math.exp(-((delta * offset / h) ** 2) / 2.0)
                fac = 1.0
                ss[j, 0] += xcounts[k] * w
                for ii in range(2, ppp + 1):
                    fac *= delta * offset
                    ss[j, ii - 1] += xcounts[k] * w * fac

    # For every grid point, build the pp x pp moment matrix "Smat" from
    # "ss", invert it (originally via LINPACK's dgefa/dgedi with job=01,
    # i.e. compute the matrix inverse without the determinant;
    # np.linalg.inv is equivalent), and take its (1, 1) entry (Smat(1,1)
    # in R's 1-based indexing, i.e. index [0, 0] here) as the diagonal
    # smoother-matrix value at that grid point.
    Sdg = np.zeros(M, dtype=np.float64)
    row_idx = np.arange(pp)
    col_idx = row_idx[:, None] + row_idx[None, :]
    for k in range(M):
        Smat = ss[k][col_idx]
        Sinv = np.linalg.inv(Smat)
        Sdg[k] = Sinv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Note: `drv` is accepted for interface parity with the R function, but
    # (exactly as in the original R/Fortran source) it is never referenced in
    # the computation below -- "sstdiag" always uses row/column 1 (index 0)
    # of the inverted moment matrix at each grid point when combining with
    # the squared-weight moment matrix "Umat".
    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    degree = int(degree)
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
    else:
        xcounts = x_arr
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth" (1-based, matching R's "indic").
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * float(bandwidth_arr[0]) / delta), dtype=np.int64)
        hdisc = np.full(Q, float(bandwidth_arr[0]), dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Convert "indic" to 0-based bandwidth-level indices for use below.
    indic0 = indic - 1

    # Reimplementation of Fortran routine "sstdg": for every discretised
    # bandwidth level i, accumulate weighted moment sums "ss" (from the
    # xcounts weights, ippp columns) and their squared-weight counterparts
    # "uu" at each grid point j that is assigned to level i and falls
    # within that level's window [j - Lvec[i], j + Lvec[i]].
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)

    for k in range(M):
        if xcounts[k] == 0:
            continue
        for i in range(Q):
            L = int(Lvec[i])
            h = hdisc[i]
            lo = max(0, k - L)
            hi = min(M - 1, k + L)
            for j in range(lo, hi + 1):
                if indic0[j] != i:
                    continue
                offset = k - j
                w = math.exp(-((delta * offset / h) ** 2) / 2.0)
                fac = 1.0
                ss[j, 0] += xcounts[k] * w
                uu[j, 0] += xcounts[k] * (w ** 2)
                for ii in range(2, ppp + 1):
                    fac *= delta * offset
                    ss[j, ii - 1] += xcounts[k] * w * fac
                    uu[j, ii - 1] += xcounts[k] * (w ** 2) * fac

    # For every grid point, build the pp x pp moment matrices "Smat" (from
    # "ss") and "Umat" (from "uu"), invert "Smat" (originally via
    # LINPACK's dgefa/dgedi with job=01, i.e. compute the matrix inverse
    # without the determinant; np.linalg.inv is equivalent), and combine
    # them as SSTd[k] = Smat^{-1}[0, :] @ Umat @ Smat^{-1}[:, 0], matching
    # the Fortran loop "SSTd(k) = SSTd(k) + Smat(1,i)*Umat(i,j)*Smat(j,1)"
    # (where Smat there already holds the inverted matrix in-place).
    SSTd = np.zeros(M, dtype=np.float64)
    row_idx = np.arange(pp)
    col_idx = row_idx[:, None] + row_idx[None, :]
    for k in range(M):
        Smat = ss[k][col_idx]
        Umat = uu[k][col_idx]
        Sinv = np.linalg.inv(Smat)
        SSTd[k] = Sinv[0, :] @ Umat @ Sinv[:, 0]

    return {"x": gpoints, "y": SSTd}


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x_arr = np.asarray(x, dtype=np.float64)

    ## Rename variables

    n = x_arr.shape[0]
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data
    ##
    ## NOTE: R's source computes `gpoints <- seq(a, b, length.out = M)` and
    ## `gcounts <- linbin(x, gpoints, truncate)` here, but both are
    ## immediately discarded/overwritten once the standardised-data
    ## `gpoints`/`gcounts` are computed a few lines below. That first pair
    ## of statements has no side effects, so it is genuinely dead code and
    ## is omitted here for a cleaner translation.

    ## Compute scale estimate

    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'arg' should be one of " + ", ".join(repr(s) for s in valid_scalest))

    if scalest == "stdev":
        scale = math.sqrt(float(np.var(x_arr, ddof=1)))
    elif scalest == "iqr":
        scale = (float(np.quantile(x_arr, 0.75)) - float(np.quantile(x_arr, 0.25))) / 1.349
    else:  # "minim"
        scale = min(
            (float(np.quantile(x_arr, 0.75)) - float(np.quantile(x_arr, 0.25))) / 1.349,
            math.sqrt(float(np.var(x_arr, ddof=1))),
        )

    if scale == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    xmean = float(np.mean(x_arr))
    sx = (x_arr - xmean) / scale
    sa = (a - xmean) / scale
    sb = (b - xmean) / scale

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps

    if level == 0:
        hpi = (24 * math.sqrt(math.pi) / n) ** (1 / 3)
    elif level == 1:
        alpha = (2 / (3 * n)) ** (1 / 5) * math.sqrt(2)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    elif level == 2:
        alpha = ((2 / (5 * n)) ** (1 / 7)) * math.sqrt(2)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2 / math.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    elif level == 3:
        alpha = ((2 / (7 * n)) ** (1 / 9)) * math.sqrt(2)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2 / math.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    elif level == 4:
        alpha = ((2 / (9 * n)) ** (1 / 11)) * math.sqrt(2)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * math.sqrt(2 / math.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2 / math.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    else:  # level == 5
        alpha = ((2 / (11 * n)) ** (1 / 13)) * math.sqrt(2)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * math.sqrt(2 / math.pi) / (psi10hat * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * math.sqrt(2 / math.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2 / math.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)

    return scale * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    valid_kernel = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernel:
        raise ValueError("'arg' should be one of " + ", ".join(repr(s) for s in valid_kernel))

    ## Set kernel constants

    if canonical:
        del0 = 1.0
    else:
        del0 = {
            "normal": 1 / ((4 * math.pi) ** (1 / 10)),
            "box": (9 / 2) ** (1 / 5),
            "epanech": 15 ** (1 / 5),
            "biweight": 35 ** (1 / 5),
            "triweight": (9450 / 143) ** (1 / 5),
        }[kernel]

    ## Rename variables

    x_arr = np.asarray(x, dtype=np.float64)
    n = x_arr.shape[0]
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data
    ##
    ## NOTE: R's source computes `gpoints <- seq(a, b, length.out = M)` and
    ## `gcounts <- linbin(x, gpoints, truncate)` here, but both are
    ## immediately discarded/overwritten once the standardised-data
    ## `gpoints`/`gcounts` are computed a few lines below. That first pair
    ## of statements has no side effects, so it is genuinely dead code and
    ## is omitted here for a cleaner translation.

    ## Compute scale estimate

    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'arg' should be one of " + ", ".join(repr(s) for s in valid_scalest))

    if scalest == "stdev":
        scale = math.sqrt(float(np.var(x_arr, ddof=1)))
    elif scalest == "iqr":
        scale = (float(np.quantile(x_arr, 0.75)) - float(np.quantile(x_arr, 0.25))) / 1.349
    else:  # "minim"
        scale = min(
            (float(np.quantile(x_arr, 0.75)) - float(np.quantile(x_arr, 0.25))) / 1.349,
            math.sqrt(float(np.var(x_arr, ddof=1))),
        )

    if scale == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    xmean = float(np.mean(x_arr))
    sx = (x_arr - xmean) / scale
    sa = (a - xmean) / scale
    sb = (b - xmean) / scale

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps:

    if level == 0:
        psi4hat = 3 / (8 * math.sqrt(math.pi))
    elif level == 1:
        alpha = (2 * (math.sqrt(2)) ** 7 / (5 * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2 * (math.sqrt(2)) ** 9 / (7 * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2 * (math.sqrt(2)) ** 11 / (9 * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * math.sqrt(2 / math.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2 * (math.sqrt(2)) ** 13 / (11 * n)) ** (1 / 13)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * math.sqrt(2 / math.pi) / (psi10hat * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * math.sqrt(2 / math.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:  # level == 5
        alpha = (2 * (math.sqrt(2)) ** 15 / (13 * n)) ** (1 / 15)  # bandwidth for psi_12
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = (945 * math.sqrt(2 / math.pi) / (psi12hat * n)) ** (1 / 13)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * math.sqrt(2 / math.pi) / (psi10hat * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * math.sqrt(2 / math.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return scale * del0 * (1 / (psi4hat * n)) ** (1 / 5)


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # cbind(x, y); xy[sort.list(xy[, 1]), ] -- sort both x and y jointly by x,
    # using a stable sort, matching the convention already established in
    # the converted "cpblock"/"blkest" (np.argsort(..., kind="stable")).
    order_idx = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order_idx]
    y_sorted = y_arr[order_idx]

    n_full = x_sorted.shape[0]
    indlow = math.floor(trim * n_full) + 1
    indupp = n_full - math.floor(trim * n_full)

    # R's x[indlow:indupp] is an inclusive 1-based range; convert to a
    # 0-based, end-inclusive Python slice as x_sorted[indlow-1:indupp].
    x = x_sorted[indlow - 1:indupp]
    y = y_sorted[indlow - 1:indupp]

    ## Rename common parameters
    n = x.shape[0]
    M = gridsize

    # NOTE on R's lazy evaluation of `range.x = range(x)`: default arguments
    # in R are evaluated in the function's own execution frame the first
    # time they are referenced, not at call time in the caller's frame.
    # Since the body reassigns `x <- xy[, 1L]` (the trimmed x) before
    # `range.x` is ever used, an unsupplied `range.x` resolves to
    # `range()` of the TRIMMED x, not the original untrimmed x. We
    # replicate that here by computing the default range from the
    # already-trimmed `x` below.
    if range_x is None:
        a = float(np.min(x))
        b = float(np.max(x))
    else:
        a = float(range_x[0])
        b = float(range_x[1])

    ## Bin the data
    gpoints = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    ## Choose the value of N using Mallow's C_p
    Nmax = max(min(math.floor(n / divisor), blockmax), 1)
    Nval = cpblock(x, y, Nmax, 4)

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.
    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    ## Estimate theta_22 using a local cubic fit
    ## with a "rule-of-thumb" bandwidth: "gamseh"
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1 / 7)

    mddest = locpoly(
        xcounts, ycounts, drv=2, bandwidth=gamseh,
        range_x=(a, b), binned=True,
    )["y"]

    llow = math.floor(proptrun * M) + 1
    lupp = M - math.floor(proptrun * M)
    # R's mddest[llow:lupp] is an inclusive 1-based range; convert to a
    # 0-based, end-inclusive Python slice as mddest[llow-1:lupp].
    th22kn = float(
        np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n
    )

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / math.sqrt(2 * math.pi)) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    ## Now compute a local linear kernel estimate of
    ## the variance.
    mest = locpoly(
        xcounts, ycounts, bandwidth=lamseh, range_x=(a, b), binned=True,
    )["y"]
    Sdg = sdiag(
        xcounts, bandwidth=lamseh, range_x=(a, b), binned=True,
    )["y"]
    SSTdg = sstdiag(
        xcounts, bandwidth=lamseh, range_x=(a, b), binned=True,
    )["y"]

    sigsqn = float(np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts))
    sigsqd = float(n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts))
    sigsqkn = sigsqn / sigsqd

    ## Combine to obtain final answer.
    return (sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1 / 5)


def on_attach(libname: str, pkgname: str) -> None:
    # Equivalent of R's packageStartupMessage(): emit the package startup
    # message. `libname` and `pkgname` are kept in the signature for
    # interface parity with the original R ".onAttach" hook, even though
    # they are unused in the body, matching the original R behavior.
    warnings.warn("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")


def on_unload(libpath: str) -> None:
    # Equivalent of R's ".onUnload" package-detach hook, which originally
    # called library.dynam.unload("KernSmooth", libpath) to unload the
    # compiled shared library backing the package. This Python port has no
    # compiled extension to unload (all Fortran routines have been
    # reimplemented in pure Python/NumPy), so this is a no-op that merely
    # preserves the original interface, accepting `libpath` for parity.
    pass
