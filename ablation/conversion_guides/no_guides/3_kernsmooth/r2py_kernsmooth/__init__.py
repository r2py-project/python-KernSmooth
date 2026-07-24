import math
import warnings
from typing import Any

import numpy as np
from scipy.stats import beta, norm

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

    # Equivalent of the Fortran subroutine linbin:
    # obtains bin counts for univariate data via the linear binning strategy.
    gcnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)

    lxi = ((X - a) / delta) + 1.0

    # Fortran's int() truncates toward zero, matching np.trunc here.
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    mask_mid = (li >= 1) & (li < M)
    li_mid = li[mask_mid]
    rem_mid = rem[mask_mid]
    # Convert 1-based Fortran indices (li, li+1) to 0-based Python indices.
    np.add.at(gcnts, li_mid - 1, 1.0 - rem_mid)
    np.add.at(gcnts, li_mid, rem_mid)

    if trun == 0:
        mask_low = li < 1
        gcnts[0] += np.count_nonzero(mask_low)

        mask_high = li >= M
        gcnts[M - 1] += np.count_nonzero(mask_high)

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)

    n = X.shape[0]
    X1 = X[:, 0]
    X2 = X[:, 1]

    M1 = gpoints1.shape[0]
    M2 = gpoints2.shape[0]
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Equivalent of the Fortran subroutine lbtwod:
    # obtains bin counts for bivariate data via the linear binning
    # strategy. Observations outside the mesh are ignored.
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    lxi1 = ((X1 - a1) / delta1) + 1.0
    lxi2 = ((X2 - a2) / delta2) + 1.0

    # Fortran's int() truncates toward zero, matching np.trunc here.
    li1 = np.trunc(lxi1).astype(np.int64)
    li2 = np.trunc(lxi2).astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    mask = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)

    li1_m = li1[mask]
    li2_m = li2[mask]
    rem1_m = rem1[mask]
    rem2_m = rem2[mask]

    # Convert 1-based Fortran indices to 0-based Python indices for the
    # four surrounding grid cell corners of each observation.
    i1 = li1_m - 1
    i2 = li2_m - 1

    np.add.at(gcnts, (i1, i2), (1.0 - rem1_m) * (1.0 - rem2_m))
    np.add.at(gcnts, (i1 + 1, i2), rem1_m * (1.0 - rem2_m))
    np.add.at(gcnts, (i1, i2 + 1), (1.0 - rem1_m) * rem2_m)
    np.add.at(gcnts, (i1 + 1, i2 + 1), rem1_m * rem2_m)

    return gcnts


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Equivalent of the Fortran subroutine rlbin:
    # obtains bin counts for a regression data set (X, Y) via the
    # linear binning strategy. If trun == 0, weight from end
    # observations is given to the corresponding end grid points;
    # if trun == 1, end observations are truncated.
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)

    lxi = ((X - a) / delta) + 1.0

    # Fortran's int() truncates toward zero, matching np.trunc here.
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Correction for right endpoint (not included if li == M).
    mask_right = (X == b)
    li = np.where(mask_right, M - 1, li)
    rem = np.where(mask_right, 1.0, rem)

    mask_mid = (li >= 1) & (li < M)
    li_mid = li[mask_mid]
    rem_mid = rem[mask_mid]
    y_mid = Y[mask_mid]
    # Convert 1-based Fortran indices (li, li+1) to 0-based Python indices.
    np.add.at(xcnts, li_mid - 1, 1.0 - rem_mid)
    np.add.at(xcnts, li_mid, rem_mid)
    np.add.at(ycnts, li_mid - 1, (1.0 - rem_mid) * y_mid)
    np.add.at(ycnts, li_mid, rem_mid * y_mid)

    if trun == 0:
        mask_low = li < 1
        xcnts[0] += np.count_nonzero(mask_low)
        ycnts[0] += np.sum(Y[mask_low])

        mask_high = li >= M
        xcnts[M - 1] += np.count_nonzero(mask_high)
        ycnts[M - 1] += np.sum(Y[mask_high])

    return {"xcounts": xcnts, "ycounts": ycnts}


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'kernel' should be one of " + ", ".join(valid_kernels))

    x = np.asarray(x, dtype=np.float64)

    # Rename common variables.
    n = x.shape[0]
    M = gridsize

    # Set canonical scaling factors.
    if kernel == "normal":
        del0 = (1 / (4 * np.pi)) ** (1 / 10)
    elif kernel == "box":
        del0 = (9 / 2) ** (1 / 5)
    elif kernel == "epanech":
        del0 = 15 ** (1 / 5)
    elif kernel == "biweight":
        del0 = 35 ** (1 / 5)
    else:  # triweight
        del0 = (9450 / 143) ** (1 / 5)

    if not isinstance(canonical, bool):
        raise ValueError("'canonical' must be a length-1 logical vector")

    # Set default bandwidth.
    if bandwidth is None:
        h = del0 * (243 / (35 * n)) ** (1 / 5) * np.sqrt(np.var(x, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values.
    tau = 4 if kernel == "normal" else 1

    if range_x is None:
        range_x = (float(np.min(x)) - tau * h, float(np.max(x)) + tau * h)
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data.
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute kernel weights.
    delta = (b - a) / (h * (M - 1))
    L = int(min(np.floor(tau / delta), M))
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    if kernel == "normal":
        kappa = norm.pdf(lvec * delta) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate.
    # We need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)

    # R's fft(z, inverse = TRUE) is the unnormalized inverse transform,
    # i.e. numpy.fft.ifft(z) * len(z); dividing that result by P
    # (== len(z)) is therefore mathematically equivalent to just taking
    # numpy.fft.ifft(z) directly, which is what is done here.
    est = np.real(np.fft.ifft(kappa_fft * gcounts_fft))[:M]

    return {"x": gpoints, "y": est}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | list[float] | None, gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: list[tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]]] | tuple[tuple[float, float], tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Equivalent of KernSmooth::bkde2D: a bivariate binned kernel density
    # estimator based on a bivariate normal kernel, evaluated efficiently
    # via the 2D FFT. Note that, exactly as in the original R
    # implementation, the "truncate" argument is accepted for interface
    # compatibility but is never actually referenced inside the body
    # (linbin2D is always called without it).
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None:
        bandwidth_check = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
        if np.min(bandwidth_check) <= 0:
            raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables.
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h_in = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction if only a single bandwidth is
    # given.
    if h_in.shape[0] == 1:
        h = np.array([h_in[0], h_in[0]], dtype=np.float64)
    else:
        h = h_in.astype(np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x_list = [None, None]
        for idd in range(2):
            range_x_list[idd] = (
                float(np.min(x[:, idd]) - 1.5 * h[idd]),
                float(np.max(x[:, idd]) + 1.5 * h[idd]),
            )
    else:
        range_x_list = range_x

    a = np.array([range_x_list[0][0], range_x_list[1][0]], dtype=np.float64)
    b = np.array([range_x_list[0][1], range_x_list[1][1]], dtype=np.float64)

    # Set up grid points and bin the data.
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights.
    L = np.zeros(2, dtype=np.int64)
    kapid = [None, None]
    for idd in range(2):
        L[idd] = int(min(
            np.floor(tau * h[idd] * (M[idd] - 1) / (b[idd] - a[idd])),
            M[idd] - 1,
        ))
        lvecid = np.arange(0, L[idd] + 1, dtype=np.float64)
        facid = (b[idd] - a[idd]) / (h[idd] * (M[idd] - 1))
        z = norm.pdf(lvecid * facid) / h[idd]
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * facid * h[idd]
        kapid[idd] = z / tot

    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate.
    P = (2 ** np.ceil(
        np.log(M.astype(np.float64) + L.astype(np.float64)) / np.log(2)
    )).astype(np.int64)  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0:L1 + 1, 0:L2 + 1] = kapp
    if L1:
        rp[P1 - L1:P1, 0:L2 + 1] = kapp[L1:0:-1, 0:L2 + 1]
    if L2:
        rp[:, P2 - L2:P2] = rp[:, L2:0:-1]
    # "rp" is now the wrap-around version of "kapp".

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # "sp" is the zero-padded version of "gcounts".

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s.
    sp_fft = np.fft.fft2(sp)

    # R's fft(z, inverse = TRUE) is the unnormalized inverse transform,
    # i.e. numpy.fft.ifft2(z) * z.size; dividing that result by (P1 * P2)
    # (== z.size) is therefore mathematically equivalent to just taking
    # numpy.fft.ifft2(z) directly, which is what is done here.
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    # Invert element-wise product of FFT's and truncate it.

    # Ensure that rp is non-negative.
    rp = np.where(rp > 0.0, rp, 0.0)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        x = np.asarray(x, dtype=np.float64)
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename variables.
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])
    h = bandwidth

    # Bin the data if not already binned.
    if not binned:
        x = np.asarray(x, dtype=np.float64)
        gpoints = np.linspace(a, b, M)
        gcounts = linbin(x, gpoints, truncate)
    else:
        gcounts = np.asarray(x, dtype=np.float64)
        M = gcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width.
    n = float(np.sum(gcounts))
    delta = (b - a) / (M - 1)

    # Obtain kernel weights.
    tau = 4 + drv
    L = int(min(np.floor(tau * h / delta), M))

    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = norm.pdf(arg) / (h ** (drv + 1))
    hmold0 = 1.0
    hmold1 = arg
    hmnew = 1.0
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1        # Compute mth degree Hermite polynomial
            hmold1 = hmnew         # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate.
    # We need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam_fft = np.fft.fft(kappam)
    Gcounts_fft = np.fft.fft(Gcounts)

    # R's fft(z, inverse = TRUE) is the unnormalized inverse transform,
    # i.e. numpy.fft.ifft(z) * len(z); dividing that result by P
    # (== len(z)) is therefore mathematically equivalent to just taking
    # numpy.fft.ifft(z) directly, which is what is done here.
    conv = np.real(np.fft.ifft(kappam_fft * Gcounts_fft))

    return float(np.sum(gcounts * conv[:M]) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # For obtaining preliminary estimates of quantities required for the
    # "direct plug-in" regression bandwidth selector based on blocked
    # qth degree polynomial fits.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]

    # Sort the (x, y) data with respect to the x's.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    # Equivalent of the Fortran subroutine blkest:
    # divides the sorted (x, y) data into Nval contiguous blocks, fits a
    # q'th degree polynomial to each block by least squares (equivalent to
    # the LINPACK dqrdc/dqrsl QR decomposition used in the original Fortran
    # code), and accumulates the residual sum of squares plus the
    # functionals of the 2nd and 4th derivatives of the fitted polynomials.
    qq = q + 1
    idiv = n // Nval

    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    for j in range(1, Nval + 1):
        # For each member of the partition (1-based Fortran-style bounds).
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        nj = iupp - ilow + 1

        # 0-based slice corresponding to the 1-based Fortran range
        # ilow..iupp.
        Xj = x[ilow - 1:iupp]
        Yj = y[ilow - 1:iupp]

        # Obtain a q'th degree fit over the current member of the
        # partition. Set up the design matrix.
        Xmat = np.ones((nj, qq), dtype=np.float64)
        for k in range(2, qq + 1):
            Xmat[:, k - 1] = Xj ** (k - 1)

        coef, _resid, _rank, _sv = np.linalg.lstsq(Xmat, Yj, rcond=None)

        for i in range(nj):
            fiti = coef[0]
            ddm = 2.0 * coef[2]
            ddddm = 24.0 * coef[4]
            for k in range(2, qq + 1):
                fiti = fiti + coef[k - 1] * Xj[i] ** (k - 1)
                if k <= q - 1:
                    ddm = ddm + k * (k + 1) * coef[k + 1] * Xj[i] ** (k - 1)
                    if k <= q - 3:
                        ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef[k + 3] * Xj[i] ** (k - 1)
            th22e = th22e + ddm ** 2
            th24e = th24e + ddm * ddddm
            RSS = RSS + (Yj[i] - fiti) ** 2

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]

    # Sort the (X, Y) data with respect to the X's.
    order_idx = np.argsort(X, kind="stable")
    X = X[order_idx]
    Y = Y[order_idx]

    # Set up dimensions for the equivalent of FORTRAN subroutine "cp".
    qq = q + 1
    RSS = np.zeros(Nmax, dtype=np.float64)

    # Equivalent of the Fortran subroutine cp:
    # for each candidate number of blocks Nval = 1..Nmax, partition the
    # sorted (X, Y) data into Nval blocks, fit a degree-q polynomial to
    # each block via least squares, and accumulate the residual sum of
    # squares (RSS) over all blocks for that Nval.
    for Nval in range(1, Nmax + 1):
        idiv = n // Nval
        RSSval = 0.0
        for j in range(1, Nval + 1):
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n
            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]
            nj = Xj.shape[0]

            # Set up the design matrix for a q'th degree polynomial fit
            # over the current member of the partition.
            Xmat = np.empty((nj, qq), dtype=np.float64)
            Xmat[:, 0] = 1.0
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            # Least-squares fit (equivalent of dqrdc/dqrsl QR solve).
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)
            fitted = Xmat @ coef
            RSSval += float(np.sum((Yj - fitted) ** 2))

        RSS[Nval - 1] = RSSval

    # Now compute the array of Mallow's C_p values.
    Cpvals = np.empty(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    # R's order(Cpvec)[1L] returns the (1-based) index of the smallest
    # Cp value. Since Cpvec is conceptually indexed by the number of
    # blocks N = 1..Nmax, this index is the chosen block count itself,
    # so the returned integer is that block count N, not a 0-based
    # array index.
    Nopt = int(np.argmin(Cpvals)) + 1

    return Nopt


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    # Rename variables

    n = x.shape[0]
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate

    choices = ("minim", "stdev", "iqr")
    if scalest not in choices:
        raise ValueError("'arg' should be one of " + ", ".join(repr(c) for c in choices))

    if scalest == "stdev":
        scale_val = math.sqrt(float(np.var(x, ddof=1)))
    elif scalest == "iqr":
        scale_val = (float(np.quantile(x, 0.75)) - float(np.quantile(x, 0.25))) / 1.349
    else:  # "minim"
        scale_val = min(
            (float(np.quantile(x, 0.75)) - float(np.quantile(x, 0.25))) / 1.349,
            math.sqrt(float(np.var(x, ddof=1))),
        )

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:

    x_mean = float(np.mean(x))
    sx = (x - x_mean) / scale_val
    sa = (a - x_mean) / scale_val
    sb = (b - x_mean) / scale_val

    # Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps

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
    elif level == 5:
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
    else:
        hpi = None

    return scale_val * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    # match.arg(kernel, choices) with (unambiguous prefix) partial matching
    kernel_choices = ["normal", "box", "epanech", "biweight", "triweight"]
    kernel_matches = [k for k in kernel_choices if k.startswith(kernel)]
    if len(kernel_matches) != 1:
        raise ValueError(f"'kernel' should be one of {kernel_choices}")
    kernel = kernel_matches[0]

    x = np.asarray(x, dtype=np.float64)

    # Default range.x = range(x)
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Set kernel constants
    if canonical:
        del0 = 1.0
    elif kernel == "normal":
        del0 = 1 / ((4 * np.pi) ** (1 / 10))
    elif kernel == "box":
        del0 = (9 / 2) ** (1 / 5)
    elif kernel == "epanech":
        del0 = 15 ** (1 / 5)
    elif kernel == "biweight":
        del0 = 35 ** (1 / 5)
    else:  # "triweight"
        del0 = (9450 / 143) ** (1 / 5)

    # Rename variables
    n = x.shape[0]
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate: match.arg(scalest, choices) with partial matching
    scalest_choices = ["minim", "stdev", "iqr"]
    scalest_matches = [s for s in scalest_choices if s.startswith(scalest)]
    if len(scalest_matches) != 1:
        raise ValueError(f"'scalest' should be one of {scalest_choices}")
    scalest = scalest_matches[0]

    if scalest == "stdev":
        scale_val = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")) / 1.349
    else:  # "minim"
        iqr_val = (np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")) / 1.349
        scale_val = min(iqr_val, np.sqrt(np.var(x, ddof=1)))

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    x_mean = np.mean(x)
    sx = (x - x_mean) / scale_val
    sa = (a - x_mean) / scale_val
    sb = (b - x_mean) / scale_val

    # Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps:
    if level == 0:
        psi4hat = 3 / (8 * np.sqrt(np.pi))
    elif level == 1:
        alpha = (2 * (np.sqrt(2)) ** 7 / (5 * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2 * (np.sqrt(2)) ** 9 / (7 * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2 * (np.sqrt(2)) ** 11 / (9 * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2 * (np.sqrt(2)) ** 13 / (11 * n)) ** (1 / 13)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * np.sqrt(2 / np.pi) / (psi10hat * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 5:
        alpha = (2 * (np.sqrt(2)) ** 15 / (13 * n)) ** (1 / 15)  # bandwidth for psi_12
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = (945 * np.sqrt(2 / np.pi) / (psi12hat * n)) ** (1 / 13)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * np.sqrt(2 / np.pi) / (psi10hat * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:
        raise ValueError("Level should be between 0 and 5")

    return float(scale_val * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    # Computes a direct plug-in selector of the
    # bandwidth for local linear regression as
    # described in the 1996 J. Amer. Statist. Assoc.
    # paper by Ruppert, Sheather and Wand.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Trim the 100(trim)% of the data from each end (in the x-direction).
    # Sort the (x, y) pairs with respect to x (stable sort, mirroring
    # R's sort.list on the first column of cbind(x, y)).
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    n_orig = x.shape[0]
    indlow = int(math.floor(trim * n_orig)) + 1
    indupp = n_orig - int(math.floor(trim * n_orig))

    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    # Rename common parameters.
    n = x.shape[0]
    M = int(gridsize)

    # NOTE: R's default argument 'range.x = range(x)' is a promise that
    # is only evaluated the first time it is used, which happens after
    # 'x' has already been reassigned to the trimmed data above. Hence
    # the default range below is computed from the TRIMMED x, matching
    # R's lazy-evaluation semantics exactly.
    if range_x is None:
        a = float(np.min(x))
        b = float(np.max(x))
    else:
        range_x_arr = np.asarray(range_x, dtype=np.float64)
        a = float(range_x_arr[0])
        b = float(range_x_arr[1])

    # Bin the data.
    gpoints = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    # Choose the value of N using Mallow's C_p.
    Nmax = int(max(min(math.floor(n / divisor), blockmax), 1))
    Nval = cpblock(x, y, Nmax, 4)

    # Estimate sig^2, theta_22 and theta_24 using quartic fits
    # on "Nval" blocks.
    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    # Estimate theta_22 using a local cubic fit
    # with a "rule-of-thumb" bandwidth: "gamseh".
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=np.array([a, b], dtype=np.float64), binned=True)["y"]

    llow = int(math.floor(proptrun * M)) + 1
    lupp = M - int(math.floor(proptrun * M))
    th22kn = float(np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n)

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh".
    C3K = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / (math.sqrt(2 * math.pi))) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    # Now compute a local linear kernel estimate of
    # the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                    range_x=np.array([a, b], dtype=np.float64), binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh,
                range_x=np.array([a, b], dtype=np.float64), binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh,
                    range_x=np.array([a, b], dtype=np.float64), binned=True)["y"]

    sigsqn = float(np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts))
    sigsqd = float(n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts))
    sigsqkn = sigsqn / sigsqd

    # Combine to obtain final answer.
    return (sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1 / 5)


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For computing a binned local polynomial regression estimator of a
    # univariate regression function or its derivative. The data are
    # discretised on an equally spaced grid. The bandwidths are
    # discretised on a logarithmically spaced grid.
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None:
        bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
        if np.any(bandwidth_arr <= 0):
            raise ValueError("'bandwidth' must be strictly positive")
    else:
        bandwidth_arr = None

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x) - np.min(x))
            range_x = (np.min(x) - extra, np.max(x) + extra)
        else:
            range_x = (np.min(x), np.max(x))

    # Rename common variables.
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Decide whether a density estimate or regression estimate is required.
    if y is None:
        # Obtain density estimate.
        n = x.shape[0]
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:
        # Obtain regression estimate.
        y = np.asarray(y, dtype=np.float64)
        # Bin the data if not already binned.
        if not binned:
            gpoints = np.linspace(a, b, M)
            out = rlbin(x, y, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = np.asarray(x, dtype=np.float64)
            ycounts = np.asarray(y, dtype=np.float64)
            M = xcounts.shape[0]
            gpoints = np.linspace(a, b, M)

    # Set the bin width.
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths.
    if bandwidth_arr is not None and bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc".
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc" to each member of
        # "bandwidth".
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
    elif bandwidth_arr is not None and bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Allocate space for the kernel vector and final estimate. All of the
    # arrays below are padded with an unused element at index 0 so that
    # the original 1-based Fortran indices from "locpol.f" can be used
    # verbatim without further translation.
    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    curvest = np.zeros(M + 1, dtype=np.float64)
    midpts = np.zeros(Q + 1, dtype=np.int64)
    ss = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    tt = np.zeros((M + 1, pp + 1), dtype=np.float64)
    Smat = np.zeros((pp + 1, pp + 1), dtype=np.float64)
    Tvec = np.zeros(pp + 1, dtype=np.float64)

    xcnts = np.concatenate(([0.0], np.asarray(xcounts, dtype=np.float64)))
    ycnts = np.concatenate(([0.0], np.asarray(ycounts, dtype=np.float64)))
    hdisc1 = np.concatenate(([0.0], np.asarray(hdisc, dtype=np.float64)))
    Lvec1 = np.concatenate(([0], np.asarray(Lvec, dtype=np.int64)))
    indic1 = np.concatenate(([0], np.asarray(indic, dtype=np.int64)))

    # Equivalent of the FORTRAN subroutine "locpol" (locpol.f):
    # obtain the kernel weights.
    mid = int(Lvec1[1]) + 1
    for i in range(1, Q):
        midpts[i] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec1[i]) + 1):
            fkap[mid + j] = np.exp(-((delta * j / hdisc1[i]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lvec1[i]) + int(Lvec1[i + 1]) + 1
    midpts[Q] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lvec1[Q]) + 1):
        fkap[mid + j] = np.exp(-((delta * j / hdisc1[Q]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts.
    for k in range(1, M + 1):
        if xcnts[k] != 0:
            for i in range(1, Q + 1):
                lo = max(1, k - int(Lvec1[i]))
                hi = min(M, k + int(Lvec1[i]))
                for j in range(lo, hi + 1):
                    if indic1[j] == i:
                        fac = 1.0
                        ss[j, 1] += xcnts[k] * fkap[k - j + midpts[i]]
                        tt[j, 1] += ycnts[k] * fkap[k - j + midpts[i]]
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcnts[k] * fkap[k - j + midpts[i]] * fac
                            if ii <= pp:
                                tt[j, ii] += ycnts[k] * fkap[k - j + midpts[i]] * fac

    # Solve the local (weighted) polynomial least-squares system at each
    # grid point (equivalent of the LINPACK dgefa/dgesl solve).
    for k in range(1, M + 1):
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i, j] = ss[k, indss]
            Tvec[i] = tt[k, i]

        sol = np.linalg.solve(Smat[1:pp + 1, 1:pp + 1], Tvec[1:pp + 1])

        curvest[k] = sol[drv]

    curvest_final = math.gamma(drv + 1) * curvest[1:M + 1]

    return {"x": gpoints, "y": curvest_final}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For computing the binned diagonal entries of a smoother
    # matrix for local polynomial kernel regression.
    # Note: 'drv' and 'kernel' are accepted for signature compatibility
    # with the original R function but are not otherwise referenced --
    # the underlying Fortran routine only ever uses the Gaussian kernel.
    x = np.asarray(x, dtype=np.float64)

    if range_x is None:
        if not binned:
            range_x = np.array([np.min(x), np.max(x)], dtype=np.float64)
        else:
            raise ValueError("'range_x' is required when 'binned' is True")
    else:
        range_x = np.asarray(range_x, dtype=np.float64)

    # Rename common variables.
    M = gridsize
    Q = int(bwdisc)
    a = range_x[0]
    b = range_x[1]
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4.0

    # Bin the data if not already binned.
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width.
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths.
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'.
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of 'hdisc'
        # to each member of 'bandwidth'.
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Equivalent of the Fortran subroutine sdiag:
    # obtains kernel weights on the discretised grid, combines them with
    # the bin counts to build up, for each grid point, the 'ppp' running
    # sums needed to assemble the local (ipp x ipp) weighted
    # least-squares system matrix, and extracts the (1,1) entry of its
    # inverse -- the diagonal entry of the smoother ('hat') matrix at
    # that grid point (in place of the LINPACK dgefa/dgedi calls used by
    # the original Fortran code, np.linalg.inv is used to invert the
    # local system matrix directly).
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)

    # Obtain kernel weights (1-based Fortran indices into 'fkap' are
    # tracked explicitly and converted to 0-based numpy indices on use).
    mid = int(Lvec[0]) + 1
    for i in range(1, Q):
        midpts[i - 1] = mid
        fkap[mid - 1] = 1.0
        Li = int(Lvec[i - 1])
        for j in range(1, Li + 1):
            val = np.exp(-((delta * j / hdisc[i - 1]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    LQ = int(Lvec[Q - 1])
    for j in range(1, LQ + 1):
        val = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    # Combine kernel weights and grid counts.
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i in range(1, Q + 1):
                Li = int(Lvec[i - 1])
                j_lo = max(1, k - Li)
                j_hi = min(M, k + Li)
                for j in range(j_lo, j_hi + 1):
                    if indic[j - 1] == i:
                        fac = 1.0
                        w = fkap[k - j + midpts[i - 1] - 1]
                        ss[j - 1, 0] = ss[j - 1, 0] + xcounts[k - 1] * w
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] = ss[j - 1, ii - 1] + xcounts[k - 1] * w * fac

    # Build the local system matrix at each grid point and invert it to
    # extract the diagonal entry of the smoother matrix.
    Sdg = np.zeros(M, dtype=np.float64)
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k - 1, indss - 1]
        Smat_inv = np.linalg.inv(Smat)
        Sdg[k - 1] = Smat_inv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
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
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Obtain kernel weights.
    # fkap[i][d] holds the kernel weight for the i-th discretised bandwidth
    # at an integer bin-distance d (0 <= d <= Lvec[i]).
    fkap = [np.zeros(int(Lvec[i]) + 1, dtype=np.float64) for i in range(Q)]
    for i in range(Q):
        fkap[i][0] = 1.0
        for j in range(1, int(Lvec[i]) + 1):
            fkap[i][j] = np.exp(-((delta * j / hdisc[i]) ** 2) / 2)

    # Combine kernel weights and grid counts to build, for every grid
    # point j, the moment sums used to form the local weighted least
    # squares system Smat (weights) and Umat (squared weights).
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)

    indic0 = indic - 1  # convert to 0-based bandwidth-group index

    for k in range(M):
        if xcounts[k] != 0:
            for i in range(Q):
                Li = int(Lvec[i])
                jlo = max(0, k - Li)
                jhi = min(M - 1, k + Li)
                for j in range(jlo, jhi + 1):
                    if indic0[j] == i:
                        d = abs(k - j)
                        wk = fkap[i][d]
                        fac = 1.0
                        cw = xcounts[k] * wk
                        ss[j, 0] += cw
                        uu[j, 0] += cw * wk
                        for ii in range(1, ppp):
                            fac *= delta * (k - j)
                            ss[j, ii] += cw * fac
                            uu[j, ii] += cw * wk * fac

    # At each grid point, assemble the (pp x pp) moment matrices from
    # the accumulated sums and compute the quadratic form giving the
    # diagonal entry of S * S^T.
    SSTd = np.zeros(M, dtype=np.float64)
    Smat = np.zeros((pp, pp), dtype=np.float64)
    Umat = np.zeros((pp, pp), dtype=np.float64)

    for k in range(M):
        for i in range(pp):
            for j in range(pp):
                indss = i + j
                Smat[i, j] = ss[k, indss]
                Umat[i, j] = uu[k, indss]

        Smat_inv = np.linalg.inv(Smat)
        row1 = Smat_inv[0, :]
        SSTd[k] = row1 @ Umat @ row1

    return {"x": gpoints, "y": SSTd}


def on_attach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")


def _on_unload(libpath: str) -> None:
    # In the original R package, this hook unloads the compiled
    # shared library ("KernSmooth") from the given libpath when the
    # package is detached, via library.dynam.unload("KernSmooth", libpath).
    #
    # This Python port is pure Python/NumPy and does not load any
    # compiled shared library, so there is nothing to unload here.
    # The function is kept as a no-op stub for interface parity with
    # the original R package.
    pass
