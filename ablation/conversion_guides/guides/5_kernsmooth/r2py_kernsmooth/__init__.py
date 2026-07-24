import math
import warnings
from typing import Any, Literal

import numpy as np
from scipy.stats import beta as beta_dist, norm

from . import _KernSmooth

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


def on_attach(libname: str | None = None, pkgname: str | None = None) -> None:
    warnings.warn(
        "KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009",
        category=UserWarning,
        stacklevel=2,
    )


def on_unload(libpath: str) -> None:
    # Python's import system does not support safely unloading compiled
    # extension modules (no equivalent of R's library.dynam.unload),
    # so this hook is a no-op stub preserving the original intent.
    pass


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    X_d = np.asarray(X, dtype=np.float64)
    n_i = np.int32(n)
    a_d = np.float64(a)
    b_d = np.float64(b)
    M_i = np.int32(M)
    trun_i = np.int32(trun)
    gcounts = np.zeros(M, dtype=np.float64)

    _KernSmooth.linbin(X_d, n_i, a_d, b_d, M_i, trun_i, gcounts)

    return gcounts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = X.shape[0]
    X_flat = np.concatenate([X[:, 0], X[:, 1]]).astype(np.float64)
    M1 = len(gpoints1)
    M2 = len(gpoints2)
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    n_i = np.int32(n)
    a1_d = np.float64(a1)
    a2_d = np.float64(a2)
    b1_d = np.float64(b1)
    b2_d = np.float64(b2)
    M1_i = np.int32(M1)
    M2_i = np.int32(M2)
    out = np.zeros(M1 * M2, dtype=np.float64)

    _KernSmooth.lbtwod(X_flat, n_i, a1_d, a2_d, b1_d, b2_d, M1_i, M2_i, out)

    return out.reshape((M1, M2), order='F')


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    xcounts = np.zeros(M, dtype=np.float64)
    ycounts = np.zeros(M, dtype=np.float64)

    _KernSmooth.rlbin(x, y, n, a, b, M, trun, xcounts, ycounts)

    return {"xcounts": xcounts, "ycounts": ycounts}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    ## Install safeguard against non-positive bandwidths:

    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        range_x = (np.min(x), np.max(x))

    ## Rename variables

    M = gridsize
    a = range_x[0]
    b = range_x[1]
    h = bandwidth

    ## Bin the data if not already binned

    if not binned:
        gpoints = np.linspace(a, b, gridsize)
        gcounts = linbin(x, gpoints, truncate)
    else:
        gcounts = np.asarray(x, dtype=np.float64)
        M = len(gcounts)
        gpoints = np.linspace(a, b, M)

    ## Set the sample size and bin width

    n = np.sum(gcounts)
    delta = (b - a) / (M - 1)

    ## Obtain kernel weights

    tau = 4 + drv
    L = min(int(np.floor(tau * h / delta)), M)

    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = norm.pdf(arg) / (h ** (drv + 1))
    hmold0 = 1
    hmold1 = arg
    hmnew = 1
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0  # Compute mth degree Hermite polynomial
            hmold0 = hmold1                          # by recurrence.
            hmold1 = hmnew
    kappam = hmnew * kappam

    ## Now combine weights and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    return float(np.sum(gcounts * (np.fft.ifft(kappam * Gcounts).real)[:M]) / (n ** 2))


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: Literal["normal", "box", "epanech", "biweight", "triweight"] = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        matches = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError(
                f"'kernel' should be one of {_kernel_choices}"
            )

    ## Rename common variables

    n = len(x)
    M = gridsize

    ## Set canonical scaling factors

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

    ## Set default bandwidth

    if bandwidth is None:
        h = del0 * (243 / (35 * n)) ** (1 / 5) * np.std(np.asarray(x, dtype=np.float64), ddof=1)
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    ## Set kernel support values

    tau = 4 if kernel == "normal" else 1

    if range_x is None:
        range_x = (np.min(x) - tau * h, np.max(x) + tau * h)
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute kernel weights

    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    if kernel == "normal":
        kappa = norm.pdf(lvec * delta) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([np.asarray(gcounts, dtype=np.float64), np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)
    return {
        "x": gpoints,
        "y": np.real(np.fft.ifft(kappa * gcounts))[:M],
    }


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: tuple[int, int] = (51, 51), range_x: list[tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.min(np.atleast_1d(bandwidth)) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.float64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([h[0], h[0]])

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [None, None]
        for id in range(2):
            range_x[id] = (float(np.min(x[:, id]) - 1.5 * h[id]),
                           float(np.max(x[:, id]) + 1.5 * h[id]))

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.float64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]] | None] = [None, None]
    for id in range(2):
        L[id] = min(np.floor(tau * h[id] * (M[id] - 1) / (b[id] - a[id])), M[id] - 1)
        lvecid = np.arange(0, int(L[id]) + 1)
        facid = (b[id] - a[id]) / (h[id] * (M[id] - 1))
        z = (norm.pdf(lvecid * facid) / h[id]).reshape(-1, 1)
        tot = np.sum(np.concatenate([z.flatten(), z.flatten()[1:][::-1]])) * facid * h[id]
        kapid[id] = z / tot
    kapp = kapid[0] @ kapid[1].T / n

    if np.min(L) == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: "
                      "consider increasing 'gridsize'")

    # Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log(M + L) / np.log(2))).astype(np.int64)  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0:(L1 + 1), 0:(L2 + 1)] = kapp
    if L1:
        rp[(P1 - L1):P1, 0:(L2 + 1)] = kapp[L1:0:-1, :]
    if L2:
        rp[:, (P2 - L2):P2] = rp[:, L2:0:-1]
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    datmat = np.column_stack((x, y))
    sort_idx = np.argsort(datmat[:, 0], kind="quicksort")
    datmat = datmat[sort_idx, :]
    x = datmat[:, 0]
    y = datmat[:, 1]

    # Set up arrays for FORTRAN programme "blkest"
    qq = q + 1
    xj = np.zeros(n, dtype=np.float64)
    yj = np.zeros(n, dtype=np.float64)
    coef = np.zeros(qq, dtype=np.float64)
    Xmat = np.zeros((n, qq), dtype=np.float64)
    wk = np.zeros(n, dtype=np.float64)
    qraux = np.zeros(qq, dtype=np.float64)
    sigsqe = np.zeros(1, dtype=np.float64)
    th22e = np.zeros(1, dtype=np.float64)
    th24e = np.zeros(1, dtype=np.float64)

    x_d = np.asarray(x, dtype=np.float64)
    y_d = np.asarray(y, dtype=np.float64)
    n_i = np.int32(n)
    q_i = np.int32(q)
    qq_i = np.int32(qq)
    Nval_i = np.int32(Nval)

    _KernSmooth.blkest(x_d, y_d, n_i, q_i, qq_i, Nval_i, xj, yj, coef,
                       Xmat, wk, qraux, sigsqe, th22e, th24e)

    return {"sigsqe": float(sigsqe[0]), "th22e": float(th22e[0]), "th24e": float(th24e[0])}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    datmat = np.column_stack((X, Y))
    sort_idx = np.argsort(datmat[:, 0], kind="quicksort")
    datmat = datmat[sort_idx, :]
    X = datmat[:, 0]
    Y = datmat[:, 1]

    # Set up arrays for FORTRAN subroutine "cp"
    qq = q + 1
    RSS = np.zeros(Nmax, dtype=np.float64)
    Xj = np.zeros(n, dtype=np.float64)
    Yj = np.zeros(n, dtype=np.float64)
    coef = np.zeros(qq, dtype=np.float64)
    Xmat = np.zeros((n, qq), dtype=np.float64)
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    wk = np.zeros(n, dtype=np.float64)
    qraux = np.zeros(qq, dtype=np.float64)

    X_d = np.asarray(X, dtype=np.float64)
    Y_d = np.asarray(Y, dtype=np.float64)
    n_i = np.int32(n)
    qq_i = np.int32(qq)
    Nmax_i = np.int32(Nmax)

    # remove unused 'q' 2007-07-10
    _KernSmooth.cp(X_d, Y_d, n_i, qq_i, Nmax_i, RSS, Xj, Yj, coef, Xmat, wk,
                   qraux, Cpvals)

    Cpvec = Cpvals

    return int(np.argmin(Cpvec)) + 1


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.any(np.asarray(bandwidth, dtype=np.float64) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

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

    # Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = range_x[0]
    b = range_x[1]
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Decide whether a density estimate or regression estimate is required.
    if y is None:  # obtain density estimate
        n = len(x)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        # Bin the data if not already binned
        if not binned:
            gpoints = np.linspace(a, b, M)
            out = rlbin(x, y, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = np.asarray(x, dtype=np.float64)
            ycounts = np.asarray(y, dtype=np.float64)
            M = len(xcounts)
            gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if len(bandwidth_arr) == M:
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
                raw = ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                indic = np.round(raw).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    # Allocate space for the kernel vector and final estimate
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    curvest = np.zeros(M, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int32)
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)
    Smat = np.zeros((pp, pp), dtype=np.float64)
    Tvec = np.zeros(pp, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int32)

    # Call FORTRAN routine "locpol"
    xcounts_d = np.asarray(xcounts, dtype=np.float64)
    ycounts_d = np.asarray(ycounts, dtype=np.float64)
    drv_i = np.int32(drv)
    delta_d = np.float64(delta)
    hdisc_d = np.asarray(hdisc, dtype=np.float64)
    Lvec_i = np.asarray(Lvec, dtype=np.int32)
    indic_i = np.asarray(indic, dtype=np.int32)
    M_i = np.int32(M)
    Q_i = np.int32(Q)
    pp_i = np.int32(pp)
    ppp_i = np.int32(ppp)

    _KernSmooth.locpol(xcounts_d, ycounts_d, drv_i, delta_d, hdisc_d,
                       Lvec_i, indic_i, midpts, M_i, Q_i, fkap, pp_i,
                       ppp_i, ss, tt, Smat, Tvec, ipvt, curvest)

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables

    M = gridsize
    Q = int(bwdisc)
    a = range_x[0]
    b = range_x[1]
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned

    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    # Set the bin width

    delta = (b - a) / (M - 1)

    # Discretise the bandwidths

    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int32)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int32)
            else:
                raw = ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                indic = np.round(raw).astype(np.int32)
        else:
            indic = np.ones(M, dtype=np.int32)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int32)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int32)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int32)
    ss = np.zeros((M, ppp), dtype=np.float64)
    Smat = np.zeros((pp, pp), dtype=np.float64)
    work = np.zeros(pp, dtype=np.float64)
    det = np.zeros(2, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int32)
    Sdg = np.zeros(M, dtype=np.float64)

    xcounts_d = np.asarray(xcounts, dtype=np.float64)
    delta_d = np.float64(delta)
    hdisc_d = np.asarray(hdisc, dtype=np.float64)
    Lvec_i = np.asarray(Lvec, dtype=np.int32)
    indic_i = np.asarray(indic, dtype=np.int32)
    M_i = np.int32(M)
    Q_i = np.int32(Q)
    pp_i = np.int32(pp)
    ppp_i = np.int32(ppp)

    _KernSmooth.sdiag(xcounts_d, delta_d, hdisc_d, Lvec_i, indic_i, midpts,
                      M_i, Q_i, fkap, pp_i, ppp_i, ss, Smat, work, det,
                      ipvt, Sdg)

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = range_x[0]
    b = range_x[1]
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int32)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int32)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int32)
        else:
            indic = np.ones(M, dtype=np.int32)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int32)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int32)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int32)
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    Smat = np.zeros((pp, pp), dtype=np.float64)
    Umat = np.zeros((pp, pp), dtype=np.float64)
    work = np.zeros(pp, dtype=np.float64)
    det = np.zeros(2, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int32)
    SSTd = np.zeros(M, dtype=np.float64)

    xcounts_d = np.asarray(xcounts, dtype=np.float64)
    delta_d = np.float64(delta)
    hdisc_d = np.asarray(hdisc, dtype=np.float64)
    Lvec_i = np.asarray(Lvec, dtype=np.int32)
    indic_i = np.asarray(indic, dtype=np.int32)
    M_i = np.int32(M)
    Q_i = np.int32(Q)
    pp_i = np.int32(pp)
    ppp_i = np.int32(ppp)

    _KernSmooth.sstdg(xcounts_d, delta_d, hdisc_d, Lvec_i, indic_i,
                       midpts, M_i, Q_i, fkap, pp_i, ppp_i, ss, uu,
                       Smat, Umat, work, det, ipvt, SSTd)

    return {"x": gpoints, "y": SSTd}


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: Literal["minim", "stdev", "iqr"] = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    ## Rename variables

    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        _matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(_matches) == 1:
            scalest = _matches[0]  # type: ignore[assignment]
        else:
            raise ValueError(
                "'arg' should be one of " + ", ".join(repr(c) for c in _scalest_choices)
            )

    if scalest == "stdev":
        scale_value = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_value = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scale_value = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

    if scale_value == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = np.mean(x)
    sx = (x - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps

    if level == 0:
        hpi = (24 * np.sqrt(np.pi) / n) ** (1 / 3)
    elif level == 1:
        alpha = (2 / (3 * n)) ** (1 / 5) * np.sqrt(2)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    elif level == 2:
        alpha = ((2 / (5 * n)) ** (1 / 7)) * np.sqrt(2)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    elif level == 3:
        alpha = ((2 / (7 * n)) ** (1 / 9)) * np.sqrt(2)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    elif level == 4:
        alpha = ((2 / (9 * n)) ** (1 / 11)) * np.sqrt(2)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    elif level == 5:
        alpha = ((2 / (11 * n)) ** (1 / 13)) * np.sqrt(2)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * np.sqrt(2 / np.pi) / (psi10hat * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    else:
        raise ValueError("Level should be between 0 and 5")

    return float(scale_value * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: Literal["minim", "stdev", "iqr"] = "minim", level: int = 2, kernel: Literal["normal", "box", "epanech", "biweight", "triweight"] = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        matches = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError(f"'kernel' should be one of {_kernel_choices}")

    ## Set kernel constants

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
    else:  # triweight
        del0 = (9450 / 143) ** (1 / 5)

    x = np.asarray(x, dtype=np.float64)

    ## Rename variables

    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (np.min(x), np.max(x))
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError(f"'scalest' should be one of {_scalest_choices}")

    if scalest == "stdev":
        scale_val = np.std(x, ddof=1)
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # minim
        scale_val = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            np.std(x, ddof=1),
        )

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = np.mean(x)
    sx = (x - x_mean) / scale_val
    sa = (a - x_mean) / scale_val
    sb = (b - x_mean) / scale_val

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M-1)

    ## Perform plug-in steps:

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
    else:  # level == 5
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

    return float(scale_val * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    # Trim the 100(trim)% of the data from each end (in the x-direction).
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    xy = np.column_stack((x, y))
    sort_idx = np.argsort(xy[:, 0], kind="quicksort")
    xy = xy[sort_idx, :]
    x = xy[:, 0]
    y = xy[:, 1]

    indlow = int(np.floor(trim * len(x)))
    indupp = len(x) - int(np.floor(trim * len(x)))

    x = x[indlow:indupp]
    y = y[indlow:indupp]

    # Rename common parameters
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    # Bin the data
    gpoints = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    # Choose the value of N using Mallow's C_p
    Nmax = max(min(int(np.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x, y, Nmax, 4)

    # Estimate sig^2, theta_22 and theta_24 using quartic fits
    # on "Nval" blocks.
    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    # Estimate theta_22 using a local cubic fit
    # with a "rule-of-thumb" bandwidth: "gamseh"
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=range_x, binned=True)["y"]

    llow = int(np.floor(proptrun * M))
    lupp = M - int(np.floor(proptrun * M))
    th22kn = np.sum((mddest[llow:lupp] ** 2) * xcounts[llow:lupp]) / n

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / (math.sqrt(2 * math.pi))) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    # Now compute a local linear kernel estimate of
    # the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                    range_x=range_x, binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh,
                range_x=range_x, binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh,
                    range_x=range_x, binned=True)["y"]
    sigsqn = np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    # Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1 / 5))
