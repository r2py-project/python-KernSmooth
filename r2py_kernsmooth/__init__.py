import math
import warnings
from typing import Any

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


def _resolve_choice(val: str, choices: tuple[str, ...]) -> str:
    if val in choices:
        return val
    matches = [c for c in choices if c.startswith(val)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("'arg' should be one of " + ", ".join(f'"{c}"' for c in choices))


def _discretize_bandwidth(
    bandwidth: float | np.ndarray[Any, np.dtype[np.float64]],
    M: int,
    delta: float,
    Q: int,
    tau: float,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray, np.ndarray, int]:
    bw = np.asarray(bandwidth, dtype=np.float64)
    if bw.ndim == 0 or len(bw) == 1:
        scalar = float(bw.ravel()[0])
        Q = 1
        hdisc = np.full(1, scalar, dtype=np.float64)
        Lvec = np.full(1, int(np.floor(tau * scalar / delta)), dtype=np.int32)
        indic = np.ones(M, dtype=np.int32)
    elif len(bw) == M:
        hlow = float(np.min(bw))
        hupp = float(np.max(bw))
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))
        Lvec = np.floor(tau * hdisc / delta).astype(np.int32)
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int32)
            else:
                indic = np.round(((np.log(bw) - np.log(hlow)) / gap) + 1).astype(
                    np.int32
                )
        else:
            indic = np.ones(M, dtype=np.int32)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )
    return hdisc, Lvec, indic, Q


def linbin(
    X: np.ndarray[Any, np.dtype[np.float64]],
    gpoints: np.ndarray[Any, np.dtype[np.float64]],
    truncate: bool = True,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = len(X)
    M = len(gpoints)
    trun = np.int32(1) if truncate else np.int32(0)
    a = np.float64(gpoints[0])
    b = np.float64(gpoints[-1])
    gcnts = np.zeros(M, dtype=np.float64)
    _KernSmooth.linbin(
        np.asarray(X, dtype=np.float64), np.int32(n), a, b, np.int32(M), trun, gcnts
    )
    return gcnts


def rlbin(
    X: np.ndarray[Any, np.dtype[np.float64]],
    Y: np.ndarray[Any, np.dtype[np.float64]],
    gpoints: np.ndarray[Any, np.dtype[np.float64]],
    truncate: bool = True,
) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    n = len(X)
    M = len(gpoints)
    trun = np.int32(1) if truncate else np.int32(0)
    a = np.float64(gpoints[0])
    b = np.float64(gpoints[-1])
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)
    _KernSmooth.rlbin(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        np.int32(n),
        a,
        b,
        np.int32(M),
        trun,
        xcnts,
        ycnts,
    )
    return {"xcounts": xcnts, "ycounts": ycnts}


def bkfe(
    x: np.ndarray[Any, np.dtype[np.float64]],
    drv: int,
    bandwidth: float | None = None,
    gridsize: int = 401,
    range_x: tuple[float, float] | None = None,
    binned: bool = False,
    truncate: bool = True,
) -> np.float64:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        range_x = (np.min(x), np.max(x))

    # Rename variables
    M = gridsize
    a = range_x[0]
    b = range_x[1]
    h = bandwidth

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, gridsize)
        gcounts = linbin(x, gpoints, truncate)
    else:
        gcounts = x
        M = len(gcounts)
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width
    n = np.sum(gcounts)
    delta = (b - a) / (M - 1)

    # Obtain kernel weights
    tau = 4 + drv
    L = int(min(np.floor(tau * h / delta), M))

    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'",
            UserWarning,
            stacklevel=2,
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = norm.pdf(arg) / (h ** (drv + 1))
    hmold0 = 1
    hmold1 = arg
    hmnew = 1
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1  # Compute mth degree Hermite polynomial
            hmold1 = hmnew  # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log2(M + L + 1)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    return np.sum(gcounts * np.fft.ifft(kappam * Gcounts).real[:M]) / (n**2)


def blkest(
    x: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, np.dtype[np.float64]],
    Nval: int,
    q: int,
) -> dict[str, np.float64]:
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    sort_idx = np.argsort(x)
    x = np.asarray(x, dtype=np.float64)[sort_idx]
    y = np.asarray(y, dtype=np.float64)[sort_idx]

    # Set up arrays for Fortran programme 'blkest'
    qq = q + 1
    xj = np.zeros(n, dtype=np.float64)
    yj = np.zeros(n, dtype=np.float64)
    coef = np.zeros(qq, dtype=np.float64)
    Xmat = np.zeros((n, qq), dtype=np.float64, order="F")
    wk = np.zeros(n, dtype=np.float64)
    qraux = np.zeros(qq, dtype=np.float64)

    # The f2py interface (via src/_KernSmooth.pyf with intent(out) directives) returns
    # sigsqe, th22e, th24e as a tuple rather than requiring them as input arrays.
    sigsqe, th22e, th24e = _KernSmooth.blkest(
        x,
        y,
        np.int32(q),
        np.int32(Nval),
        xj,
        yj,
        coef,
        Xmat,
        wk,
        qraux,
    )

    return {"sigsqe": sigsqe, "th22e": th22e, "th24e": th24e}


def cpblock(
    X: np.ndarray[Any, np.dtype[np.float64]],
    Y: np.ndarray[Any, np.dtype[np.float64]],
    Nmax: int,
    q: int,
) -> int:
    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    sort_idx = np.argsort(X)
    X = np.asarray(X, dtype=np.float64)[sort_idx]
    Y = np.asarray(Y, dtype=np.float64)[sort_idx]

    # Set up arrays for Fortran subroutine 'cp'
    qq = q + 1
    RSS = np.zeros(Nmax, dtype=np.float64)
    Xj = np.zeros(n, dtype=np.float64)
    Yj = np.zeros(n, dtype=np.float64)
    coef = np.zeros(qq, dtype=np.float64)
    Xmat = np.zeros((n, qq), dtype=np.float64, order="F")
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    wk = np.zeros(n, dtype=np.float64)
    qraux = np.zeros(qq, dtype=np.float64)

    _KernSmooth.cp(
        X,
        Y,
        RSS,
        Xj,
        Yj,
        coef,
        Xmat,
        wk,
        qraux,
        Cpvals,
    )

    return int(np.argmin(Cpvals)) + 1


def linbin2D(
    X: np.ndarray[Any, np.dtype[np.float64]],
    gpoints1: np.ndarray[Any, np.dtype[np.float64]],
    gpoints2: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = X.shape[0]
    X_flat = np.concatenate([X[:, 0], X[:, 1]])
    M1 = len(gpoints1)
    M2 = len(gpoints2)
    a1 = np.float64(gpoints1[0])
    a2 = np.float64(gpoints2[0])
    b1 = np.float64(gpoints1[-1])
    b2 = np.float64(gpoints2[-1])
    gcnts = np.zeros(M1 * M2, dtype=np.float64)
    _KernSmooth.lbtwod(
        np.asarray(X_flat, dtype=np.float64),
        np.int32(n),
        a1,
        a2,
        b1,
        b2,
        np.int32(M1),
        np.int32(M2),
        gcnts,
    )
    return gcnts.reshape((M1, M2), order="F")


def locpoly(
    x: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    drv: int = 0,
    degree: int | None = None,
    kernel: str = "normal",
    bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None,
    gridsize: int = 401,
    bwdisc: int = 25,
    range_x: tuple[float, float] | None = None,
    binned: bool = False,
    truncate: bool = True,
) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.any(np.asarray(bandwidth) <= 0):
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
            out_bin = rlbin(x, y, gpoints, truncate)
            xcounts = out_bin["xcounts"]
            ycounts = out_bin["ycounts"]
        else:
            xcounts = np.asarray(x, dtype=np.float64)
            ycounts = np.asarray(y, dtype=np.float64)
            M = len(xcounts)
            gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    hdisc, Lvec, indic, Q = _discretize_bandwidth(bandwidth, M, delta, Q, tau)

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Allocate space for the kernel vector and final estimate
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    curvest = np.zeros(M, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int32)
    ss = np.zeros((M, ppp), dtype=np.float64, order="F")
    tt = np.zeros((M, pp), dtype=np.float64, order="F")
    Smat = np.zeros((pp, pp), dtype=np.float64, order="F")
    Tvec = np.zeros(pp, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int32)

    # Call Fortran routine 'locpol'
    # f2py infers m/ipp/ippp from ss/tt shapes; iq=Q must immediately follow midpts
    _KernSmooth.locpol(
        np.asarray(xcounts, dtype=np.float64),
        np.asarray(ycounts, dtype=np.float64),
        np.int32(drv),
        np.float64(delta),
        hdisc,
        Lvec,
        indic,
        midpts,
        np.int32(Q),
        fkap,
        ss,
        tt,
        Smat,
        Tvec,
        ipvt,
        curvest,
    )

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def sdiag(
    x: np.ndarray[Any, np.dtype[np.float64]],
    drv: int = 0,
    degree: int = 1,
    kernel: str = "normal",
    bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None,
    gridsize: int = 401,
    bwdisc: int = 25,
    range_x: tuple[float, float] | None = None,
    binned: bool = False,
    truncate: bool = True,
) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
        range_x = (np.min(x), np.max(x))

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
    hdisc, Lvec, indic, Q = _discretize_bandwidth(bandwidth, M, delta, Q, tau)

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int32)
    ss = np.zeros((M, ppp), dtype=np.float64, order="F")
    Smat = np.zeros((pp, pp), dtype=np.float64, order="F")
    work = np.zeros(pp, dtype=np.float64)
    det = np.zeros(2, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int32)
    Sdg = np.zeros(M, dtype=np.float64)

    _KernSmooth.sdiag(
        np.asarray(xcounts, dtype=np.float64),
        np.float64(delta),
        hdisc,
        Lvec,
        indic,
        midpts,
        np.int32(Q),
        fkap,
        ss,
        Smat,
        work,
        det,
        ipvt,
        Sdg,
    )

    return {"x": gpoints, "y": Sdg}


def sstdiag(
    x: np.ndarray[Any, np.dtype[np.float64]],
    drv: int = 0,
    degree: int = 1,
    kernel: str = "normal",
    bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None,
    gridsize: int = 401,
    bwdisc: int = 25,
    range_x: tuple[float, float] | None = None,
    binned: bool = False,
    truncate: bool = True,
) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
        range_x = (np.min(x), np.max(x))

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
    hdisc, Lvec, indic, Q = _discretize_bandwidth(bandwidth, M, delta, Q, tau)

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int32)
    ss = np.zeros((M, ppp), dtype=np.float64, order="F")
    uu = np.zeros((M, ppp), dtype=np.float64, order="F")
    Smat = np.zeros((pp, pp), dtype=np.float64, order="F")
    Umat = np.zeros((pp, pp), dtype=np.float64, order="F")
    work = np.zeros(pp, dtype=np.float64)
    det = np.zeros(2, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int32)
    SSTd = np.zeros(M, dtype=np.float64)

    _KernSmooth.sstdg(
        np.asarray(xcounts, dtype=np.float64),
        np.float64(delta),
        hdisc,
        Lvec,
        indic,
        midpts,
        np.int32(Q),
        fkap,
        ss,
        uu,
        Smat,
        Umat,
        work,
        det,
        ipvt,
        SSTd,
    )

    return {"x": gpoints, "y": SSTd}


def bkde(
    x: np.ndarray[Any, np.dtype[np.float64]],
    kernel: str = "normal",
    canonical: bool = False,
    bandwidth: float | None = None,
    gridsize: int = 401,
    range_x: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    truncate: bool = True,
) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")
    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    kernel = _resolve_choice(kernel, valid_kernels)
    n = len(x)
    M = gridsize
    del0_map = {
        "normal": (1.0 / (4.0 * np.pi)) ** (1.0 / 10.0),
        "box": (9.0 / 2.0) ** (1.0 / 5.0),
        "epanech": 15.0 ** (1.0 / 5.0),
        "biweight": 35.0 ** (1.0 / 5.0),
        "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
    }
    del0 = del0_map[kernel]
    if not isinstance(canonical, (bool, np.bool_)):
        raise ValueError("'canonical' must be a length-1 logical vector")
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.std(x, ddof=1)
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth
    tau = 4.0 if kernel == "normal" else 1.0
    if range_x is None:
        range_x = np.array([np.min(x) - tau * h, np.max(x) + tau * h])
    a = range_x[0]
    b = range_x[1]
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'",
            UserWarning,
            stacklevel=2,
        )
    lvec = np.arange(0, L + 1)
    if kernel == "normal":
        kappa = norm.pdf(lvec * delta) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1.0), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1.0), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1.0), 3, 3) / (n * h)
    else:
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1.0), 4, 4) / (n * h)
    P = int(2 ** np.ceil(np.log2(M + L + 1)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)
    return {"x": gpoints, "y": np.fft.ifft(kappa * gcounts).real[:M]}


def bkde2D(
    x: np.ndarray[Any, np.dtype[np.float64]],
    bandwidth: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    gridsize: tuple[int, int] = (51, 51),
    range_x: list[tuple[float, float]] | None = None,
    truncate: bool = True,
) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables
    n = x.shape[0]
    M = np.array(gridsize, dtype=int)
    h = np.asarray(bandwidth, dtype=np.float64)
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction if only a single bandwidth is given.
    if h.ndim == 0 or len(h) == 1:
        h = np.array([float(h.ravel()[0]), float(h.ravel()[0])])

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [None, None]
        for id in range(2):
            range_x[id] = (
                np.min(x[:, id]) - 1.5 * h[id],
                np.max(x[:, id]) + 1.5 * h[id],
            )

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], M[0])
    gpoints2 = np.linspace(a[1], b[1], M[1])

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=float)
    kapid = [None, None]
    for id in range(2):
        L[id] = min(np.floor(tau * h[id] * (M[id] - 1) / (b[id] - a[id])), M[id] - 1)
        lvecid = np.arange(0, int(L[id]) + 1)
        facid = (b[id] - a[id]) / (h[id] * (M[id] - 1))
        z = (norm.pdf(lvecid * facid) / h[id]).reshape(-1, 1)
        z_flat = z.ravel()
        tot = np.sum(np.concatenate([z_flat, z_flat[1:][::-1]])) * facid * h[id]
        kapid[id] = z / tot
    kapp = (kapid[0] @ kapid[1].T) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'",
            UserWarning,
            stacklevel=2,
        )

    # Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log2(M + L))).astype(int)
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[: L1 + 1, : L2 + 1] = kapp
    if L1:
        rp[P1 - L1 : P1, : L2 + 1] = kapp[L1:0:-1, : L2 + 1]
    if L2:
        rp[:, P2 - L2 : P2] = rp[:, L2:0:-1]
    # wrap-around version of kapp

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[:M1, :M2] = gcounts
    # zero-padded version of gcounts

    rp = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp = np.fft.fft2(sp)
    rp = np.fft.ifft2(rp * sp).real[:M1, :M2]
    # invert element-wise product of FFT's and truncate and normalise it

    # Ensure that rp is non-negative
    rp = np.maximum(rp, 0.0)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def dpih(
    x: np.ndarray[Any, np.dtype[np.float64]],
    scalest: str = "minim",
    level: int = 2,
    gridsize: int = 401,
    range_x: tuple[float, float] | None = None,
    truncate: bool = True,
) -> np.float64:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    # Rename variables
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (np.min(x), np.max(x))
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate
    _SCALEST_CHOICES = ("minim", "stdev", "iqr")
    scalest = _resolve_choice(scalest, _SCALEST_CHOICES)

    std_val = np.std(x, ddof=1)
    iqr_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    scale_map = {
        "stdev": std_val,
        "iqr": iqr_val,
        "minim": min(iqr_val, std_val),
    }
    scalest_val = scale_map[scalest]

    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    x_mean = np.mean(x)
    sx = (x - x_mean) / scalest_val
    sa = (a - x_mean) / scalest_val
    sb = (b - x_mean) / scalest_val

    # Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps
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
        _val = -3 * np.sqrt(2 / np.pi) / (psi6hat * n)
        alpha = math.copysign(abs(_val) ** (1 / 7), _val)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    elif level == 4:
        alpha = ((2 / (9 * n)) ** (1 / 11)) * np.sqrt(2)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        _val = 15 * np.sqrt(2 / np.pi) / (psi8hat * n)
        alpha = math.copysign(abs(_val) ** (1 / 9), _val)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        _val = -3 * np.sqrt(2 / np.pi) / (psi6hat * n)
        alpha = math.copysign(abs(_val) ** (1 / 7), _val)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)
    elif level == 5:
        alpha = ((2 / (11 * n)) ** (1 / 13)) * np.sqrt(2)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        _val = -105 * np.sqrt(2 / np.pi) / (psi10hat * n)
        alpha = math.copysign(abs(_val) ** (1 / 11), _val)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        _val = 15 * np.sqrt(2 / np.pi) / (psi8hat * n)
        alpha = math.copysign(abs(_val) ** (1 / 9), _val)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        _val = -3 * np.sqrt(2 / np.pi) / (psi6hat * n)
        alpha = math.copysign(abs(_val) ** (1 / 7), _val)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1 / 3)

    return np.float64(scalest_val * hpi)


def dpik(
    x: np.ndarray[Any, np.dtype[np.float64]],
    scalest: str = "minim",
    level: int = 2,
    kernel: str = "normal",
    canonical: bool = False,
    gridsize: int = 401,
    range_x: tuple[float, float] | None = None,
    truncate: bool = True,
) -> np.float64:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    # Validate kernel argument
    _KERNEL_CHOICES = ("normal", "box", "epanech", "biweight", "triweight")
    kernel = _resolve_choice(kernel, _KERNEL_CHOICES)

    # Set kernel constants
    if canonical:
        del0 = 1.0
    else:
        _KERNEL_DEL0 = {
            "normal": 1.0 / ((4.0 * math.pi) ** (1.0 / 10.0)),
            "box": (9.0 / 2.0) ** (1.0 / 5.0),
            "epanech": 15.0 ** (1.0 / 5.0),
            "biweight": 35.0 ** (1.0 / 5.0),
            "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
        }
        del0 = _KERNEL_DEL0[kernel]

    # Rename variables
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (np.min(x), np.max(x))
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate
    _SCALEST_CHOICES = ("minim", "stdev", "iqr")
    scalest = _resolve_choice(scalest, _SCALEST_CHOICES)

    std_val = np.std(x, ddof=1)
    iqr_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    scale_map = {
        "stdev": float(std_val),
        "iqr": float(iqr_val),
        "minim": float(min(iqr_val, std_val)),
    }
    scalest_val = scale_map[scalest]

    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    x_mean = np.mean(x)
    sx = (x - x_mean) / scalest_val
    sa = (a - x_mean) / scalest_val
    sb = (b - x_mean) / scalest_val

    # Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps:
    if level == 0:
        psi4hat = 3.0 / (8.0 * math.sqrt(math.pi))
    elif level == 1:
        alpha = (2.0 * (math.sqrt(2.0)) ** 7 / (5.0 * n)) ** (
            1.0 / 7.0
        )  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2.0 * (math.sqrt(2.0)) ** 9 / (7.0 * n)) ** (
            1.0 / 9.0
        )  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        _val = -3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 7.0), _val)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2.0 * (math.sqrt(2.0)) ** 11 / (9.0 * n)) ** (
            1.0 / 11.0
        )  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        _val = 15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 9.0), _val)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        _val = -3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 7.0), _val)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2.0 * (math.sqrt(2.0)) ** 13 / (11.0 * n)) ** (
            1.0 / 13.0
        )  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        _val = -105.0 * math.sqrt(2.0 / math.pi) / (psi10hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 11.0), _val)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        _val = 15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 9.0), _val)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        _val = -3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 7.0), _val)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 5:
        alpha = (2.0 * (math.sqrt(2.0)) ** 15 / (13.0 * n)) ** (
            1.0 / 15.0
        )  # bandwidth for psi_12
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        _val = 945.0 * math.sqrt(2.0 / math.pi) / (psi12hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 13.0), _val)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        _val = -105.0 * math.sqrt(2.0 / math.pi) / (psi10hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 11.0), _val)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        _val = 15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 9.0), _val)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        _val = -3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)
        alpha = math.copysign(abs(_val) ** (1.0 / 7.0), _val)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return np.float64(scalest_val * del0 * (1.0 / (psi4hat * n)) ** (1.0 / 5.0))


def dpill(
    x: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, np.dtype[np.float64]],
    blockmax: int = 5,
    divisor: int = 20,
    trim: float = 0.01,
    proptrun: float = 0.05,
    gridsize: int = 401,
    range_x: tuple[float, float] | None = None,
    truncate: bool = True,
) -> np.float64:
    # Trim the 100(trim)% of the data from each end (in the x-direction).
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    indlow = int(np.floor(trim * len(x)))
    indupp = len(x) - int(np.floor(trim * len(x)))
    x = x[indlow:indupp]
    y = y[indlow:indupp]

    # In R, range.x = range(x) is a lazy default evaluated after trimming.
    # Match that behaviour: compute range from the trimmed x when not supplied.
    if range_x is None:
        range_x = (x[0], x[-1])

    # Rename common parameters
    n = len(x)
    M = gridsize
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
    # on 'Nval' blocks.
    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    # Estimate theta_22 using a local cubic fit
    # with a 'rule-of-thumb' bandwidth: 'gamseh'
    gamseh = sigsqQ * (b - a) / (np.abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * np.sqrt(np.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * np.sqrt(np.pi))) ** (1 / 7)

    mddest = locpoly(
        xcounts, ycounts, drv=2, bandwidth=gamseh, range_x=range_x, binned=True
    )["y"]

    llow = int(np.floor(proptrun * M))
    lupp = M - int(np.floor(proptrun * M))
    th22kn = np.sum((mddest[llow:lupp] ** 2) * xcounts[llow:lupp]) / n

    # Estimate sigma^2 using a local linear fit
    # with a 'direct plug-in' bandwidth: 'lamseh'
    C3K = (1 / 2) + 2 * np.sqrt(2) - (4 / 3) * np.sqrt(3)
    C3K = (4 * C3K / np.sqrt(2 * np.pi)) ** (1 / 9)
    lamseh = C3K * (((sigsqQ**2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    # Now compute a local linear kernel estimate of the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh, range_x=range_x, binned=True)[
        "y"
    ]
    Sdg = sdiag(xcounts, bandwidth=lamseh, range_x=range_x, binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh, range_x=range_x, binned=True)["y"]
    sigsqn = np.sum(y**2) - 2 * np.sum(mest * ycounts) + np.sum((mest**2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    # Combine to obtain final answer.
    return (sigsqkn * (b - a) / (2 * np.sqrt(np.pi) * th22kn * n)) ** (1 / 5)
