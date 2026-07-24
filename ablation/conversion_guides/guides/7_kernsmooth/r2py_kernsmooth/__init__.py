import math
import sys
from typing import Any
import warnings

import numpy as np
from scipy.stats import beta as beta_dist, norm


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
    n: int = len(X)
    M: int = len(gpoints)
    trun: int = 1 if truncate else 0
    a: float = float(gpoints[0])
    b: float = float(gpoints[M - 1])

    gcnts: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M, dtype=np.float64)

    delta: float = (b - a) / (M - 1)
    for i in range(n):
        lxi: float = ((X[i] - a) / delta) + 1

        # Find integer part of "lxi" (truncation toward zero, as in Fortran INT())
        li: int = int(lxi)

        rem: float = lxi - li
        if li >= 1 and li < M:
            gcnts[li - 1] = gcnts[li - 1] + (1 - rem)
            gcnts[li] = gcnts[li] + rem

        if li < 1 and trun == 0:
            gcnts[0] = gcnts[0] + 1

        if li >= M and trun == 0:
            gcnts[M - 1] = gcnts[M - 1] + 1

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)

    n: int = X.shape[0]
    x1: np.ndarray[Any, np.dtype[np.float64]] = X[:, 0]
    x2: np.ndarray[Any, np.dtype[np.float64]] = X[:, 1]

    M1: int = len(gpoints1)
    M2: int = len(gpoints2)
    a1: float = float(gpoints1[0])
    a2: float = float(gpoints2[0])
    b1: float = float(gpoints1[M1 - 1])
    b2: float = float(gpoints2[M2 - 1])

    # Grid counts, laid out as an M1 x M2 matrix (equivalent to the
    # column-major reshape of the flat Fortran output vector in R's
    # matrix(out[[9L]], M1, M2))
    gcnts: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M1, M2), dtype=np.float64)

    delta1: float = (b1 - a1) / (M1 - 1)
    delta2: float = (b2 - a2) / (M2 - 1)
    for i in range(n):
        lxi1: float = ((x1[i] - a1) / delta1) + 1
        lxi2: float = ((x2[i] - a2) / delta2) + 1

        # Find the integer part of "lxi1" and "lxi2" (truncation toward
        # zero, as in Fortran INT())
        li1: int = int(lxi1)
        li2: int = int(lxi2)
        rem1: float = lxi1 - li1
        rem2: float = lxi2 - li2

        if li1 >= 1 and li2 >= 1 and li1 < M1 and li2 < M2:
            gcnts[li1 - 1, li2 - 1] = gcnts[li1 - 1, li2 - 1] + (1 - rem1) * (1 - rem2)
            gcnts[li1, li2 - 1] = gcnts[li1, li2 - 1] + rem1 * (1 - rem2)
            gcnts[li1 - 1, li2] = gcnts[li1 - 1, li2] + (1 - rem1) * rem2
            gcnts[li1, li2] = gcnts[li1, li2] + rem1 * rem2

    return gcnts


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    n: int = len(X)
    M: int = len(gpoints)
    trun: int = 1 if truncate else 0
    a: float = float(gpoints[0])
    b: float = float(gpoints[M - 1])

    xcnts: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M, dtype=np.float64)
    ycnts: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M, dtype=np.float64)

    delta: float = (b - a) / (M - 1)
    for i in range(n):
        lxi: float = ((X[i] - a) / delta) + 1

        # Find integer part of "lxi" (truncation toward zero, as in Fortran INT())
        li: int = int(lxi)
        rem: float = lxi - li

        # Correction for right endpoint (not included if li == M)
        if X[i] == b:
            li = M - 1
            rem = 1.0

        if li >= 1 and li < M:
            xcnts[li - 1] = xcnts[li - 1] + (1 - rem)
            xcnts[li] = xcnts[li] + rem
            ycnts[li - 1] = ycnts[li - 1] + (1 - rem) * Y[i]
            ycnts[li] = ycnts[li] + rem * Y[i]

        if li < 1 and trun == 0:
            xcnts[0] = xcnts[0] + 1
            ycnts[0] = ycnts[0] + Y[i]

        if li >= M and trun == 0:
            xcnts[M - 1] = xcnts[M - 1] + 1
            ycnts[M - 1] = ycnts[M - 1] + Y[i]

    return {"xcounts": xcnts, "ycounts": ycnts}


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    _kernel_choices: tuple[str, ...] = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        matches: list[str] = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError(
                f"'kernel' = '{kernel}' is not a valid choice; must be one of {_kernel_choices}."
            )

    x = np.asarray(x, dtype=np.float64)

    ## Rename common variables
    n: int = len(x)
    M: int = gridsize

    ## Set canonical scaling factors
    _kernel_del0: dict[str, float] = {
        "normal": (1.0 / (4.0 * math.pi)) ** (1.0 / 10.0),
        "box": (9.0 / 2.0) ** (1.0 / 5.0),
        "epanech": 15.0 ** (1.0 / 5.0),
        "biweight": 35.0 ** (1.0 / 5.0),
        "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
    }
    del0: float = _kernel_del0[kernel]

    if not isinstance(canonical, bool):
        raise ValueError("'canonical' must be a length-1 logical vector")

    ## Set default bandwidth
    h: float
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * float(np.std(x, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    ## Set kernel support values
    tau: float = 4.0 if kernel == "normal" else 1.0

    if range_x is None:
        range_x = (float(np.min(x)) - tau * h, float(np.max(x)) + tau * h)
    a: float = float(range_x[0])
    b: float = float(range_x[1])

    ## Set up grid points and bin the data
    gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
    gcounts: np.ndarray[Any, np.dtype[np.float64]] = linbin(x, gpoints, truncate)

    ## Compute kernel weights
    delta: float = (b - a) / (h * (M - 1))
    L: int = int(min(math.floor(tau / delta), M))
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec: np.ndarray[Any, np.dtype[np.float64]] = np.arange(0, L + 1, dtype=np.float64)
    kappa: np.ndarray[Any, np.dtype[np.float64]]
    if kernel == "normal":
        kappa = norm.pdf(lvec * delta) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # kernel == "triweight"
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P: int = int(2 ** (math.ceil(math.log(M + L + 1) / math.log(2))))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1, dtype=np.float64), kappa[1:][::-1]])
    tot: float = float(np.sum(kappa)) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M, dtype=np.float64)])
    kappa_fft: np.ndarray[Any, np.dtype[np.complex128]] = np.fft.fft(kappa / tot)
    gcounts_fft: np.ndarray[Any, np.dtype[np.complex128]] = np.fft.fft(gcounts)

    y: np.ndarray[Any, np.dtype[np.float64]] = (
        np.fft.ifft(kappa_fft * gcounts_fft).real
    )[:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float], gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: list[tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    ## Install safeguard against non-positive bandwidths:
    x = np.asarray(x, dtype=np.float64)
    bandwidth_arr: np.ndarray[Any, np.dtype[np.float64]] = np.atleast_1d(
        np.asarray(bandwidth, dtype=np.float64)
    )
    if float(np.min(bandwidth_arr)) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    ## Rename common variables
    n: int = x.shape[0]
    M: np.ndarray[Any, np.dtype[np.int64]] = np.asarray(gridsize, dtype=np.int64)
    h: np.ndarray[Any, np.dtype[np.float64]] = bandwidth_arr
    tau: float = 3.4  # For bivariate normal kernel.

    ## Use same bandwidth in each direction
    ## if only a single bandwidth is given.
    if h.shape[0] == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    ## If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [
            (float(np.min(x[:, 0])) - 1.5 * h[0], float(np.max(x[:, 0])) + 1.5 * h[0]),
            (float(np.min(x[:, 1])) - 1.5 * h[1], float(np.max(x[:, 1])) + 1.5 * h[1]),
        ]

    a: np.ndarray[Any, np.dtype[np.float64]] = np.array(
        [range_x[0][0], range_x[1][0]], dtype=np.float64
    )
    b: np.ndarray[Any, np.dtype[np.float64]] = np.array(
        [range_x[0][1], range_x[1][1]], dtype=np.float64
    )

    ## Set up grid points and bin the data
    gpoints1: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a[0], b[0], int(M[0]))
    gpoints2: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a[1], b[1], int(M[1]))

    gcounts: np.ndarray[Any, np.dtype[np.float64]] = linbin2D(x, gpoints1, gpoints2)

    ## Compute kernel weights
    L: np.ndarray[Any, np.dtype[np.int64]] = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [
        np.zeros((1, 1), dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
    ]
    for id_ in range(2):
        L[id_] = min(
            math.floor(float(tau * h[id_] * (M[id_] - 1) / (b[id_] - a[id_]))),
            int(M[id_]) - 1,
        )
        lvecid: np.ndarray[Any, np.dtype[np.float64]] = np.arange(
            0, int(L[id_]) + 1, dtype=np.float64
        )
        facid: float = float((b[id_] - a[id_]) / (h[id_] * (int(M[id_]) - 1)))
        z: np.ndarray[Any, np.dtype[np.float64]] = norm.pdf(lvecid * facid) / h[id_]
        tot: float = float(np.sum(np.concatenate([z, z[1:][::-1]]))) * facid * h[id_]
        kapid[id_] = (z / tot).reshape(-1, 1)

    kapp: np.ndarray[Any, np.dtype[np.float64]] = (kapid[0] @ kapid[1].T) / n

    if int(np.min(L)) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    ## Now combine weight and counts using the FFT to obtain estimate
    P: np.ndarray[Any, np.dtype[np.int64]] = (
        2.0 ** np.ceil(np.log((M + L).astype(np.float64)) / np.log(2.0))
    ).astype(np.int64)  # smallest powers of 2 >= M+L
    L1: int = int(L[0])
    L2: int = int(L[1])
    M1: int = int(M[0])
    M2: int = int(M[1])
    P1: int = int(P[0])
    P2: int = int(P[1])

    rp: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((P1, P2), dtype=np.float64)
    rp[0 : L1 + 1, 0 : L2 + 1] = kapp
    if L1:
        rp[P1 - L1 : P1, 0 : L2 + 1] = kapp[L1:0:-1, 0 : L2 + 1]
    if L2:
        rp[:, P2 - L2 : P2] = rp[:, L2:0:-1]
    ## wrap-around version of "kapp"

    sp: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    ## zero-padded version of "gcounts"

    rp_fft: np.ndarray[Any, np.dtype[np.complex128]] = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft: np.ndarray[Any, np.dtype[np.complex128]] = np.fft.fft2(sp)
    rp = np.fft.ifft2(rp_fft * sp_fft).real[0:M1, 0:M2]
    ## invert element-wise product of FFT's
    ## and truncate and normalise it

    ## Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        x = np.asarray(x, dtype=np.float64)
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename variables
    M: int = gridsize
    a: float = float(range_x[0])
    b: float = float(range_x[1])
    h: float = float(bandwidth)

    # Bin the data if not already binned
    if not binned:
        gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, gridsize)
        gcounts: np.ndarray[Any, np.dtype[np.float64]] = linbin(x, gpoints, truncate)
    else:
        gcounts = np.asarray(x, dtype=np.float64)
        M = len(gcounts)
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width
    n: float = float(np.sum(gcounts))
    delta: float = (b - a) / (M - 1)

    # Obtain kernel weights
    tau: float = 4 + drv
    L: int = int(min(math.floor(tau * h / delta), M))

    if L == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    lvec: np.ndarray[Any, np.dtype[np.float64]] = np.arange(0, L + 1, dtype=np.float64)
    arg: np.ndarray[Any, np.dtype[np.float64]] = lvec * delta / h

    kappam: np.ndarray[Any, np.dtype[np.float64]] = norm.pdf(arg) / (h ** (drv + 1))
    hmold0: float | np.ndarray[Any, np.dtype[np.float64]] = 1.0
    hmold1: np.ndarray[Any, np.dtype[np.float64]] = arg.copy()
    hmnew: float | np.ndarray[Any, np.dtype[np.float64]] = 1.0
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1        # Compute mth degree Hermite polynomial
            hmold1 = hmnew         # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P: int = int(2 ** (math.ceil(math.log(M + L + 1) / math.log(2))))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1, dtype=np.float64), kappam[1:][::-1]])
    Gcounts: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate([gcounts, np.zeros(P - M, dtype=np.float64)])
    kappam_fft: np.ndarray[Any, np.dtype[np.complex128]] = np.fft.fft(kappam)
    Gcounts_fft: np.ndarray[Any, np.dtype[np.complex128]] = np.fft.fft(Gcounts)

    result: float = float(np.sum(gcounts * (np.fft.ifft(kappam_fft * Gcounts_fft).real)[:M]) / (n ** 2))
    return result


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n: int = len(x)

    # Sort the (x, y) data with respect to the x's.
    datmat: np.ndarray[Any, np.dtype[np.float64]] = np.column_stack((x, y))
    sort_idx: np.ndarray[Any, np.dtype[np.int64]] = np.argsort(datmat[:, 0])
    datmat = datmat[sort_idx, :]
    x = datmat[:, 0]
    y = datmat[:, 1]

    # Set up dimensions for the blocked polynomial fits
    qq: int = q + 1

    RSS: float = 0.0
    th22e: float = 0.0
    th24e: float = 0.0

    idiv: int = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition
        ilow: int = (j - 1) * idiv + 1
        iupp: int = j * idiv
        if j == Nval:
            iupp = n
        nj: int = iupp - ilow + 1

        xj: np.ndarray[Any, np.dtype[np.float64]] = x[ilow - 1:iupp]
        yj: np.ndarray[Any, np.dtype[np.float64]] = y[ilow - 1:iupp]

        # Obtain a q'th degree fit over the current member of the partition.
        # Set up the design ("X") matrix: column k (0-based) holds x^k.
        Xmat: np.ndarray[Any, np.dtype[np.float64]] = np.empty((nj, qq), dtype=np.float64)
        Xmat[:, 0] = 1.0
        for k in range(2, qq + 1):
            Xmat[:, k - 1] = xj ** (k - 1)

        # Least-squares solution via QR decomposition (equivalent to the
        # LINPACK dqrdc/dqrsl calls used by the original Fortran routine).
        coef, _, _, _ = np.linalg.lstsq(Xmat, yj, rcond=None)

        for i in range(nj):
            fiti: float = coef[0]
            ddm: float = 2 * coef[2]
            ddddm: float = 24 * coef[4]
            for k in range(2, qq + 1):
                fiti = fiti + coef[k - 1] * xj[i] ** (k - 1)
                if k <= (q - 1):
                    ddm = ddm + k * (k + 1) * coef[k + 1] * xj[i] ** (k - 1)
                    if k <= (q - 3):
                        ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef[k + 3] * xj[i] ** (k - 1)
            th22e = th22e + ddm ** 2
            th24e = th24e + ddm * ddddm
            RSS = RSS + (yj[i] - fiti) ** 2

    sigsqe: float = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": sigsqe, "th22e": th22e, "th24e": th24e}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n: int = len(X)

    # Sort the (X, Y) data with respect to the X's.
    datmat: np.ndarray[Any, np.dtype[np.float64]] = np.column_stack((X, Y))
    sort_idx: np.ndarray[Any, np.dtype[np.int64]] = np.argsort(datmat[:, 0])
    datmat = datmat[sort_idx, :]
    X = datmat[:, 0]
    Y = datmat[:, 1]

    # Set up dimensions for the blocked polynomial fits.
    qq: int = q + 1

    RSS: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # For each number of partitions
        idiv: int = n // Nval
        RSS_Nval: float = 0.0

        for j in range(1, Nval + 1):
            # For each member of the partition
            ilow: int = (j - 1) * idiv + 1
            iupp: int = j * idiv
            if j == Nval:
                iupp = n
            nj: int = iupp - ilow + 1

            Xj: np.ndarray[Any, np.dtype[np.float64]] = X[ilow - 1:iupp]
            Yj: np.ndarray[Any, np.dtype[np.float64]] = Y[ilow - 1:iupp]

            # Obtain a q'th degree fit over the current member of the partition.
            # Set up the design ("X") matrix: column k (0-based) holds x^k.
            Xmat: np.ndarray[Any, np.dtype[np.float64]] = np.empty((nj, qq), dtype=np.float64)
            Xmat[:, 0] = 1.0
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            # Least-squares solution via QR decomposition (equivalent to the
            # LINPACK dqrdc/dqrsl calls used by the original Fortran routine).
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            RSSj: float = 0.0
            for i in range(nj):
                fiti: float = coef[0]
                for k in range(2, qq + 1):
                    fiti = fiti + coef[k - 1] * Xj[i] ** (k - 1)
                RSSj = RSSj + (Yj[i] - fiti) ** 2

            RSS_Nval = RSS_Nval + RSSj

        RSS[Nval - 1] = RSS_Nval

    # Now compute array of Mallow's C_p values.
    Cpvals: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    # order(Cpvec)[1L] in R returns the 1-based index of the minimum C_p value,
    # which is used directly as the chosen number of blocks.
    return int(np.argmin(Cpvals)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    ## Rename variables

    n: int = len(x)
    M: int = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a: float = float(range_x[0])
    b: float = float(range_x[1])

    ## Set up grid points and bin the data

    gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
    gcounts: np.ndarray[Any, np.dtype[np.float64]] = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    _scalest_choices: tuple[str, ...] = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        matches: list[str] = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError(
                f"'scalest' = '{scalest}' is not a valid choice; must be one of {_scalest_choices}."
            )

    scalest_value: float
    if scalest == "stdev":
        scalest_value = float(np.sqrt(np.var(x, ddof=1)))
    elif scalest == "iqr":
        scalest_value = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    else:  # "minim"
        scalest_value = float(
            min(
                (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
                np.sqrt(np.var(x, ddof=1)),
            )
        )

    if scalest_value == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean: float = float(np.mean(x))
    sx: np.ndarray[Any, np.dtype[np.float64]] = (x - x_mean) / scalest_value
    sa: float = (a - x_mean) / scalest_value
    sb: float = (b - x_mean) / scalest_value

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ##    delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps

    hpi: float
    if level == 0:
        hpi = (24.0 * math.sqrt(math.pi) / n) ** (1.0 / 3.0)
    elif level == 1:
        alpha: float = (2.0 / (3.0 * n)) ** (1.0 / 5.0) * math.sqrt(2.0)  # bandwidth for psi_2
        psi2hat: float = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 2:
        alpha = ((2.0 / (5.0 * n)) ** (1.0 / 7.0)) * math.sqrt(2.0)  # bandwidth for psi_4
        psi4hat: float = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2.0 / math.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 3:
        alpha = ((2.0 / (7.0 * n)) ** (1.0 / 9.0)) * math.sqrt(2.0)  # bandwidth for psi_6
        psi6hat: float = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2.0 / math.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 4:
        alpha = ((2.0 / (9.0 * n)) ** (1.0 / 11.0)) * math.sqrt(2.0)  # bandwidth for psi_8
        psi8hat: float = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2.0 / math.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    else:  # level == 5
        alpha = ((2.0 / (11.0 * n)) ** (1.0 / 13.0)) * math.sqrt(2.0)  # bandwidth for psi_10
        psi10hat: float = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * math.sqrt(2.0 / math.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2.0 / math.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)

    return scalest_value * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    ## Resolve 'kernel' argument (partial matching as in R's match.arg)

    _kernel_choices: tuple[str, ...] = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        kernel_matches: list[str] = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(kernel_matches) == 1:
            kernel = kernel_matches[0]
        else:
            raise ValueError(
                f"'kernel' = '{kernel}' is not a valid choice; must be one of {_kernel_choices}."
            )

    ## Set kernel constants

    del0: float
    if canonical:
        del0 = 1.0
    elif kernel == "normal":
        del0 = 1.0 / ((4.0 * math.pi) ** (1.0 / 10.0))
    elif kernel == "box":
        del0 = (9.0 / 2.0) ** (1.0 / 5.0)
    elif kernel == "epanech":
        del0 = 15.0 ** (1.0 / 5.0)
    elif kernel == "biweight":
        del0 = 35.0 ** (1.0 / 5.0)
    else:  # "triweight"
        del0 = (9450.0 / 143.0) ** (1.0 / 5.0)

    ## Rename variables

    n: int = len(x)
    M: int = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a: float = float(range_x[0])
    b: float = float(range_x[1])

    ## Set up grid points and bin the data

    gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
    gcounts: np.ndarray[Any, np.dtype[np.float64]] = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    _scalest_choices: tuple[str, ...] = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        matches: list[str] = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError(
                f"'scalest' = '{scalest}' is not a valid choice; must be one of {_scalest_choices}."
            )

    scalest_value: float
    if scalest == "stdev":
        scalest_value = float(np.sqrt(np.var(x, ddof=1)))
    elif scalest == "iqr":
        scalest_value = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    else:  # "minim"
        scalest_value = float(
            min(
                (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
                np.sqrt(np.var(x, ddof=1)),
            )
        )

    if scalest_value == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean: float = float(np.mean(x))
    sx: np.ndarray[Any, np.dtype[np.float64]] = (x - x_mean) / scalest_value
    sa: float = (a - x_mean) / scalest_value
    sb: float = (b - x_mean) / scalest_value

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ##    delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps:

    psi4hat: float
    if level == 0:
        psi4hat = 3.0 / (8.0 * math.sqrt(math.pi))
    elif level == 1:
        alpha: float = (2.0 * (math.sqrt(2.0)) ** 7 / (5.0 * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2.0 * (math.sqrt(2.0)) ** 9 / (7.0 * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat: float = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2.0 * (math.sqrt(2.0)) ** 11 / (9.0 * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat: float = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2.0 * (math.sqrt(2.0)) ** 13 / (11.0 * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat: float = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * math.sqrt(2.0 / math.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:  # level == 5
        alpha = (2.0 * (math.sqrt(2.0)) ** 15 / (13.0 * n)) ** (1.0 / 15.0)  # bandwidth for psi_12
        psi12hat: float = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = (945.0 * math.sqrt(2.0 / math.pi) / (psi12hat * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * math.sqrt(2.0 / math.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return scalest_value * del0 * (1.0 / (psi4hat * n)) ** (1.0 / 5.0)


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # range.x defaults to range(x) computed on the *original*, untrimmed x.
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    datmat: np.ndarray[Any, np.dtype[np.float64]] = np.column_stack((x, y))
    sort_idx: np.ndarray[Any, np.dtype[np.int64]] = np.argsort(datmat[:, 0])
    datmat = datmat[sort_idx, :]
    x = datmat[:, 0]
    y = datmat[:, 1]

    indlow: int = math.floor(trim * len(x)) + 1
    indupp: int = len(x) - math.floor(trim * len(x))

    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    ## Rename common parameters
    n: int = len(x)
    M: int = gridsize
    a: float = float(range_x[0])
    b: float = float(range_x[1])

    ## Bin the data
    gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts: np.ndarray[Any, np.dtype[np.float64]] = out["xcounts"]
    ycounts: np.ndarray[Any, np.dtype[np.float64]] = out["ycounts"]

    ## Choose the value of N using Mallow's C_p
    Nmax: int = max(min(math.floor(n / divisor), blockmax), 1)
    Nval: int = cpblock(x, y, Nmax, 4)

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.
    blk_out = blkest(x, y, Nval, 4)
    sigsqQ: float = blk_out["sigsqe"]
    th24Q: float = blk_out["th24e"]

    ## Estimate theta_22 using a local cubic fit
    ## with a "rule-of-thumb" bandwidth: "gamseh"
    gamseh: float = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1 / 7)

    mddest: np.ndarray[Any, np.dtype[np.float64]] = locpoly(
        xcounts, ycounts, drv=2, bandwidth=gamseh, range_x=range_x, binned=True
    )["y"]

    llow: int = math.floor(proptrun * M) + 1
    lupp: int = M - math.floor(proptrun * M)
    th22kn: float = float(
        np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n
    )

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    C3K: float = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / math.sqrt(2 * math.pi)) ** (1 / 9)
    lamseh: float = C3K * (
        ((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9)
    )

    ## Now compute a local linear kernel estimate of
    ## the variance.
    mest: np.ndarray[Any, np.dtype[np.float64]] = locpoly(
        xcounts, ycounts, bandwidth=lamseh, range_x=range_x, binned=True
    )["y"]
    Sdg: np.ndarray[Any, np.dtype[np.float64]] = sdiag(
        xcounts, bandwidth=lamseh, range_x=range_x, binned=True
    )["y"]
    SSTdg: np.ndarray[Any, np.dtype[np.float64]] = sstdiag(
        xcounts, bandwidth=lamseh, range_x=range_x, binned=True
    )["y"]
    sigsqn: float = float(
        np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    )
    sigsqd: float = float(
        n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    )
    sigsqkn: float = sigsqn / sigsqd

    ## Combine to obtain final answer.
    return (sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1 / 5)


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.any(np.asarray(bandwidth, dtype=np.float64) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        if y is None:
            extra: float = 0.05 * (float(np.max(x)) - float(np.min(x)))
            range_x = (float(np.min(x)) - extra, float(np.max(x)) + extra)
        else:
            range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M: int = int(gridsize)
    Q: int = int(bwdisc)
    a: float = float(range_x[0])
    b: float = float(range_x[1])
    pp: int = degree + 1
    ppp: int = 2 * degree + 1
    tau: float = 4.0

    # Decide whether a density estimate or regression estimate is required.
    if y is None:
        # obtain density estimate
        n: int = len(x)
        gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
        xcounts: np.ndarray[Any, np.dtype[np.float64]] = linbin(x, gpoints, truncate)
        ycounts: np.ndarray[Any, np.dtype[np.float64]] = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:
        # obtain regression estimate
        y = np.asarray(y, dtype=np.float64)
        if not binned:
            gpoints = np.linspace(a, b, M)
            out: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = rlbin(x, y, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = x
            ycounts = y
            M = len(xcounts)
            gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta: float = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr: np.ndarray[Any, np.dtype[np.float64]] = np.atleast_1d(
        np.asarray(bandwidth, dtype=np.float64)
    )

    if len(bandwidth_arr) == M:
        sorted_bw: np.ndarray[Any, np.dtype[np.float64]] = np.sort(bandwidth_arr)
        hlow: float = float(sorted_bw[0])
        hupp: float = float(sorted_bw[M - 1])
        hdisc: np.ndarray[Any, np.dtype[np.float64]] = np.exp(
            np.linspace(math.log(hlow), math.log(hupp), Q)
        )

        # Determine value of L for each member of "hdisc"
        Lvec: np.ndarray[Any, np.dtype[np.int64]] = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc" to each member of "bandwidth"
        if Q > 1:
            lhdisc: np.ndarray[Any, np.dtype[np.float64]] = np.log(hdisc)
            gap: float = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic: np.ndarray[Any, np.dtype[np.int64]] = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - math.log(hlow)) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * float(bandwidth_arr[0]) / delta), dtype=np.int64)
        hdisc = np.full(Q, float(bandwidth_arr[0]), dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if int(np.min(Lvec)) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Allocate space for the kernel vector/final estimate and call the local
    # polynomial kernel-weighted least-squares core (translated line-for-line
    # from the FORTRAN subroutine "locpol" in KernSmooth/src/locpoly.f).
    # Arrays below are padded with an unused index 0 so that the indexing
    # exactly mirrors the FORTRAN (1-based) source.
    def _locpol(
        xcnts: np.ndarray[Any, np.dtype[np.float64]],
        ycnts: np.ndarray[Any, np.dtype[np.float64]],
        idrv: int,
        delta_: float,
        hdisc_: np.ndarray[Any, np.dtype[np.float64]],
        Lvec_: np.ndarray[Any, np.dtype[np.int64]],
        indic_: np.ndarray[Any, np.dtype[np.int64]],
        M_: int,
        iQ: int,
        ipp: int,
        ippp: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        Lvec1: np.ndarray[Any, np.dtype[np.int64]] = np.concatenate(([0], Lvec_))
        hdisc1: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate(([0.0], hdisc_))
        indic1: np.ndarray[Any, np.dtype[np.int64]] = np.concatenate(([0], indic_))
        xcnts1: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate(([0.0], xcnts))
        ycnts1: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate(([0.0], ycnts))

        dimfkap: int = 2 * int(np.sum(Lvec_)) + iQ
        fkap: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(dimfkap + 1, dtype=np.float64)
        midpts: np.ndarray[Any, np.dtype[np.int64]] = np.zeros(iQ + 1, dtype=np.int64)
        ss: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M_ + 1, ippp + 1), dtype=np.float64)
        tt: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M_ + 1, ipp + 1), dtype=np.float64)
        cvest: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M_ + 1, dtype=np.float64)

        # Obtain kernel weights
        mid: int = int(Lvec1[1]) + 1
        for i in range(1, iQ):
            midpts[i] = mid
            fkap[mid] = 1.0
            for j in range(1, int(Lvec1[i]) + 1):
                fkap[mid + j] = math.exp(-((delta_ * j / hdisc1[i]) ** 2) / 2)
                fkap[mid - j] = fkap[mid + j]
            mid = mid + int(Lvec1[i]) + int(Lvec1[i + 1]) + 1
        midpts[iQ] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec1[iQ]) + 1):
            fkap[mid + j] = math.exp(-((delta_ * j / hdisc1[iQ]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]

        # Combine kernel weights and grid counts
        for k in range(1, M_ + 1):
            if xcnts1[k] != 0:
                for i in range(1, iQ + 1):
                    lo: int = max(1, k - int(Lvec1[i]))
                    hi: int = min(M_, k + int(Lvec1[i]))
                    for j in range(lo, hi + 1):
                        if indic1[j] == i:
                            fac: float = 1.0
                            weight: float = fkap[k - j + midpts[i]]
                            ss[j, 1] += xcnts1[k] * weight
                            tt[j, 1] += ycnts1[k] * weight
                            for ii in range(2, ippp + 1):
                                fac = fac * delta_ * (k - j)
                                ss[j, ii] += xcnts1[k] * weight * fac
                                if ii <= ipp:
                                    tt[j, ii] += ycnts1[k] * weight * fac

        # Solve the (ipp x ipp) weighted least-squares system at each grid point
        for k in range(1, M_ + 1):
            Smat: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((ipp, ipp), dtype=np.float64)
            Tvec: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(ipp, dtype=np.float64)
            for i in range(1, ipp + 1):
                for j in range(1, ipp + 1):
                    indss: int = i + j - 1
                    Smat[i - 1, j - 1] = ss[k, indss]
                Tvec[i - 1] = tt[k, i]

            Tsol: np.ndarray[Any, np.dtype[np.float64]] = np.linalg.solve(Smat, Tvec)
            cvest[k] = Tsol[idrv]

        return cvest[1 : M_ + 1]

    curvest: np.ndarray[Any, np.dtype[np.float64]] = _locpol(
        xcounts, ycounts, drv, delta, hdisc, Lvec, indic, M, Q, pp, ppp
    )

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # NOTE: 'drv' and 'kernel' are accepted for signature parity with the
    # original R function, but (as in the R source) they are never actually
    # used/passed on to the underlying Fortran computation.
    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M: int = int(gridsize)
    Q: int = int(bwdisc)
    a: float = float(range_x[0])
    b: float = float(range_x[1])
    pp: int = int(degree) + 1
    ppp: int = 2 * int(degree) + 1
    tau: float = 4.0

    # Bin the data if not already binned
    if not binned:
        gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
        xcounts: np.ndarray[Any, np.dtype[np.float64]] = linbin(x, gpoints, truncate)
    else:
        xcounts = x
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta: float = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr: np.ndarray[Any, np.dtype[np.float64]] = np.atleast_1d(
        np.asarray(bandwidth, dtype=np.float64)
    )

    if len(bandwidth_arr) == M:
        sorted_bw: np.ndarray[Any, np.dtype[np.float64]] = np.sort(bandwidth_arr)
        hlow: float = float(sorted_bw[0])
        hupp: float = float(sorted_bw[M - 1])
        hdisc: np.ndarray[Any, np.dtype[np.float64]] = np.exp(
            np.linspace(math.log(hlow), math.log(hupp), Q)
        )

        # Determine value of L for each member of "hdisc"
        Lvec: np.ndarray[Any, np.dtype[np.int64]] = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc: np.ndarray[Any, np.dtype[np.float64]] = np.log(hdisc)
            gap: float = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic: np.ndarray[Any, np.dtype[np.int64]] = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - math.log(hlow)) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * float(bandwidth_arr[0]) / delta), dtype=np.int64)
        hdisc = np.full(Q, float(bandwidth_arr[0]), dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Compute the diagonal entries of the binned smoother matrix. This is a
    # line-for-line translation of the FORTRAN subroutine "sdiag" in
    # KernSmooth/src/sdiag.f. Arrays below are padded with an unused index 0
    # so that the indexing exactly mirrors the FORTRAN (1-based) source.
    def _sdiag_core(
        xcnts: np.ndarray[Any, np.dtype[np.float64]],
        delta_: float,
        hdisc_: np.ndarray[Any, np.dtype[np.float64]],
        Lvec_: np.ndarray[Any, np.dtype[np.int64]],
        indic_: np.ndarray[Any, np.dtype[np.int64]],
        M_: int,
        iQ: int,
        ipp: int,
        ippp: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        Lvec1: np.ndarray[Any, np.dtype[np.int64]] = np.concatenate(([0], Lvec_))
        hdisc1: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate(([0.0], hdisc_))
        indic1: np.ndarray[Any, np.dtype[np.int64]] = np.concatenate(([0], indic_))
        xcnts1: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate(([0.0], xcnts))

        dimfkap: int = 2 * int(np.sum(Lvec_)) + iQ
        fkap: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(dimfkap + 1, dtype=np.float64)
        midpts: np.ndarray[Any, np.dtype[np.int64]] = np.zeros(iQ + 1, dtype=np.int64)
        ss: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M_ + 1, ippp + 1), dtype=np.float64)
        Sdg: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M_ + 1, dtype=np.float64)

        # Obtain kernel weights
        mid: int = int(Lvec1[1]) + 1
        for i in range(1, iQ):
            midpts[i] = mid
            fkap[mid] = 1.0
            for j in range(1, int(Lvec1[i]) + 1):
                fkap[mid + j] = math.exp(-((delta_ * j / hdisc1[i]) ** 2) / 2)
                fkap[mid - j] = fkap[mid + j]
            mid = mid + int(Lvec1[i]) + int(Lvec1[i + 1]) + 1
        midpts[iQ] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec1[iQ]) + 1):
            fkap[mid + j] = math.exp(-((delta_ * j / hdisc1[iQ]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]

        # Combine kernel weights and grid counts
        for k in range(1, M_ + 1):
            if xcnts1[k] != 0:
                for i in range(1, iQ + 1):
                    lo: int = max(1, k - int(Lvec1[i]))
                    hi: int = min(M_, k + int(Lvec1[i]))
                    for j in range(lo, hi + 1):
                        if indic1[j] == i:
                            fac: float = 1.0
                            weight: float = fkap[k - j + midpts[i]]
                            ss[j, 1] += xcnts1[k] * weight
                            for ii in range(2, ippp + 1):
                                fac = fac * delta_ * (k - j)
                                ss[j, ii] += xcnts1[k] * weight * fac

        # Invert the (ipp x ipp) system at each grid point and take the
        # (1,1) entry of the inverse -- this is the value that the local
        # weighted least-squares fit places on the grid point's own
        # observation, i.e. the diagonal entry of the smoother matrix.
        for k in range(1, M_ + 1):
            Smat: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((ipp, ipp), dtype=np.float64)
            for i in range(1, ipp + 1):
                for j in range(1, ipp + 1):
                    indss: int = i + j - 1
                    Smat[i - 1, j - 1] = ss[k, indss]

            Smat_inv: np.ndarray[Any, np.dtype[np.float64]] = np.linalg.inv(Smat)
            Sdg[k] = Smat_inv[0, 0]

        return Sdg[1 : M_ + 1]

    sdg: np.ndarray[Any, np.dtype[np.float64]] = _sdiag_core(
        xcounts, delta, hdisc, Lvec, indic, M, Q, pp, ppp
    )

    return {"x": gpoints, "y": sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # NOTE: 'drv' and 'kernel' are accepted for signature parity with the
    # original R function, but (as in the R source) they are never actually
    # used/passed on to the underlying Fortran computation.
    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M: int = int(gridsize)
    Q: int = int(bwdisc)
    a: float = float(range_x[0])
    b: float = float(range_x[1])
    pp: int = int(degree) + 1
    ppp: int = 2 * int(degree) + 1
    tau: float = 4.0

    # Bin the data if not already binned
    if not binned:
        gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
        xcounts: np.ndarray[Any, np.dtype[np.float64]] = linbin(x, gpoints, truncate)
    else:
        xcounts = x
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta: float = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr: np.ndarray[Any, np.dtype[np.float64]] = np.atleast_1d(
        np.asarray(bandwidth, dtype=np.float64)
    )

    if len(bandwidth_arr) == M:
        sorted_bw: np.ndarray[Any, np.dtype[np.float64]] = np.sort(bandwidth_arr)
        hlow: float = float(sorted_bw[0])
        hupp: float = float(sorted_bw[M - 1])
        hdisc: np.ndarray[Any, np.dtype[np.float64]] = np.exp(
            np.linspace(math.log(hlow), math.log(hupp), Q)
        )

        # Determine value of L for each member of "hdisc"
        Lvec: np.ndarray[Any, np.dtype[np.int64]] = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc: np.ndarray[Any, np.dtype[np.float64]] = np.log(hdisc)
            gap: float = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic: np.ndarray[Any, np.dtype[np.int64]] = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - math.log(hlow)) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * float(bandwidth_arr[0]) / delta), dtype=np.int64)
        hdisc = np.full(Q, float(bandwidth_arr[0]), dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Compute the diagonal entries of the binned SS^T matrix, where S is the
    # smoother matrix for local polynomial fitting. This is a line-for-line
    # translation of the FORTRAN subroutine "sstdg" in
    # KernSmooth/src/sstdiag.f. Arrays below are padded with an unused index
    # 0 so that the indexing exactly mirrors the FORTRAN (1-based) source.
    # In addition to the "ss"/"Smat" accumulation used by "sdiag", an
    # extra "uu"/"Umat" accumulation of squared kernel weights is built up
    # and combined with the inverse of "Smat" to give diag(S S^T).
    def _sstdiag_core(
        xcnts: np.ndarray[Any, np.dtype[np.float64]],
        delta_: float,
        hdisc_: np.ndarray[Any, np.dtype[np.float64]],
        Lvec_: np.ndarray[Any, np.dtype[np.int64]],
        indic_: np.ndarray[Any, np.dtype[np.int64]],
        M_: int,
        iQ: int,
        ipp: int,
        ippp: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        Lvec1: np.ndarray[Any, np.dtype[np.int64]] = np.concatenate(([0], Lvec_))
        hdisc1: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate(([0.0], hdisc_))
        indic1: np.ndarray[Any, np.dtype[np.int64]] = np.concatenate(([0], indic_))
        xcnts1: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate(([0.0], xcnts))

        dimfkap: int = 2 * int(np.sum(Lvec_)) + iQ
        fkap: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(dimfkap + 1, dtype=np.float64)
        midpts: np.ndarray[Any, np.dtype[np.int64]] = np.zeros(iQ + 1, dtype=np.int64)
        ss: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M_ + 1, ippp + 1), dtype=np.float64)
        uu: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M_ + 1, ippp + 1), dtype=np.float64)
        SSTd: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M_ + 1, dtype=np.float64)

        # Obtain kernel weights
        mid: int = int(Lvec1[1]) + 1
        for i in range(1, iQ):
            midpts[i] = mid
            fkap[mid] = 1.0
            for j in range(1, int(Lvec1[i]) + 1):
                fkap[mid + j] = math.exp(-((delta_ * j / hdisc1[i]) ** 2) / 2)
                fkap[mid - j] = fkap[mid + j]
            mid = mid + int(Lvec1[i]) + int(Lvec1[i + 1]) + 1
        midpts[iQ] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec1[iQ]) + 1):
            fkap[mid + j] = math.exp(-((delta_ * j / hdisc1[iQ]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]

        # Combine kernel weights and grid counts
        for k in range(1, M_ + 1):
            if xcnts1[k] != 0:
                for i in range(1, iQ + 1):
                    lo: int = max(1, k - int(Lvec1[i]))
                    hi: int = min(M_, k + int(Lvec1[i]))
                    for j in range(lo, hi + 1):
                        if indic1[j] == i:
                            fac: float = 1.0
                            weight: float = fkap[k - j + midpts[i]]
                            ss[j, 1] += xcnts1[k] * weight
                            uu[j, 1] += xcnts1[k] * (weight ** 2)
                            for ii in range(2, ippp + 1):
                                fac = fac * delta_ * (k - j)
                                ss[j, ii] += xcnts1[k] * weight * fac
                                uu[j, ii] += xcnts1[k] * (weight ** 2) * fac

        # Invert the (ipp x ipp) system at each grid point and combine the
        # first row/column of the inverse with "Umat" to obtain the k-th
        # diagonal entry of S S^T.
        for k in range(1, M_ + 1):
            Smat: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((ipp, ipp), dtype=np.float64)
            Umat: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((ipp, ipp), dtype=np.float64)
            for i in range(1, ipp + 1):
                for j in range(1, ipp + 1):
                    indss: int = i + j - 1
                    Smat[i - 1, j - 1] = ss[k, indss]
                    Umat[i - 1, j - 1] = uu[k, indss]

            Smat_inv: np.ndarray[Any, np.dtype[np.float64]] = np.linalg.inv(Smat)

            sstd_k: float = 0.0
            for i in range(1, ipp + 1):
                for j in range(1, ipp + 1):
                    sstd_k += Smat_inv[0, i - 1] * Umat[i - 1, j - 1] * Smat_inv[j - 1, 0]
            SSTd[k] = sstd_k

        return SSTd[1 : M_ + 1]

    sstd: np.ndarray[Any, np.dtype[np.float64]] = _sstdiag_core(
        xcounts, delta, hdisc, Lvec, indic, M, Q, pp, ppp
    )

    return {"x": gpoints, "y": sstd}


def on_attach(libname: str, pkgname: str) -> None:
    # Mirrors R's .onAttach package hook, which calls packageStartupMessage().
    # packageStartupMessage() writes an informational message to stderr, so we
    # replicate that here with print(..., file=sys.stderr).
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)
    return None


def on_unload(libpath: str) -> None:
    # R's .onUnload hook calls library.dynam.unload("KernSmooth", libpath) to
    # unload the compiled shared library when the package namespace is detached.
    # CPython does not provide a safe, supported mechanism for unloading a
    # compiled extension module (dlclose on a still-referenced extension can
    # segfault), so there is no direct Python equivalent. Per the conversion
    # guide, this is intentionally translated as a no-op.
    pass
