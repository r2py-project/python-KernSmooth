import math
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import beta, norm

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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: Optional[float] = None, gridsize: int = 401, range_x: Optional[Tuple[float, float]] = None, truncate: bool = True) -> Dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    ## Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if kernel not in ("normal", "box", "epanech", "biweight", "triweight"):
        raise ValueError(
            "'kernel' should be one of \"normal\", \"box\", \"epanech\", \"biweight\", \"triweight\""
        )

    x = np.asarray(x, dtype=np.float64)

    ## Rename common variables
    n = len(x)
    M = gridsize

    ## Set canonical scaling factors
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
        raise ValueError("'canonical' must be a length-1 logical vector")

    ## Set default bandwidth
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.sqrt(np.var(x, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    ## Set kernel support values
    tau = 4 if kernel == "normal" else 1

    if range_x is None:
        range_x = (float(np.min(x) - tau * h), float(np.max(x) + tau * h))
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = int(min(np.floor(tau / delta), M))
    if L == 0:
        import warnings
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    if kernel == "normal":
        kappa = np.exp(-0.5 * (lvec * delta) ** 2) / np.sqrt(2.0 * np.pi) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    return {"x": gpoints, "y": np.real(np.fft.ifft(kappa * gcounts))[:M]}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: Optional[Union[float, Sequence[float]]] = None, gridsize: Tuple[int, int] = (51, 51), range_x: Optional[List[Tuple[float, float]]] = None, truncate: bool = True) -> Dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    ## Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x = np.asarray(x, dtype=np.float64)

    ## Rename common variables
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    ## Use same bandwidth in each direction
    ## if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([h[0], h[0]])

    ## If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [
            (float(np.min(x[:, 0]) - 1.5 * h[0]), float(np.max(x[:, 0]) + 1.5 * h[0])),
            (float(np.min(x[:, 1]) - 1.5 * h[1]), float(np.max(x[:, 1]) + 1.5 * h[1])),
        ]

    a = np.array([range_x[0][0], range_x[1][0]])
    b = np.array([range_x[0][1], range_x[1][1]])

    ## Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    ## Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: List[np.ndarray[Any, np.dtype[np.float64]]] = [
        np.zeros((1, 1)),
        np.zeros((1, 1)),
    ]
    for id_ in range(2):
        L[id_] = int(
            min(np.floor(tau * h[id_] * (M[id_] - 1) / (b[id_] - a[id_])), M[id_] - 1)
        )
        lvecid = np.arange(0, L[id_] + 1)
        facid = (b[id_] - a[id_]) / (h[id_] * (M[id_] - 1))
        z = (norm.pdf(lvecid * facid) / h[id_]).reshape(-1, 1)
        tot = np.sum(np.concatenate([z.flatten(), z.flatten()[1:][::-1]])) * facid * h[id_]
        kapid[id_] = z / tot

    kapp = kapid[0] @ kapid[1].T / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    ## Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log(M + L) / np.log(2))).astype(np.int64)  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0 : L1 + 1, 0 : L2 + 1] = kapp
    if L1:
        rp[P1 - L1 : P1, 0 : L2 + 1] = kapp[L1:0:-1, 0 : L2 + 1]
    if L2:
        rp[:, P2 - L2 : P2] = rp[:, L2:0:-1]
    ## wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    ## zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    ## invert element-wise product of FFT's
    ## and truncate and normalise it

    ## Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: Optional[float] = None, gridsize: int = 401, range_x: Optional[Tuple[float, float]] = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

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
    L = int(min(np.floor(tau * h / delta), M))

    if L == 0:
        import warnings
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = np.exp(-0.5 * arg ** 2) / np.sqrt(2.0 * np.pi)
    kappam = kappam / (h ** (drv + 1))
    hmold0 = np.ones_like(arg)
    hmold1 = arg.copy()
    hmnew = np.ones_like(arg)
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1        # Compute mth degree Hermite polynomial
            hmold1 = hmnew         # by recurrence.
    kappam = hmnew * kappam

    ## Now combine weights and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    return float(
        np.sum(gcounts * np.real(np.fft.ifft(kappam * Gcounts))[:M]) / (n ** 2)
    )


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    # Sort the (x, y) data with respect to the x's (stable, as in R's sort.list).
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    qq = q + 1

    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    # Equivalent of Fortran subroutine blkest: blocked q'th degree
    # polynomial fits over Nval contiguous partitions of the sorted data.
    idiv = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition (1-based bounds, as in the Fortran source).
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        nj = iupp - ilow + 1

        Xj = x[ilow - 1:iupp]
        Yj = y[ilow - 1:iupp]

        # Obtain a q'th degree fit over the current member of the partition.
        # Xmat columns are 1, Xj, Xj^2, ..., Xj^q (matches the Fortran Xmat setup).
        Xmat = np.vander(Xj, N=qq, increasing=True)

        # Least squares solve, equivalent to the Fortran dqrdc/dqrsl QR fit.
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        for i in range(nj):
            xi = Xj[i]
            fiti = coef[0]
            ddm = 2.0 * coef[2]
            ddddm = 24.0 * coef[4]
            for k in range(2, qq + 1):
                fiti = fiti + coef[k - 1] * xi ** (k - 1)
                if k <= q - 1:
                    ddm = ddm + k * (k + 1) * coef[k + 1] * xi ** (k - 1)
                    if k <= q - 3:
                        ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef[k + 3] * xi ** (k - 1)
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
    n = len(X)

    # Sort the (X, Y) data with respect to the X's (stable, as in R's sort.list).
    order = np.argsort(X, kind="stable")
    X = X[order]
    Y = Y[order]

    qq = q + 1

    # Equivalent of Fortran subroutine cp: for each candidate number of
    # blocks Nval = 1..Nmax, partition the sorted data into Nval contiguous
    # blocks, fit a q'th degree polynomial per block, and accumulate the RSS.
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        idiv = n // Nval
        RSSval = 0.0

        for j in range(1, Nval + 1):
            # 1-based partition bounds, as in the Fortran source.
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n

            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]

            # Xmat columns are 1, Xj, Xj^2, ..., Xj^q (matches the Fortran
            # Xmat setup); solve the q'th degree least-squares fit, equivalent
            # to the Fortran dqrdc/dqrsl QR fit.
            Xmat = np.vander(Xj, N=qq, increasing=True)
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)
            fit = Xmat @ coef
            RSSval += np.sum((Yj - fit) ** 2)

        RSS[Nval - 1] = RSSval

    # Compute the array of Mallow's C_p values.
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = (n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1] + 2 * qq * i - n

    # R's order(Cpvec)[1L] returns the smallest 1-based index achieving the
    # minimum C_p value; np.argmin already returns the first (lowest) index
    # among ties, so adding 1 reproduces R's 1-based result exactly. This
    # 1-based block-count convention is preserved because callers (e.g.
    # dpill) treat the return value directly as a number of blocks.
    return int(np.argmin(Cpvals) + 1)


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: Optional[Tuple[float, float]] = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename variables

    n = len(x)
    M = gridsize
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError("'scalest' should be one of \"minim\", \"stdev\", \"iqr\"")

    if scalest == "stdev":
        scale_val = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scale_val = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    mean_x = np.mean(x)
    sx = (x - mean_x) / scale_val
    sa = (a - mean_x) / scale_val
    sb = (b - mean_x) / scale_val

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
    else:  # level == 5
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

    return float(scale_val * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: Optional[Tuple[float, float]] = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    if kernel not in ("normal", "box", "epanech", "biweight", "triweight"):
        raise ValueError(
            "'kernel' should be one of \"normal\", \"box\", \"epanech\", \"biweight\", \"triweight\""
        )

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
    else:  # "triweight"
        del0 = (9450 / 143) ** (1 / 5)

    x = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename variables

    n = len(x)
    M = gridsize
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError("'scalest' should be one of \"minim\", \"stdev\", \"iqr\"")

    if scalest == "stdev":
        scale_val = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scale_val = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    mean_x = np.mean(x)
    sx = (x - mean_x) / scale_val
    sa = (a - mean_x) / scale_val
    sb = (b - mean_x) / scale_val

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M - 1)

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


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: Optional[Tuple[float, float]] = None, truncate: bool = True) -> float:
    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    indlow = int(np.floor(trim * len(x))) + 1
    indupp = len(x) - int(np.floor(trim * len(x)))

    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    ## Rename common parameters.
    ## Note: R's default `range.x = range(x)` is evaluated lazily, i.e. only
    ## the first time `range.x` is used in the body -- which happens after
    ## the trimming above has already reassigned `x`. Replicate that by
    ## computing the default range from the trimmed `x` here.
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    ## Bin the data.
    gpoints = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    ## Choose the value of N using Mallow's C_p.
    Nmax = max(min(int(np.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x, y, Nmax, 4)

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.
    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    ## Estimate theta_22 using a local cubic fit
    ## with a "rule-of-thumb" bandwidth: "gamseh".
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * np.sqrt(np.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * np.sqrt(np.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=range_x, binned=True)["y"]

    llow = int(np.floor(proptrun * M)) + 1
    lupp = M - int(np.floor(proptrun * M))
    th22kn = np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh".
    C3K = (1 / 2) + 2 * np.sqrt(2) - (4 / 3) * np.sqrt(3)
    C3K = (4 * C3K / np.sqrt(2 * np.pi)) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    ## Now compute a local linear kernel estimate of
    ## the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                    range_x=range_x, binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh,
                range_x=range_x, binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh,
                     range_x=range_x, binned=True)["y"]
    sigsqn = np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    ## Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * np.sqrt(np.pi) * th22kn * n)) ** (1 / 5))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    gcnts = np.zeros(M, dtype=np.float64)

    # Equivalent of Fortran subroutine linbin: linear binning of X onto
    # the M grid points spanning [a, b].
    delta = (b - a) / (M - 1)

    # lxi is the (1-based) fractional grid position of each data point.
    lxi = ((X - a) / delta) + 1.0

    # Fortran's int() truncates toward zero, which matches Python's int()
    # / np.trunc() behaviour (as opposed to np.floor()).
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    in_range = (li >= 1) & (li < M)
    li_in = li[in_range]
    rem_in = rem[in_range]

    # Convert the 1-based Fortran grid indices to 0-based Python indices
    # and accumulate with np.add.at so that repeated indices are summed
    # rather than overwritten.
    idx0 = li_in - 1
    np.add.at(gcnts, idx0, 1.0 - rem_in)
    np.add.at(gcnts, idx0 + 1, rem_in)

    if trun == 0:
        below = li < 1
        gcnts[0] += np.count_nonzero(below)

        above = li >= M
        gcnts[M - 1] += np.count_nonzero(above)

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)

    M1 = len(gpoints1)
    M2 = len(gpoints2)
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Equivalent of Fortran subroutine lbtwod: bivariate linear (bilinear)
    # binning of X onto the M1 x M2 grid spanning [a1, b1] x [a2, b2].
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    x1 = X[:, 0]
    x2 = X[:, 1]

    # lxi1/lxi2 are the (1-based) fractional grid positions of each
    # data point along each dimension.
    lxi1 = ((x1 - a1) / delta1) + 1.0
    lxi2 = ((x2 - a2) / delta2) + 1.0

    # Fortran's int() truncates toward zero, which matches np.trunc().
    li1 = np.trunc(lxi1).astype(np.int64)
    li2 = np.trunc(lxi2).astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    # Observations outside the mesh (in either dimension) are ignored,
    # matching the nested Fortran IF statements.
    in_range = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)

    li1_in = li1[in_range]
    li2_in = li2[in_range]
    rem1_in = rem1[in_range]
    rem2_in = rem2[in_range]

    # Convert the 1-based Fortran grid indices to 0-based Python indices
    # and accumulate with np.add.at so that repeated indices are summed
    # rather than overwritten. Weight is distributed bilinearly across
    # the 4 nearest grid points, matching gcnts(ind1..ind4) in the
    # Fortran source (row = gpoints1 index, col = gpoints2 index).
    row0 = li1_in - 1
    col0 = li2_in - 1

    np.add.at(gcnts, (row0, col0), (1.0 - rem1_in) * (1.0 - rem2_in))
    np.add.at(gcnts, (row0 + 1, col0), rem1_in * (1.0 - rem2_in))
    np.add.at(gcnts, (row0, col0 + 1), (1.0 - rem1_in) * rem2_in)
    np.add.at(gcnts, (row0 + 1, col0 + 1), rem1_in * rem2_in)

    return gcnts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None, drv: int = 0, degree: Optional[int] = None, kernel: str = "normal", bandwidth: Optional[Union[float, np.ndarray[Any, np.dtype[np.float64]]]] = None, gridsize: int = 401, bwdisc: int = 25, range_x: Optional[Tuple[float, float]] = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    ## Install safeguard against non-positive bandwidths:
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
            extra = 0.05 * (np.max(x) - np.min(x))
            range_x = (float(np.min(x) - extra), float(np.max(x) + extra))
        else:
            range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = range_x[0]
    b = range_x[1]
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Decide whether a density estimate or regression estimate is required.
    if y is None:  # obtain density estimate
        n = len(x)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        y = np.asarray(y, dtype=np.float64)
        ## Bin the data if not already binned
        if not binned:
            gpoints = np.linspace(a, b, M)
            out = rlbin(x, y, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = x
            ycounts = y
            M = len(xcounts)
            gpoints = np.linspace(a, b, M)

    ## Set the bin width
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = (
        np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
        if bandwidth is not None
        else None
    )
    if bandwidth_arr is not None and bandwidth_arr.size == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
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
    elif bandwidth_arr is not None and bandwidth_arr.size == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.array([np.floor(tau * float(bandwidth_arr[0]) / delta)], dtype=np.int64)
        hdisc = np.array([float(bandwidth_arr[0])], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    ## Allocate space for the kernel vector and final estimate
    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap, dtype=np.float64)
    curvest = np.zeros(M, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)

    ## Equivalent of the FORTRAN routine "locpol": obtain kernel weights.
    ## (indices below follow the original 1-based FORTRAN convention and
    ## are converted to 0-based array access only at the point of use.)
    mid = int(Lvec[0]) + 1
    for i_f in range(1, Q):
        midpts[i_f - 1] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec[i_f - 1]) + 1):
            val = math.exp(-((delta * j / hdisc[i_f - 1]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i_f - 1]) + int(Lvec[i_f]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        val = math.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    ## Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i_f in range(1, Q + 1):
                lo = max(1, k - int(Lvec[i_f - 1]))
                hi = min(M, k + int(Lvec[i_f - 1]))
                for j in range(lo, hi + 1):
                    if indic[j - 1] == i_f:
                        fk = fkap[k - j + midpts[i_f - 1] - 1]
                        fac = 1.0
                        ss[j - 1, 0] += xcounts[k - 1] * fk
                        tt[j - 1, 0] += ycounts[k - 1] * fk
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xcounts[k - 1] * fk * fac
                            if ii <= pp:
                                tt[j - 1, ii - 1] += ycounts[k - 1] * fk * fac

    ## Solve the local weighted least-squares system at each grid point
    ## (equivalent of the LINPACK dgefa/dgesl solve in the FORTRAN source).
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Tvec = np.zeros(pp, dtype=np.float64)
        for i_f in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i_f + j - 1
                Smat[i_f - 1, j - 1] = ss[k - 1, indss - 1]
            Tvec[i_f - 1] = tt[k - 1, i_f - 1]

        Tsol = np.linalg.solve(Smat, Tvec)
        curvest[k - 1] = Tsol[drv]

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)

    # Equivalent of Fortran subroutine rlbin: linear binning of the
    # regression data (X, Y) onto the M grid points spanning [a, b],
    # accumulating both the bin counts (xcnts) and the linearly-binned
    # weighted sums of Y (ycnts).
    delta = (b - a) / (M - 1)

    # lxi is the (1-based) fractional grid position of each data point.
    lxi = ((X - a) / delta) + 1.0

    # Fortran's int() truncates toward zero, which matches Python's int()
    # / np.trunc() behaviour (as opposed to np.floor()).
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Correction for right endpoint: not included if li == M, so points
    # exactly at the right boundary are forced onto the last grid cell
    # with full weight, regardless of truncation.
    at_b = (X == b)
    li = np.where(at_b, M - 1, li)
    rem = np.where(at_b, 1.0, rem)

    in_range = (li >= 1) & (li < M)
    idx0 = li[in_range] - 1
    rem_in = rem[in_range]
    y_in = Y[in_range]

    # Convert the 1-based Fortran grid indices to 0-based Python indices
    # and accumulate with np.add.at so that repeated indices are summed
    # rather than overwritten.
    np.add.at(xcnts, idx0, 1.0 - rem_in)
    np.add.at(xcnts, idx0 + 1, rem_in)
    np.add.at(ycnts, idx0, (1.0 - rem_in) * y_in)
    np.add.at(ycnts, idx0 + 1, rem_in * y_in)

    if trun == 0:
        below = li < 1
        xcnts[0] += np.count_nonzero(below)
        ycnts[0] += np.sum(Y[below])

        above = li >= M
        xcnts[M - 1] += np.count_nonzero(above)
        ycnts[M - 1] += np.sum(Y[above])

    return {"xcounts": xcnts, "ycounts": ycnts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: Optional[Union[float, np.ndarray[Any, np.dtype[np.float64]]]] = None, gridsize: int = 401, bwdisc: int = 25, range_x: Optional[Tuple[float, float]] = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
        x = np.asarray(x, dtype=np.float64)
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = range_x[0]
    b = range_x[1]
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.size == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
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
    elif bandwidth_arr.size == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.array([np.floor(tau * float(bandwidth_arr[0]) / delta)], dtype=np.int64)
        hdisc = np.array([float(bandwidth_arr[0])], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    ## Allocate space for the kernel vector and final estimate
    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    Sdg = np.zeros(M, dtype=np.float64)

    ## Equivalent of the FORTRAN routine "sdiag": obtain kernel weights.
    ## (indices below follow the original 1-based FORTRAN convention and
    ## are converted to 0-based array access only at the point of use.)
    mid = int(Lvec[0]) + 1
    for i_f in range(1, Q):
        midpts[i_f - 1] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec[i_f - 1]) + 1):
            val = math.exp(-((delta * j / hdisc[i_f - 1]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i_f - 1]) + int(Lvec[i_f]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        val = math.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    ## Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i_f in range(1, Q + 1):
                lo = max(1, k - int(Lvec[i_f - 1]))
                hi = min(M, k + int(Lvec[i_f - 1]))
                for j in range(lo, hi + 1):
                    if indic[j - 1] == i_f:
                        fk = fkap[k - j + midpts[i_f - 1] - 1]
                        fac = 1.0
                        ss[j - 1, 0] += xcounts[k - 1] * fk
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xcounts[k - 1] * fk * fac

    ## Build the local design/moment matrix at each grid point and take
    ## the (1,1) entry of its inverse, i.e. the diagonal entry of the
    ## smoother matrix (equivalent of the LINPACK dgefa/dgedi inversion
    ## in the FORTRAN source).
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        for i_f in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i_f + j - 1
                Smat[i_f - 1, j - 1] = ss[k - 1, indss - 1]

        Smat_inv = np.linalg.inv(Smat)
        Sdg[k - 1] = Smat_inv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: Optional[Union[float, np.ndarray[Any, np.dtype[np.float64]]]] = None, gridsize: int = 401, bwdisc: int = 25, range_x: Optional[Tuple[float, float]] = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
        x = np.asarray(x, dtype=np.float64)
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = range_x[0]
    b = range_x[1]
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.size == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
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
    elif bandwidth_arr.size == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.array([np.floor(tau * float(bandwidth_arr[0]) / delta)], dtype=np.int64)
        hdisc = np.array([float(bandwidth_arr[0])], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    ## Allocate space for the kernel vector and final estimate
    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    SSTd = np.zeros(M, dtype=np.float64)

    ## Equivalent of the FORTRAN routine "sstdg": obtain kernel weights.
    ## (indices below follow the original 1-based FORTRAN convention and
    ## are converted to 0-based array access only at the point of use.)
    mid = int(Lvec[0]) + 1
    for i_f in range(1, Q):
        midpts[i_f - 1] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec[i_f - 1]) + 1):
            val = math.exp(-((delta * j / hdisc[i_f - 1]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i_f - 1]) + int(Lvec[i_f]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        val = math.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    ## Combine kernel weights and grid counts. "ss" accumulates the
    ## design-matrix moments (as in sdiag) while "uu" accumulates the
    ## corresponding moments weighted by the *squared* kernel weight,
    ## which is what is needed for the SS^T diagonal.
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i_f in range(1, Q + 1):
                lo = max(1, k - int(Lvec[i_f - 1]))
                hi = min(M, k + int(Lvec[i_f - 1]))
                for j in range(lo, hi + 1):
                    if indic[j - 1] == i_f:
                        fk = fkap[k - j + midpts[i_f - 1] - 1]
                        fac = 1.0
                        ss[j - 1, 0] += xcounts[k - 1] * fk
                        uu[j - 1, 0] += xcounts[k - 1] * (fk ** 2)
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xcounts[k - 1] * fk * fac
                            uu[j - 1, ii - 1] += xcounts[k - 1] * (fk ** 2) * fac

    ## Build the local design/moment matrices at each grid point.
    ## "Smat" is inverted (equivalent of the LINPACK dgefa/dgedi
    ## inversion in the FORTRAN source) and combined with "Umat" to
    ## obtain the diagonal entries of SS^T: SSTd[k] = e1' Smat^-1 Umat
    ## Smat^-1 e1, where e1 = (1, 0, ..., 0)'.
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Umat = np.zeros((pp, pp), dtype=np.float64)
        for i_f in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i_f + j - 1
                Smat[i_f - 1, j - 1] = ss[k - 1, indss - 1]
                Umat[i_f - 1, j - 1] = uu[k - 1, indss - 1]

        Smat_inv = np.linalg.inv(Smat)

        SSTd[k - 1] = float(Smat_inv[0, :] @ Umat @ Smat_inv[:, 0])

    return {"x": gpoints, "y": SSTd}


def on_attach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def on_unload(libpath: str) -> None:
    # In R, this hook explicitly unloads the package's compiled
    # dynamic library (library.dynam.unload("KernSmooth", libpath))
    # when the package is detached. Python has no equivalent explicit
    # unloading step for compiled extension modules: the _KernSmooth
    # extension is managed by Python's import system and garbage
    # collector, so there is nothing to do here.
    pass

