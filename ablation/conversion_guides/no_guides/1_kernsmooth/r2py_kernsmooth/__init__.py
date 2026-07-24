import math
import sys
import warnings
from typing import Any, Literal

import numpy as np
from scipy.stats import beta as _beta

from . import _KernSmooth
from .bkfe import bkfe
from .blkest import blkest
from .cpblock import cpblock
from .linbin import linbin
from .linbin2D import linbin2D
from .locpoly import locpoly
from .rlbin import rlbin
from .sdiag import sdiag
from .sstdiag import sstdiag


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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
    # (bandwidth is None mimics R's `missing(bandwidth)`.)
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError(
            "'kernel' should be one of " + ", ".join(repr(k) for k in valid_kernels)
        )

    # Rename common variables
    n = len(x)
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
        raise ValueError("'canonical' must be a length-1 logical vector")

    # Set default bandwidth
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.sqrt(np.var(x, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values
    tau = 4 if kernel == "normal" else 1

    if range_x is None:
        range_x = (float(np.min(x) - tau * h), float(np.max(x) + tau * h))
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    if kernel == "normal":
        arg = lvec * delta
        dnorm_arg = np.exp(-(arg ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
        kappa = dnorm_arg / (n * h)
    elif kernel == "box":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)

    # R's fft(z, inverse=TRUE) is unnormalized: it equals P * np.fft.ifft(z).
    conv = np.fft.ifft(kappa_fft * gcounts_fft) * P
    y = (conv.real / P)[0:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: tuple[int, int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: list[tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
    h_check: np.ndarray[Any, np.dtype[np.float64]] = np.atleast_1d(
        np.asarray(bandwidth, dtype=np.float64)
    )
    if np.min(h_check) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables
    n: int = x.shape[0]
    M: np.ndarray[Any, np.dtype[np.int64]] = np.asarray(gridsize, dtype=np.int64)
    h: np.ndarray[Any, np.dtype[np.float64]] = np.atleast_1d(
        np.asarray(bandwidth, dtype=np.float64)
    )
    tau: float = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    if len(h) == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [
            (
                float(np.min(x[:, 0]) - 1.5 * h[0]),
                float(np.max(x[:, 0]) + 1.5 * h[0]),
            ),
            (
                float(np.min(x[:, 1]) - 1.5 * h[1]),
                float(np.max(x[:, 1]) + 1.5 * h[1]),
            ),
        ]

    a: np.ndarray[Any, np.dtype[np.float64]] = np.array(
        [range_x[0][0], range_x[1][0]], dtype=np.float64
    )
    b: np.ndarray[Any, np.dtype[np.float64]] = np.array(
        [range_x[0][1], range_x[1][1]], dtype=np.float64
    )

    # Set up grid points and bin the data
    gpoints1: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a[0], b[0], int(M[0]))
    gpoints2: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a[1], b[1], int(M[1]))

    gcounts: np.ndarray[Any, np.dtype[np.float64]] = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    L: np.ndarray[Any, np.dtype[np.int64]] = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.zeros(0), np.zeros(0)]
    for id_ in range(2):
        L[id_] = min(
            int(np.floor(tau * h[id_] * (int(M[id_]) - 1) / (b[id_] - a[id_]))),
            int(M[id_]) - 1,
        )
        lvecid: np.ndarray[Any, np.dtype[np.float64]] = np.arange(
            0, int(L[id_]) + 1, dtype=np.float64
        )
        facid: float = (b[id_] - a[id_]) / (h[id_] * (int(M[id_]) - 1))
        z: np.ndarray[Any, np.dtype[np.float64]] = (
            np.exp(-((lvecid * facid) ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
        ) / h[id_]
        tot: float = (float(z[0]) + 2.0 * float(np.sum(z[1:]))) * facid * h[id_]
        kapid[id_] = z / tot

    kapp: np.ndarray[Any, np.dtype[np.float64]] = np.outer(kapid[0], kapid[1]) / n

    if int(np.min(L)) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    P: np.ndarray[Any, np.dtype[np.int64]] = (
        2.0
        ** np.ceil(np.log(M.astype(np.float64) + L.astype(np.float64)) / np.log(2.0))
    ).astype(np.int64)  # smallest powers of 2 >= M+L
    L1: int = int(L[0])
    L2: int = int(L[1])
    M1: int = int(M[0])
    M2: int = int(M[1])
    P1: int = int(P[0])
    P2: int = int(P[1])

    rp: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((P1, P2), dtype=np.float64)
    rp[0 : (L1 + 1), 0 : (L2 + 1)] = kapp
    if L1:
        rp[(P1 - L1) : P1, 0 : (L2 + 1)] = kapp[L1:0:-1, 0 : (L2 + 1)]
    if L2:
        rp[:, (P2 - L2) : P2] = rp[:, L2:0:-1]
    # wrap-around version of "kapp"

    sp: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft: np.ndarray[Any, np.dtype[np.complex128]] = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft: np.ndarray[Any, np.dtype[np.complex128]] = np.fft.fft2(sp)
    # R's fft(z, inverse = TRUE) is unnormalized: it equals (P1*P2) * np.fft.ifft2(z).
    conv: np.ndarray[Any, np.dtype[np.complex128]] = np.fft.ifft2(rp_fft * sp_fft) * (P1 * P2)
    rp = (conv.real / (P1 * P2))[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths.
    # (bandwidth is None mimics R's `missing(bandwidth)`.)
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

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
        gcounts = np.asarray(x, dtype=np.float64)
        M = len(gcounts)
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width
    n = np.sum(gcounts)
    delta = (b - a) / (M - 1)

    # Obtain kernel weights
    tau = 4 + drv
    L = min(int(np.floor(tau * h / delta)), M)

    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    dnorm_arg = np.exp(-(arg ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
    kappam = dnorm_arg / (h ** (drv + 1))
    hmold0 = 1.0
    hmold1 = arg
    hmnew = 1.0
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1        # Compute mth degree Hermite polynomial
            hmold1 = hmnew         # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    # R's fft(z, inverse=TRUE) is unnormalized: it equals P * np.fft.ifft(z).
    conv = np.fft.ifft(kappam * Gcounts) * P
    estimate = np.sum(gcounts * (conv.real / P)[:M]) / (n ** 2)

    return float(estimate)


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    n: int = len(x)

    # Sort the (x, y) data with respect to the x's.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order: np.ndarray[Any, np.dtype[np.intp]] = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    # Set up arrays for FORTRAN programme "blkest"
    qq: int = q + 1
    xj: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(n, dtype=np.float64)
    yj: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(n, dtype=np.float64)
    coef: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(qq, dtype=np.float64)
    Xmat: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((n, qq), dtype=np.float64)
    wk: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(n, dtype=np.float64)
    qraux: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(qq, dtype=np.float64)
    sigsqe: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(1, dtype=np.float64)
    th22e: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(1, dtype=np.float64)
    th24e: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(1, dtype=np.float64)

    _KernSmooth.blkest(
        x,
        y,
        n,
        q,
        qq,
        Nval,
        xj,
        yj,
        coef,
        Xmat,
        wk,
        qraux,
        sigsqe,
        th22e,
        th24e,
    )

    return {"sigsqe": float(sigsqe[0]), "th22e": float(th22e[0]), "th24e": float(th24e[0])}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    order_idx = np.argsort(X, kind="stable")
    X = np.asarray(X, dtype=np.float64)[order_idx]
    Y = np.asarray(Y, dtype=np.float64)[order_idx]

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

    # remove unused 'q' 2007-07-10
    _KernSmooth.cp(
        X.astype(np.float64),
        Y.astype(np.float64),
        int(n),
        int(qq),
        int(Nmax),
        RSS,
        Xj,
        Yj,
        coef,
        Xmat,
        wk,
        qraux,
        Cpvals,
    )

    Cpvec = Cpvals

    return int(np.argmin(Cpvec)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: Literal["minim", "stdev", "iqr"] = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    ## Rename variables

    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError('scalest must be one of "minim", "stdev", "iqr"')

    if scalest == "stdev":
        scale_val = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scale_val = min((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349, np.sqrt(np.var(x, ddof=1)))

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

    return float(scale_val * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: Literal["minim", "stdev", "iqr"] = "minim", level: int = 2, kernel: Literal["normal", "box", "epanech", "biweight", "triweight"] = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    if kernel not in ("normal", "box", "epanech", "biweight", "triweight"):
        raise ValueError('kernel must be one of "normal", "box", "epanech", "biweight", "triweight"')

    x = np.asarray(x, dtype=np.float64)

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

    ## Rename variables

    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError('scalest must be one of "minim", "stdev", "iqr"')

    if scalest == "stdev":
        scale_val = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scale_val = min((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349, np.sqrt(np.var(x, ddof=1)))

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

    return float(scale_val * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Trim the 100(trim)% of the data from each end (in the x-direction).

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    indlow = int(np.floor(trim * len(x))) + 1
    indupp = len(x) - int(np.floor(trim * len(x)))

    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    ## Rename common parameters
    n = len(x)
    M = gridsize
    a = range_x[0]
    b = range_x[1]

    ## Bin the data

    gpoints = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    ## Choose the value of N using Mallow's C_p
    Nmax = max(min(int(np.floor(n / divisor)), blockmax), 1)
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
        gamseh = (3 * gamseh / (8 * np.sqrt(np.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * np.sqrt(np.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=range_x, binned=True)["y"]

    llow = int(np.floor(proptrun * M)) + 1
    lupp = M - int(np.floor(proptrun * M))
    th22kn = np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * np.sqrt(2) - (4 / 3) * np.sqrt(3)
    C3K = (4 * C3K / (np.sqrt(2 * np.pi))) ** (1 / 9)
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
    n: int = len(X)
    M: int = len(gpoints)
    trun: int = 1 if truncate else 0
    a: float = gpoints[0]
    b: float = gpoints[M - 1]
    gcounts: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M, dtype=np.float64)
    _KernSmooth.linbin(
        np.asarray(X, dtype=np.float64),
        n,
        a,
        b,
        M,
        trun,
        gcounts,
    )
    return gcounts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    n: int = X.shape[0]
    X_flat: np.ndarray[Any, np.dtype[np.float64]] = np.concatenate(
        [np.asarray(X[:, 0], dtype=np.float64), np.asarray(X[:, 1], dtype=np.float64)]
    )
    M1: int = len(gpoints1)
    M2: int = len(gpoints2)
    a1: float = gpoints1[0]
    a2: float = gpoints2[0]
    b1: float = gpoints1[M1 - 1]
    b2: float = gpoints2[M2 - 1]
    out: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M1 * M2, dtype=np.float64)
    _KernSmooth.lbtwod(
        X_flat,
        n,
        a1,
        a2,
        b1,
        b2,
        M1,
        M2,
        out,
    )
    return np.reshape(out, (M1, M2), order="F")


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths.
    # (bandwidth is None mimics R's `missing(bandwidth)`.)
    if bandwidth is not None and np.any(np.asarray(bandwidth) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x = np.asarray(x, dtype=np.float64)

    # (range_x is None mimics R's `missing(range.x)`.)
    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x) - np.min(x))
            range_x = (float(np.min(x) - extra), float(np.max(x) + extra))
        else:
            range_x = (float(np.min(x)), float(np.max(x)))

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
        gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        y = np.asarray(y, dtype=np.float64)
        # Bin the data if not already binned
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

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc: np.ndarray[Any, np.dtype[np.float64]] = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc"
        Lvec: np.ndarray[Any, np.dtype[np.int_]] = np.floor(tau * hdisc / delta).astype(np.int_)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic: np.ndarray[Any, np.dtype[np.int_]] = np.ones(M, dtype=np.int_)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int_)
        else:
            indic = np.ones(M, dtype=np.int_)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int_)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * bandwidth_arr[0] / delta), dtype=np.int_)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Allocate space for the kernel vector and final estimate
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(dimfkap, dtype=np.float64)
    curvest: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M, dtype=np.float64)
    midpts: np.ndarray[Any, np.dtype[np.int_]] = np.zeros(Q, dtype=np.int_)
    ss: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M, ppp), dtype=np.float64)
    tt: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M, pp), dtype=np.float64)
    Smat: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((pp, pp), dtype=np.float64)
    Tvec: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(pp, dtype=np.float64)
    ipvt: np.ndarray[Any, np.dtype[np.int_]] = np.zeros(pp, dtype=np.int_)

    # Call FORTRAN routine "locpol"
    _KernSmooth.locpol(
        np.asarray(xcounts, dtype=np.float64),
        np.asarray(ycounts, dtype=np.float64),
        drv,
        delta,
        hdisc,
        Lvec,
        indic,
        midpts,
        M,
        Q,
        fkap,
        pp,
        ppp,
        ss,
        tt,
        Smat,
        Tvec,
        ipvt,
        curvest,
    )

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    n: int = len(X)
    M: int = len(gpoints)
    trun: int = 1 if truncate else 0
    a: float = gpoints[0]
    b: float = gpoints[M - 1]
    xcounts: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M, dtype=np.float64)
    ycounts: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M, dtype=np.float64)
    _KernSmooth.rlbin(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        n,
        a,
        b,
        M,
        trun,
        xcounts,
        ycounts,
    )
    return {"xcounts": xcounts, "ycounts": ycounts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # (range_x is None mimics R's `missing(range.x)`.)
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
        gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = x
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
        hdisc: np.ndarray[Any, np.dtype[np.float64]] = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc"
        Lvec: np.ndarray[Any, np.dtype[np.int_]] = np.floor(tau * hdisc / delta).astype(np.int_)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic: np.ndarray[Any, np.dtype[np.int_]] = np.ones(M, dtype=np.int_)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int_)
        else:
            indic = np.ones(M, dtype=np.int_)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int_)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * bandwidth_arr[0] / delta), dtype=np.int_)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(dimfkap, dtype=np.float64)
    midpts: np.ndarray[Any, np.dtype[np.int_]] = np.zeros(Q, dtype=np.int_)
    ss: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M, ppp), dtype=np.float64)
    Smat: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((pp, pp), dtype=np.float64)
    work: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(pp, dtype=np.float64)
    det: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(2, dtype=np.float64)
    ipvt: np.ndarray[Any, np.dtype[np.int_]] = np.zeros(pp, dtype=np.int_)
    Sdg: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M, dtype=np.float64)

    # Call FORTRAN routine "sdiag"
    _KernSmooth.sdiag(
        np.asarray(xcounts, dtype=np.float64),
        delta,
        hdisc,
        Lvec,
        indic,
        midpts,
        M,
        Q,
        fkap,
        pp,
        ppp,
        ss,
        Smat,
        work,
        det,
        ipvt,
        Sdg,
    )

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # (range_x is None mimics R's `missing(range.x)`.)
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
        gpoints: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = x
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
        hdisc: np.ndarray[Any, np.dtype[np.float64]] = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc"
        Lvec: np.ndarray[Any, np.dtype[np.int_]] = np.floor(tau * hdisc / delta).astype(np.int_)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic: np.ndarray[Any, np.dtype[np.int_]] = np.ones(M, dtype=np.int_)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int_)
        else:
            indic = np.ones(M, dtype=np.int_)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int_)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * bandwidth_arr[0] / delta), dtype=np.int_)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(dimfkap, dtype=np.float64)
    midpts: np.ndarray[Any, np.dtype[np.int_]] = np.zeros(Q, dtype=np.int_)
    ss: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M, ppp), dtype=np.float64)
    uu: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((M, ppp), dtype=np.float64)
    Smat: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((pp, pp), dtype=np.float64)
    Umat: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((pp, pp), dtype=np.float64)
    work: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(pp, dtype=np.float64)
    det: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(2, dtype=np.float64)
    ipvt: np.ndarray[Any, np.dtype[np.int_]] = np.zeros(pp, dtype=np.int_)
    SSTd: np.ndarray[Any, np.dtype[np.float64]] = np.zeros(M, dtype=np.float64)

    # Call FORTRAN routine "sstdg"
    _KernSmooth.sstdg(
        np.asarray(xcounts, dtype=np.float64),
        delta,
        hdisc,
        Lvec,
        indic,
        midpts,
        M,
        Q,
        fkap,
        pp,
        ppp,
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


def on_attach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def on_unload(libpath: str) -> None:
    # R's library.dynam.unload("KernSmooth", libpath) unloads the package's
    # compiled native library from the current session. Python has no direct,
    # safe equivalent for forcibly unloading a C/Fortran extension module at
    # runtime, so we mirror the intent by removing the native extension module
    # from sys.modules if it has been imported, so that a subsequent import
    # will reload it.
    module_name = "_KernSmooth"
    if module_name in sys.modules:
        del sys.modules[module_name]
