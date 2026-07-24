import math
import warnings
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import beta as _beta

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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> "BkdeResult":
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'kernel' should be one of " + repr(valid_kernels))

    x_arr = np.asarray(x, dtype=np.float64)

    # Rename common variables
    n = len(x_arr)
    M = gridsize

    # Set canonical scaling factors
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

    # Set default bandwidth
    if bandwidth is None:
        h = del0 * (243 / (35 * n)) ** (1 / 5) * np.sqrt(np.var(x_arr, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values
    tau = 4 if kernel == "normal" else 1

    if range_x is None:
        range_x = (float(np.min(x_arr) - tau * h), float(np.max(x_arr) + tau * h))
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    # Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    if kernel == "normal":
        kappa = np.exp(-0.5 * (lvec * delta) ** 2) / np.sqrt(2 * np.pi) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P = int(2 ** (np.ceil(np.log(M + L + 1) / np.log(2))))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    # R's fft(z, inverse=TRUE) is unnormalized; numpy's ifft divides by
    # length P automatically, so (Re(fft(kappa*gcounts, TRUE))/P) in R
    # corresponds exactly to np.real(np.fft.ifft(kappa * gcounts)) here.
    y = np.real(np.fft.ifft(kappa * gcounts))[0:M]

    return BkdeResult(x=gpoints, y=y)


class BkdeResult(NamedTuple):
    x: np.ndarray[Any, np.dtype[np.float64]]
    y: np.ndarray[Any, np.dtype[np.float64]]


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.integer[Any]]] = (51, 51), range_x: list[np.ndarray[Any, np.dtype[np.float64]]] | None = None, truncate: bool = True) -> "Bkde2DResult":
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.min(np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)]
        for id in range(2):
            range_x[id] = np.array(
                [np.min(x[:, id]) - 1.5 * h[id], np.max(x[:, id]) + 1.5 * h[id]]
            )

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    ]
    for id in range(2):
        L[id] = min(
            int(np.floor(tau * h[id] * (M[id] - 1) / (b[id] - a[id]))),
            int(M[id] - 1),
        )
        lvecid = np.arange(0, L[id] + 1)
        facid = (b[id] - a[id]) / (h[id] * (M[id] - 1))
        z = np.exp(-0.5 * (lvecid * facid) ** 2) / np.sqrt(2 * np.pi) / h[id]
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * facid * h[id]
        kapid[id] = z / tot

    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    P = (
        2 ** np.ceil(np.log(M.astype(np.float64) + L.astype(np.float64)) / np.log(2))
    ).astype(np.int64)  # smallest powers of 2 >= M+L
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
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[:M1, :M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[:M1, :M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return Bkde2DResult(x1=gpoints1, x2=gpoints2, fhat=rp)


class Bkde2DResult(NamedTuple):
    x1: np.ndarray[Any, np.dtype[np.float64]]
    x2: np.ndarray[Any, np.dtype[np.float64]]
    fhat: np.ndarray[Any, np.dtype[np.float64]]


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
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
        import warnings
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = np.exp(-0.5 * arg**2) / np.sqrt(2 * np.pi) / (h ** (drv + 1))
    hmold0 = np.ones_like(arg)
    hmold1 = arg.copy()
    hmnew = np.ones_like(arg)
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1  # Compute mth degree Hermite polynomial
            hmold1 = hmnew   # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P = int(2 ** (np.ceil(np.log(M + L + 1) / np.log(2))))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    # R's fft(z, inverse=TRUE) is unnormalized; numpy's ifft divides by
    # length P automatically, so multiply by P to undo that and match R,
    # then the original code divides by P again -- the two cancel out.
    inv = np.fft.ifft(kappam * Gcounts)
    return float(np.sum(gcounts * np.real(inv)[0:M]) / (n**2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> "BlkestResult":
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    # Set up arrays for FORTRAN routine "blkest"
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

    _KernSmooth.blkest(
        x_sorted,
        y_sorted,
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

    return BlkestResult(sigsqe=float(sigsqe[0]), th22e=float(th22e[0]), th24e=float(th24e[0]))


class BlkestResult(NamedTuple):
    sigsqe: float
    th22e: float
    th24e: float


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    order_idx = np.argsort(X_arr, kind="stable")
    X_sorted = X_arr[order_idx].copy()
    Y_sorted = Y_arr[order_idx].copy()

    # Set up arrays for FORTRAN subroutine "cp"
    qq = q + 1
    RSS = np.zeros(Nmax, dtype=np.float64)
    Xj = np.zeros(n, dtype=np.float64)
    Yj = np.zeros(n, dtype=np.float64)
    coef = np.zeros(qq, dtype=np.float64)
    Xmat = np.zeros(n * qq, dtype=np.float64)
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    wk = np.zeros(n, dtype=np.float64)
    qraux = np.zeros(qq, dtype=np.float64)

    # remove unused 'q' 2007-07-10
    _KernSmooth.cp(
        X_sorted,
        Y_sorted,
        n,
        qq,
        Nmax,
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

    return int(np.argmin(Cpvec))


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    x = np.asarray(x, dtype=np.float64)

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
        raise ValueError("'scalest' should be one of 'minim', 'stdev', 'iqr'")

    if scalest == "stdev":
        scale_value = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_value = (np.quantile(x, 3 / 4) - np.quantile(x, 1 / 4)) / 1.349
    else:  # "minim"
        scale_value = min(
            (np.quantile(x, 3 / 4) - np.quantile(x, 1 / 4)) / 1.349,
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


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    if kernel not in ("normal", "box", "epanech", "biweight", "triweight"):
        raise ValueError(
            "'kernel' should be one of 'normal', 'box', 'epanech', 'biweight', 'triweight'"
        )

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

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
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError("'scalest' should be one of 'minim', 'stdev', 'iqr'")

    if scalest == "stdev":
        scale_value = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_value = (np.quantile(x, 3 / 4) - np.quantile(x, 1 / 4)) / 1.349
    else:  # "minim"
        scale_value = min(
            (np.quantile(x, 3 / 4) - np.quantile(x, 1 / 4)) / 1.349,
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

    return float(scale_value * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # R evaluates 'range.x = range(x)' lazily using the *original* x,
    # before the body below reassigns/trims x. Replicate that timing here.
    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    order_idx = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order_idx]
    y_sorted = y_arr[order_idx]

    n_full = len(x_sorted)
    indlow = int(np.floor(trim * n_full))
    indupp = n_full - int(np.floor(trim * n_full))

    x_trim = x_sorted[indlow:indupp]
    y_trim = y_sorted[indlow:indupp]

    ## Rename common parameters
    n = len(x_trim)
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])

    ## Bin the data
    gpoints = np.linspace(a, b, M)
    rlbin_out = rlbin(x_trim, y_trim, gpoints, truncate)
    xcounts = rlbin_out.xcounts
    ycounts = rlbin_out.ycounts

    ## Choose the value of N using Mallow's C_p
    Nmax = max(min(int(np.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x_trim, y_trim, Nmax, 4) + 1

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.
    blkest_out = blkest(x_trim, y_trim, Nval, 4)
    sigsqQ = blkest_out.sigsqe
    th24Q = blkest_out.th24e

    ## Estimate theta_22 using a local cubic fit
    ## with a "rule-of-thumb" bandwidth: "gamseh"
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * np.sqrt(np.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * np.sqrt(np.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh, range_x=range_x, binned=True).y

    llow = int(np.floor(proptrun * M))
    lupp = M - int(np.floor(proptrun * M))
    th22kn = np.sum((mddest[llow:lupp] ** 2) * xcounts[llow:lupp]) / n

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * np.sqrt(2) - (4 / 3) * np.sqrt(3)
    C3K = (4 * C3K / np.sqrt(2 * np.pi)) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    ## Now compute a local linear kernel estimate of
    ## the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh, range_x=range_x, binned=True).y
    Sdg = sdiag(xcounts, bandwidth=lamseh, range_x=range_x, binned=True).y
    SSTdg = sstdiag(xcounts, bandwidth=lamseh, range_x=range_x, binned=True).y
    sigsqn = np.sum(y_trim ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    ## Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * np.sqrt(np.pi) * th22kn * n)) ** (1 / 5))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]
    gcounts = np.zeros(M, dtype=np.float64)
    _KernSmooth.linbin(np.asarray(X, dtype=np.float64), n, a, b, M, trun, gcounts)
    return gcounts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = X.shape[0]
    x_vec = np.concatenate((np.asarray(X[:, 0], dtype=np.float64), np.asarray(X[:, 1], dtype=np.float64)))
    M1 = len(gpoints1)
    M2 = len(gpoints2)
    a1 = float(gpoints1[0])
    a2 = float(gpoints2[0])
    b1 = float(gpoints1[M1 - 1])
    b2 = float(gpoints2[M2 - 1])
    out9 = np.zeros(M1 * M2, dtype=np.float64)
    _KernSmooth.lbtwod(x_vec, n, a1, a2, b1, b2, M1, M2, out9)
    return out9.reshape((M1, M2), order='F')


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> "LocpolyResult":
    # Install safeguard against non-positive bandwidths
    if bandwidth is not None:
        bw_check = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
        if np.any(bw_check <= 0):
            raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64) if y is not None else None

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
        n = len(x_arr)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.full(M, 1.0, dtype=np.float64)
    else:  # obtain regression estimate
        # Bin the data if not already binned
        if not binned:
            gpoints = np.linspace(a, b, M)
            rlbin_out = rlbin(x_arr, y_arr, gpoints, truncate)
            xcounts = rlbin_out.xcounts
            ycounts = rlbin_out.ycounts
        else:
            xcounts = x_arr
            ycounts = y_arr
            M = len(xcounts)
            gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    if bandwidth is None:
        raise TypeError("argument 'bandwidth' is missing, with no default")

    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.full(M, 1.0, dtype=np.float64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                )
        else:
            indic = np.full(M, 1.0, dtype=np.float64)
    elif len(bandwidth_arr) == 1:
        indic = np.full(M, 1.0, dtype=np.float64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.float64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Allocate space for the kernel vector and final estimate
    dimfkap = int(2 * np.sum(Lvec) + Q)
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
    Lvec_i = Lvec.astype(np.int32)
    indic_i = indic.astype(np.int32)

    _KernSmooth.locpol(
        xcounts_d,
        ycounts_d,
        drv,
        delta,
        hdisc,
        Lvec_i,
        indic_i,
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

    return LocpolyResult(x=gpoints, y=curvest)


class LocpolyResult(NamedTuple):
    x: np.ndarray[Any, np.dtype[np.float64]]
    y: np.ndarray[Any, np.dtype[np.float64]]


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> "RlbinResult":
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = float(gpoints[0])
    b = float(gpoints[M - 1])

    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(Y, dtype=np.float64)
    xcounts = np.zeros(M, dtype=np.float64)
    ycounts = np.zeros(M, dtype=np.float64)

    _KernSmooth.rlbin(x_arr, y_arr, n, a, b, M, trun, xcounts, ycounts)

    return RlbinResult(xcounts=xcounts, ycounts=ycounts)


class RlbinResult(NamedTuple):
    xcounts: np.ndarray[Any, np.dtype[np.float64]]
    ycounts: np.ndarray[Any, np.dtype[np.float64]]


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> "SdiagResult":
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
        xcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
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
    Smat = np.zeros((pp, pp), dtype=np.float64)
    work = np.zeros(pp, dtype=np.float64)
    det = np.zeros(2, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int32)
    Sdg = np.zeros(M, dtype=np.float64)

    _KernSmooth.sdiag(
        np.asarray(xcounts, dtype=np.float64),
        delta,
        np.asarray(hdisc, dtype=np.float64),
        np.asarray(Lvec, dtype=np.int32),
        np.asarray(indic, dtype=np.int32),
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

    return SdiagResult(x=gpoints, y=Sdg)


class SdiagResult(NamedTuple):
    x: np.ndarray[Any, np.dtype[np.float64]]
    y: np.ndarray[Any, np.dtype[np.float64]]


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> "SstdiagResult":
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
        xcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
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

    _KernSmooth.sstdg(
        np.asarray(xcounts, dtype=np.float64),
        delta,
        np.asarray(hdisc, dtype=np.float64),
        np.asarray(Lvec, dtype=np.int32),
        np.asarray(indic, dtype=np.int32),
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

    return SstdiagResult(x=gpoints, y=SSTd)


class SstdiagResult(NamedTuple):
    x: np.ndarray[Any, np.dtype[np.float64]]
    y: np.ndarray[Any, np.dtype[np.float64]]


def onAttach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")


def _on_unload(libpath: str) -> None:
    # R's .onUnload package hook calls library.dynam.unload("KernSmooth", libpath)
    # to unload the compiled shared library when the package is detached.
    # Python does not unload compiled extension modules (e.g. _KernSmooth) in
    # the same manner, so there is no direct equivalent action to perform here.
    # This stub is kept for interface/signature compatibility only.
    pass
