from collections.abc import Sequence
from typing import Any
import math
import sys
import warnings

from scipy.stats import beta, norm
import numpy as np

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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError(
            "'kernel' must be one of " + ", ".join(repr(k) for k in valid_kernels)
        )

    x = np.asarray(x, dtype=np.float64)

    ## Rename common variables
    n = x.shape[0]
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
    tau = 4.0 if kernel == "normal" else 1.0

    if range_x is None:
        a = float(np.min(x)) - tau * h
        b = float(np.max(x)) + tau * h
    else:
        a = float(range_x[0])
        b = float(range_x[1])

    ## Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
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

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)

    # R's fft(z, inverse=TRUE) is the *unnormalized* inverse FFT, i.e.
    # numpy.fft.ifft(z) * P. The original code then divides that result
    # by P again, so the two factors of P cancel exactly, leaving plain
    # numpy.fft.ifft(...) with no extra scaling needed here.
    y = np.real(np.fft.ifft(kappa_fft * gcounts_fft))[:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: tuple[tuple[float, float], tuple[float, float]] | list[list[float]] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    ## Install safeguard against non-positive bandwidths
    if bandwidth is not None and np.min(np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))) <= 0:
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
        h = np.array([h[0], h[0]], dtype=np.float64)
    else:
        h = h[:2].astype(np.float64)

    ## If range_x is not specified then set it at its default value.
    ## range_x, when given, is a length-2 sequence of (low, high) pairs,
    ## one per dimension: range_x = ((a1, b1), (a2, b2)).
    if range_x is None:
        a = np.array([
            np.min(x[:, 0]) - 1.5 * h[0],
            np.min(x[:, 1]) - 1.5 * h[1],
        ])
        b = np.array([
            np.max(x[:, 0]) + 1.5 * h[0],
            np.max(x[:, 1]) + 1.5 * h[1],
        ])
    else:
        a = np.array([float(range_x[0][0]), float(range_x[1][0])])
        b = np.array([float(range_x[0][1]), float(range_x[1][1])])

    ## Set up grid points and bin the data
    M1 = int(M[0])
    M2 = int(M[1])
    gpoints1 = np.linspace(a[0], b[0], M1)
    gpoints2 = np.linspace(a[1], b[1], M2)

    gcounts = linbin2D(x, gpoints1, gpoints2)

    ## Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.zeros(0), np.zeros(0)]
    for i in range(2):
        L[i] = min(
            int(np.floor(tau * h[i] * (M[i] - 1) / (b[i] - a[i]))),
            int(M[i]) - 1,
        )
        lvec = np.arange(0, L[i] + 1)
        fac = (b[i] - a[i]) / (h[i] * (M[i] - 1))
        z = norm.pdf(lvec * fac) / h[i]
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * fac * h[i]
        kapid[i] = z / tot

    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    ## Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log(M.astype(np.float64) + L.astype(np.float64)) / np.log(2))).astype(np.int64)
    L1 = int(L[0])
    L2 = int(L[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2))
    rp[0:(L1 + 1), 0:(L2 + 1)] = kapp
    if L1:
        rp[(P1 - L1):P1, 0:(L2 + 1)] = kapp[L1:0:-1, 0:(L2 + 1)]
    if L2:
        rp[:, (P2 - L2):P2] = rp[:, L2:0:-1]
    ## wrap-around version of "kapp"

    sp = np.zeros((P1, P2))
    sp[0:M1, 0:M2] = gcounts
    ## zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    # R's fft(z, inverse=TRUE) is the *unnormalized* inverse FFT, i.e.
    # numpy.fft.ifft2(z) * (P1 * P2). The original code then divides that
    # result by (P1 * P2), so the two factors cancel exactly, leaving
    # plain numpy.fft.ifft2(...) with no extra scaling needed here.
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    ## invert element-wise product of FFT's
    ## and truncate and normalise it

    ## Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths.
    # ('bandwidth' has no default in R and must always be supplied by the
    # caller; the missing() check there is purely a validation step.)
    if bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename variables
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])
    h = float(bandwidth)

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
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = norm.pdf(arg) / (h ** (drv + 1))
    hmold0: np.ndarray[Any, np.dtype[np.float64]] | float = 1.0
    hmold1: np.ndarray[Any, np.dtype[np.float64]] | float = arg
    hmnew: np.ndarray[Any, np.dtype[np.float64]] | float = 1.0
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1       # Compute mth degree Hermite polynomial
            hmold1 = hmnew        # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam_fft = np.fft.fft(kappam)
    Gcounts_fft = np.fft.fft(Gcounts)

    # R's fft(z, inverse=TRUE) is the *unnormalized* inverse FFT, i.e.
    # numpy.fft.ifft(z) * P. The original code then divides that result
    # by P again, so the two factors of P cancel exactly, leaving plain
    # numpy.fft.ifft(...) with no extra scaling needed here.
    conv = np.real(np.fft.ifft(kappam_fft * Gcounts_fft))[:M]

    return float(np.sum(gcounts * conv) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # For obtaining preliminary estimates of quantities required for the
    # "direct plug-in" regression bandwidth selector based on blocked
    # q'th degree polynomial fits.
    #
    # This reimplements the Fortran routine F_blkest natively, since no
    # compiled _KernSmooth.blkest() binding is available. The block-wise
    # least-squares polynomial fit performed there via LINPACK's
    # dqrdc/dqrsl (QR decomposition) is replaced here by numpy's
    # numpy.linalg.lstsq, which solves the same least-squares problem.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]

    # Sort the (x, y) data with respect to the x's.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    qq = q + 1
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0
    idiv = n // Nval

    # It is assumed that the (x, y) data are sorted with respect to the x's.
    for j in range(1, Nval + 1):
        # For each member of the partition
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        nj = iupp - ilow + 1

        Xj = x[ilow - 1:iupp]
        Yj = y[ilow - 1:iupp]

        # Obtain a q'th degree fit over current member of partition.
        # Set up "X" matrix.
        Xmat = np.empty((nj, qq), dtype=np.float64)
        Xmat[:, 0] = 1.0
        for k in range(2, qq + 1):
            Xmat[:, k - 1] = Xj ** (k - 1)

        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)
        fiti = Xmat @ coef

        ddm = np.full(nj, 2.0 * coef[2]) if qq >= 3 else np.zeros(nj, dtype=np.float64)
        ddddm = np.full(nj, 24.0 * coef[4]) if qq >= 5 else np.zeros(nj, dtype=np.float64)

        for k in range(2, qq + 1):
            if k <= (q - 1):
                ddm = ddm + k * (k + 1) * coef[k + 1] * Xj ** (k - 1)
                if k <= (q - 3):
                    ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef[k + 3] * Xj ** (k - 1)

        th22e += np.sum(ddm ** 2)
        th24e += np.sum(ddm * ddddm)
        RSS += np.sum((Yj - fiti) ** 2)

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # Chooses the number of blocks for the preliminary step of a
    # plug-in rule using Mallows' C_p.
    #
    # This reimplements the Fortran routine F_cp natively, since no
    # compiled _KernSmooth.cp() binding is available. The block-wise
    # least-squares polynomial fit performed there via LINPACK's
    # dqrdc/dqrsl (QR decomposition) is replaced here by numpy's
    # numpy.linalg.lstsq, which solves the same least-squares problem.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]

    # Sort the (X, Y) data with respect to the X's.
    order = np.argsort(X, kind="stable")
    X = X[order]
    Y = Y[order]

    qq = q + 1

    # Compute vector of RSS values, one per candidate number of blocks
    # Nval = 1, ..., Nmax. RSS[Nval - 1] holds the total residual sum of
    # squares for that number of blocks.
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # For each number of partitions
        idiv = n // Nval
        RSSj_total = 0.0
        for j in range(1, Nval + 1):
            # For each member of the partition
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n

            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]

            # Obtain a q'th degree fit over current member of partition.
            # Set up "X" matrix.
            nj = iupp - ilow + 1
            Xmat = np.empty((nj, qq), dtype=np.float64)
            Xmat[:, 0] = 1.0
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)
            fitj = Xmat @ coef

            RSSj_total += np.sum((Yj - fitj) ** 2)

        RSS[Nval - 1] = RSS[Nval - 1] + RSSj_total

    # Now compute array of Mallow's C_p values.
    Cpvals = np.empty(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    Cpvec = Cpvals

    # R's order(Cpvec)[1L] returns the 1-based index of the minimum
    # element of Cpvec, which here coincides with the (1-based) number
    # of blocks Nval since Cpvec[k] (0-based) was computed for Nval =
    # k + 1. This value is later used by dpill() as a magnitude (a
    # count of blocks), not as an array index, so we return it in its
    # natural 1-based form: argmin + 1.
    return int(np.argmin(Cpvec)) + 1


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

    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError("'scalest' should be one of 'minim', 'stdev', 'iqr'")

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

    # Replace input data by standardised data for numerical stability:

    x_mean = np.mean(x)
    sx = (x - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    # Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    # delta <- (sb-sa)/(M - 1)

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

    return float(scale_value * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError(
            "'kernel' must be one of " + ", ".join(repr(k) for k in valid_kernels)
        )

    x = np.asarray(x, dtype=np.float64)

    ## Set kernel constants
    if canonical:
        del0 = 1.0
    elif kernel == "normal":
        del0 = 1.0 / ((4.0 * np.pi) ** (1.0 / 10.0))
    elif kernel == "box":
        del0 = (9.0 / 2.0) ** (1.0 / 5.0)
    elif kernel == "epanech":
        del0 = 15.0 ** (1.0 / 5.0)
    elif kernel == "biweight":
        del0 = 35.0 ** (1.0 / 5.0)
    else:  # triweight
        del0 = (9450.0 / 143.0) ** (1.0 / 5.0)

    ## Rename variables

    n = x.shape[0]
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError("'scalest' should be one of 'minim', 'stdev', 'iqr'")

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
    # delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps:

    if level == 0:
        psi4hat = 3.0 / (8.0 * np.sqrt(np.pi))
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

    return float(scale_value * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    # Computes a direct plug-in selector of the bandwidth for local linear
    # regression as described in Ruppert, Sheather and Wand (1996),
    # J. Amer. Statist. Assoc.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # cbind(x, y); xy <- xy[sort.list(xy[, 1L]), ] -- jointly sort (x, y)
    # by x using a stable sort, matching R's default 'sort.list' method.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    # Trim the 100(trim)% of the data from each end (in the x-direction).
    # R's 1-based inclusive x[indlow:indupp] with indlow = floor(trim*n)+1
    # and indupp = n - floor(trim*n) becomes the 0-based Python slice
    # x[ntrim : n - ntrim] with ntrim = floor(trim*n).
    n_full = x.shape[0]
    ntrim = int(math.floor(trim * n_full))
    x = x[ntrim:n_full - ntrim]
    y = y[ntrim:n_full - ntrim]

    # Rename common parameters
    n = x.shape[0]
    M = int(gridsize)

    # range.x = range(x) is a default argument evaluated lazily in R; by
    # the time it is first referenced in the body, 'x' has already been
    # reassigned to the trimmed data, so the default range is computed
    # from the TRIMMED x (not the original, untrimmed x).
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    # Bin the data
    gpoints = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    # Choose the value of N using Mallow's C_p
    Nmax = max(min(int(math.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x, y, Nmax, 4)

    # Estimate sig^2, theta_22 and theta_24 using quartic fits
    # on "Nval" blocks.
    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    # Estimate theta_22 using a local cubic fit with a "rule-of-thumb"
    # bandwidth: "gamseh"
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1 / 7)

    mddest = locpoly(
        xcounts, ycounts, drv=2, bandwidth=gamseh, range_x=range_x, binned=True
    )["y"]

    # R's 1-based inclusive mddest[llow:lupp] with
    # llow = floor(proptrun*M)+1, lupp = M - floor(proptrun*M) becomes the
    # 0-based Python slice [floor(proptrun*M) : M - floor(proptrun*M)].
    llow = int(math.floor(proptrun * M))
    lupp = M - int(math.floor(proptrun * M))
    th22kn = float(np.sum((mddest[llow:lupp] ** 2) * xcounts[llow:lupp]) / n)

    # Estimate sigma^2 using a local linear fit with a "direct plug-in"
    # bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / math.sqrt(2 * math.pi)) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    # Now compute a local linear kernel estimate of the variance.
    mest = locpoly(
        xcounts, ycounts, bandwidth=lamseh, range_x=range_x, binned=True
    )["y"]
    Sdg = sdiag(
        xcounts, bandwidth=lamseh, range_x=range_x, binned=True
    )["y"]
    SSTdg = sstdiag(
        xcounts, bandwidth=lamseh, range_x=range_x, binned=True
    )["y"]
    sigsqn = np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    # Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1 / 5))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    # For application of linear binning to a univariate data set.
    # This reimplements the Fortran routine F_linbin natively, since no
    # compiled _KernSmooth.linbin() binding is available.
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    gcnts = np.zeros(M, dtype=np.float64)
    if n == 0:
        return gcnts

    delta = (b - a) / (M - 1)
    lxi = ((X - a) / delta) + 1

    # Fortran's int() truncates toward zero, matching NumPy's float-to-int
    # array cast; this is not the same as floor() for negative values.
    li = lxi.astype(np.int64)
    rem = lxi - li

    in_range = (li >= 1) & (li < M)
    li_in = li[in_range]
    rem_in = rem[in_range]
    np.add.at(gcnts, li_in - 1, 1 - rem_in)
    np.add.at(gcnts, li_in, rem_in)

    if trun == 0:
        gcnts[0] += np.count_nonzero(li < 1)
        gcnts[M - 1] += np.count_nonzero(li >= M)

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    # Creates the grid counts from a bivariate data set X over an
    # equally-spaced set of grid points contained in gpoints1/gpoints2
    # using the linear binning strategy. This reimplements the Fortran
    # routine F_lbtwod natively, since no compiled
    # _KernSmooth.lbtwod() binding is available.
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)

    n = X.shape[0]
    M1 = gpoints1.shape[0]
    M2 = gpoints2.shape[0]
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Column-major layout: matrix(out[[9]], M1, M2) is (M1, M2)-shaped,
    # so we build the result directly with that shape (no order='F'
    # reshape needed since we index it directly rather than flattening).
    gcnts = np.zeros((M1, M2), dtype=np.float64)
    if n == 0:
        return gcnts

    x1 = X[:, 0]
    x2 = X[:, 1]

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    lxi1 = ((x1 - a1) / delta1) + 1
    lxi2 = ((x2 - a2) / delta2) + 1

    # Fortran's int() truncates toward zero, matching NumPy's
    # float-to-int array cast; this is not the same as floor() for
    # negative values.
    li1 = lxi1.astype(np.int64)
    li2 = lxi2.astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    # Observations outside the mesh are ignored. Note that the original
    # Fortran uses a strict li1 < M1 / li2 < M2 test (not <=), so a
    # point falling exactly on the upper boundary (b1 or b2) is also
    # dropped entirely, matching the reference implementation's quirk.
    in_range = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)
    li1_0 = li1[in_range] - 1
    li2_0 = li2[in_range] - 1
    r1 = rem1[in_range]
    r2 = rem2[in_range]

    np.add.at(gcnts, (li1_0, li2_0), (1 - r1) * (1 - r2))
    np.add.at(gcnts, (li1_0 + 1, li2_0), r1 * (1 - r2))
    np.add.at(gcnts, (li1_0, li2_0 + 1), (1 - r1) * r2)
    np.add.at(gcnts, (li1_0 + 1, li2_0 + 1), r1 * r2)

    return gcnts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'kernel' is accepted for API compatibility with the R signature but is
    # not used: KernSmooth's Fortran backend for locpoly only implements the
    # normal (Gaussian) kernel, regardless of what is passed here.
    del kernel

    x = np.asarray(x, dtype=np.float64)
    if y is not None:
        y = np.asarray(y, dtype=np.float64)

    # 'bandwidth' has no default in R; referencing it when not supplied
    # eventually raises an error there too, so we validate it up front.
    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.any(bandwidth_arr <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x) - np.min(x))
            range_x = (float(np.min(x) - extra), float(np.max(x) + extra))
        else:
            range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    if range_x is None:
        raise ValueError("argument 'range.x' is missing, with no default")
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4.0

    # Decide whether a density estimate or regression estimate is required.
    if y is None:  # obtain density estimate
        n = x.shape[0]
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
            xcounts = x
            ycounts = y
            M = xcounts.shape[0]
            gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine (1-based, matching R's 'indic') index of the closest
        # entry of 'hdisc' to each member of 'bandwidth'
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
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
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

    # Convert the 1-based bandwidth-level index ('indic' in R) to 0-based
    # for use as a NumPy index into 'hdisc'/'Lvec'.
    indic0 = indic - 1

    # --- Native reimplementation of Fortran routine F_locpol -----------
    # For each target grid point j, F_locpol accumulates weighted moments
    # of the counts (ss/tt) using a Gaussian kernel whose bandwidth is the
    # discretised value selected by 'indic0[j]', then solves the local
    # weighted-least-squares normal equations at each grid point.
    def _locpol(
        xcnts: np.ndarray[Any, np.dtype[np.float64]],
        ycnts: np.ndarray[Any, np.dtype[np.float64]],
        idrv: int,
        delta_: float,
        hdisc_: np.ndarray[Any, np.dtype[np.float64]],
        Lvec_: np.ndarray[Any, np.dtype[np.int64]],
        indic0_: np.ndarray[Any, np.dtype[np.int64]],
        M_: int,
        Q_: int,
        pp_: int,
        ppp_: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        ss = np.zeros((M_, ppp_), dtype=np.float64)
        tt = np.zeros((M_, pp_), dtype=np.float64)
        powers_idx = np.arange(ppp_, dtype=np.float64)

        for k in np.nonzero(xcnts)[0]:
            for i in range(Q_):
                L = int(Lvec_[i])
                jlo = max(0, k - L)
                jhi = min(M_ - 1, k + L)
                if jlo > jhi:
                    continue
                j_arr = np.arange(jlo, jhi + 1)
                mask = indic0_[j_arr] == i
                if not np.any(mask):
                    continue
                j_sel = j_arr[mask]
                diff = (k - j_sel).astype(np.float64)
                w = np.exp(-((delta_ * diff / hdisc_[i]) ** 2) / 2.0)
                dd = delta_ * diff
                powers = dd[:, None] ** powers_idx[None, :]
                ss[j_sel, :] += (xcnts[k] * w)[:, None] * powers
                tt[j_sel, :] += (ycnts[k] * w)[:, None] * powers[:, :pp_]

        curvest_ = np.zeros(M_, dtype=np.float64)
        for k in range(M_):
            Smat = np.empty((pp_, pp_), dtype=np.float64)
            for ii in range(pp_):
                Smat[ii, :] = ss[k, ii:ii + pp_]
            Tvec = tt[k, :].copy()
            sol = np.linalg.solve(Smat, Tvec)
            curvest_[k] = sol[idrv]
        return curvest_

    curvest = _locpol(
        xcounts, ycounts, drv, delta, hdisc, Lvec, indic0, M, Q, pp, ppp
    )

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For application of linear binning to a regression data set.
    # This reimplements the Fortran routine F_rlbin natively, since no
    # compiled _KernSmooth.rlbin() binding is available.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)
    if n == 0:
        return {"xcounts": xcnts, "ycounts": ycnts}

    delta = (b - a) / (M - 1)
    lxi = ((X - a) / delta) + 1

    # Fortran's int() truncates toward zero, matching NumPy's float-to-int
    # array cast; this is not the same as floor() for negative values.
    li = lxi.astype(np.int64)
    rem = lxi - li

    # Correction for right endpoint (not included if li == M): observations
    # exactly equal to b are always assigned full weight to the last
    # grid point, regardless of the truncate setting.
    at_b = (X == b)
    li = np.where(at_b, M - 1, li)
    rem = np.where(at_b, 1.0, rem)

    in_range = (li >= 1) & (li < M)
    li_in = li[in_range]
    rem_in = rem[in_range]
    y_in = Y[in_range]
    np.add.at(xcnts, li_in - 1, 1 - rem_in)
    np.add.at(xcnts, li_in, rem_in)
    np.add.at(ycnts, li_in - 1, (1 - rem_in) * y_in)
    np.add.at(ycnts, li_in, rem_in * y_in)

    if trun == 0:
        below = li < 1
        xcnts[0] += np.count_nonzero(below)
        ycnts[0] += np.sum(Y[below])

        above = li >= M
        xcnts[M - 1] += np.count_nonzero(above)
        ycnts[M - 1] += np.sum(Y[above])

    return {"xcounts": xcnts, "ycounts": ycnts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for API compatibility with the R
    # signature but are not used: the R 'sdiag' function never references
    # 'drv' anywhere in its body, and KernSmooth's Fortran backend for
    # 'sdiag' only implements the normal (Gaussian) kernel regardless of
    # what is passed here.
    del drv
    del kernel

    x = np.asarray(x, dtype=np.float64)

    # 'bandwidth' has no default in R; referencing it when not supplied
    # eventually raises an error there too, so we validate it up front.
    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    if range_x is None:
        raise ValueError("argument 'range.x' is missing, with no default")
    a = float(range_x[0])
    b = float(range_x[1])
    degree = int(degree)
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4.0

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = x
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine (1-based, matching R's 'indic') index of the closest
        # entry of 'hdisc' to each member of 'bandwidth'
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
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    # Convert the 1-based bandwidth-level index ('indic' in R) to 0-based
    # for use as a NumPy index into 'hdisc'/'Lvec'.
    indic0 = indic - 1

    # --- Native reimplementation of Fortran routine F_sdiag ---------------
    # For each grid point k with a nonzero bin count, F_sdiag accumulates
    # weighted moments of the counts ('ss') into every target bin j within
    # its kernel window using a Gaussian kernel whose bandwidth is the
    # discretised value selected by 'indic0[j]' (mirroring the ss-
    # accumulation logic used in 'locpoly'). It then assembles, for each
    # grid point, the Toeplitz local-regression design matrix 'Smat' from
    # those moments and returns the (1,1) entry of its inverse -- i.e. the
    # diagonal entry of the "hat"/smoother matrix for local polynomial
    # regression at that grid point (no response values are needed since
    # this is a property of the design/weights alone).
    ss = np.zeros((M, ppp), dtype=np.float64)
    powers_idx = np.arange(ppp, dtype=np.float64)

    for k in np.nonzero(xcounts)[0]:
        for i in range(Q):
            L = int(Lvec[i])
            jlo = max(0, k - L)
            jhi = min(M - 1, k + L)
            if jlo > jhi:
                continue
            j_arr = np.arange(jlo, jhi + 1)
            mask = indic0[j_arr] == i
            if not np.any(mask):
                continue
            j_sel = j_arr[mask]
            diff = (k - j_sel).astype(np.float64)
            w = np.exp(-((delta * diff / hdisc[i]) ** 2) / 2.0)
            dd = delta * diff
            powers = dd[:, None] ** powers_idx[None, :]
            ss[j_sel, :] += (xcounts[k] * w)[:, None] * powers

    Sdg = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = np.empty((pp, pp), dtype=np.float64)
        for ii in range(pp):
            Smat[ii, :] = ss[k, ii:ii + pp]
        Smat_inv = np.linalg.inv(Smat)
        Sdg[k] = Smat_inv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for API compatibility with the R
    # signature but are not used: the R 'sstdiag' function never
    # references 'drv' anywhere in its body, and KernSmooth's Fortran
    # backend for 'sstdiag' only implements the normal (Gaussian) kernel
    # regardless of what is passed here.
    del drv
    del kernel

    x = np.asarray(x, dtype=np.float64)

    # 'bandwidth' has no default in R; referencing it when not supplied
    # eventually raises an error there too, so we validate it up front.
    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    if range_x is None:
        raise ValueError("argument 'range.x' is missing, with no default")
    a = float(range_x[0])
    b = float(range_x[1])
    degree = int(degree)
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4.0

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = x
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine (1-based, matching R's 'indic') index of the closest
        # entry of 'hdisc' to each member of 'bandwidth'
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
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    # Convert the 1-based bandwidth-level index ('indic' in R) to 0-based
    # for use as a NumPy index into 'hdisc'/'Lvec'.
    indic0 = indic - 1

    # --- Native reimplementation of Fortran routine F_sstdg ---------------
    # For each grid point k with a nonzero bin count, F_sstdg accumulates
    # weighted moments of the counts into every target bin j within its
    # kernel window using a Gaussian kernel whose bandwidth is the
    # discretised value selected by 'indic0[j]'. Two accumulators are
    # built in parallel: 'ss' uses the plain kernel weight (as in
    # 'sdiag'/'locpoly'), while 'uu' uses the *squared* kernel weight, so
    # that 'Umat' captures the second-moment structure needed for the
    # diagonal of S*S^T (S being the local polynomial smoother matrix)
    # rather than just the diagonal of S itself. For each grid point, the
    # Toeplitz design matrix 'Smat' (from 'ss') and its S*S^T companion
    # 'Umat' (from 'uu') are assembled, and the diagonal entry of S*S^T is
    # computed as Sinv[0, :] @ Umat @ Sinv[:, 0], where Sinv is the
    # inverse of 'Smat' (mirroring the dgefa/dgedi inversion followed by
    # the double sum over Smat(1,i)*Umat(i,j)*Smat(j,1) in the Fortran
    # source).
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    powers_idx = np.arange(ppp, dtype=np.float64)

    for k in np.nonzero(xcounts)[0]:
        for i in range(Q):
            L = int(Lvec[i])
            jlo = max(0, k - L)
            jhi = min(M - 1, k + L)
            if jlo > jhi:
                continue
            j_arr = np.arange(jlo, jhi + 1)
            mask = indic0[j_arr] == i
            if not np.any(mask):
                continue
            j_sel = j_arr[mask]
            diff = (k - j_sel).astype(np.float64)
            w = np.exp(-((delta * diff / hdisc[i]) ** 2) / 2.0)
            dd = delta * diff
            powers = dd[:, None] ** powers_idx[None, :]
            ss[j_sel, :] += (xcounts[k] * w)[:, None] * powers
            uu[j_sel, :] += (xcounts[k] * (w ** 2))[:, None] * powers

    SSTd = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = np.empty((pp, pp), dtype=np.float64)
        Umat = np.empty((pp, pp), dtype=np.float64)
        for ii in range(pp):
            Smat[ii, :] = ss[k, ii:ii + pp]
            Umat[ii, :] = uu[k, ii:ii + pp]
        Sinv = np.linalg.inv(Smat)
        SSTd[k] = Sinv[0, :] @ Umat @ Sinv[:, 0]

    return {"x": gpoints, "y": SSTd}


def on_attach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def _on_unload(libpath: str) -> None:
    # In the original R package, .onUnload is a package-unload hook that
    # calls library.dynam.unload("KernSmooth", libpath) to unload the
    # compiled Fortran shared library (DLL/.so) associated with the package
    # when its namespace is unloaded.
    #
    # This Python translation of KernSmooth reimplements the Fortran
    # routines natively in Python/NumPy rather than loading a compiled
    # extension module, so there is no shared library to unload and no
    # equivalent unload mechanism is required. This function is kept as a
    # no-op stub purely for structural parity with the original R source.
    return None
