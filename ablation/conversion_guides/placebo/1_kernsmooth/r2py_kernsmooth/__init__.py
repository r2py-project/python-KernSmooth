import math
import warnings
from typing import Any

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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'arg' should be one of " + ", ".join(f'"{k}"' for k in valid_kernels))

    # Rename common variables.
    n = x.shape[0]
    M = gridsize

    # Set canonical scaling factors.
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

    # Set default bandwidth.
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.sqrt(np.var(x, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values.
    tau = 4.0 if kernel == "normal" else 1.0

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
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
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
    # we need P >= 2L+1, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1, dtype=np.float64), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M, dtype=np.float64)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)

    # R's fft(z, inverse = TRUE) is the *unnormalized* inverse transform,
    # i.e. P * numpy's (normalized) ifft. Dividing by P therefore reduces
    # exactly to numpy's ifft, so no extra scaling by P is needed here.
    y = np.fft.ifft(kappa_fft * gcounts_fft).real[:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float], gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: tuple[tuple[float, float], tuple[float, float]] | list[tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
    h_check = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.min(h_check) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables.
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64)).copy()
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction if only a single bandwidth is given.
    if h.shape[0] == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [
            (float(np.min(x[:, 0])) - 1.5 * h[0], float(np.max(x[:, 0])) + 1.5 * h[0]),
            (float(np.min(x[:, 1])) - 1.5 * h[1], float(np.max(x[:, 1])) + 1.5 * h[1]),
        ]

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data.
    M1 = int(M[0])
    M2 = int(M[1])
    gpoints1 = np.linspace(a[0], b[0], M1)
    gpoints2 = np.linspace(a[1], b[1], M2)

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights.
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.zeros((1, 1)), np.zeros((1, 1))]
    for id_ in range(2):
        L[id_] = min(
            int(np.floor(tau * h[id_] * (M[id_] - 1) / (b[id_] - a[id_]))),
            int(M[id_]) - 1,
        )
        lvecid = np.arange(0, L[id_] + 1)
        facid = (b[id_] - a[id_]) / (h[id_] * (M[id_] - 1))
        z = (norm.pdf(lvecid * facid) / h[id_]).reshape(-1, 1)
        tot = float(np.sum(np.concatenate([z.ravel(), z.ravel()[1:][::-1]])) * facid * h[id_])
        kapid[id_] = z / tot

    kapp = kapid[0] @ kapid[1].T / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate.
    P = (2 ** np.ceil(np.log(M + L) / np.log(2))).astype(np.int64)  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0:L1 + 1, 0:L2 + 1] = kapp
    if L1:
        rp[P1 - L1:P1, 0:L2 + 1] = kapp[L1:0:-1, :]
    if L2:
        rp[:, P2 - L2:P2] = rp[:, L2:0:-1]
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    # R's fft(z, inverse = TRUE)/(P1*P2) (2D case) is numpy's ifft2(z)
    # since numpy's ifft already includes the 1/(P1*P2) normalisation.
    rp = np.fft.ifft2(rp_fft * sp_fft).real[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative.
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        range_x = np.array([np.min(x), np.max(x)], dtype=np.float64)

    # Rename variables.
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])
    h = bandwidth

    # Bin the data if not already binned.
    if not binned:
        gpoints = np.linspace(a, b, gridsize)
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
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
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
            hmold0 = hmold1       # Compute mth degree Hermite polynomial
            hmold1 = hmnew        # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate.
    # we need P >= 2L+1, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1, dtype=np.float64), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M, dtype=np.float64)])
    kappam_fft = np.fft.fft(kappam)
    Gcounts_fft = np.fft.fft(Gcounts)

    # R's fft(z, inverse = TRUE) is the *unnormalized* inverse transform,
    # i.e. P * numpy's (normalized) ifft. Dividing by P therefore reduces
    # exactly to numpy's ifft, so no extra scaling by P is needed here.
    conv = np.fft.ifft(kappam_fft * Gcounts_fft).real

    return float(np.sum(gcounts * conv[:M]) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]

    # Sort the (x, y) data with respect to the x's.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    qq = q + 1

    # Equivalent of the Fortran subroutine `blkest`: splits the sorted
    # (x, y) data into `Nval` blocks, fits a degree-q polynomial within
    # each block via a QR-based least-squares solve, and accumulates
    # the residual sum of squares plus the block contributions to the
    # th22 and th24 functionals (based on the 2nd and 4th derivatives
    # of each block's fitted polynomial).
    idiv = n // Nval
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    exponents = np.arange(qq)  # 0, 1, ..., q (0-based power/coefficient index)

    for j in range(1, Nval + 1):
        # 1-based block bounds, as in the Fortran code.
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n

        Xj = x[ilow - 1:iupp]
        Yj = y[ilow - 1:iupp]

        # Set up the design matrix and obtain a q'th degree least-squares
        # fit over the current block (equivalent to the dqrdc/dqrsl QR
        # factorisation and solve used in the original Fortran code).
        Xmat = Xj[:, None] ** exponents[None, :]
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        fiti = Xmat @ coef
        RSS += np.sum((Yj - fiti) ** 2)

        # Second derivative of the fitted polynomial at each Xj: sum over
        # m >= 2 of m*(m-1)*coef[m]*Xj**(m-2).
        ddm = np.zeros_like(Xj)
        for m in range(2, qq):
            ddm = ddm + m * (m - 1) * coef[m] * Xj ** (m - 2)

        # Fourth derivative of the fitted polynomial at each Xj: sum over
        # m >= 4 of m*(m-1)*(m-2)*(m-3)*coef[m]*Xj**(m-4).
        ddddm = np.zeros_like(Xj)
        for m in range(4, qq):
            ddddm = ddddm + m * (m - 1) * (m - 2) * (m - 3) * coef[m] * Xj ** (m - 4)

        th22e += np.sum(ddm ** 2)
        th24e += np.sum(ddm * ddddm)

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]

    # Sort the (X, Y) data with respect to the X's.
    order = np.argsort(X, kind="stable")
    X = X[order]
    Y = Y[order]

    qq = q + 1

    # Equivalent of the Fortran subroutine `cp`: for each candidate number
    # of blocks Nval = 1..Nmax, split the sorted (X, Y) data into Nval
    # (nearly equal-sized) blocks, fit a degree-q polynomial within each
    # block via a QR-based least-squares solve, and accumulate the
    # residual sum of squares (RSS) across blocks for that Nval.
    RSS = np.zeros(Nmax, dtype=np.float64)
    exponents = np.arange(qq)  # 0, 1, ..., q (0-based power/coefficient index)

    for Nval in range(1, Nmax + 1):
        idiv = n // Nval
        RSSj_total = 0.0
        for j in range(1, Nval + 1):
            # 1-based block bounds, as in the Fortran code.
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n

            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]

            # Set up the design matrix and obtain a q'th degree
            # least-squares fit over the current block (equivalent to the
            # dqrdc/dqrsl QR factorisation and solve used in the original
            # Fortran code).
            Xmat = Xj[:, None] ** exponents[None, :]
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fiti = Xmat @ coef
            RSSj_total += np.sum((Yj - fiti) ** 2)

        RSS[Nval - 1] = RSSj_total

    # Now compute the array of Mallow's C_p values, one per candidate
    # number of blocks i = 1..Nmax.
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    # R's `order(Cpvec)[1L]` returns the 1-based *position* of the smallest
    # Cp value in Cpvec. Because position i in Cpvec was computed for a fit
    # using Nval = i blocks, that position numerically equals the optimal
    # number of blocks itself (a count, not an array index). We therefore
    # return `argmin index + 1` here: this is numerically identical to the
    # value the R function returns and matches how the caller (`dpill`)
    # uses the result -- as a block count to be plugged back into further
    # 1-based block computations -- rather than as a Python array index.
    best_Nval = int(np.argmin(Cpvals)) + 1

    return best_Nval


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    # Rename variables.
    n = x.shape[0]
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data.
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate.
    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'arg' should be one of " + ", ".join(repr(s) for s in valid_scalest))

    if scalest == "stdev":
        scale_val = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")) / 1.349
    else:
        scale_val = min(
            (np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability.
    x_mean = np.mean(x)
    sx = (x - x_mean) / scale_val
    sa = (a - x_mean) / scale_val
    sb = (b - x_mean) / scale_val

    # Set up grid points and bin the data.
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps.
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


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    valid_kernel = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernel:
        raise ValueError("'arg' should be one of " + ", ".join(repr(s) for s in valid_kernel))

    x = np.asarray(x, dtype=np.float64)

    # Set kernel constants.
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

    # Rename variables.
    n = x.shape[0]
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data.
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate.
    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'arg' should be one of " + ", ".join(repr(s) for s in valid_scalest))

    if scalest == "stdev":
        scale_val = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")) / 1.349
    else:
        scale_val = min(
            (np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability.
    x_mean = np.mean(x)
    sx = (x - x_mean) / scale_val
    sa = (a - x_mean) / scale_val
    sb = (b - x_mean) / scale_val

    # Set up grid points and bin the data.
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps.
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


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    #
    # Equivalent of R's `cbind(x, y)` followed by `xy[sort.list(xy[, 1L]), ]`:
    # sort x and y together by x, using a stable sort to match R's default
    # `sort.list` tie-breaking behaviour.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    indlow = int(np.floor(trim * x.shape[0])) + 1
    indupp = x.shape[0] - int(np.floor(trim * x.shape[0]))

    # 1-based, inclusive-on-both-ends R slice `x[indlow:indupp]` becomes
    # the 0-based, end-exclusive Python slice `x[indlow - 1:indupp]`.
    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    ## Rename common parameters
    n = x.shape[0]
    M = int(gridsize)

    # R evaluates the default `range.x = range(x)` lazily, the first time
    # `range.x` is referenced in the body (i.e. at `a <- range.x[1L]`),
    # by which point `x` has already been reassigned to the trimmed,
    # sorted subset above. Reproduce that here: only fall back to the
    # range of the (already trimmed) `x` when the caller did not supply
    # `range_x` explicitly.
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

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
    # These two branches are mutually exclusive in practice (th24Q cannot
    # be both negative and positive); they are kept as sequential `if`
    # statements -- matching the original R code -- rather than
    # `if`/`elif`, so that if `th24Q == 0` exactly, `gamseh` falls through
    # unchanged, exactly as in R.
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
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Equivalent of the Fortran subroutine `linbin`: obtains bin counts for
    # univariate data via the linear binning strategy.
    gcnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)

    # lxi and li (1-based, as in the Fortran code) computed for every
    # observation at once.
    lxi = ((X - a) / delta) + 1.0
    li = np.trunc(lxi).astype(np.int64)  # Fortran INT() truncates toward zero
    rem = lxi - li

    in_range = (li >= 1) & (li < M)
    li_in = li[in_range]
    rem_in = rem[in_range]

    # Convert the 1-based Fortran indices `li` and `li + 1` to 0-based
    # NumPy indices, accumulating contributions with np.add.at since
    # multiple observations may fall into the same bin.
    np.add.at(gcnts, li_in - 1, 1.0 - rem_in)
    np.add.at(gcnts, li_in, rem_in)

    if trun == 0:
        gcnts[0] += np.count_nonzero(li < 1)
        gcnts[M - 1] += np.count_nonzero(li >= M)

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)

    # n <- nrow(X); X <- c(X[, 1L], X[, 2L])
    x1 = X[:, 0]
    x2 = X[:, 1]

    M1 = gpoints1.shape[0]
    M2 = gpoints2.shape[0]
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Equivalent of the Fortran subroutine `lbtwod`: obtains bin counts for
    # bivariate data via the linear binning strategy. Observations outside
    # the mesh are ignored.
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    lxi1 = ((x1 - a1) / delta1) + 1.0
    lxi2 = ((x2 - a2) / delta2) + 1.0

    # Find the integer part of "lxi1" and "lxi2" (1-based, as in the
    # Fortran code); Fortran INT() truncates toward zero.
    li1 = np.trunc(lxi1).astype(np.int64)
    li2 = np.trunc(lxi2).astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    in_range = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)
    li1_in = li1[in_range] - 1  # convert to 0-based row index
    li2_in = li2[in_range] - 1  # convert to 0-based column index
    rem1_in = rem1[in_range]
    rem2_in = rem2[in_range]

    # Distribute each point's mass bilinearly among its 4 surrounding
    # grid cell corners, accumulating with np.add.at since multiple
    # observations may fall into the same cell.
    np.add.at(gcnts, (li1_in, li2_in), (1.0 - rem1_in) * (1.0 - rem2_in))
    np.add.at(gcnts, (li1_in + 1, li2_in), rem1_in * (1.0 - rem2_in))
    np.add.at(gcnts, (li1_in, li2_in + 1), (1.0 - rem1_in) * rem2_in)
    np.add.at(gcnts, (li1_in + 1, li2_in + 1), rem1_in * rem2_in)

    return gcnts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
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
            range_x = (float(np.min(x) - extra), float(np.max(x) + extra))
        else:
            range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables.
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Decide whether a density estimate or regression estimate is required.
    if y is None:  # obtain density estimate
        n = x.shape[0]
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        y = np.asarray(y, dtype=np.float64)
        # Bin the data if not already binned.
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

    # Set the bin width.
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths.
    if bandwidth is None:
        raise ValueError('argument "bandwidth" is missing, with no default')
    bw = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    if bw.shape[0] == M:
        sorted_bw = np.sort(bw)
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
                indic = np.round(((np.log(bw) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bw.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bw[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bw[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Call FORTRAN routine "locpol" -- reimplemented directly in NumPy below.
    #
    # For every target grid point j, the Fortran code accumulates, over the
    # window of source bins within Lvec[indic[j]] of j, the kernel-weighted
    # moments of the x- and y- bin counts (powers of delta*(k-j) up to
    # ppp-1 and pp-1 respectively). These moments form a small Hankel-
    # structured (pp x pp) system of normal equations whose solution's
    # (drv)-th entry (0-based) is the local polynomial estimate at j.
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)
    power_ss = np.arange(ppp, dtype=np.float64)

    for j in range(M):
        level = int(indic[j]) - 1
        L = int(Lvec[level])
        h = hdisc[level]
        lo = max(0, j - L)
        hi = min(M - 1, j + L)
        k_idx = np.arange(lo, hi + 1)
        dist = (k_idx - j).astype(np.float64)
        kernel_val = np.exp(-((delta * dist / h) ** 2) / 2.0)
        fac = (delta * dist)[:, None] ** power_ss[None, :]
        ss[j, :] = (xcounts[k_idx] * kernel_val) @ fac
        tt[j, :] = (ycounts[k_idx] * kernel_val) @ fac[:, :pp]

    idx_mat = np.add.outer(np.arange(pp), np.arange(pp))
    curvest = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = ss[k][idx_mat]
        Tvec = tt[k]
        sol = np.linalg.solve(Smat, Tvec)
        curvest[k] = sol[drv]

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def on_attach(libname: str, pkgname: str) -> None:
    # Equivalent of R's .onAttach package lifecycle hook, which is invoked
    # automatically when the package is loaded (e.g. via `library()`).
    # Python has no direct analogue of this hook, so it is exposed as a
    # plain function with the same signature that can be called manually
    # or from an `__init__.py` if desired.
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")


def _on_unload(libpath: str) -> None:
    # R's .onUnload hook called library.dynam.unload("KernSmooth", libpath)
    # to unload the package's compiled shared library when the package was
    # detached. This Python port is a pure-Python/NumPy reimplementation
    # with no compiled shared library (no `.Fortran()` calls remain -- all
    # Fortran routines were reimplemented in NumPy), so there is nothing to
    # unload here. Kept as a no-op stub with the original signature for
    # parity with the R package's lifecycle hook.
    return None


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Equivalent of the Fortran subroutine `rlbin`: obtains bin counts for
    # univariate regression data via the linear binning strategy, forming
    # both the (unweighted) x bin counts and the y-weighted bin sums.
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)

    # lxi and li (1-based, as in the Fortran code) computed for every
    # observation at once.
    lxi = ((X - a) / delta) + 1.0
    li = np.trunc(lxi).astype(np.int64)  # Fortran INT() truncates toward zero
    rem = lxi - li

    # Correction for the right endpoint (not included if li == M): force
    # observations exactly equal to `b` into the last bin interval.
    at_b = X == b
    li = np.where(at_b, M - 1, li)
    rem = np.where(at_b, 1.0, rem)

    in_range = (li >= 1) & (li < M)
    li_in = li[in_range]
    rem_in = rem[in_range]
    Y_in = Y[in_range]

    # Convert the 1-based Fortran indices `li` and `li + 1` to 0-based
    # NumPy indices, accumulating contributions with np.add.at since
    # multiple observations may fall into the same bin.
    np.add.at(xcnts, li_in - 1, 1.0 - rem_in)
    np.add.at(xcnts, li_in, rem_in)
    np.add.at(ycnts, li_in - 1, (1.0 - rem_in) * Y_in)
    np.add.at(ycnts, li_in, rem_in * Y_in)

    if trun == 0:
        below = li < 1
        above = li >= M
        xcnts[0] += np.count_nonzero(below)
        ycnts[0] += np.sum(Y[below])
        xcnts[M - 1] += np.count_nonzero(above)
        ycnts[M - 1] += np.sum(Y[above])

    return {"xcounts": xcnts, "ycounts": ycnts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables.
    #
    # NOTE: 'drv' is accepted for interface compatibility with the R
    # function but, exactly as in the original R/Fortran implementation,
    # it is never used below -- .Fortran(F_sdiag) is never passed 'drv',
    # so the diagonal entries computed always correspond to the drv = 0
    # (function estimate) self-weight, regardless of the 'drv' argument.
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned.
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = x
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width.
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths.
    if bandwidth is None:
        raise ValueError('argument "bandwidth" is missing, with no default')
    bw = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    if bw.shape[0] == M:
        sorted_bw = np.sort(bw)
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
                indic = np.round(((np.log(bw) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bw.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bw[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bw[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Call FORTRAN routine "sdiag" -- reimplemented directly in NumPy below.
    #
    # The Fortran code loops over source bins k and spreads each bin's
    # kernel-weighted moment contributions to every target bin j within
    # the window Lvec[indic[j]] of k, gated on indic(j) matching the
    # bandwidth level being processed. Because the window test
    # |k - j| <= Lvec(indic(j)) is symmetric in k and j, this is
    # mathematically identical to looping over each target j and, using
    # its own assigned bandwidth level indic[j], gathering the moment
    # contributions from every source bin k within that window -- the
    # same per-target-j moment accumulation used for 'locpoly'. Unlike
    # 'locpoly', only the x-count moments 'ss' are needed here (there is
    # no response/y-count moment vector), since 'sdiag' returns the
    # smoother matrix's diagonal self-weight rather than a fitted value.
    ss = np.zeros((M, ppp), dtype=np.float64)
    power_ss = np.arange(ppp, dtype=np.float64)

    for j in range(M):
        level = int(indic[j]) - 1
        L = int(Lvec[level])
        h = hdisc[level]
        lo = max(0, j - L)
        hi = min(M - 1, j + L)
        k_idx = np.arange(lo, hi + 1)
        dist = (k_idx - j).astype(np.float64)
        kernel_val = np.exp(-((delta * dist / h) ** 2) / 2.0)
        fac = (delta * dist)[:, None] ** power_ss[None, :]
        ss[j, :] = (xcounts[k_idx] * kernel_val) @ fac

    # For each grid point k, assemble the (pp x pp) Hankel-structured
    # moment matrix Smat from the accumulated moments 'ss', invert it
    # (equivalent to the Fortran dgefa/dgedi LU-factorisation and
    # inversion calls), and take the (1, 1) entry (0-based (0, 0)) of
    # the inverse as the diagonal self-weight Sdg[k] -- the weight the
    # local polynomial fit at grid point k places on its own bin.
    idx_mat = np.add.outer(np.arange(pp), np.arange(pp))
    Sdg = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = ss[k][idx_mat]
        Sdg[k] = np.linalg.inv(Smat)[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables.
    #
    # NOTE: 'drv' is accepted for interface compatibility with the R
    # function but, exactly as in the original R/Fortran implementation,
    # it is never used below -- .Fortran(F_sstdg) is never passed 'drv',
    # so the diagonal entries of SS^T computed always correspond to the
    # drv = 0 (function estimate) smoother, regardless of the 'drv'
    # argument.
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned.
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = x
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width.
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths.
    if bandwidth is None:
        raise ValueError('argument "bandwidth" is missing, with no default')
    bw = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    if bw.shape[0] == M:
        sorted_bw = np.sort(bw)
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
                indic = np.round(((np.log(bw) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bw.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bw[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bw[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Call FORTRAN routine "sstdg" -- reimplemented directly in NumPy below.
    #
    # As in 'sdiag', the Fortran code loops over source bins k and spreads
    # each bin's kernel-weighted moment contributions to every target bin
    # j within the window Lvec[indic[j]] of k, gated on indic(j) matching
    # the bandwidth level being processed. Because the window test
    # |k - j| <= Lvec(indic(j)) is symmetric in k and j, this is
    # mathematically identical to looping over each target j and, using
    # its own assigned bandwidth level indic[j], gathering the moment
    # contributions from every source bin k within that window. Unlike
    # 'sdiag', two moment arrays are accumulated per target j: 'ss' built
    # from the raw kernel weight fkap(k-j) (as in 'sdiag'), and 'uu' built
    # from the *squared* kernel weight fkap(k-j)**2, both sharing the same
    # polynomial factor (delta*(k-j))**power.
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    power_ss = np.arange(ppp, dtype=np.float64)

    for j in range(M):
        level = int(indic[j]) - 1
        L = int(Lvec[level])
        h = hdisc[level]
        lo = max(0, j - L)
        hi = min(M - 1, j + L)
        k_idx = np.arange(lo, hi + 1)
        dist = (k_idx - j).astype(np.float64)
        kernel_val = np.exp(-((delta * dist / h) ** 2) / 2.0)
        fac = (delta * dist)[:, None] ** power_ss[None, :]
        ss[j, :] = (xcounts[k_idx] * kernel_val) @ fac
        uu[j, :] = (xcounts[k_idx] * (kernel_val ** 2)) @ fac

    # For each grid point k, assemble the (pp x pp) Hankel-structured
    # moment matrices Smat (from 'ss') and Umat (from 'uu'), invert Smat
    # (equivalent to the Fortran dgefa/dgedi LU-factorisation and
    # inversion calls), and combine as e_0^T Smat^{-1} Umat Smat^{-1} e_0
    # -- the diagonal entry of S S^T at grid point k, i.e. the squared
    # norm of the local polynomial smoother's weight row for that point.
    idx_mat = np.add.outer(np.arange(pp), np.arange(pp))
    SSTd = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = ss[k][idx_mat]
        Umat = uu[k][idx_mat]
        Sinv = np.linalg.inv(Smat)
        SSTd[k] = Sinv[0, :] @ Umat @ Sinv[:, 0]

    return {"x": gpoints, "y": SSTd}
