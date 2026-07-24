import math
import warnings
from typing import Any

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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # match.arg: resolve (possibly abbreviated) kernel name
    kernel_choices = ["normal", "box", "epanech", "biweight", "triweight"]
    matches = [k for k in kernel_choices if k.startswith(kernel)]
    if len(matches) != 1:
        raise ValueError(
            "'kernel' should be one of " + ", ".join(repr(k) for k in kernel_choices)
        )
    kernel = matches[0]

    # Rename common variables
    n = x.shape[0]
    M = gridsize

    # Set canonical scaling factors
    if kernel == "normal":
        del0 = (1.0 / (4.0 * math.pi)) ** (1.0 / 10.0)
    elif kernel == "box":
        del0 = (9.0 / 2.0) ** (1.0 / 5.0)
    elif kernel == "epanech":
        del0 = 15.0 ** (1.0 / 5.0)
    elif kernel == "biweight":
        del0 = 35.0 ** (1.0 / 5.0)
    else:  # triweight
        del0 = (9450.0 / 143.0) ** (1.0 / 5.0)

    if not isinstance(canonical, (bool, np.bool_)):
        raise TypeError("'canonical' must be a length-1 logical vector")

    # Set default bandwidth
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * math.sqrt(float(np.var(x, ddof=1)))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values
    tau = 4.0 if kernel == "normal" else 1.0

    if range_x is None:
        a = float(np.min(x)) - tau * h
        b = float(np.max(x)) + tau * h
    else:
        a = range_x[0]
        b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = min(int(math.floor(tau / delta)), M)
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1, dtype=np.float64)

    def _dnorm(z: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)

    def _dbeta(z: np.ndarray[Any, np.dtype[np.float64]], shape1: int, shape2: int) -> np.ndarray[Any, np.dtype[np.float64]]:
        # Beta(shape1, shape2) density; shape1 == shape2 are small positive
        # integers here, so the (shape - 1) powers are safe at the
        # z == 0 / z == 1 boundaries (0 ** 0 == 1).
        norm_const = math.gamma(shape1) * math.gamma(shape2) / math.gamma(shape1 + shape2)
        return (z ** (shape1 - 1)) * ((1.0 - z) ** (shape2 - 1)) / norm_const

    if kernel == "normal":
        kappa = _dnorm(lvec * delta) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * _dbeta(0.5 * (lvec * delta + 1.0), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * _dbeta(0.5 * (lvec * delta + 1.0), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * _dbeta(0.5 * (lvec * delta + 1.0), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * _dbeta(0.5 * (lvec * delta + 1.0), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = 2 ** int(math.ceil(math.log(M + L + 1) / math.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1, dtype=np.float64), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M, dtype=np.float64)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)

    # R's fft(z, inverse = TRUE) is the unnormalised inverse transform
    # (equal to P times numpy's normalised ifft); dividing by P in the
    # original R expression exactly cancels this factor, so numpy's
    # normalised ifft can be used directly here.
    conv = np.fft.ifft(kappa_fft * gcounts_fft)

    return {"x": gpoints, "y": np.real(conv)[:M]}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] | np.ndarray[Any, np.dtype[np.integer[Any]]] = (51, 51), range_x: list[tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x_arr = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables
    n = x_arr.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    if h.shape[0] == 1:
        h = np.array([h[0], h[0]])

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [None, None]
        for idd in range(2):
            range_x[idd] = (
                float(np.min(x_arr[:, idd]) - 1.5 * h[idd]),
                float(np.max(x_arr[:, idd]) + 1.5 * h[idd]),
            )

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x_arr, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]] | None] = [None, None]
    for idd in range(2):
        L[idd] = min(
            int(np.floor(tau * h[idd] * (M[idd] - 1) / (b[idd] - a[idd]))),
            int(M[idd]) - 1,
        )
        lvecid = np.arange(0, L[idd] + 1)
        facid = (b[idd] - a[idd]) / (h[idd] * (M[idd] - 1))
        # dnorm(lvecid * facid) for the standard normal density
        z = (np.exp(-0.5 * (lvecid * facid) ** 2) / np.sqrt(2.0 * np.pi)) / h[idd]
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * facid * h[idd]
        kapid[idd] = z / tot

    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log(M + L) / np.log(2))).astype(np.int64)  # smallest powers of 2 >= M+L
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
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    # R's fft(z, inverse = TRUE) is the unnormalised inverse transform
    # (equal to P1*P2 times numpy's normalised ifft2); dividing by
    # (P1*P2) in the original R expression exactly cancels this factor,
    # so numpy's normalised ifft2 can be used directly here.
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> float:
    x_arr = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename variables
    M = gridsize
    a = range_x[0]
    b = range_x[1]
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
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    # dnorm(arg) for the standard normal density
    kappam = (np.exp(-0.5 * arg ** 2) / np.sqrt(2.0 * np.pi)) / (h ** (drv + 1))
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
    kappam_fft = np.fft.fft(kappam)
    Gcounts_fft = np.fft.fft(Gcounts)

    # R's fft(z, inverse = TRUE) is the unnormalised inverse transform
    # (equal to P times numpy's normalised ifft); dividing by P in the
    # original R expression exactly cancels this factor, so numpy's
    # normalised ifft can be used directly here.
    conv = np.fft.ifft(kappam_fft * Gcounts_fft)

    return float(np.sum(gcounts * np.real(conv)[:M]) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]

    # Sort the (x, y) data with respect to the x's (stable sort mirrors
    # R's sort.list applied to the combined data matrix).
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    qq = q + 1

    # Reimplementation of the Fortran subroutine F_blkest (blkest.f):
    # partitions the sorted (x, y) data into Nval blocks, fits a q'th
    # degree polynomial (via least squares, equivalent to the QR-based
    # dqrdc/dqrsl fit used in the Fortran code) within each block, and
    # aggregates residuals and derivative-based quantities to obtain
    # sigsqe, th22e and th24e.
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0
    idiv = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        nj = iupp - ilow + 1

        Xj = x[ilow - 1:iupp]
        Yj = y[ilow - 1:iupp]

        # Obtain a q'th degree fit over the current member of the
        # partition. Set up the design matrix.
        Xmat = np.empty((nj, qq), dtype=np.float64)
        Xmat[:, 0] = 1.0
        for k in range(2, qq + 1):
            Xmat[:, k - 1] = Xj ** (k - 1)

        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        for i in range(nj):
            fiti = coef[0]
            ddm = 2 * coef[2]
            ddddm = 24 * coef[4]
            for k in range(2, qq + 1):
                fiti = fiti + coef[k - 1] * Xj[i] ** (k - 1)
                if k <= (q - 1):
                    ddm = ddm + k * (k + 1) * coef[k + 1] * Xj[i] ** (k - 1)
                    if k <= (q - 3):
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

    # Sort the (X, Y) data with respect to the X's (stable sort mirrors
    # R's sort.list applied to the combined data matrix).
    order = np.argsort(X, kind="stable")
    X = X[order]
    Y = Y[order]

    qq = q + 1

    # Reimplementation of the Fortran subroutine F_cp (cp.f): for each
    # candidate number of blocks Nval from 1 to Nmax, partitions the
    # sorted (X, Y) data into Nval blocks, fits a q'th degree polynomial
    # per block via least squares (equivalent to the QR-based
    # dqrdc/dqrsl fit used in the Fortran code), accumulates the
    # residual sum of squares (RSS), and finally computes Mallow's C_p
    # statistic for each Nval.
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # For each number of partitions
        idiv = n // Nval
        for j in range(1, Nval + 1):
            # For each member of the partition
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n
            nj = iupp - ilow + 1

            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]

            # Obtain a q'th degree fit over the current member of the
            # partition. Set up the design matrix.
            Xmat = np.empty((nj, qq), dtype=np.float64)
            Xmat[:, 0] = 1.0
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            RSSj = 0.0
            for i in range(nj):
                fiti = coef[0]
                for k in range(2, qq + 1):
                    fiti = fiti + coef[k - 1] * Xj[i] ** (k - 1)
                RSSj = RSSj + (Yj[i] - fiti) ** 2

            RSS[Nval - 1] = RSS[Nval - 1] + RSSj

    # Now compute array of Mallow's C_p values.
    Cpvals = np.empty(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    # order(Cpvec)[1L] in R returns the 1-based index of the minimum
    # C_p value, which already equals the chosen number of blocks Nval
    # since Cpvals is indexed by Nval = 1, ..., Nmax; this corresponds
    # to the 0-based argmin index plus one in Python.
    return int(np.argmin(Cpvals) + 1)


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename variables
    n = x.shape[0]
    M = gridsize
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate
    scalest_choices = ["minim", "stdev", "iqr"]
    matches = [s for s in scalest_choices if s.startswith(scalest)]
    if len(matches) != 1:
        raise ValueError(
            "'scalest' should be one of " + ", ".join(repr(s) for s in scalest_choices)
        )
    scalest = matches[0]

    if scalest == "stdev":
        scale_value = float(np.sqrt(np.var(x, ddof=1)))
    elif scalest == "iqr":
        q75 = float(np.quantile(x, 0.75))
        q25 = float(np.quantile(x, 0.25))
        scale_value = (q75 - q25) / 1.349
    else:  # minim
        q75 = float(np.quantile(x, 0.75))
        q25 = float(np.quantile(x, 0.25))
        scale_value = min((q75 - q25) / 1.349, float(np.sqrt(np.var(x, ddof=1))))

    if scale_value == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability
    x_mean = float(np.mean(x))
    sx = (x - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    # Set up grid points and bin the data
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps
    if level == 0:
        hpi = (24.0 * np.sqrt(np.pi) / n) ** (1.0 / 3.0)
    elif level == 1:
        alpha = (2.0 / (3.0 * n)) ** (1.0 / 5.0) * np.sqrt(2.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 2:
        alpha = ((2.0 / (5.0 * n)) ** (1.0 / 7.0)) * np.sqrt(2.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2.0 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 3:
        alpha = ((2.0 / (7.0 * n)) ** (1.0 / 9.0)) * np.sqrt(2.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2.0 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 4:
        alpha = ((2.0 / (9.0 * n)) ** (1.0 / 11.0)) * np.sqrt(2.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2.0 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    else:  # level == 5
        alpha = ((2.0 / (11.0 * n)) ** (1.0 / 13.0)) * np.sqrt(2.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2.0 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)

    return scale_value * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    # match.arg: resolve (possibly abbreviated) kernel name
    kernel_choices = ["normal", "box", "epanech", "biweight", "triweight"]
    kernel_matches = [k for k in kernel_choices if k.startswith(kernel)]
    if len(kernel_matches) != 1:
        raise ValueError(
            "'kernel' should be one of " + ", ".join(repr(k) for k in kernel_choices)
        )
    kernel = kernel_matches[0]

    # Set kernel constants
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

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename variables
    n = x.shape[0]
    M = gridsize
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate
    scalest_choices = ["minim", "stdev", "iqr"]
    scalest_matches = [s for s in scalest_choices if s.startswith(scalest)]
    if len(scalest_matches) != 1:
        raise ValueError(
            "'scalest' should be one of " + ", ".join(repr(s) for s in scalest_choices)
        )
    scalest = scalest_matches[0]

    if scalest == "stdev":
        scale_value = float(np.sqrt(np.var(x, ddof=1)))
    elif scalest == "iqr":
        q75 = float(np.quantile(x, 0.75))
        q25 = float(np.quantile(x, 0.25))
        scale_value = (q75 - q25) / 1.349
    else:  # minim
        q75 = float(np.quantile(x, 0.75))
        q25 = float(np.quantile(x, 0.25))
        scale_value = min((q75 - q25) / 1.349, float(np.sqrt(np.var(x, ddof=1))))

    if scale_value == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability
    x_mean = float(np.mean(x))
    sx = (x - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    # Set up grid points and bin the data
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps
    if level == 0:
        psi4hat = 3.0 / (8.0 * np.sqrt(np.pi))
    elif level == 1:
        alpha = (2.0 * (np.sqrt(2.0)) ** 7 / (5.0 * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2.0 * (np.sqrt(2.0)) ** 9 / (7.0 * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2.0 * (np.sqrt(2.0)) ** 11 / (9.0 * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2.0 * (np.sqrt(2.0)) ** 13 / (11.0 * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:  # level == 5
        alpha = (2.0 * (np.sqrt(2.0)) ** 15 / (13.0 * n)) ** (1.0 / 15.0)  # bandwidth for psi_12
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = (945.0 * np.sqrt(2.0 / np.pi) / (psi12hat * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return scale_value * del0 * (1.0 / (psi4hat * n)) ** (1.0 / 5.0)


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    # Trim the 100(trim)% of the data from each end (in the x-direction).
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Sort the (x, y) data with respect to the x's (stable sort mirrors
    # R's sort.list applied to the combined data matrix).
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    n_full = x.shape[0]
    indlow = int(np.floor(trim * n_full)) + 1
    indupp = n_full - int(np.floor(trim * n_full))

    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    # Rename common parameters
    n = x.shape[0]
    M = gridsize

    # Note: R evaluates the default 'range.x = range(x)' lazily, at the
    # point it is first used in the body (after 'x' has already been
    # reassigned to the trimmed data above), so the default range is that
    # of the *trimmed* x, not the original untrimmed x. This is mirrored
    # here by computing the default only after trimming, using the
    # sentinel value 'None'.
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
        gamseh = (3 * gamseh / (8 * np.sqrt(np.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * np.sqrt(np.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh, range_x=range_x, binned=True)["y"]

    llow = int(np.floor(proptrun * M)) + 1
    lupp = M - int(np.floor(proptrun * M))
    th22kn = float(np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n)

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * np.sqrt(2) - (4 / 3) * np.sqrt(3)
    C3K = (4 * C3K / np.sqrt(2 * np.pi)) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    # Now compute a local linear kernel estimate of
    # the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh, range_x=range_x, binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh, range_x=range_x, binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh, range_x=range_x, binned=True)["y"]
    sigsqn = float(np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts))
    sigsqd = n - 2 * float(np.sum(Sdg * xcounts)) + float(np.sum(SSTdg * xcounts))
    sigsqkn = sigsqn / sigsqd

    # Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * np.sqrt(np.pi) * th22kn * n)) ** (1 / 5))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Reimplementation of the Fortran subroutine F_linbin (linbin.f):
    # performs linear binning of the univariate data X onto the grid
    # defined by M equally spaced points between a and b.
    gcnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)
    for i in range(n):
        lxi = ((X[i] - a) / delta) + 1

        # Find integer part of "lxi" (Fortran int() truncates toward zero)
        li = int(lxi)

        rem = lxi - li
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
    n = X.shape[0]
    M1 = gpoints1.shape[0]
    M2 = gpoints2.shape[0]
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Reimplementation of the Fortran subroutine F_lbtwod (linbin2D.f):
    # obtains bin counts for bivariate data via the linear binning
    # strategy on the M1 x M2 grid defined by gpoints1 x gpoints2.
    # Observations outside the mesh [a1, b1] x [a2, b2] are ignored.
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)
    for i in range(n):
        lxi1 = ((X[i, 0] - a1) / delta1) + 1
        lxi2 = ((X[i, 1] - a2) / delta2) + 1

        # Find integer part of "lxi1" and "lxi2" (Fortran int() truncates toward zero)
        li1 = int(lxi1)
        li2 = int(lxi2)
        rem1 = lxi1 - li1
        rem2 = lxi2 - li2

        if li1 >= 1 and li2 >= 1 and li1 < M1 and li2 < M2:
            gcnts[li1 - 1, li2 - 1] = gcnts[li1 - 1, li2 - 1] + (1 - rem1) * (1 - rem2)
            gcnts[li1, li2 - 1] = gcnts[li1, li2 - 1] + rem1 * (1 - rem2)
            gcnts[li1 - 1, li2] = gcnts[li1 - 1, li2] + (1 - rem1) * rem2
            gcnts[li1, li2] = gcnts[li1, li2] + rem1 * rem2

    return gcnts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64) if y is not None else None

    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.any(np.asarray(bandwidth, dtype=np.float64) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    if range_x is None and not binned:
        if y_arr is None:
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
    if y_arr is None:  # obtain density estimate
        n = x.shape[0]
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.full(M, 1.0, dtype=np.float64)
    else:  # obtain regression estimate
        # Bin the data if not already binned
        if not binned:
            gpoints = np.linspace(a, b, M)
            out = rlbin(x, y_arr, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = x
            ycounts = y_arr
            M = xcounts.shape[0]
            gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.shape[0] == M:
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
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1)
        else:
            indic = np.full(M, 1.0, dtype=np.float64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.full(M, 1.0, dtype=np.float64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.float64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    Lvec = Lvec.astype(np.int64)
    indic = indic.astype(np.int64)

    if int(np.min(Lvec)) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Reimplementation of the Fortran subroutine F_locpol (locpoly.f):
    # for each grid point, combine the (log-)discretised Gaussian kernel
    # weights with the binned counts to build weighted local-polynomial
    # moment accumulators "ss"/"tt", then solve the resulting pp x pp
    # weighted normal-equations system (equivalent to the LINPACK
    # dgefa/dgesl solve used in the Fortran code) at each grid point to
    # extract the drv-th derivative coefficient (scaled below by drv!).
    #
    # All arrays below are padded with an unused index 0 entry so that
    # the original 1-based Fortran indices (and index arithmetic such
    # as "k - j + midpts(i)") can be mirrored directly without having
    # to re-derive the offsets.
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    midpts = np.zeros(Q + 1, dtype=np.int64)
    ss = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    tt = np.zeros((M + 1, pp + 1), dtype=np.float64)
    cvest = np.zeros(M + 1, dtype=np.float64)

    xcnts = np.concatenate(([0.0], np.asarray(xcounts, dtype=np.float64)))
    ycnts = np.concatenate(([0.0], np.asarray(ycounts, dtype=np.float64)))
    Lvec_p = np.concatenate(([0], Lvec))
    hdisc_p = np.concatenate(([0.0], hdisc))
    indic_p = np.concatenate(([0], indic))

    # Obtain kernel weights
    mid = int(Lvec_p[1]) + 1
    for i in range(1, Q):
        midpts[i] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec_p[i]) + 1):
            fkap[mid + j] = math.exp(-((delta * j / hdisc_p[i]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lvec_p[i]) + int(Lvec_p[i + 1]) + 1
    midpts[Q] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lvec_p[Q]) + 1):
        fkap[mid + j] = math.exp(-((delta * j / hdisc_p[Q]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcnts[k] != 0:
            for i in range(1, Q + 1):
                lo = max(1, k - int(Lvec_p[i]))
                hi = min(M, k + int(Lvec_p[i]))
                for j in range(lo, hi + 1):
                    if indic_p[j] == i:
                        fac = 1.0
                        ss[j, 1] += xcnts[k] * fkap[k - j + midpts[i]]
                        tt[j, 1] += ycnts[k] * fkap[k - j + midpts[i]]
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcnts[k] * fkap[k - j + midpts[i]] * fac
                            if ii <= pp:
                                tt[j, ii] += ycnts[k] * fkap[k - j + midpts[i]] * fac

    # Solve the weighted local polynomial normal equations at each grid point
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Tvec = np.zeros(pp, dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k, indss]
            Tvec[i - 1] = tt[k, i]
        Tsol = np.linalg.solve(Smat, Tvec)
        cvest[k] = Tsol[drv]

    curvest = math.gamma(drv + 1) * cvest[1:M + 1]

    return {"x": gpoints, "y": curvest}


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Reimplementation of the Fortran subroutine F_rlbin (rlbin.f):
    # obtains bin counts for univariate regression data via the linear
    # binning strategy. If trun == 0 then weight from end observations
    # is given to corresponding end grid points. If trun == 1 then end
    # observations are truncated.
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)
    for i in range(n):
        lxi = ((X[i] - a) / delta) + 1

        # Find integer part of "lxi" (Fortran int() truncates toward zero)
        li = int(lxi)
        rem = lxi - li

        # Correction for right endpoint (not included if li == M)
        if X[i] == b:
            li = M - 1
            rem = 1

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


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted only for signature compatibility with
    # the R function: the underlying Fortran routine F_sdiag does not use
    # them (it always uses a Gaussian kernel and does not depend on 'drv').
    x = np.asarray(x, dtype=np.float64)

    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")

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
        xcounts = x
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
        Lvec = np.floor(tau * hdisc / delta)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.full(M, 1.0, dtype=np.float64)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1)
        else:
            indic = np.full(M, 1.0, dtype=np.float64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.full(M, 1.0, dtype=np.float64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.float64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    Lvec = Lvec.astype(np.int64)
    indic = indic.astype(np.int64)

    # Reimplementation of the Fortran subroutine F_sdiag (sdiag.f):
    # for each grid point, combine the (log-)discretised Gaussian kernel
    # weights with the binned counts to build the weighted local-polynomial
    # moment accumulator "ss", then form the pp x pp weighted normal-
    # equations matrix Smat at each grid point (same construction as in
    # locpoly) and invert it (equivalent to the LINPACK dgefa/dgedi
    # inversion used in the Fortran code), taking the (1,1) entry of the
    # inverse as the diagonal smoother-matrix weight Sdg.
    #
    # All arrays below are padded with an unused index 0 entry so that
    # the original 1-based Fortran indices (and index arithmetic such
    # as "k - j + midpts(i)") can be mirrored directly without having
    # to re-derive the offsets.
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    midpts = np.zeros(Q + 1, dtype=np.int64)
    ss = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    Sdg = np.zeros(M + 1, dtype=np.float64)

    xcnts = np.concatenate(([0.0], np.asarray(xcounts, dtype=np.float64)))
    Lvec_p = np.concatenate(([0], Lvec))
    hdisc_p = np.concatenate(([0.0], hdisc))
    indic_p = np.concatenate(([0], indic))

    # Obtain kernel weights
    mid = int(Lvec_p[1]) + 1
    for i in range(1, Q):
        midpts[i] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec_p[i]) + 1):
            fkap[mid + j] = math.exp(-((delta * j / hdisc_p[i]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lvec_p[i]) + int(Lvec_p[i + 1]) + 1
    midpts[Q] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lvec_p[Q]) + 1):
        fkap[mid + j] = math.exp(-((delta * j / hdisc_p[Q]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcnts[k] != 0:
            for i in range(1, Q + 1):
                lo = max(1, k - int(Lvec_p[i]))
                hi = min(M, k + int(Lvec_p[i]))
                for j in range(lo, hi + 1):
                    if indic_p[j] == i:
                        fac = 1.0
                        ss[j, 1] += xcnts[k] * fkap[k - j + midpts[i]]
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcnts[k] * fkap[k - j + midpts[i]] * fac

    # Solve for the diagonal entries of the (binned) smoother matrix at
    # each grid point
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k, indss]
        Sinv = np.linalg.inv(Smat)
        Sdg[k] = Sinv[0, 0]

    return {"x": gpoints, "y": Sdg[1:M + 1]}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted only for signature compatibility with
    # the R function: the underlying Fortran routine F_sstdg does not use
    # them (it always uses a Gaussian kernel and does not depend on 'drv').
    x = np.asarray(x, dtype=np.float64)

    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")

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
        xcounts = x
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
        Lvec = np.floor(tau * hdisc / delta)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.full(M, 1.0, dtype=np.float64)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1)
        else:
            indic = np.full(M, 1.0, dtype=np.float64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.full(M, 1.0, dtype=np.float64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.float64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    Lvec = Lvec.astype(np.int64)
    indic = indic.astype(np.int64)

    # Reimplementation of the Fortran subroutine F_sstdg (sstdiag.f):
    # for each grid point, combine the (log-)discretised Gaussian kernel
    # weights with the binned counts to build the weighted local-polynomial
    # moment accumulators "ss" (uses fkap) and "uu" (uses fkap**2), then
    # form the pp x pp weighted normal-equations matrix Smat and the
    # corresponding Umat at each grid point (same construction as in
    # locpoly/sdiag) and invert Smat (equivalent to the LINPACK
    # dgefa/dgedi inversion used in the Fortran code). The diagonal entry
    # of S*S^T at each grid point is then
    # sum_i sum_j Sinv[0, i] * Umat[i, j] * Sinv[j, 0].
    #
    # All arrays below are padded with an unused index 0 entry so that
    # the original 1-based Fortran indices (and index arithmetic such
    # as "k - j + midpts(i)") can be mirrored directly without having
    # to re-derive the offsets.
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    midpts = np.zeros(Q + 1, dtype=np.int64)
    ss = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    uu = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    SSTd = np.zeros(M + 1, dtype=np.float64)

    xcnts = np.concatenate(([0.0], np.asarray(xcounts, dtype=np.float64)))
    Lvec_p = np.concatenate(([0], Lvec))
    hdisc_p = np.concatenate(([0.0], hdisc))
    indic_p = np.concatenate(([0], indic))

    # Obtain kernel weights
    mid = int(Lvec_p[1]) + 1
    for i in range(1, Q):
        midpts[i] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec_p[i]) + 1):
            fkap[mid + j] = math.exp(-((delta * j / hdisc_p[i]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lvec_p[i]) + int(Lvec_p[i + 1]) + 1
    midpts[Q] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lvec_p[Q]) + 1):
        fkap[mid + j] = math.exp(-((delta * j / hdisc_p[Q]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcnts[k] != 0:
            for i in range(1, Q + 1):
                lo = max(1, k - int(Lvec_p[i]))
                hi = min(M, k + int(Lvec_p[i]))
                for j in range(lo, hi + 1):
                    if indic_p[j] == i:
                        fac = 1.0
                        w = fkap[k - j + midpts[i]]
                        ss[j, 1] += xcnts[k] * w
                        uu[j, 1] += xcnts[k] * (w ** 2)
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcnts[k] * w * fac
                            uu[j, ii] += xcnts[k] * (w ** 2) * fac

    # Solve for the diagonal entries of the (binned) S*S^T matrix at
    # each grid point
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Umat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k, indss]
                Umat[i - 1, j - 1] = uu[k, indss]
        Sinv = np.linalg.inv(Smat)
        SSTd[k] = float(Sinv[0, :] @ Umat @ Sinv[:, 0])

    return {"x": gpoints, "y": SSTd[1:M + 1]}


def onAttach(libname: str, pkgname: str) -> None:
    # Reimplementation of R's .onAttach package load-hook, which used
    # packageStartupMessage() to print a non-fatal startup message when
    # the package was attached. Python modules have no direct analogue
    # of .onAttach, so this is exposed as a plain function that prints
    # the equivalent startup message when invoked (e.g. from the
    # package's __init__.py).
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")


def onUnload(libpath: str) -> None:
    # Original R implementation called library.dynam.unload("KernSmooth", libpath)
    # to unload the package's compiled Fortran/C shared library when the
    # package was detached. This Python port is a pure NumPy reimplementation
    # with no compiled shared library to unload, so this hook is a no-op.
    pass
