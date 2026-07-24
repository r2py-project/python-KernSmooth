import math
import sys
from typing import Any
import warnings

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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'arg' should be one of " + ", ".join(repr(k) for k in valid_kernels))

    x = np.asarray(x, dtype=np.float64)

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

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    # R's fft(z, inverse = TRUE) is the unnormalized inverse transform, i.e.
    # P * np.fft.ifft(z); dividing by P below therefore corresponds exactly
    # to using NumPy's (normalized) np.fft.ifft directly.
    y = np.real(np.fft.ifft(kappa * gcounts))[:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: tuple[int, int] | np.ndarray[Any, np.dtype[np.integer[Any]]] = (51, 51), range_x: list[np.ndarray[Any, np.dtype[np.float64]]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths. Unlike R, `bandwidth`
    # has no default in this Python signature, so it is never "missing"; the
    # check below always runs once a value has been supplied by the caller.
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.min(bandwidth_arr) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    ## Rename common variables

    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = bandwidth_arr.astype(np.float64)
    tau = 3.4  # For bivariate normal kernel.

    ## Use same bandwidth in each direction
    ## if only a single bandwidth is given.

    if h.shape[0] == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    ## If range_x is not specified then set it at its default value.

    if range_x is None:
        range_x = [np.zeros(2), np.zeros(2)]
        for id_ in range(2):
            range_x[id_] = np.array(
                [np.min(x[:, id_]) - 1.5 * h[id_], np.max(x[:, id_]) + 1.5 * h[id_]]
            )

    a = np.array([range_x[0][0], range_x[1][0]])
    b = np.array([range_x[0][1], range_x[1][1]])

    ## Set up grid points and bin the data

    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    ## Compute kernel weights

    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.zeros((1, 1)), np.zeros((1, 1))]
    for id_ in range(2):
        L[id_] = int(min(np.floor(tau * h[id_] * (M[id_] - 1) / (b[id_] - a[id_])), M[id_] - 1))
        lvecid = np.arange(0, L[id_] + 1, dtype=np.float64)
        facid = (b[id_] - a[id_]) / (h[id_] * (M[id_] - 1))
        z = (norm.pdf(lvecid * facid) / h[id_]).reshape(-1, 1)
        tot = np.sum(np.concatenate([z.ravel(), z.ravel()[1:][::-1]])) * facid * h[id_]
        kapid[id_] = z / tot

    kapp = (kapid[0] @ kapid[1].T) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    ## Now combine weight and counts using the FFT to obtain estimate

    P = 2 ** np.ceil(np.log2(M.astype(np.float64) + L.astype(np.float64)))  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
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

    rp = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp = np.fft.fft2(sp)
    rp = np.real(np.fft.ifft2(rp * sp))[0:M1, 0:M2]
    ## invert element-wise product of FFT's
    ## and truncate and normalise it
    ## (numpy's ifft2 already divides by P1*P2, unlike R's fft(..., inverse=TRUE))

    ## Ensure that rp is non-negative

    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x = np.asarray(x, dtype=np.float64)

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
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = (np.exp(-0.5 * arg ** 2) / np.sqrt(2.0 * np.pi)) / (h ** (drv + 1))
    hmold0 = np.ones_like(arg)
    hmold1 = arg.copy()
    hmnew = np.ones_like(arg)
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0  # Compute mth degree Hermite polynomial
            hmold0 = hmold1                          # by recurrence.
            hmold1 = hmnew
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    # R's fft(z, inverse = TRUE) is the unnormalized inverse transform, i.e.
    # P * np.fft.ifft(z); dividing by P below therefore corresponds exactly
    # to using NumPy's (normalized) np.fft.ifft directly.
    conv = np.real(np.fft.ifft(kappam * Gcounts))[:M]

    return float(np.sum(gcounts * conv) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    # Set up quantities analogous to the FORTRAN routine "blkest"
    qq = q + 1
    idiv = n // Nval

    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    # It is assumed that the (x, y) data are sorted with respect to x.
    for j in range(1, Nval + 1):
        # For each member of the partition
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        lo = ilow - 1
        hi = iupp
        xj = x[lo:hi]
        yj = y[lo:hi]
        nj = hi - lo

        # Obtain a q'th degree fit over the current member of the partition.
        # Set up the design matrix: column k (0-based) holds xj ** k.
        xmat = np.vander(xj, N=qq, increasing=True)

        # Least-squares polynomial fit (equivalent to the QR-based dqrdc/dqrsl
        # solve used by the FORTRAN routine).
        coef, _, _, _ = np.linalg.lstsq(xmat, yj, rcond=None)

        for i in range(nj):
            xi = xj[i]
            fiti = coef[0]
            ddm = 2.0 * coef[2]
            ddddm = 24.0 * coef[4]
            for kk in range(1, q + 1):
                fiti = fiti + coef[kk] * xi ** kk
                if kk <= q - 2:
                    ddm = ddm + (kk + 1) * (kk + 2) * coef[kk + 2] * xi ** kk
                    if kk <= q - 4:
                        ddddm = ddddm + (kk + 1) * (kk + 2) * (kk + 3) * (kk + 4) * coef[kk + 4] * xi ** kk
            th22e = th22e + ddm ** 2
            th24e = th24e + ddm * ddddm
            RSS = RSS + (yj[i] - fiti) ** 2

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    sort_order = np.argsort(X, kind="stable")
    X = X[sort_order]
    Y = Y[sort_order]

    # Set up arrays analogous to the FORTRAN subroutine "cp".
    qq = q + 1
    RSS = np.zeros(Nmax, dtype=np.float64)

    # It is assumed that the (X, Y) data are sorted with respect to the X's.
    for Nval in range(1, Nmax + 1):
        # For each number of partitions.
        idiv = n // Nval
        for j in range(1, Nval + 1):
            # For each member of the partition.
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n
            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]
            nj = iupp - ilow + 1

            # Obtain a q'th degree fit over the current member of the partition.
            # Set up the design ("X") matrix.
            Xmat = np.empty((nj, qq), dtype=np.float64)
            Xmat[:, 0] = 1.0
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)
            fiti = Xmat @ coef
            RSSj = float(np.sum((Yj - fiti) ** 2))

            RSS[Nval - 1] += RSSj

    # Now compute the array of Mallow's C_p values.
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    return int(np.argmin(Cpvals) + 1)


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    # R's default range.x = range(x) is evaluated lazily against the
    # parameter 'x'; Python has no equivalent, so compute it explicitly here.
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

    # Compute scale estimate (equivalent to match.arg(scalest, c('minim','stdev','iqr')))
    choices = ("minim", "stdev", "iqr")
    matches = [choice for choice in choices if choice.startswith(scalest)]
    if len(matches) != 1:
        raise ValueError("'scalest' should be one of " + repr(list(choices)))
    scalest_name = matches[0]

    if scalest_name == "stdev":
        scale = math.sqrt(float(np.var(x, ddof=1)))
    elif scalest_name == "iqr":
        q75 = float(np.quantile(x, 0.75, method="linear"))
        q25 = float(np.quantile(x, 0.25, method="linear"))
        scale = (q75 - q25) / 1.349
    else:
        q75 = float(np.quantile(x, 0.75, method="linear"))
        q25 = float(np.quantile(x, 0.25, method="linear"))
        scale = min((q75 - q25) / 1.349, math.sqrt(float(np.var(x, ddof=1))))

    if scale == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    mean_x = float(np.mean(x))
    sx = (x - mean_x) / scale
    sa = (a - mean_x) / scale
    sb = (b - mean_x) / scale

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
        raise ValueError("Level should be between 0 and 5")

    return scale * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'kernel' should be one of " + repr(valid_kernels))

    ## Set kernel constants
    if canonical:
        del0 = 1.0
    else:
        del0 = {
            "normal": 1.0 / ((4.0 * np.pi) ** (1.0 / 10.0)),
            "box": (9.0 / 2.0) ** (1.0 / 5.0),
            "epanech": 15.0 ** (1.0 / 5.0),
            "biweight": 35.0 ** (1.0 / 5.0),
            "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
        }[kernel]

    ## Rename variables
    n = x.shape[0]
    M = gridsize
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate
    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'scalest' should be one of " + repr(valid_scalest))

    stdev_val = float(np.std(x, ddof=1))
    iqr_val = float(
        (np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")) / 1.349
    )
    if scalest == "stdev":
        scale_value = stdev_val
    elif scalest == "iqr":
        scale_value = iqr_val
    else:  # "minim"
        scale_value = min(iqr_val, stdev_val)

    if scale_value == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:
    x_mean = float(np.mean(x))
    sx = (x - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    ## Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    ## Perform plug-in steps:
    if level == 0:
        psi4hat = 3.0 / (8.0 * np.sqrt(np.pi))
    elif level == 1:
        alpha = (2.0 * np.sqrt(2.0) ** 7 / (5.0 * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2.0 * np.sqrt(2.0) ** 9 / (7.0 * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2.0 * np.sqrt(2.0) ** 11 / (9.0 * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2.0 * np.sqrt(2.0) ** 13 / (11.0 * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:  # level == 5
        alpha = (2.0 * np.sqrt(2.0) ** 15 / (13.0 * n)) ** (1.0 / 15.0)  # bandwidth for psi_12
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = (945.0 * np.sqrt(2.0 / np.pi) / (psi12hat * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return float(scale_value * del0 * (1.0 / (psi4hat * n)) ** (1.0 / 5.0))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # R's default 'range.x = range(x)' is evaluated lazily from the
    # original (untrimmed) 'x' before any trimming occurs below.
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Sort the (x, y) data with respect to the x's (equivalent to
    # 'xy <- cbind(x, y); xy <- xy[sort.list(xy[, 1L]), ]').
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    # Trim the 100(trim)% of the data from each end (in the x-direction).
    n_full = len(x)
    ntrim = int(math.floor(trim * n_full))
    x = x[ntrim:n_full - ntrim]
    y = y[ntrim:n_full - ntrim]

    # Rename common parameters
    n = len(x)
    M = gridsize
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

    # Estimate theta_22 using a local cubic fit
    # with a "rule-of-thumb" bandwidth: "gamseh"
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=range_x, binned=True)["y"]

    llow = int(math.floor(proptrun * M))
    lupp = M - int(math.floor(proptrun * M))
    th22kn = np.sum((mddest[llow:lupp] ** 2) * xcounts[llow:lupp]) / n

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / math.sqrt(2 * math.pi)) ** (1 / 9)
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


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Direct translation of KernSmooth's Fortran `linbin` routine (src/linbin.f):
    # linearly distributes the weight of each observation between the two
    # nearest grid points, optionally truncating observations outside [a, b].
    gcnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)

    # lxi is the (1-based) fractional grid position of each observation.
    lxi = ((X - a) / delta) + 1.0

    # Integer part of lxi (equivalent to Fortran's int() truncation towards
    # zero; only differs from floor() for negative lxi, where the exact value
    # of li is never used numerically below, only compared against 1).
    li = np.floor(lxi).astype(np.int64)
    rem = lxi - li

    in_range = (li >= 1) & (li < M)
    li_in_range = li[in_range]
    rem_in_range = rem[in_range]

    # Convert the 1-based Fortran grid index to a 0-based Python index.
    idx = li_in_range - 1
    np.add.at(gcnts, idx, 1.0 - rem_in_range)
    np.add.at(gcnts, idx + 1, rem_in_range)

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

    n = X.shape[0]
    x1 = X[:, 0]
    x2 = X[:, 1]

    M1 = gpoints1.shape[0]
    M2 = gpoints2.shape[0]
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Direct translation of KernSmooth's Fortran `lbtwod` routine (src/linbin2D.f):
    # bilinearly distributes the unit mass of each observation among the four
    # grid points surrounding it; observations falling outside the mesh
    # defined by [a1, b1] x [a2, b2] are truncated (ignored).
    gcnts = np.zeros(M1 * M2, dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    # lxi1/lxi2 are the (1-based) fractional grid positions of each observation.
    lxi1 = ((x1 - a1) / delta1) + 1.0
    lxi2 = ((x2 - a2) / delta2) + 1.0

    # Integer part of lxi1/lxi2 (equivalent to Fortran's int() truncation
    # towards zero; only differs from floor() for negative lxi, where the
    # exact value of li1/li2 is never used numerically below, only compared
    # against 1 and M1/M2).
    li1 = np.floor(lxi1).astype(np.int64)
    li2 = np.floor(lxi2).astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    in_range = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)

    li1_in = li1[in_range]
    li2_in = li2[in_range]
    rem1_in = rem1[in_range]
    rem2_in = rem2[in_range]

    # Convert the 1-based Fortran grid indices to 0-based row/column indices
    # and then to a 0-based flat index into the column-major (Fortran-order)
    # gcnts array, mirroring `ind1 = M1*(li2-1) + li1` etc. in the Fortran code.
    row0 = li1_in - 1
    col0 = li2_in - 1

    ind1 = col0 * M1 + row0
    ind2 = ind1 + 1
    ind3 = ind1 + M1
    ind4 = ind3 + 1

    np.add.at(gcnts, ind1, (1.0 - rem1_in) * (1.0 - rem2_in))
    np.add.at(gcnts, ind2, rem1_in * (1.0 - rem2_in))
    np.add.at(gcnts, ind3, (1.0 - rem1_in) * rem2_in)
    np.add.at(gcnts, ind4, rem1_in * rem2_in)

    # R's `matrix(out, M1, M2)` fills column-major, matching the Fortran-order
    # layout of gcnts constructed above.
    return gcnts.reshape(M1, M2, order='F')


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths.
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.any(bandwidth_arr <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x = np.asarray(x, dtype=np.float64)
    if y is not None:
        y = np.asarray(y, dtype=np.float64)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x) - np.min(x))
            range_x = (np.min(x) - extra, np.max(x) + extra)
        else:
            range_x = (np.min(x), np.max(x))

    # Rename common variables.
    M = gridsize
    Q = int(bwdisc)
    a = range_x[0]
    b = range_x[1]
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Decide whether a density estimate or a regression estimate is required.
    if y is None:  # obtain density estimate
        n = len(x)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        # Bin the data if not already binned.
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

    # Set the bin width.
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths.
    if bandwidth_arr.size == M:
        sorted_bandwidth = np.sort(bandwidth_arr)
        hlow = sorted_bandwidth[0]
        hupp = sorted_bandwidth[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc".
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc" to each member of "bandwidth".
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bandwidth[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.size == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if Lvec.min() == 0:
        raise ValueError("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    # Direct translation of KernSmooth's Fortran "locpol" routine (src/locpoly.f):
    # for each grid point "j", accumulate weighted moments of the binned counts
    # over a window of source bins "k" within Lvec[indic[j]-1] of "j", using a
    # Gaussian kernel weight based on the distance delta*(k-j) scaled by the
    # bandwidth assigned to bin "j" (hdisc[indic[j]-1]). The resulting Hankel-like
    # moment matrix "Smat" and moment vector "Tvec" define the weighted least
    # squares normal equations for the local polynomial coefficients at "j";
    # solving them and taking the drv-th coefficient (scaled by drv!) gives the
    # local polynomial estimate of the drv-th derivative at that grid point.
    curvest = np.zeros(M, dtype=np.float64)
    powers = np.arange(ppp)
    for j in range(M):
        i = int(indic[j]) - 1
        L = int(Lvec[i])
        hd = hdisc[i]

        k_lo = max(0, j - L)
        k_hi = min(M - 1, j + L)
        ks = np.arange(k_lo, k_hi + 1)
        ks = ks[xcounts[ks] != 0]

        dist = delta * (ks - j)
        w = np.exp(-((dist / hd) ** 2) / 2)
        dist_pow = dist[:, None] ** powers[None, :]

        ss = np.sum((xcounts[ks] * w)[:, None] * dist_pow, axis=0)
        tt = np.sum((ycounts[ks] * w)[:, None] * dist_pow[:, :pp], axis=0)

        Smat = np.array([[ss[row + col] for col in range(pp)] for row in range(pp)], dtype=np.float64)
        Tvec = tt.astype(np.float64)

        beta = np.linalg.solve(Smat, Tvec)
        curvest[j] = beta[drv]

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


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

    delta = (b - a) / (M - 1)
    for i in range(n):
        lxi = ((x[i] - a) / delta) + 1

        # Find integer part of "lxi" (Fortran int() truncates toward zero,
        # matching Python's int() applied to a float)
        li = int(lxi)
        rem = lxi - li

        # Correction for right endpoint (not included if li == M)
        if x[i] == b:
            li = M - 1
            rem = 1.0

        if li >= 1 and li < M:
            xcounts[li - 1] += (1 - rem)
            xcounts[li] += rem
            ycounts[li - 1] += (1 - rem) * y[i]
            ycounts[li] += rem * y[i]

        if li < 1 and trun == 0:
            xcounts[0] += 1
            ycounts[0] += y[i]

        if li >= M and trun == 0:
            xcounts[M - 1] += 1
            ycounts[M - 1] += y[i]

    return {"xcounts": xcounts, "ycounts": ycounts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for interface consistency with the
    # other KernSmooth routines but, exactly as in the original R code,
    # are not actually used anywhere in the body of this function.
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
                indic = np.repeat(1, M)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.repeat(1, M)
    elif len(bandwidth_arr) == 1:
        indic = np.repeat(1, M)
        Q = 1
        Lvec = np.repeat(np.floor(tau * bandwidth_arr[0] / delta).astype(np.int64), Q)
        hdisc = np.repeat(bandwidth_arr[0], Q)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    Sdg = np.zeros(M, dtype=np.float64)

    # --- Direct translation of KernSmooth's Fortran `sdiag` routine ---
    # (src/sdiag.f). All array indices below are kept as 1-based "Fortran"
    # positions and translated to 0-based NumPy indices with an explicit
    # "- 1" at the point of access, to make the translation easy to audit
    # line by line against the original Fortran source.

    # Obtain kernel weights
    mid = int(Lvec[0]) + 1
    for qi in range(Q - 1):
        midpts[qi] = mid
        fkap[mid - 1] = 1.0
        Lq = int(Lvec[qi])
        for j in range(1, Lq + 1):
            val = np.exp(-((delta * j / hdisc[qi]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[qi]) + int(Lvec[qi + 1]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    LQ = int(Lvec[Q - 1])
    for j in range(1, LQ + 1):
        val = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    # Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i in range(1, Q + 1):
                jlo = max(1, k - int(Lvec[i - 1]))
                jhi = min(M, k + int(Lvec[i - 1]))
                for j in range(jlo, jhi + 1):
                    if indic[j - 1] == i:
                        fkap_val = fkap[k - j + int(midpts[i - 1]) - 1]
                        fac = 1.0
                        ss[j - 1, 0] += xcounts[k - 1] * fkap_val
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xcounts[k - 1] * fkap_val * fac

    # Solve the local weighted least-squares system at each grid point and
    # extract the (1, 1) entry of the inverse moment matrix (equivalent to
    # the LINPACK dgefa/dgedi calls in the Fortran routine, which factor
    # "Smat" and then overwrite it with its inverse).
    idx = np.add.outer(np.arange(pp), np.arange(pp))
    for k in range(1, M + 1):
        Smat = ss[k - 1, idx]
        Sinv = np.linalg.inv(Smat)
        Sdg[k - 1] = Sinv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], *, drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = np.array([np.min(x), np.max(x)], dtype=np.float64)

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
        M = len(xcounts)
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

        # Determine index of closest entry of "hdisc" to each member of "bandwidth"
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

    dimfkap = int(2 * np.sum(Lvec) + Q)

    # --- Direct translation of the FORTRAN subroutine "sstdg" (src/sstdiag.f). ---
    # 1-based padded copies of the inputs are used (index 0 is an unused
    # placeholder) so that the loop/index arithmetic mirrors the original
    # FORTRAN code exactly and is easy to audit line by line.
    xc = np.concatenate(([0.0], xcounts))
    hd = np.concatenate(([0.0], hdisc))
    Lv = np.concatenate(([0], Lvec))
    ind = np.concatenate(([0], indic))

    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    midpts = np.zeros(Q + 1, dtype=np.int64)
    ss = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    uu = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    SSTd = np.zeros(M + 1, dtype=np.float64)

    # Obtain kernel weights
    mid = int(Lv[1]) + 1
    for i in range(1, Q):
        midpts[i] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lv[i]) + 1):
            fkap[mid + j] = np.exp(-((delta * j / hd[i]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lv[i]) + int(Lv[i + 1]) + 1
    midpts[Q] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lv[Q]) + 1):
        fkap[mid + j] = np.exp(-((delta * j / hd[Q]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xc[k] != 0:
            for i in range(1, Q + 1):
                lo = max(1, k - int(Lv[i]))
                hi = min(M, k + int(Lv[i]))
                for j in range(lo, hi + 1):
                    if ind[j] == i:
                        fac = 1.0
                        w = fkap[k - j + midpts[i]]
                        ss[j, 1] += xc[k] * w
                        uu[j, 1] += xc[k] * (w ** 2)
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xc[k] * w * fac
                            uu[j, ii] += xc[k] * (w ** 2) * fac

    # Build the local moment matrices at each grid point, invert the
    # S-matrix (replacing the LINPACK dgefa/dgedi(job=01) inverse-only
    # call) and combine with the U-matrix to obtain the diagonal of S S^T.
    Smat = np.zeros((pp + 1, pp + 1), dtype=np.float64)
    Umat = np.zeros((pp + 1, pp + 1), dtype=np.float64)
    for k in range(1, M + 1):
        SSTd[k] = 0.0
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i, j] = ss[k, indss]
                Umat[i, j] = uu[k, indss]

        Smat[1:pp + 1, 1:pp + 1] = np.linalg.inv(Smat[1:pp + 1, 1:pp + 1])

        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                SSTd[k] += Smat[1, i] * Umat[i, j] * Smat[j, 1]

    return {"x": gpoints, "y": SSTd[1:M + 1]}


def on_attach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def _on_unload(libpath: str) -> None:
    # R's .onUnload package hook calls library.dynam.unload("KernSmooth", libpath)
    # to explicitly unlink the compiled shared library from the R session when
    # the package is detached. Python has no equivalent mechanism: compiled
    # extension modules (here, the `_KernSmooth` module) are managed by the
    # Python import system and garbage collector, and are not explicitly
    # unloaded when a module goes out of scope. This function is therefore
    # retained only as a structural placeholder for the R unload hook and is
    # a no-op in the Python port.
    pass
