import math
import sys
from typing import Any, Sequence
import warnings

import numpy as np
from scipy.stats import beta, norm

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
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    allowed_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in allowed_kernels:
        raise ValueError("'arg' should be one of " + ", ".join(repr(k) for k in allowed_kernels))

    x_arr = np.asarray(x, dtype=np.float64)

    ## Rename common variables
    n = x_arr.shape[0]
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
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.sqrt(np.var(x_arr, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    ## Set kernel support values
    tau = 4.0 if kernel == "normal" else 1.0

    if range_x is None:
        a = float(np.min(x_arr)) - tau * h
        b = float(np.max(x_arr)) + tau * h
    else:
        a = float(range_x[0])
        b = float(range_x[1])

    ## Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    ## Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
    if L == 0:
        import warnings
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    lvec = np.arange(0, L + 1, dtype=np.float64)
    if kernel == "normal":
        kappa = norm.pdf(lvec * delta) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1.0), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1.0), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1.0), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1.0), 4, 4) / (n * h)

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1, dtype=np.float64), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M, dtype=np.float64)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)

    # R's fft(z, inverse = TRUE) is the *unnormalized* inverse transform
    # (equal to P * numpy.fft.ifft(z)); dividing that result by P, as the
    # R code does, is therefore exactly numpy.fft.ifft(z).
    conv = np.fft.ifft(kappa_fft * gcounts_fft)
    y = conv.real[0:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]], gridsize: Sequence[int] = (51, 51), range_x: Sequence[Sequence[float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths.
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.min(bandwidth_arr) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables.
    x_arr = np.asarray(x, dtype=np.float64)
    n = x_arr.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = bandwidth_arr
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [
            (float(np.min(x_arr[:, 0])) - 1.5 * h[0], float(np.max(x_arr[:, 0])) + 1.5 * h[0]),
            (float(np.min(x_arr[:, 1])) - 1.5 * h[1], float(np.max(x_arr[:, 1])) + 1.5 * h[1]),
        ]

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data.
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x_arr, gpoints1, gpoints2)

    # Compute kernel weights.
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.zeros((1, 1)), np.zeros((1, 1))]
    for idx in range(2):
        L[idx] = min(
            int(np.floor(tau * h[idx] * (M[idx] - 1) / (b[idx] - a[idx]))),
            int(M[idx]) - 1,
        )
        lvecid = np.arange(0, L[idx] + 1, dtype=np.float64)
        facid = (b[idx] - a[idx]) / (h[idx] * (M[idx] - 1))
        # Standard normal density (dnorm).
        dens = np.exp(-0.5 * (lvecid * facid) ** 2) / np.sqrt(2.0 * np.pi)
        z = (dens / h[idx]).reshape(-1, 1)
        tot = float(np.sum(np.concatenate([z.flatten(), z.flatten()[1:][::-1]]))) * facid * h[idx]
        kapid[idx] = z / tot

    kapp = (kapid[0] @ kapid[1].T) / n

    if int(np.min(L)) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate.
    P = (2 ** np.ceil(np.log(M.astype(np.float64) + L.astype(np.float64)) / np.log(2.0))).astype(np.int64)
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
    # invert element-wise product of FFT's (numpy's ifft2 is already
    # normalised by 1/(P1*P2), matching R's fft(..., inverse=TRUE)/(P1*P2))
    # and truncate it
    rp_result = np.fft.ifft2(rp_fft * sp_fft).real[0:M1, 0:M2]

    # Ensure that rp is non-negative.
    rp_result = np.where(rp_result > 0, rp_result, 0.0)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp_result}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename variables
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])
    h = bandwidth

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, gridsize)
        gcounts = linbin(x_arr, gpoints, truncate)
    else:
        gcounts = x_arr
        M = len(gcounts)
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width
    n = np.sum(gcounts)
    delta = (b - a) / (M - 1)

    # Obtain kernel weights
    tau = 4 + drv
    L = int(min(np.floor(tau * h / delta), M))

    if L == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = norm.pdf(arg) / (h ** (drv + 1))
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

    conv = (np.fft.ifft(kappam * Gcounts) * P).real / P

    return float(np.sum(gcounts * conv[0:M]) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # It is assumed that the (x, y) data are sorted with respect to the x's.
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = x_arr.shape[0]

    # Sort the (x, y) data with respect to the x's (stable, mirroring R's sort.list).
    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    qq = q + 1

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

        xj = x_sorted[ilow - 1:iupp]
        yj = y_sorted[ilow - 1:iupp]

        # Obtain a q'th degree least-squares fit over the current member
        # of the partition (coefficients are ordered from the constant
        # term up to the q'th degree term, as in the R/Fortran code).
        coef = np.polynomial.polynomial.polyfit(xj, yj, q)

        fitj = np.polynomial.polynomial.polyval(xj, coef)

        # Second and fourth derivatives of the fitted polynomial,
        # evaluated at the block's x values.
        d2coef = np.polynomial.polynomial.polyder(coef, 2)
        d4coef = np.polynomial.polynomial.polyder(coef, 4)
        ddm = np.polynomial.polynomial.polyval(xj, d2coef)
        ddddm = np.polynomial.polynomial.polyval(xj, d4coef)

        th22e = th22e + np.sum(ddm ** 2)
        th24e = th24e + np.sum(ddm * ddddm)
        RSS = RSS + np.sum((yj - fitj) ** 2)

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    n = X_arr.shape[0]

    # Sort the (X, Y) data with respect to the X's (stable, like R's sort.list).
    sort_idx = np.argsort(X_arr, kind="mergesort")
    X_sorted = X_arr[sort_idx]
    Y_sorted = Y_arr[sort_idx]

    # Number of parameters in the q'th degree polynomial fit (intercept + q terms).
    qq = q + 1

    # RSS[Nval - 1] holds the pooled residual sum of squares for a partition
    # of the data into Nval roughly-equal blocks, each fitted with a q'th
    # degree polynomial least-squares regression (mirrors Fortran routine 'cp').
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        idiv = n // Nval
        RSSval = 0.0
        for j in range(1, Nval + 1):
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n
            # Convert the 1-based inclusive Fortran range [ilow, iupp] to a
            # 0-based Python slice.
            Xj = X_sorted[ilow - 1:iupp]
            Yj = Y_sorted[ilow - 1:iupp]

            # Design matrix with columns 1, x, x^2, ..., x^(qq - 1).
            Xmat = np.vander(Xj, N=qq, increasing=True)

            # Least-squares polynomial fit, equivalent to the QR-based
            # regression performed by dqrdc/dqrsl in the Fortran code.
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)
            fitted = Xmat @ coef
            RSSval += float(np.sum((Yj - fitted) ** 2))

        RSS[Nval - 1] = RSSval

    # Mallow's C_p statistic for each candidate number of blocks, using the
    # finest partition (Nmax blocks) RSS as the estimate of pure error.
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = (n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1] + 2 * qq * i - n

    # order(Cpvec)[1L] in R: the (1-based) number of blocks minimizing C_p,
    # with ties broken by the smallest block count (first occurrence), which
    # matches np.argmin's behaviour on a 0-based array.
    return int(np.argmin(Cpvals)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename variables
    n = len(x_arr)
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    # Compute scale estimate
    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError("'scalest' should be one of 'minim', 'stdev', 'iqr'")

    if scalest == "stdev":
        scale_val = float(np.sqrt(np.var(x_arr, ddof=1)))
    elif scalest == "iqr":
        scale_val = float(
            (np.quantile(x_arr, 0.75, method="linear") - np.quantile(x_arr, 0.25, method="linear")) / 1.349
        )
    else:
        iqr_scale = (np.quantile(x_arr, 0.75, method="linear") - np.quantile(x_arr, 0.25, method="linear")) / 1.349
        stdev_scale = np.sqrt(np.var(x_arr, ddof=1))
        scale_val = float(min(iqr_scale, stdev_scale))

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    mean_x = float(np.mean(x_arr))
    sx = (x_arr - mean_x) / scale_val
    sa = (a - mean_x) / scale_val
    sb = (b - mean_x) / scale_val

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

    return float(scale_val * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    if kernel not in ("normal", "box", "epanech", "biweight", "triweight"):
        raise ValueError("'kernel' should be one of \"normal\", \"box\", \"epanech\", \"biweight\", \"triweight\"")

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Set kernel constants
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

    # Rename variables
    n = len(x_arr)
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    # Compute scale estimate
    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError("'scalest' should be one of \"minim\", \"stdev\", \"iqr\"")

    if scalest == "stdev":
        scale_est = np.sqrt(np.var(x_arr, ddof=1))
    elif scalest == "iqr":
        scale_est = (np.quantile(x_arr, 3 / 4, method="linear") - np.quantile(x_arr, 1 / 4, method="linear")) / 1.349
    else:  # minim
        scale_est = min(
            (np.quantile(x_arr, 3 / 4, method="linear") - np.quantile(x_arr, 1 / 4, method="linear")) / 1.349,
            np.sqrt(np.var(x_arr, ddof=1)),
        )

    if scale_est == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    x_mean = np.mean(x_arr)
    sx = (x_arr - x_mean) / scale_est
    sa = (a - x_mean) / scale_est
    sb = (b - x_mean) / scale_est

    # Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps:
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

    return float(scale_est * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # R's default `range.x = range(x)` is evaluated lazily against the
    # ORIGINAL (untrimmed) x, before any trimming takes place below.
    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    n_full = x_sorted.shape[0]
    indlow = int(np.floor(trim * n_full))
    indupp = n_full - int(np.floor(trim * n_full))

    x_trim = x_sorted[indlow:indupp]
    y_trim = y_sorted[indlow:indupp]

    ## Rename common parameters
    n = x_trim.shape[0]
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])

    ## Bin the data
    gpoints = np.linspace(a, b, M)
    out = rlbin(x_trim, y_trim, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    ## Choose the value of N using Mallow's C_p
    Nmax = max(min(int(np.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x_trim, y_trim, Nmax, 4)

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.
    out = blkest(x_trim, y_trim, Nval, 4)
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
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                    range_x=range_x, binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh,
                range_x=range_x, binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh,
                     range_x=range_x, binned=True)["y"]
    sigsqn = np.sum(y_trim ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    ## Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * np.sqrt(np.pi) * th22kn * n)) ** (1 / 5))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    X_arr = np.asarray(X, dtype=np.float64)
    n = X_arr.shape[0]
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)
    M = gpoints_arr.shape[0]
    trun = 1 if truncate else 0
    a = gpoints_arr[0]
    b = gpoints_arr[M - 1]

    gcounts = np.zeros(M, dtype=np.float64)

    if n == 0 or M < 2:
        return gcounts

    delta = (b - a) / (M - 1)

    # 1-based grid coordinate, mirroring the Fortran routine's "lxi"
    lxi = (X_arr - a) / delta + 1.0

    # Fortran's int() truncates toward zero (not floor), so use np.trunc
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Points that fall strictly inside the grid range contribute to two
    # neighbouring bins via linear binning weights.
    mask_in = (li >= 1) & (li < M)
    li_in = li[mask_in]
    rem_in = rem[mask_in]
    idx_in = li_in - 1  # convert 1-based Fortran index to 0-based Python index

    np.add.at(gcounts, idx_in, 1.0 - rem_in)
    np.add.at(gcounts, idx_in + 1, rem_in)

    if trun == 0:
        mask_lo = li < 1
        if np.any(mask_lo):
            gcounts[0] += np.count_nonzero(mask_lo)

        mask_hi = li >= M
        if np.any(mask_hi):
            gcounts[M - 1] += np.count_nonzero(mask_hi)

    return gcounts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    X_arr = np.asarray(X, dtype=np.float64)
    n = X_arr.shape[0]
    x1 = X_arr[:, 0]
    x2 = X_arr[:, 1]

    gpoints1_arr = np.asarray(gpoints1, dtype=np.float64)
    gpoints2_arr = np.asarray(gpoints2, dtype=np.float64)
    M1 = gpoints1_arr.shape[0]
    M2 = gpoints2_arr.shape[0]

    a1 = gpoints1_arr[0]
    a2 = gpoints2_arr[0]
    b1 = gpoints1_arr[M1 - 1]
    b2 = gpoints2_arr[M2 - 1]

    gcounts = np.zeros((M1, M2), dtype=np.float64)

    if n == 0 or M1 < 2 or M2 < 2:
        return gcounts

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    # 1-based grid coordinates, mirroring the Fortran routine's "lxi1"/"lxi2"
    lxi1 = (x1 - a1) / delta1 + 1.0
    lxi2 = (x2 - a2) / delta2 + 1.0

    # Fortran's int() truncates toward zero (not floor), so use np.trunc
    li1 = np.trunc(lxi1).astype(np.int64)
    li2 = np.trunc(lxi2).astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    # Points that fall strictly inside the grid range in both dimensions
    # contribute to four neighbouring bins via bilinear binning weights.
    valid = (li1 >= 1) & (li1 < M1) & (li2 >= 1) & (li2 < M2)

    i1 = li1[valid] - 1  # convert 1-based Fortran index to 0-based Python index
    i2 = li2[valid] - 1
    r1 = rem1[valid]
    r2 = rem2[valid]

    np.add.at(gcounts, (i1, i2), (1.0 - r1) * (1.0 - r2))
    np.add.at(gcounts, (i1 + 1, i2), r1 * (1.0 - r2))
    np.add.at(gcounts, (i1, i2 + 1), (1.0 - r1) * r2)
    np.add.at(gcounts, (i1 + 1, i2 + 1), r1 * r2)

    return gcounts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    ## Install safeguard against non-positive bandwidths:
    if bandwidth is None:
        raise ValueError("argument \"bandwidth\" is missing, with no default")
    if np.any(np.atleast_1d(np.asarray(bandwidth, dtype=np.float64)) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    x_arr = np.asarray(x, dtype=np.float64)

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (float(np.max(x_arr)) - float(np.min(x_arr)))
            range_x = (float(np.min(x_arr)) - extra, float(np.max(x_arr)) + extra)
        else:
            range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    ## Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    tau = 4.0

    ## Decide whether a density estimate or regression estimate is required.
    if y is None:  # obtain density estimate
        n = x_arr.shape[0]
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        y_arr = np.asarray(y, dtype=np.float64)
        ## Bin the data if not already binned
        if not binned:
            gpoints = np.linspace(a, b, M)
            out = rlbin(x_arr, y_arr, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = x_arr
            ycounts = y_arr
            M = xcounts.shape[0]
            gpoints = np.linspace(a, b, M)

    ## Set the bin width
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.shape[0] == M:
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

    if np.min(Lvec) == 0:
        raise ValueError("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    ## Allocate space for the final estimate
    curvest = np.zeros(M, dtype=np.float64)

    ## Native reimplementation of the FORTRAN routine "locpol": for each grid
    ## point, build the local weighted polynomial design matrix/response over
    ## the window determined by the discretised bandwidth, and solve the
    ## resulting weighted least-squares system directly.
    for g in range(M):
        q_idx = int(indic[g]) - 1  # convert 1-based R index to 0-based Python index
        h = hdisc[q_idx]
        L = int(Lvec[q_idx])

        j_lo = max(g - L, 0)
        j_hi = min(g + L, M - 1)
        j = np.arange(j_lo, j_hi + 1)

        u = (j - g) * delta
        w = xcounts[j] * np.exp(-0.5 * (u / h) ** 2) / (h * np.sqrt(2.0 * np.pi))

        Xmat = np.vander(u, N=pp, increasing=True)
        Smat = Xmat.T @ (w[:, None] * Xmat)
        Tvec = Xmat.T @ (w * ycounts[j])

        beta = np.linalg.solve(Smat, Tvec)
        curvest[g] = beta[drv]

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def on_attach(libname: str, pkgname: str) -> None:
    # Equivalent of R's packageStartupMessage(): emit a startup message
    # to stderr, mirroring the behaviour of R's condition-based message
    # system (which packageStartupMessage relies on) so that the
    # notice does not interfere with stdout output.
    startup_message = "KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009"
    print(startup_message, file=sys.stderr)


def _on_unload(libpath: str) -> None:
    # R's .onUnload package hook calls library.dynam.unload() to unload the
    # compiled shared library ('KernSmooth') from the R session when the
    # package is detached. The Python port of this package is a pure
    # NumPy implementation with no compiled/dynamically loaded extension
    # module to unload, so this hook is retained only for structural
    # parity with the original R package and performs no action.
    return None


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    n = X_arr.shape[0]
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)
    M = gpoints_arr.shape[0]
    trun = 1 if truncate else 0
    a = gpoints_arr[0]
    b = gpoints_arr[M - 1]

    xcounts = np.zeros(M, dtype=np.float64)
    ycounts = np.zeros(M, dtype=np.float64)

    if n == 0 or M < 2:
        return {"xcounts": xcounts, "ycounts": ycounts}

    delta = (b - a) / (M - 1)

    # 1-based grid coordinate, mirroring the Fortran routine's "lxi"
    lxi = (X_arr - a) / delta + 1.0

    # Fortran's int() truncates toward zero (not floor), so use np.trunc
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Correction for right endpoint: not included in the "li" bin unless
    # li equals M, so points exactly at "b" are forced into the last bin.
    at_b = X_arr == b
    li = np.where(at_b, M - 1, li).astype(np.int64)
    rem = np.where(at_b, 1.0, rem)

    # Points that fall strictly inside the grid range contribute to two
    # neighbouring bins via linear binning weights.
    mask_in = (li >= 1) & (li < M)
    li_in = li[mask_in]
    rem_in = rem[mask_in]
    y_in = Y_arr[mask_in]
    idx_in = li_in - 1  # convert 1-based Fortran index to 0-based Python index

    np.add.at(xcounts, idx_in, 1.0 - rem_in)
    np.add.at(xcounts, idx_in + 1, rem_in)
    np.add.at(ycounts, idx_in, (1.0 - rem_in) * y_in)
    np.add.at(ycounts, idx_in + 1, rem_in * y_in)

    if trun == 0:
        mask_lo = li < 1
        if np.any(mask_lo):
            xcounts[0] += np.count_nonzero(mask_lo)
            ycounts[0] += np.sum(Y_arr[mask_lo])

        mask_hi = li >= M
        if np.any(mask_hi):
            xcounts[M - 1] += np.count_nonzero(mask_hi)
            ycounts[M - 1] += np.sum(Y_arr[mask_hi])

    return {"xcounts": xcounts, "ycounts": ycounts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for interface compatibility with the
    # R function's signature but, exactly as in the original R code, they
    # are never actually referenced in the body (only the Gaussian kernel
    # is supported by the underlying Fortran routine).

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    ## Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4.0

    ## Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.shape[0] == M:
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
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    ## Native re-implementation of the Fortran routine F_sdiag.
    ##
    ## For each grid point g, the Fortran code accumulates, over data bins
    ## k lying within Lvec[indic[g]] bins of g (using the bandwidth level
    ## assigned to g itself, i.e. hdisc[indic[g]]), the power-weighted sums
    ##   ss[g, m] = sum_k xcounts[k] * dnorm((k-g)*delta / h) * ((k-g)*delta)^(m-1)
    ## for m = 1..ppp. These are then assembled into the pp x pp local Gram
    ## matrix Smat[i,j] = ss[g, i+j-1] (1-based), which is inverted; the
    ## diagonal smoother weight at g is the (1,1) entry of that inverse.
    xcounts_arr = np.asarray(xcounts, dtype=np.float64)
    powers = np.arange(ppp, dtype=np.float64)
    Sdg = np.zeros(M, dtype=np.float64)

    for g in range(M):
        level = int(indic[g]) - 1  # convert 1-based R index to 0-based Python index
        h = hdisc[level]
        L = int(Lvec[level])

        k_lo = max(0, g - L)
        k_hi = min(M - 1, g + L)
        k_idx = np.arange(k_lo, k_hi + 1)

        dist = (k_idx - g).astype(np.float64)  # matches Fortran's (k - j)
        kernel_w = np.exp(-0.5 * (delta * dist / h) ** 2)
        weighted = xcounts_arr[k_idx] * kernel_w

        ss_g = np.sum(weighted[:, None] * (delta * dist)[:, None] ** powers[None, :], axis=0)

        Smat = np.empty((pp, pp), dtype=np.float64)
        for ii in range(pp):
            for jj in range(pp):
                Smat[ii, jj] = ss_g[ii + jj]

        Smat_inv = np.linalg.inv(Smat)
        Sdg[g] = Smat_inv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for interface compatibility with the
    # R function's signature but, exactly as in the original R code, they
    # are never actually referenced in the body (only the Gaussian kernel
    # is supported by the underlying Fortran routine).

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    ## Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4.0

    ## Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.shape[0] == M:
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
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    ## Native re-implementation of the Fortran routine F_sstdg.
    ##
    ## For each grid point g, the Fortran code accumulates, over data bins
    ## k lying within Lvec[indic[g]] bins of g (using the bandwidth level
    ## assigned to g itself, i.e. hdisc[indic[g]]), the power-weighted sums
    ##   ss[g, m] = sum_k xcounts[k]   * dnorm((k-g)*delta / h) * ((k-g)*delta)^(m-1)
    ##   uu[g, m] = sum_k xcounts[k]   * dnorm((k-g)*delta / h)^2 * ((k-g)*delta)^(m-1)
    ## for m = 1..ppp. These are then assembled into the pp x pp local Gram
    ## matrices Smat[i,j] = ss[g, i+j-1] and Umat[i,j] = uu[g, i+j-1] (1-based).
    ## Smat is the local weighted-least-squares design matrix (X^T W X) and
    ## Umat is the analogous matrix built from the squared kernel weights
    ## (needed because each grid bin represents 'xcounts[k]' coincident data
    ## points sharing the same fitted-smoother weight). The diagonal entry
    ## of S S^T at g is then the (1,1) entry of Smat^{-1} Umat Smat^{-1},
    ## i.e. e0^T Smat^{-1} Umat Smat^{-1} e0 where e0 = (1, 0, ..., 0).
    xcounts_arr = np.asarray(xcounts, dtype=np.float64)
    powers = np.arange(ppp, dtype=np.float64)
    SSTd = np.zeros(M, dtype=np.float64)

    for g in range(M):
        level = int(indic[g]) - 1  # convert 1-based R index to 0-based Python index
        h = hdisc[level]
        L = int(Lvec[level])

        k_lo = max(0, g - L)
        k_hi = min(M - 1, g + L)
        k_idx = np.arange(k_lo, k_hi + 1)

        dist = (k_idx - g).astype(np.float64)  # matches Fortran's (k - j)
        kernel_w = np.exp(-0.5 * (delta * dist / h) ** 2)
        w1 = xcounts_arr[k_idx] * kernel_w
        w2 = xcounts_arr[k_idx] * kernel_w ** 2

        u_pow = (delta * dist)[:, None] ** powers[None, :]  # shape (n, ppp)
        ss_g = np.sum(w1[:, None] * u_pow, axis=0)
        uu_g = np.sum(w2[:, None] * u_pow, axis=0)

        Smat = np.empty((pp, pp), dtype=np.float64)
        Umat = np.empty((pp, pp), dtype=np.float64)
        for ii in range(pp):
            for jj in range(pp):
                Smat[ii, jj] = ss_g[ii + jj]
                Umat[ii, jj] = uu_g[ii + jj]

        Smat_inv = np.linalg.inv(Smat)
        SSTd[g] = Smat_inv[0, :] @ Umat @ Smat_inv[:, 0]

    return {"x": gpoints, "y": SSTd}
