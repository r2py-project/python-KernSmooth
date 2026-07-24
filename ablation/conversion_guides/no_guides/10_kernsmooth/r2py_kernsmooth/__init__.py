import math
import warnings
from typing import Any, Sequence

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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'kernel' should be one of " + str(valid_kernels))

    ## Rename common variables

    x = np.asarray(x, dtype=np.float64)
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
    else:  # "triweight"
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
        a = float(np.min(x) - tau * h)
        b = float(np.max(x) + tau * h)
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
    else:  # "triweight"
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)
    y = (np.real(np.fft.ifft(kappa_fft * gcounts_fft) * P) / P)[:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]], gridsize: tuple[int, int] | Sequence[int] = (51, 51), range_x: Sequence[tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    ## Install safeguard against non-positive bandwidths:
    if np.min(np.asarray(bandwidth, dtype=np.float64)) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    ## Rename common variables

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    ## Use same bandwidth in each direction
    ## if only a single bandwidth is given.

    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    ## If range_x is not specified then set it at its default value.

    if range_x is None:
        range_x_list: list[tuple[float, float]] = [(0.0, 0.0), (0.0, 0.0)]
        for idd in range(2):
            range_x_list[idd] = (
                float(np.min(x[:, idd]) - 1.5 * h[idd]),
                float(np.max(x[:, idd]) + 1.5 * h[idd]),
            )
    else:
        range_x_list = [tuple(range_x[0]), tuple(range_x[1])]

    a = np.array([range_x_list[0][0], range_x_list[1][0]], dtype=np.float64)
    b = np.array([range_x_list[0][1], range_x_list[1][1]], dtype=np.float64)

    ## Set up grid points and bin the data

    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    ## Compute kernel weights

    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    ]
    for idd in range(2):
        L[idd] = min(
            int(np.floor(tau * h[idd] * (M[idd] - 1) / (b[idd] - a[idd]))),
            int(M[idd]) - 1,
        )
        lvecid = np.arange(0, L[idd] + 1, dtype=np.float64)
        facid = (b[idd] - a[idd]) / (h[idd] * (M[idd] - 1))
        z = norm.pdf(lvecid * facid) / h[idd]
        tot = (np.sum(z) + np.sum(z[1:][::-1])) * facid * h[idd]
        kapid[idd] = z / tot

    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    ## Now combine weight and counts using the FFT to obtain estimate

    P = (2.0 ** np.ceil(np.log(M + L) / np.log(2.0))).astype(np.int64)  # smallest powers of 2 >= M+L
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
    ## invert element-wise product of FFT's (numpy's ifft2 already divides by P1*P2,
    ## matching R's fft(..., inverse = TRUE) / (P1 * P2))
    ## and truncate and normalise it
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]

    ## Ensure that rp is non-negative

    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename variables
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])
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
    L = int(min(math.floor(tau * h / delta), M))

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

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = int(2 ** (math.ceil(math.log(M + L + 1) / math.log(2))))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    # R's fft(z, inverse=TRUE) is the unnormalized inverse transform, i.e.
    # P * np.fft.ifft(z); dividing by P below therefore reduces to np.fft.ifft(z).
    result = np.sum(gcounts * np.real(np.fft.ifft(kappam * Gcounts))[:M]) / (n ** 2)

    return float(result)


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    qq = q + 1

    # It is assumed that the (x, y) data are sorted with respect to the x's.
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

        # Obtain a q'th degree fit over current member of partition

        # Set up "X" matrix
        Xmat = np.empty((nj, qq), dtype=np.float64)
        Xmat[:, 0] = 1.0
        for k in range(2, qq + 1):
            Xmat[:, k - 1] = Xj ** (k - 1)

        # Least squares polynomial fit (equivalent to the LINPACK
        # dqrdc/dqrsl QR decomposition and solve used by the original
        # Fortran routine).
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        for i in range(nj):
            xi = Xj[i]
            fiti = coef[0]
            ddm = 2.0 * coef[2] if qq > 2 else 0.0
            ddddm = 24.0 * coef[4] if qq > 4 else 0.0
            for k in range(2, qq + 1):
                fiti = fiti + coef[k - 1] * xi ** (k - 1)
                if k <= (q - 1):
                    ddm = ddm + k * (k + 1) * coef[k + 1] * xi ** (k - 1)
                    if k <= (q - 3):
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

    # Sort the (X, Y) data with respect to the X's.
    order = np.argsort(X, kind="stable")
    X = X[order]
    Y = Y[order]

    # Set up arrays for the FORTRAN subroutine "cp" (removed unused 'q').
    qq = q + 1
    RSS = np.zeros(Nmax, dtype=np.float64)

    # It is assumed that the (X, Y) data are sorted with respect to the X's.
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

            # Obtain a q'th degree fit over current member of partition

            # Set up "X" matrix
            Xmat = np.empty((nj, qq), dtype=np.float64)
            Xmat[:, 0] = 1.0
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            # Least squares polynomial fit (equivalent to the LINPACK
            # dqrdc/dqrsl QR decomposition and solve used by the original
            # Fortran routine).
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fiti = Xmat @ coef
            RSSj = float(np.sum((Yj - fiti) ** 2))

            RSS[Nval - 1] += RSSj

    # Now compute array of Mallow's C_p values.
    ivals = np.arange(1, Nmax + 1, dtype=np.float64)
    Cpvals = ((n - qq * Nmax) * RSS / RSS[Nmax - 1]) + 2 * qq * ivals - n

    # order(Cpvec)[1L] in R returns the 1-based position of the smallest
    # Cp value; since that position directly equals the number of blocks
    # 'Nval' that minimizes Cp (not merely an array index used for further
    # 0-based indexing downstream), the 1-based semantics are preserved.
    return int(np.argmin(Cpvals)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    def rpow(base: float, exp: float) -> float:
        # Mimic R's '^' operator: a negative base raised to a non-integer
        # power yields NaN in R rather than a complex number.
        if base < 0 and exp != int(exp):
            return float("nan")
        return float(base) ** exp

    x = np.asarray(x, dtype=np.float64)

    # Rename variables
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate (mimic R's match.arg partial matching)
    candidates = ["minim", "stdev", "iqr"]
    matches = [c for c in candidates if c.startswith(scalest)]
    if len(matches) != 1:
        raise ValueError("'scalest' should be one of " + str(candidates))
    scalest = matches[0]

    if scalest == "stdev":
        scale_val = math.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scale_val = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            math.sqrt(np.var(x, ddof=1)),
        )
    scale_val = float(scale_val)

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability
    mean_x = float(np.mean(x))
    sx = (x - mean_x) / scale_val
    sa = (a - mean_x) / scale_val
    sb = (b - mean_x) / scale_val

    # Set up grid points and bin the data
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps
    if level == 0:
        hpi = rpow(24 * math.sqrt(math.pi) / n, 1 / 3)
    elif level == 1:
        alpha = rpow(2 / (3 * n), 1 / 5) * math.sqrt(2)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = rpow(6 / (-psi2hat * n), 1 / 3)
    elif level == 2:
        alpha = rpow(2 / (5 * n), 1 / 7) * math.sqrt(2)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(math.sqrt(2 / math.pi) / (psi4hat * n), 1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = rpow(6 / (-psi2hat * n), 1 / 3)
    elif level == 3:
        alpha = rpow(2 / (7 * n), 1 / 9) * math.sqrt(2)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-3 * math.sqrt(2 / math.pi) / (psi6hat * n), 1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(math.sqrt(2 / math.pi) / (psi4hat * n), 1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = rpow(6 / (-psi2hat * n), 1 / 3)
    elif level == 4:
        alpha = rpow(2 / (9 * n), 1 / 11) * math.sqrt(2)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(15 * math.sqrt(2 / math.pi) / (psi8hat * n), 1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-3 * math.sqrt(2 / math.pi) / (psi6hat * n), 1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(math.sqrt(2 / math.pi) / (psi4hat * n), 1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = rpow(6 / (-psi2hat * n), 1 / 3)
    elif level == 5:
        alpha = rpow(2 / (11 * n), 1 / 13) * math.sqrt(2)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-105 * math.sqrt(2 / math.pi) / (psi10hat * n), 1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(15 * math.sqrt(2 / math.pi) / (psi8hat * n), 1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-3 * math.sqrt(2 / math.pi) / (psi6hat * n), 1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(math.sqrt(2 / math.pi) / (psi4hat * n), 1 / 5)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = rpow(6 / (-psi2hat * n), 1 / 3)
    else:
        raise ValueError("Level should be between 0 and 5")

    return scale_val * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    def rpow(base: float, exp: float) -> float:
        # Mimic R's '^' operator: a negative base raised to a non-integer
        # power yields NaN in R rather than a complex number.
        if base < 0 and exp != int(exp):
            return float("nan")
        return float(base) ** exp

    # Mimic R's match.arg partial matching for 'kernel'
    kernel_candidates = ["normal", "box", "epanech", "biweight", "triweight"]
    kernel_matches = [c for c in kernel_candidates if c.startswith(kernel)]
    if len(kernel_matches) != 1:
        raise ValueError("'kernel' should be one of " + str(kernel_candidates))
    kernel = kernel_matches[0]

    ## Set kernel constants
    if canonical:
        del0 = 1.0
    elif kernel == "normal":
        del0 = 1 / rpow(4 * math.pi, 1 / 10)
    elif kernel == "box":
        del0 = rpow(9 / 2, 1 / 5)
    elif kernel == "epanech":
        del0 = rpow(15, 1 / 5)
    elif kernel == "biweight":
        del0 = rpow(35, 1 / 5)
    else:  # "triweight"
        del0 = rpow(9450 / 143, 1 / 5)

    x = np.asarray(x, dtype=np.float64)

    ## Rename variables
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate (mimic R's match.arg partial matching)
    scale_candidates = ["minim", "stdev", "iqr"]
    scale_matches = [c for c in scale_candidates if c.startswith(scalest)]
    if len(scale_matches) != 1:
        raise ValueError("'scalest' should be one of " + str(scale_candidates))
    scalest = scale_matches[0]

    if scalest == "stdev":
        scale_val = math.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scale_val = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            math.sqrt(np.var(x, ddof=1)),
        )
    scale_val = float(scale_val)

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:
    mean_x = float(np.mean(x))
    sx = (x - mean_x) / scale_val
    sa = (a - mean_x) / scale_val
    sb = (b - mean_x) / scale_val

    ## Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    ## Perform plug-in steps:
    if level == 0:
        psi4hat = 3 / (8 * math.sqrt(math.pi))
    elif level == 1:
        alpha = rpow(2 * rpow(math.sqrt(2), 7) / (5 * n), 1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = rpow(2 * rpow(math.sqrt(2), 9) / (7 * n), 1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-3 * math.sqrt(2 / math.pi) / (psi6hat * n), 1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = rpow(2 * rpow(math.sqrt(2), 11) / (9 * n), 1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(15 * math.sqrt(2 / math.pi) / (psi8hat * n), 1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-3 * math.sqrt(2 / math.pi) / (psi6hat * n), 1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = rpow(2 * rpow(math.sqrt(2), 13) / (11 * n), 1 / 13)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-105 * math.sqrt(2 / math.pi) / (psi10hat * n), 1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(15 * math.sqrt(2 / math.pi) / (psi8hat * n), 1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-3 * math.sqrt(2 / math.pi) / (psi6hat * n), 1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:  # level == 5
        alpha = rpow(2 * rpow(math.sqrt(2), 15) / (13 * n), 1 / 15)  # bandwidth for psi_12
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(945 * math.sqrt(2 / math.pi) / (psi12hat * n), 1 / 13)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-105 * math.sqrt(2 / math.pi) / (psi10hat * n), 1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(15 * math.sqrt(2 / math.pi) / (psi8hat * n), 1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = rpow(-3 * math.sqrt(2 / math.pi) / (psi6hat * n), 1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return scale_val * del0 * rpow(1 / (psi4hat * n), 1 / 5)


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    ## Trim the 100(trim)% of the data from each end (in the x-direction).

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    n_full = len(x)
    indlow = int(np.floor(trim * n_full))
    indupp = n_full - int(np.floor(trim * n_full))

    x = x[indlow:indupp]
    y = y[indlow:indupp]

    ## Rename common parameters
    ## NOTE: as in the original R code, the default 'range.x = range(x)'
    ## is evaluated (lazily) only when first accessed below, i.e. AFTER
    ## 'x' has been re-sorted and trimmed. Hence, when the caller does not
    ## supply 'range_x', it must be computed from the trimmed 'x', not the
    ## original input.
    n = len(x)
    M = int(gridsize)
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
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                     range_x=range_x, binned=True)["y"]

    llow = int(np.floor(proptrun * M))
    lupp = M - int(np.floor(proptrun * M))
    th22kn = float(np.sum((mddest[llow:lupp] ** 2) * xcounts[llow:lupp]) / n)

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / math.sqrt(2 * math.pi)) ** (1 / 9)
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
    return float((sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1 / 5))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0

    a = float(gpoints[0])
    b = float(gpoints[M - 1])

    gcnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    for i in range(n):
        xi = float(X[i])
        lxi = ((xi - a) / delta) + 1

        # Find integer part of "lxi" (truncation toward zero, as in Fortran's int())
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
    n = X.shape[0]
    M1 = len(gpoints1)
    M2 = len(gpoints2)

    a1 = float(gpoints1[0])
    a2 = float(gpoints2[0])
    b1 = float(gpoints1[M1 - 1])
    b2 = float(gpoints2[M2 - 1])

    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    for i in range(n):
        lxi1 = ((float(X[i, 0]) - a1) / delta1) + 1
        lxi2 = ((float(X[i, 1]) - a2) / delta2) + 1

        # Find integer part of "lxi1" and "lxi2" (truncation toward zero, as in Fortran's int())
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


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    ## Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.any(np.atleast_1d(np.asarray(bandwidth, dtype=np.float64)) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    x = np.asarray(x, dtype=np.float64)
    if y is not None:
        y = np.asarray(y, dtype=np.float64)

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

    ## Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
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
        ## Bin the data if not already binned
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

    ## Set the bin width
    delta = (b - a) / (M - 1)

    if bandwidth is None:
        raise ValueError('argument "bandwidth" is missing, with no default')

    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    ## Discretise the bandwidths
    if len(bandwidth_arr) == M:
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
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.array([int(np.floor(tau * bandwidth_arr[0] / delta))], dtype=np.int64)
        hdisc = np.array([bandwidth_arr[0]], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    ## Allocate space for the kernel vector and final estimate.
    ## Arrays below use 1-based indexing (index 0 is unused padding) to
    ## mirror the original FORTRAN routine "locpol" as closely as possible.
    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = [0.0] * (dimfkap + 1)
    curvest = [0.0] * (M + 1)
    midpts = [0] * (Q + 1)
    ss = [[0.0] * (ppp + 1) for _ in range(M + 1)]
    tt = [[0.0] * (pp + 1) for _ in range(M + 1)]

    xcnts = [0.0] + [float(v) for v in xcounts]
    ycnts = [0.0] + [float(v) for v in ycounts]
    Lvec1 = [0] + [int(v) for v in Lvec]
    indic1 = [0] + [int(v) for v in indic]
    hdisc1 = [0.0] + [float(v) for v in hdisc]

    def _locpol_core() -> None:
        ## Obtain kernel weights
        mid = Lvec1[1] + 1
        for i in range(1, Q):
            midpts[i] = mid
            fkap[mid] = 1.0
            for j in range(1, Lvec1[i] + 1):
                fkap[mid + j] = math.exp(-((delta * j / hdisc1[i]) ** 2) / 2)
                fkap[mid - j] = fkap[mid + j]
            mid = mid + Lvec1[i] + Lvec1[i + 1] + 1
        midpts[Q] = mid
        fkap[mid] = 1.0
        for j in range(1, Lvec1[Q] + 1):
            fkap[mid + j] = math.exp(-((delta * j / hdisc1[Q]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]

        ## Combine kernel weights and grid counts
        for k in range(1, M + 1):
            if xcnts[k] != 0:
                for i in range(1, Q + 1):
                    lo = max(1, k - Lvec1[i])
                    hi = min(M, k + Lvec1[i])
                    for j in range(lo, hi + 1):
                        if indic1[j] == i:
                            fac = 1.0
                            w = fkap[k - j + midpts[i]]
                            ss[j][1] += xcnts[k] * w
                            tt[j][1] += ycnts[k] * w
                            for ii in range(2, ppp + 1):
                                fac = fac * delta * (k - j)
                                ss[j][ii] += xcnts[k] * w * fac
                                if ii <= pp:
                                    tt[j][ii] += ycnts[k] * w * fac

        ## Solve the local weighted least-squares system at each grid point
        ## (replaces the LINPACK dgefa/dgesl calls with an equivalent solve).
        for k in range(1, M + 1):
            Smat = np.zeros((pp, pp), dtype=np.float64)
            Tvec = np.zeros(pp, dtype=np.float64)
            for i in range(1, pp + 1):
                for j in range(1, pp + 1):
                    indss = i + j - 1
                    Smat[i - 1, j - 1] = ss[k][indss]
                Tvec[i - 1] = tt[k][i]
            solution = np.linalg.solve(Smat, Tvec)
            curvest[k] = solution[drv]

    _locpol_core()

    curvest_arr = math.gamma(drv + 1) * np.array(curvest[1:M + 1], dtype=np.float64)

    return {"x": gpoints, "y": curvest_arr}


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0

    a = float(gpoints[0])
    b = float(gpoints[M - 1])

    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    for i in range(n):
        xi = float(X[i])
        yi = float(Y[i])
        lxi = ((xi - a) / delta) + 1

        # Find integer part of "lxi" (truncation toward zero, as in Fortran's int())
        li = int(lxi)
        rem = lxi - li

        # Correction for right endpoint (not included if li == M)
        if xi == b:
            li = M - 1
            rem = 1.0

        if li >= 1 and li < M:
            xcnts[li - 1] = xcnts[li - 1] + (1 - rem)
            xcnts[li] = xcnts[li] + rem
            ycnts[li - 1] = ycnts[li - 1] + (1 - rem) * yi
            ycnts[li] = ycnts[li] + rem * yi

        if li < 1 and trun == 0:
            xcnts[0] = xcnts[0] + 1
            ycnts[0] = ycnts[0] + yi

        if li >= M and trun == 0:
            xcnts[M - 1] = xcnts[M - 1] + 1
            ycnts[M - 1] = ycnts[M - 1] + yi

    return {"xcounts": xcnts, "ycounts": ycnts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # NOTE: 'drv' and 'kernel' are accepted for signature compatibility with
    # the R original but, exactly as in the R source, are not used by the body.
    if bandwidth is None:
        raise ValueError("'bandwidth' must be supplied")

    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename common variables

    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
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

    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(int)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=int)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(int)
        else:
            indic = np.ones(M, dtype=int)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=int)
        Q = 1
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=int)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    ## Allocate working arrays

    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=int)
    ss = np.zeros((M, ppp), dtype=np.float64)
    Sdg = np.zeros(M, dtype=np.float64)

    ## --- Native reimplementation of the Fortran 'sdiag' routine ---

    ## Obtain kernel weights

    mid = int(Lvec[0]) + 1
    for i in range(1, Q):
        midpts[i - 1] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec[i - 1]) + 1):
            fkap[mid + j - 1] = np.exp(-((delta * j / hdisc[i - 1]) ** 2) / 2)
            fkap[mid - j - 1] = fkap[mid + j - 1]
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        fkap[mid + j - 1] = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid - j - 1] = fkap[mid + j - 1]

    ## Combine kernel weights and grid counts

    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i in range(1, Q + 1):
                j_lo = max(1, k - int(Lvec[i - 1]))
                j_hi = min(M, k + int(Lvec[i - 1]))
                for j in range(j_lo, j_hi + 1):
                    if indic[j - 1] == i:
                        fkap_val = fkap[k - j + midpts[i - 1] - 1]
                        ss[j - 1, 0] += xcounts[k - 1] * fkap_val
                        fac = 1.0
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xcounts[k - 1] * fkap_val * fac

    ## Invert the local "S" matrix at each grid point and take the (1,1)
    ## entry of the inverse as the diagonal smoother-matrix entry Sdg(k)

    for k in range(M):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(pp):
            for j in range(pp):
                Smat[i, j] = ss[k, i + j]
        Sdg[k] = np.linalg.inv(Smat)[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if bandwidth is None:
        raise ValueError("'bandwidth' is missing, with no default")

    if range_x is None:
        if not binned:
            range_x = (float(np.min(x)), float(np.max(x)))
        else:
            raise ValueError("'range.x' is missing, with no default")

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = int(degree) + 1
    ppp = 2 * int(degree) + 1
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
    if bandwidth_arr.size == M:
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
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = int(2 * np.sum(Lvec) + Q)

    # Obtain kernel weights.
    # "fkap" and "midpts" use 1-based indexing (index 0 is unused padding)
    # to mirror the Fortran routine "sstdg" as closely as possible.
    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    midpts = np.zeros(Q + 1, dtype=np.int64)

    mid = int(Lvec[0]) + 1
    for i in range(1, Q):
        midpts[i] = mid
        fkap[mid] = 1.0
        Li = int(Lvec[i - 1])
        hi = hdisc[i - 1]
        for j in range(1, Li + 1):
            fkap[mid + j] = np.exp(-((delta * j / hi) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + Li + int(Lvec[i]) + 1
    midpts[Q] = mid
    fkap[mid] = 1.0
    LQ = int(Lvec[Q - 1])
    hQ = hdisc[Q - 1]
    for j in range(1, LQ + 1):
        fkap[mid + j] = np.exp(-((delta * j / hQ) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts.
    # "ss"/"uu" use 1-based row/column indexing (index 0 is unused padding).
    ss = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    uu = np.zeros((M + 1, ppp + 1), dtype=np.float64)

    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i in range(1, Q + 1):
                Li = int(Lvec[i - 1])
                lo = max(1, k - Li)
                hi_ = min(M, k + Li)
                for j in range(lo, hi_ + 1):
                    if indic[j - 1] == i:
                        fac = 1.0
                        kern_val = fkap[k - j + midpts[i]]
                        ss[j, 1] += xcounts[k - 1] * kern_val
                        uu[j, 1] += xcounts[k - 1] * kern_val ** 2
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcounts[k - 1] * kern_val * fac
                            uu[j, ii] += xcounts[k - 1] * (kern_val ** 2) * fac

    # For each grid point, assemble the local (pp x pp) moment matrices
    # "Smat" and "Umat", invert "Smat" (as dgefa/dgedi do in the Fortran
    # code) and combine to obtain the SS^T diagonal entry.
    SSTd = np.zeros(M, dtype=np.float64)
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Umat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k, indss]
                Umat[i - 1, j - 1] = uu[k, indss]

        Sinv = np.linalg.inv(Smat)

        total = 0.0
        for i in range(pp):
            for j in range(pp):
                total += Sinv[0, i] * Umat[i, j] * Sinv[j, 0]
        SSTd[k - 1] = total

    return {"x": gpoints, "y": SSTd}


def on_attach(libname: str | None = None, pkgname: str | None = None) -> None:
    # R's .onAttach is a package-load hook invoked automatically when the
    # package is attached via library()/require(). Python has no direct
    # analogue of this hook, so this function merely reproduces the
    # message-printing behaviour of packageStartupMessage(), which writes
    # a non-fatal, suppressible message.
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")


def _on_unload(libpath: str) -> None:
    # In R, this hook calls library.dynam.unload("KernSmooth", libpath) to
    # unload the package's compiled Fortran/C shared library when the
    # KernSmooth package is detached/unloaded.
    #
    # This is a pure-Python reimplementation with no compiled extension
    # module backing it, so there is no native shared library to unload.
    # The function is retained as a no-op placeholder purely to preserve
    # the original signature/intent (e.g. for symmetry with a package
    # unload hook), and does nothing.
    pass
