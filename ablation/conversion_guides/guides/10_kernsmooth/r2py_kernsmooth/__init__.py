import math
import sys
import warnings
from typing import Any, Sequence

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm

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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        matches = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError(
                f"'kernel' should be one of {_kernel_choices}"
            )

    ## Rename common variables
    n = len(x)
    M = gridsize

    ## Set canonical scaling factors
    _del0_map = {
        "normal": (1.0 / (4.0 * math.pi)) ** (1.0 / 10.0),
        "box": (9.0 / 2.0) ** (1.0 / 5.0),
        "epanech": 15.0 ** (1.0 / 5.0),
        "biweight": 35.0 ** (1.0 / 5.0),
        "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
    }
    del0 = _del0_map[kernel]

    if not isinstance(canonical, bool):
        raise ValueError("'canonical' must be a length-1 logical vector")

    ## Set default bandwidth
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * math.sqrt(np.var(x, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    ## Set kernel support values
    tau = 4 if kernel == "normal" else 1

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
    L = int(min(math.floor(tau / delta), M))
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    if kernel == "normal":
        kappa = norm.pdf(lvec * delta) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # "triweight"
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    return {
        "x": gpoints,
        "y": (np.fft.ifft(kappa * gcounts).real)[:M],
    }


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.integer[Any]]] = (51, 51), range_x: tuple[tuple[float, float], tuple[float, float]] | list[list[float]] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'truncate' is accepted for interface compatibility with the original
    # R function but is not referenced anywhere in its body.
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.any(np.asarray(bandwidth, dtype=np.float64) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x_arr = np.empty((2, 2), dtype=np.float64)
        for idd in range(2):
            range_x_arr[idd, 0] = np.min(x[:, idd]) - 1.5 * h[idd]
            range_x_arr[idd, 1] = np.max(x[:, idd]) + 1.5 * h[idd]
    else:
        range_x_arr = np.asarray(range_x, dtype=np.float64)

    a = np.array([range_x_arr[0][0], range_x_arr[1][0]], dtype=np.float64)
    b = np.array([range_x_arr[0][1], range_x_arr[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.zeros((1, 1)), np.zeros((1, 1))]
    for idd in range(2):
        L[idd] = min(int(np.floor(tau * h[idd] * (M[idd] - 1) / (b[idd] - a[idd]))), int(M[idd]) - 1)
        lvecid = np.arange(0, L[idd] + 1, dtype=np.float64)
        facid = (b[idd] - a[idd]) / (h[idd] * (M[idd] - 1))
        z = (norm.pdf(lvecid * facid) / h[idd]).reshape(-1, 1)
        tot = np.sum(np.concatenate([z.ravel(), z.ravel()[1:][::-1]])) * facid * h[idd]
        kapid[idd] = z / tot

    kapp = kapid[0] @ kapid[1].T / n

    if np.min(L) == 0:
        import warnings
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log2((M + L).astype(np.float64)))).astype(np.int64)  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0:(L1 + 1), 0:(L2 + 1)] = kapp
    if L1:
        rp[(P1 - L1):P1, 0:(L2 + 1)] = kapp[L1:0:-1, 0:(L2 + 1)]
    if L2:
        rp[:, (P2 - L2):P2] = rp[:, L2:0:-1]
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    rp_out = np.fft.ifft2(rp_fft * sp_fft).real[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp_out = rp_out * (rp_out > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp_out}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths.
    if bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename variables.
    M = gridsize
    a = range_x[0]
    b = range_x[1]
    h = bandwidth

    # Bin the data if not already binned.
    if not binned:
        gpoints = np.linspace(a, b, gridsize)
        gcounts = linbin(x, gpoints, truncate)
    else:
        gcounts = x
        M = len(gcounts)
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width.
    n = np.sum(gcounts)
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

    kappam = (np.exp(-(arg ** 2) / 2) / np.sqrt(2 * np.pi)) / (h ** (drv + 1))
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
    # we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    return float(
        np.sum(gcounts * (np.fft.ifft(kappam * Gcounts).real)[:M]) / (n ** 2)
    )


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    order_idx = np.argsort(x, kind='stable')
    x = x[order_idx]
    y = y[order_idx]

    # Set up quantities equivalent to the FORTRAN subroutine "blkest".
    # It is assumed that the (x, y) data are sorted with respect to the x's.
    qq = q + 1
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0
    idiv = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition.
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        nj = iupp - ilow + 1

        Xj = x[ilow - 1:iupp]
        Yj = y[ilow - 1:iupp]

        # Obtain a q'th degree fit over the current member of the
        # partition. Set up the design ("X") matrix: columns are
        # 1, Xj, Xj^2, ..., Xj^(qq-1) (increasing powers).
        Xmat = np.vander(Xj, N=qq, increasing=True)

        # Least-squares polynomial fit, equivalent to the QR-based
        # solve performed by dqrdc/dqrsl in the original Fortran code.
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        for i in range(nj):
            # fiti: fitted value of the q'th degree polynomial at Xj[i].
            # ddm: 2nd derivative of the fitted polynomial at Xj[i].
            # ddddm: 4th derivative of the fitted polynomial at Xj[i].
            fiti = coef[0]
            ddm = 2.0 * coef[2]
            ddddm = 24.0 * coef[4]
            for k in range(2, qq + 1):
                fiti = fiti + coef[k - 1] * Xj[i] ** (k - 1)
                if k <= q - 1:
                    ddm = ddm + k * (k + 1) * coef[k + 1] * Xj[i] ** (k - 1)
                    if k <= q - 3:
                        ddddm = (
                            ddddm
                            + k * (k + 1) * (k + 2) * (k + 3)
                            * coef[k + 3] * Xj[i] ** (k - 1)
                        )

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
    order_idx = np.argsort(X, kind='stable')
    X = X[order_idx]
    Y = Y[order_idx]

    # Set up arrays equivalent to the FORTRAN subroutine "cp".
    qq = q + 1
    RSS = np.zeros(Nmax, dtype=np.float64)
    Cpvals = np.zeros(Nmax, dtype=np.float64)

    # It is assumed that the (X, Y) data are sorted with respect to the X's.
    # Compute vector of RSS values for each candidate number of blocks.
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

            # Obtain a q'th degree fit over the current member of the
            # partition. Set up the design ("X") matrix: columns are
            # 1, Xj, Xj^2, ..., Xj^(qq-1) (increasing powers).
            Xmat = np.vander(Xj, N=qq, increasing=True)

            # Least-squares polynomial fit, equivalent to the QR-based
            # solve performed by dqrdc/dqrsl in the original Fortran code.
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fitj = Xmat @ coef
            RSSj = np.sum((Yj - fitj) ** 2)

            RSS[Nval - 1] = RSS[Nval - 1] + RSSj

    # Now compute array of Mallow's C_p values.
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    # order(Cpvec)[1L] in R returns the 1-based index of the minimum
    # C_p value; since Cpvals[i - 1] corresponds to Nval == i, adding 1
    # to the 0-based argmin yields the chosen number of blocks (Nval).
    return int(np.argmin(Cpvals)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)

    if level > 5:
        raise ValueError("Level should be between 0 and 5")

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

    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError(f"'scalest' should be one of {_scalest_choices}")

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


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)

    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'arg' should be one of " + ", ".join(repr(k) for k in valid_kernels))

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
    elif kernel == "triweight":
        del0 = (9450 / 143) ** (1 / 5)

    # Rename variables
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate
    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'arg' should be one of " + ", ".join(repr(s) for s in valid_scalest))

    if scalest == "stdev":
        scale = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale = (np.quantile(x, 3 / 4) - np.quantile(x, 1 / 4)) / 1.349
    else:
        scale = min(
            (np.quantile(x, 3 / 4) - np.quantile(x, 1 / 4)) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

    if scale == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    xmean = np.mean(x)
    sx = (x - xmean) / scale
    sa = (a - xmean) / scale
    sb = (b - xmean) / scale

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

    return float(scale * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | Sequence[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    order_idx = np.argsort(x, kind='stable')
    x = x[order_idx]
    y = y[order_idx]

    n0 = len(x)
    indlow = int(np.floor(trim * n0)) + 1
    indupp = n0 - int(np.floor(trim * n0))

    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    ## Rename common parameters
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
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

    # Equivalent of Fortran subroutine linbin: linear binning of the data
    # in X onto the equally spaced grid defined by [a, b] with M points.
    gcnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    for i in range(n):
        lxi = ((X[i] - a) / delta) + 1

        # Find integer part of "lxi" (truncation toward zero, as Fortran int())
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
    # R flattens X column-major into c(X[, 1], X[, 2]); here we keep the
    # two columns separate and index them directly, which is equivalent.
    x1 = X[:, 0]
    x2 = X[:, 1]

    M1 = len(gpoints1)
    M2 = len(gpoints2)
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Equivalent of Fortran subroutine lbtwod: bivariate linear binning of
    # the data (x1, x2) onto the M1 x M2 grid defined by
    # [a1, b1] x [a2, b2]. Observations outside the mesh are ignored.
    # The result is stored as an M1 x M2 array, matching the column-major
    # M1 x M2 matrix produced by the R/Fortran routine (rows indexed by
    # the first grid, columns indexed by the second grid).
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    for i in range(n):
        lxi1 = ((x1[i] - a1) / delta1) + 1
        lxi2 = ((x2[i] - a2) / delta2) + 1

        # Find the integer part of "lxi1" and "lxi2" (truncation toward
        # zero, as Fortran int())
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
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths:
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
            range_x = (np.min(x) - extra, np.max(x) + extra)
        else:
            range_x = (np.min(x), np.max(x))

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
        n = len(x)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        y = np.asarray(y, dtype=np.float64)
        # Bin the data if not already binned
        if not binned:
            gpoints = np.linspace(a, b, M)
            out_bin = rlbin(x, y, gpoints, truncate)
            xcounts = out_bin["xcounts"]
            ycounts = out_bin["ycounts"]
        else:
            xcounts = x
            ycounts = y
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
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, int(np.floor(tau * float(bandwidth_arr[0]) / delta)), dtype=np.int64)
        hdisc = np.full(Q, float(bandwidth_arr[0]), dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Allocate space for the kernel vector and final estimate.
    #
    # NOTE: the arrays below are allocated one element larger than strictly
    # needed and index 0 of each is left unused. This mirrors the 1-based
    # indexing of the original FORTRAN routine "locpol" (KernSmooth/src/
    # locpoly.f) index-for-index, which greatly reduces the risk of an
    # off-by-one translation error in this numerically dense routine.
    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    curvest = np.zeros(M + 1, dtype=np.float64)
    midpts = np.zeros(Q + 1, dtype=np.int64)
    ss = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    tt = np.zeros((M + 1, pp + 1), dtype=np.float64)
    Smat = np.zeros((pp + 1, pp + 1), dtype=np.float64)
    Tvec = np.zeros(pp + 1, dtype=np.float64)

    xcounts_p = np.concatenate(([0.0], np.asarray(xcounts, dtype=np.float64)))
    ycounts_p = np.concatenate(([0.0], np.asarray(ycounts, dtype=np.float64)))
    indic_p = np.concatenate(([0], np.asarray(indic, dtype=np.int64)))
    Lvec_p = np.concatenate(([0], np.asarray(Lvec, dtype=np.int64)))
    hdisc_p = np.concatenate(([0.0], np.asarray(hdisc, dtype=np.float64)))

    # --- Call FORTRAN routine "locpol" (KernSmooth/src/locpoly.f) ---
    # (translated directly below, 1-based indices preserved via padding)

    # Obtain kernel weights
    mid = int(Lvec_p[1]) + 1
    for i in range(1, Q):
        midpts[i] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec_p[i]) + 1):
            fkap[mid + j] = np.exp(-((delta * j / hdisc_p[i]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lvec_p[i]) + int(Lvec_p[i + 1]) + 1
    midpts[Q] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lvec_p[Q]) + 1):
        fkap[mid + j] = np.exp(-((delta * j / hdisc_p[Q]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcounts_p[k] != 0:
            for i in range(1, Q + 1):
                lo = max(1, k - int(Lvec_p[i]))
                hi = min(M, k + int(Lvec_p[i]))
                for j in range(lo, hi + 1):
                    if indic_p[j] == i:
                        fac = 1.0
                        ss[j, 1] += xcounts_p[k] * fkap[k - j + midpts[i]]
                        tt[j, 1] += ycounts_p[k] * fkap[k - j + midpts[i]]
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcounts_p[k] * fkap[k - j + midpts[i]] * fac
                            if ii <= pp:
                                tt[j, ii] += ycounts_p[k] * fkap[k - j + midpts[i]] * fac

    for k in range(1, M + 1):
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i, j] = ss[k, indss]
            Tvec[i] = tt[k, i]

        Smat_sub = Smat[1:pp + 1, 1:pp + 1]
        Tvec_sub = Tvec[1:pp + 1]
        Tvec_sol = np.linalg.solve(Smat_sub, Tvec_sub)

        curvest[k] = Tvec_sol[drv]

    curvest_out = curvest[1:M + 1]
    curvest_out = math.gamma(drv + 1) * curvest_out

    return {"x": gpoints, "y": curvest_out}


def on_attach(libname: str, pkgname: str) -> None:
    # R's .onAttach hook calls packageStartupMessage() when the package is
    # attached via library()/require(). packageStartupMessage() writes its
    # text to stderr (not stdout) and is purely informational, so the most
    # faithful Python translation is a print to sys.stderr.
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def on_unload(libpath: str) -> None:
    # R's .onUnload hook calls library.dynam.unload("KernSmooth", libpath) to
    # explicitly detach the package's compiled shared library from the R session.
    #
    # Python's import system has no safe, supported way to unload a compiled
    # C/Fortran extension module (e.g. _KernSmooth) once it has been imported:
    # CPython does not guarantee that dlclose()-ing an extension is safe while
    # live references to its symbols may still exist. Consequently there is no
    # meaningful Python equivalent of library.dynam.unload, and this function is
    # kept only as a no-op placeholder mirroring the R package-unload hook.
    pass


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Equivalent of Fortran subroutine rlbin: linear binning of a bivariate
    # regression data set (X, Y) onto the equally spaced grid defined by
    # [a, b] with M points, accumulating both xcounts (bin mass) and
    # ycounts (X-weighted-Y bin mass).
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    for i in range(n):
        lxi = ((X[i] - a) / delta) + 1

        # Find integer part of "lxi" (truncation toward zero, as Fortran int())
        li = int(lxi)
        rem = lxi - li

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


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for interface compatibility with the
    # original R function but are not used in its body (only 'degree' is).
    x = np.asarray(x, dtype=np.float64)

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
        Lvec = np.floor(tau * hdisc / delta).astype(int)

        # Determine index of closest entry of "hdisc"
        # to each member of "bandwidth"
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

    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=int)
    ss = np.zeros((M, ppp), dtype=np.float64)
    Sdg = np.zeros(M, dtype=np.float64)

    # Equivalent of Fortran subroutine sdiag: binned diagonal entries of
    # the smoother (hat) matrix for local polynomial kernel regression.
    #
    # Obtain kernel weights (all indices below mirror the 1-based Fortran
    # indexing of Lvec/hdisc/midpts/fkap; array reads/writes use '- 1' to
    # translate into 0-based numpy indices).
    mid = Lvec[0] + 1
    for i in range(Q - 1):
        midpts[i] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, Lvec[i] + 1):
            fkap[mid + j - 1] = np.exp(-((delta * j / hdisc[i]) ** 2) / 2)
            fkap[mid - j - 1] = fkap[mid + j - 1]
        mid = mid + Lvec[i] + Lvec[i + 1] + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, Lvec[Q - 1] + 1):
        fkap[mid + j - 1] = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid - j - 1] = fkap[mid + j - 1]

    # Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i in range(1, Q + 1):
                jlo = max(1, k - Lvec[i - 1])
                jhi = min(M, k + Lvec[i - 1])
                for j in range(jlo, jhi + 1):
                    if indic[j - 1] == i:
                        fac = 1.0
                        ss[j - 1, 0] = ss[j - 1, 0] + xcounts[k - 1] * fkap[k - j + midpts[i - 1] - 1]
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] = (
                                ss[j - 1, ii - 1]
                                + xcounts[k - 1] * fkap[k - j + midpts[i - 1] - 1] * fac
                            )

    # For each grid point, assemble the local moment matrix Smat from the
    # accumulated ss values, invert it (LINPACK's dgefa/dgedi computed the
    # inverse of Smat in-place with job=01; np.linalg.inv is equivalent),
    # and take its (1,1) entry as the diagonal entry of the smoother matrix.
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k - 1, indss - 1]

        Smat_inv = np.linalg.inv(Smat)
        Sdg[k - 1] = Smat_inv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

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
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    SSTd = np.zeros(M, dtype=np.float64)

    # Equivalent of Fortran subroutine sstdg: computes the binned
    # diagonal entries of SS^T, where S is the local polynomial
    # smoother matrix. First obtain the (discretised) kernel weights.
    mid = int(Lvec[0])
    for i in range(Q - 1):
        midpts[i] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec[i]) + 1):
            fkap[mid + j] = np.exp(-((delta * j / hdisc[i]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lvec[i]) + int(Lvec[i + 1]) + 1
    midpts[Q - 1] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        fkap[mid + j] = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts
    for k in range(M):
        if xcounts[k] != 0:
            for i in range(Q):
                lower = max(0, k - int(Lvec[i]))
                upper = min(M - 1, k + int(Lvec[i]))
                for j in range(lower, upper + 1):
                    if indic[j] == i + 1:
                        fkap_val = fkap[k - j + midpts[i]]
                        ss[j, 0] += xcounts[k] * fkap_val
                        uu[j, 0] += xcounts[k] * fkap_val ** 2
                        fac = 1.0
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j, ii - 1] += xcounts[k] * fkap_val * fac
                            uu[j, ii - 1] += xcounts[k] * (fkap_val ** 2) * fac

    # For each grid point, build the local (pp x pp) moment matrices
    # Smat and Umat from ss/uu, invert Smat (equivalent to dgefa/dgedi
    # with job=01, i.e. inverse only) and accumulate the SS^T diagonal
    # entry: SSTd(k) = Smat^{-1}[0, :] . Umat . Smat^{-1}[:, 0]
    for k in range(M):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Umat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k, indss - 1]
                Umat[i - 1, j - 1] = uu[k, indss - 1]

        Sinv = np.linalg.inv(Smat)
        total = 0.0
        for i in range(pp):
            for j in range(pp):
                total += Sinv[0, i] * Umat[i, j] * Sinv[j, 0]
        SSTd[k] = total

    return {"x": gpoints, "y": SSTd}
