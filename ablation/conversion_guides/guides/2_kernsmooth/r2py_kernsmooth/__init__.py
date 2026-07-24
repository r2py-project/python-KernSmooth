import math
import sys
import warnings
from typing import Any

import numpy as np
from scipy.stats import beta as _beta_dist
from scipy.stats import norm

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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in kernel_choices:
        matches = [c for c in kernel_choices if c.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError(
                "'kernel' should be one of " + ", ".join(repr(c) for c in kernel_choices)
            )

    ## Rename common variables

    n = len(x)
    M = gridsize

    ## Set canonical scaling factors

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

    ## Set default bandwidth

    if bandwidth is None:
        h = del0 * (243 / (35 * n)) ** (1 / 5) * np.std(x, ddof=1)
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    ## Set kernel support values

    tau = 4 if kernel == "normal" else 1

    if range_x is None:
        range_x = (np.min(x) - tau * h, np.max(x) + tau * h)
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute kernel weights

    delta = (b - a) / (h * (M - 1))
    L = int(min(np.floor(tau / delta), M))
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    if kernel == "normal":
        kappa = (np.exp(-0.5 * (lvec * delta) ** 2) / np.sqrt(2 * np.pi)) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * _beta_dist.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * _beta_dist.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * _beta_dist.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * _beta_dist.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    return {"x": gpoints, "y": np.real(np.fft.ifft(kappa * gcounts))[:M]}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: tuple[int, int] | np.ndarray[Any, np.dtype[np.integer[Any]]] = (51, 51), range_x: list[tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]]] | tuple[tuple[float, float], tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    ## Install safeguard against non-positive bandwidths:

    if np.min(np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    ## Rename common variables

    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64)).copy()
    tau = 3.4  # For bivariate normal kernel.

    ## Use same bandwidth in each direction
    ## if only a single bandwidth is given.

    if h.shape[0] == 1:
        h = np.array([h[0], h[0]])

    ## If range_x is not specified then set it at its default value.

    if range_x is None:
        range_x = [
            (np.min(x[:, 0]) - 1.5 * h[0], np.max(x[:, 0]) + 1.5 * h[0]),
            (np.min(x[:, 1]) - 1.5 * h[1], np.max(x[:, 1]) + 1.5 * h[1]),
        ]

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    ## Set up grid points and bin the data

    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    ## Compute kernel weights

    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [None, None]  # type: ignore[list-item]
    for id_ in range(2):
        L[id_] = int(min(np.floor(tau * h[id_] * (M[id_] - 1) / (b[id_] - a[id_])), M[id_] - 1))
        lvecid = np.arange(0, int(L[id_]) + 1)
        facid = (b[id_] - a[id_]) / (h[id_] * (M[id_] - 1))
        z = (norm.pdf(lvecid * facid) / h[id_]).reshape(-1, 1)
        z_flat = z.reshape(-1)
        tot = np.sum(np.concatenate([z_flat, z_flat[1:][::-1]])) * facid * h[id_]
        kapid[id_] = z / tot

    kapp = (kapid[0] @ kapid[1].T) / n

    if int(np.min(L)) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    ## Now combine weight and counts using the FFT to obtain estimate

    P = (2 ** np.ceil(np.log(M.astype(np.float64) + L.astype(np.float64)) / np.log(2))).astype(np.int64)  # smallest powers of 2 >= M+L
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
    ## wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    ## zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    rp = np.fft.ifft2(rp_fft * sp_fft).real[0:M1, 0:M2]
    ## invert element-wise product of FFT's
    ## and truncate and normalise it

    ## Ensure that rp is non-negative

    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        x = np.asarray(x, dtype=np.float64)
        range_x = (np.min(x), np.max(x))

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
    L = int(min(np.floor(tau * h / delta), M))

    if L == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = (np.exp(-0.5 * arg ** 2) / np.sqrt(2 * np.pi)) / (h ** (drv + 1))
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
    P = int(2 ** np.ceil(np.log2(M + L + 1)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    return float(np.sum(gcounts * np.real(np.fft.ifft(kappam * Gcounts))[:M]) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    qq = q + 1

    # Native Python/NumPy re-implementation of the Fortran routine
    # `blkest` (KernSmooth/src/blkest.f), since no pre-existing Python
    # port of F_blkest is available. The Fortran routine partitions the
    # sorted (x, y) data into Nval contiguous blocks, fits a degree-q
    # polynomial via least squares within each block (using dqrdc/dqrsl,
    # here replaced by numpy's least-squares solver), and aggregates the
    # residuals and derivative-based quantities across blocks.
    def _deriv_poly_eval(
        coef: np.ndarray[Any, np.dtype[np.float64]], order: int, xv: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        # Evaluate the `order`-th derivative of the polynomial with
        # ascending-order coefficients `coef` (coef[m] is the coefficient
        # of x**m) at the points in `xv`.
        deg = len(coef) - 1
        result = np.zeros_like(xv, dtype=np.float64)
        for m in range(order, deg + 1):
            factor = 1.0
            for t in range(order):
                factor *= (m - t)
            result = result + factor * coef[m] * xv ** (m - order)
        return result

    RSS = 0.0
    th22e = 0.0
    th24e = 0.0
    idiv = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition (converted to 0-based slicing)
        ilow = (j - 1) * idiv
        iupp = j * idiv
        if j == Nval:
            iupp = n
        Xj = x[ilow:iupp]
        Yj = y[ilow:iupp]

        # Obtain a q'th degree fit over the current member of the partition.
        # Set up the design matrix: columns are 1, Xj, Xj**2, ..., Xj**q
        Xmat = np.vander(Xj, N=qq, increasing=True)

        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        fiti = Xmat @ coef
        ddm = _deriv_poly_eval(coef, 2, Xj)
        ddddm = _deriv_poly_eval(coef, 4, Xj)

        th22e = th22e + np.sum(ddm ** 2)
        th24e = th24e + np.sum(ddm * ddddm)
        RSS = RSS + np.sum((Yj - fiti) ** 2)

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    sort_idx = np.argsort(X)
    X = X[sort_idx]
    Y = Y[sort_idx]

    qq = q + 1

    # Native Python/NumPy re-implementation of the Fortran routine
    # `cp` (KernSmooth/src/cp.f), since no pre-existing Python port of
    # F_cp is available. For each candidate number of blocks Nval from 1
    # to Nmax, the sorted (X, Y) data are partitioned into Nval
    # contiguous blocks, a degree-q polynomial (qq = q + 1 coefficients)
    # is fit within each block via least squares (using numpy's
    # least-squares solver in place of dqrdc/dqrsl), and the residual
    # sum of squares (RSS) is accumulated across blocks. Mallow's C_p
    # statistic is then computed for each Nval, and the (1-based) Nval
    # that minimizes C_p is returned.
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # For each number of partitions
        idiv = n // Nval
        RSSj_total = 0.0

        for j in range(1, Nval + 1):
            # For each member of the partition (converted to 0-based slicing)
            ilow = (j - 1) * idiv
            iupp = j * idiv
            if j == Nval:
                iupp = n
            Xj = X[ilow:iupp]
            Yj = Y[ilow:iupp]

            # Obtain a q'th degree fit over the current member of the
            # partition. Set up the design matrix: columns are
            # 1, Xj, Xj**2, ..., Xj**q
            Xmat = np.vander(Xj, N=qq, increasing=True)

            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fiti = Xmat @ coef
            RSSj_total = RSSj_total + np.sum((Yj - fiti) ** 2)

        RSS[Nval - 1] = RSSj_total

    # Now compute array of Mallow's C_p values.
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    Cpvec = Cpvals

    # order(Cpvec)[1L] in R: the 1-based index of the minimum element.
    return int(np.argmin(Cpvec)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    # Rename variables
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)  # noqa: F841 (result overwritten below, kept for fidelity with R source)

    # Compute scale estimate
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError(
                f"'arg' should be one of {_scalest_choices}"
            )

    std_x = float(np.std(x, ddof=1))
    iqr_x = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    if scalest == "stdev":
        scalest_val = std_x
    elif scalest == "iqr":
        scalest_val = iqr_x
    else:  # "minim"
        scalest_val = min(iqr_x, std_x)

    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    x_mean = float(np.mean(x))
    sx = (x - x_mean) / scalest_val
    sa = (a - x_mean) / scalest_val
    sb = (b - x_mean) / scalest_val

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
    # Note: the R source has a trailing 'else if (level == 5L) { }' branch
    # after the level == 5 case above; it is unreachable dead code (an empty
    # duplicate of the already-handled level == 5 branch) and is therefore
    # omitted here.

    return float(scalest_val * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    # Validate/normalise the 'kernel' argument (match.arg)
    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        matches = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError(
                f"'arg' should be one of {_kernel_choices}"
            )

    # Set kernel constants
    if canonical:
        del0 = 1.0
    else:
        _del0_map = {
            "normal": 1 / ((4 * np.pi) ** (1 / 10)),
            "box": (9 / 2) ** (1 / 5),
            "epanech": 15 ** (1 / 5),
            "biweight": 35 ** (1 / 5),
            "triweight": (9450 / 143) ** (1 / 5),
        }
        del0 = _del0_map[kernel]

    # Rename variables
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)  # noqa: F841 (result overwritten below, kept for fidelity with R source)

    # Compute scale estimate
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError(
                f"'arg' should be one of {_scalest_choices}"
            )

    std_x = float(np.std(x, ddof=1))
    iqr_x = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    if scalest == "stdev":
        scalest_val = std_x
    elif scalest == "iqr":
        scalest_val = iqr_x
    else:  # "minim"
        scalest_val = min(iqr_x, std_x)

    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    x_mean = float(np.mean(x))
    sx = (x - x_mean) / scalest_val
    sa = (a - x_mean) / scalest_val
    sb = (b - x_mean) / scalest_val

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

    return float(scalest_val * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    # Trim the 100(trim)% of the data from each end (in the x-direction).
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    sort_idx = np.argsort(x, kind="stable")
    x = x[sort_idx]
    y = y[sort_idx]

    indlow = int(np.floor(trim * len(x))) + 1
    indupp = len(x) - int(np.floor(trim * len(x)))

    # 1-based inclusive R slice x[indlow:indupp] -> 0-based half-open slice
    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    # Rename common parameters
    n = len(x)
    M = gridsize
    # NB: R evaluates the default `range.x = range(x)` lazily, i.e. only
    # the first time `range.x` is referenced -- which happens after `x`
    # has already been reassigned to the trimmed data above. Replicate
    # that behaviour here by computing the default range from the
    # trimmed x (not the original, untrimmed x).
    if range_x is None:
        a = float(np.min(x))
        b = float(np.max(x))
    else:
        a = float(range_x[0])
        b = float(range_x[1])

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

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=(a, b), binned=True)["y"]

    llow = int(np.floor(proptrun * M)) + 1
    lupp = M - int(np.floor(proptrun * M))
    # 1-based inclusive R slice mddest[llow:lupp] -> 0-based half-open slice
    th22kn = np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * np.sqrt(2) - (4 / 3) * np.sqrt(3)
    C3K = (4 * C3K / np.sqrt(2 * np.pi)) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    # Now compute a local linear kernel estimate of
    # the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                    range_x=(a, b), binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh,
                range_x=(a, b), binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh,
                     range_x=(a, b), binned=True)["y"]
    sigsqn = np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    # Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * np.sqrt(np.pi) * th22kn * n)) ** (1 / 5))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Native Python/NumPy re-implementation of the Fortran routine
    # `linbin` (KernSmooth/src/linbin.f), since no pre-existing Python
    # port of F_linbin is available.
    gcnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    for i in range(n):
        # 1-based grid position, exactly as computed in the Fortran code
        lxi = ((X[i] - a) / delta) + 1

        # Integer part of "lxi" (Fortran INT truncates toward zero,
        # matching Python's int() truncation behavior)
        li = int(lxi)
        rem = lxi - li

        if li >= 1 and li < M:
            # Convert 1-based Fortran indices (li, li+1) to 0-based
            gcnts[li - 1] += (1 - rem)
            gcnts[li] += rem

        if li < 1 and trun == 0:
            gcnts[0] += 1

        if li >= M and trun == 0:
            gcnts[M - 1] += 1

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)

    n = X.shape[0]
    M1 = len(gpoints1)
    M2 = len(gpoints2)
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Native Python/NumPy re-implementation of the Fortran routine
    # `lbtwod` (KernSmooth/src/linbin2D.f), since no pre-existing Python
    # port of F_lbtwod is available. Observations outside the mesh
    # defined by [a1, b1] x [a2, b2] are ignored, matching the Fortran
    # behaviour.
    gcnts = np.zeros((M1, M2), dtype=np.float64)
    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    for i in range(n):
        # 1-based grid positions, exactly as computed in the Fortran code
        lxi1 = ((X[i, 0] - a1) / delta1) + 1
        lxi2 = ((X[i, 1] - a2) / delta2) + 1

        # Integer part of "lxi1" and "lxi2" (Fortran INT truncates
        # toward zero, matching Python's int() truncation behavior)
        li1 = int(lxi1)
        li2 = int(lxi2)
        rem1 = lxi1 - li1
        rem2 = lxi2 - li2

        if li1 >= 1 and li2 >= 1 and li1 < M1 and li2 < M2:
            # Convert 1-based Fortran indices to 0-based NumPy indices.
            # result[i1, i2] corresponds to (gpoints1[i1], gpoints2[i2]),
            # matching R's column-major matrix(out[[9L]], M1, M2) fill.
            gcnts[li1 - 1, li2 - 1] += (1 - rem1) * (1 - rem2)
            gcnts[li1, li2 - 1] += rem1 * (1 - rem2)
            gcnts[li1 - 1, li2] += (1 - rem1) * rem2
            gcnts[li1, li2] += rem1 * rem2

    return gcnts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.any(np.atleast_1d(np.asarray(bandwidth, dtype=np.float64)) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    if range_x is None and not binned:
        x = np.asarray(x, dtype=np.float64)
        if y is None:
            extra = 0.05 * (np.max(x) - np.min(x))
            range_x = (np.min(x) - extra, np.max(x) + extra)
        else:
            range_x = (np.min(x), np.max(x))

    # Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = range_x[0]
    b = range_x[1]
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Decide whether a density estimate or regression estimate is required.
    if y is None:
        # obtain density estimate
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:
        # obtain regression estimate
        # Bin the data if not already binned
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

    if np.min(Lvec) == 0:
        raise ValueError("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    # Allocate space for the kernel vector and final estimate
    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap, dtype=np.float64)
    curvest = np.zeros(M, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)

    # Native Python/NumPy re-implementation of the Fortran routine
    # `locpol` (KernSmooth/src/locpoly.f), since no pre-existing Python
    # port of F_locpol is available. All indices below are converted
    # from the Fortran subroutine's 1-based scheme to 0-based indexing.

    # Obtain kernel weights
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
                lo = max(0, k - int(Lvec[i]))
                hi = min(M - 1, k + int(Lvec[i]))
                for j in range(lo, hi + 1):
                    if indic[j] == i + 1:
                        w = fkap[k - j + midpts[i]]
                        ss[j, 0] += xcounts[k] * w
                        tt[j, 0] += ycounts[k] * w
                        fac = 1.0
                        for ii in range(1, ppp):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcounts[k] * w * fac
                            if ii < pp:
                                tt[j, ii] += ycounts[k] * w * fac

    # Build the local design matrix/system at each grid point (Smat is
    # derived from the moment sums "ss", Tvec from the weighted-y sums
    # "tt") and solve the weighted least-squares polynomial regression.
    # This replaces the Fortran LINPACK calls dgefa/dgesl, which perform
    # an LU decomposition with partial pivoting followed by a solve;
    # numpy.linalg.solve is numerically equivalent for this well-posed
    # linear system.
    for k in range(M):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Tvec = np.zeros(pp, dtype=np.float64)
        for i in range(pp):
            for j in range(pp):
                Smat[i, j] = ss[k, i + j]
            Tvec[i] = tt[k, i]
        coeffs = np.linalg.solve(Smat, Tvec)
        curvest[k] = coeffs[drv]

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

    # Native Python/NumPy re-implementation of the Fortran routine
    # `rlbin` (KernSmooth/src/rlbin.f), since no pre-existing Python
    # port of F_rlbin is available.
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    for i in range(n):
        # 1-based grid position, exactly as computed in the Fortran code
        lxi = ((X[i] - a) / delta) + 1

        # Integer part of "lxi" (Fortran INT truncates toward zero,
        # matching Python's int() truncation behavior)
        li = int(lxi)
        rem = lxi - li

        # Correction for right endpoint (not included if li == M)
        if X[i] == b:
            li = M - 1
            rem = 1

        if li >= 1 and li < M:
            # Convert 1-based Fortran indices (li, li+1) to 0-based
            xcnts[li - 1] += (1 - rem)
            xcnts[li] += rem
            ycnts[li - 1] += (1 - rem) * Y[i]
            ycnts[li] += rem * Y[i]

        if li < 1 and trun == 0:
            xcnts[0] += 1
            ycnts[0] += Y[i]

        if li >= M and trun == 0:
            xcnts[M - 1] += 1
            ycnts[M - 1] += Y[i]

    return {"xcounts": xcnts, "ycounts": ycnts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
        x = np.asarray(x, dtype=np.float64)
        range_x = (np.min(x), np.max(x))

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
    Sdg = np.zeros(M, dtype=np.float64)

    # Native Python/NumPy re-implementation of the Fortran routine
    # `sdiag` (KernSmooth/src/sdiag.f), since no pre-existing Python
    # port of F_sdiag is available. All indices below are converted
    # from the Fortran subroutine's 1-based scheme to 0-based indexing.

    # Obtain kernel weights
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
                lo = max(0, k - int(Lvec[i]))
                hi = min(M - 1, k + int(Lvec[i]))
                for j in range(lo, hi + 1):
                    if indic[j] == i + 1:
                        w = fkap[k - j + midpts[i]]
                        ss[j, 0] += xcounts[k] * w
                        fac = 1.0
                        for ii in range(1, ppp):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcounts[k] * w * fac

    # Build the local moment matrix at each grid point and take the
    # (1,1) entry of its inverse, which is the diagonal entry of the
    # binned local-polynomial smoother ("hat") matrix. This replaces
    # the Fortran LINPACK calls dgefa/dgedi, which perform an LU
    # decomposition with partial pivoting followed by matrix
    # inversion; numpy.linalg.inv is numerically equivalent for this
    # well-posed linear system.
    for k in range(M):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(pp):
            for j in range(pp):
                Smat[i, j] = ss[k, i + j]
        Smat_inv = np.linalg.inv(Smat)
        Sdg[k] = Smat_inv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
        x = np.asarray(x, dtype=np.float64)
        range_x = (np.min(x), np.max(x))

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

    # Native Python/NumPy re-implementation of the Fortran routine
    # `sstdg` (KernSmooth/src/sstdiag.f), since no pre-existing Python
    # port of F_sstdg is available. All indices below are converted
    # from the Fortran subroutine's 1-based scheme to 0-based indexing.

    # Obtain kernel weights
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
                lo = max(0, k - int(Lvec[i]))
                hi = min(M - 1, k + int(Lvec[i]))
                for j in range(lo, hi + 1):
                    if indic[j] == i + 1:
                        w = fkap[k - j + midpts[i]]
                        ss[j, 0] += xcounts[k] * w
                        uu[j, 0] += xcounts[k] * (w ** 2)
                        fac = 1.0
                        for ii in range(1, ppp):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcounts[k] * w * fac
                            uu[j, ii] += xcounts[k] * (w ** 2) * fac

    # Build the local moment matrix "Smat" and the local squared-weight
    # moment matrix "Umat" at each grid point, invert Smat (replacing
    # the Fortran LINPACK calls dgefa/dgedi, which perform an LU
    # decomposition with partial pivoting followed by matrix inversion;
    # numpy.linalg.inv is numerically equivalent for this well-posed
    # linear system), and accumulate the diagonal entry of S U S^T
    # using only the first row/column of the inverse (equivalent to
    # extracting the diagonal entry of S S^T for the binned local
    # polynomial smoother).
    for k in range(M):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Umat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(pp):
            for j in range(pp):
                indss = i + j
                Smat[i, j] = ss[k, indss]
                Umat[i, j] = uu[k, indss]
        Smat_inv = np.linalg.inv(Smat)

        acc = 0.0
        for i in range(pp):
            for j in range(pp):
                acc += Smat_inv[0, i] * Umat[i, j] * Smat_inv[j, 0]
        SSTd[k] = acc

    return {"x": gpoints, "y": SSTd}


def on_attach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def on_unload(libpath: str) -> None:
    # R's .onUnload hook calls library.dynam.unload("KernSmooth", libpath) to
    # unload the compiled Fortran/C shared library from the package namespace.
    # CPython provides no safe, supported API for unloading a native extension
    # module once imported (dlclose()-ing a live extension can segfault if any
    # object still references its symbols), so there is no direct equivalent of
    # library.dynam.unload() here. This function is retained only for structural
    # parity with the original R package hook; it performs no action.
    pass
