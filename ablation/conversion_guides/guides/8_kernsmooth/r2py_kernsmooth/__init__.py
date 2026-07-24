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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
    kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in kernel_choices:
        matches = [c for c in kernel_choices if c.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError(
                f"'kernel' should be one of {kernel_choices}"
            )

    # Rename common variables
    n = len(x)
    M = gridsize

    # Set canonical scaling factors
    del0_map = {
        "normal": (1.0 / (4.0 * np.pi)) ** (1.0 / 10.0),
        "box": (9.0 / 2.0) ** (1.0 / 5.0),
        "epanech": 15.0 ** (1.0 / 5.0),
        "biweight": 35.0 ** (1.0 / 5.0),
        "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
    }
    del0 = del0_map[kernel]

    if not isinstance(canonical, (bool, np.bool_)):
        raise ValueError("'canonical' must be a length-1 logical vector")

    # Set default bandwidth
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.std(x, ddof=1)
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values
    tau = 4.0 if kernel == "normal" else 1.0

    if range_x is None:
        range_x = (np.min(x) - tau * h, np.max(x) + tau * h)
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = int(min(np.floor(tau / delta), M))
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
    else:  # "triweight"
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log2(M + L + 1)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    return {
        "x": gpoints,
        "y": np.fft.ifft(kappa * gcounts).real[:M],
    }


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] = (51, 51), range_x: tuple[tuple[float, float], tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.asarray(bandwidth, dtype=np.float64)
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([float(h), float(h)], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x_list: list[tuple[float, float] | None] = [None, None]
        for idx in range(2):
            range_x_list[idx] = (
                float(np.min(x[:, idx]) - 1.5 * h[idx]),
                float(np.max(x[:, idx]) + 1.5 * h[idx]),
            )
        range_x = (range_x_list[0], range_x_list[1])  # type: ignore[assignment]

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]] | None] = [None, None]
    for idx in range(2):
        L[idx] = int(
            min(
                np.floor(tau * h[idx] * (M[idx] - 1) / (b[idx] - a[idx])),
                M[idx] - 1,
            )
        )
        lvecid = np.arange(0, L[idx] + 1)
        facid = (b[idx] - a[idx]) / (h[idx] * (M[idx] - 1))
        z = norm.pdf(lvecid * facid) / h[idx]
        tot = (np.sum(z) + np.sum(z[1:][::-1])) * facid * h[idx]
        kapid[idx] = z / tot

    # kapid[0] and kapid[1] are 1-D kernel vectors along each dimension;
    # R's `kapid[[1L]] %*% t(kapid[[2L]])` is an outer product of the two.
    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log2(M + L))).astype(np.int64)  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0:L1 + 1, 0:L2 + 1] = kapp
    if L1:
        # R: rp[(P1-L1+1):P1, 1:(L2+1)] <- kapp[(L1+1):2, 1:(L2+1)]
        # 0-based rows P1-L1 .. P1-1 <- kapp rows L1 .. 1 (descending, i.e. reflected)
        rp[P1 - L1:P1, 0:L2 + 1] = kapp[L1:0:-1, 0:L2 + 1]
    if L2:
        # R: rp[, (P2-L2+1):P2] <- rp[, (L2+1):2]
        # 0-based cols P2-L2 .. P2-1 <- cols L2 .. 1 (descending, i.e. reflected)
        rp[:, P2 - L2:P2] = rp[:, L2:0:-1]
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    rp = np.fft.ifft2(rp_fft * sp_fft).real[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp = rp * (rp > 0)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
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
        import warnings
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

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
    P = int(2 ** np.ceil(np.log2(M + L + 1)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    return float(
        np.sum(gcounts * np.fft.ifft(kappam * Gcounts).real[:M]) / (n ** 2)
    )


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    n = len(x)

    # Sort the (x, y) data with respect to the x's -- equivalent to
    # cbind(x, y) followed by datmat[sort.list(datmat[, 1L]), ]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sort_idx = np.argsort(x, kind="stable")
    x = x[sort_idx]
    y = y[sort_idx]

    # Set up arrays for the compiled "blkest" routine
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

    # .Fortran(F_blkest, ...) -- writes sigsqe, th22e, th24e (and the
    # scratch arrays xj, yj, coef, Xmat, wk, qraux) by reference
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

    # Sort the (X, Y) data with respect to the X's -- equivalent to
    # cbind(X, Y) followed by datmat[sort.list(datmat[, 1L]), ]
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    sort_idx = np.argsort(X, kind="stable")
    X = X[sort_idx]
    Y = Y[sort_idx]

    # Set up arrays for the compiled "cp" routine
    qq = q + 1

    RSS = np.zeros(Nmax, dtype=np.float64)
    Xj = np.zeros(n, dtype=np.float64)
    Yj = np.zeros(n, dtype=np.float64)
    coef = np.zeros(qq, dtype=np.float64)
    Xmat = np.zeros((n, qq), dtype=np.float64)
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    wk = np.zeros(n, dtype=np.float64)
    qraux = np.zeros(qq, dtype=np.float64)

    # .Fortran(F_cp, ..., Cpvals = as.double(Cpvals)) -- writes Cpvals
    # (and the scratch arrays RSS, Xj, Yj, coef, Xmat, wk, qraux) by
    # reference; only Cpvals is used afterward
    _KernSmooth.cp(
        X,
        Y,
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

    # R's order(Cpvec)[1L] returns the 1-based index of the minimum
    # element; this value is used directly downstream as a block count
    # (Nval), so the 1-based semantics must be preserved.
    return int(np.argmin(Cpvec)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    ## Rename variables

    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (np.min(x), np.max(x))
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in scalest_choices:
        matches = [c for c in scalest_choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError(
                f"'arg' should be one of {scalest_choices}"
            )

    if scalest == "stdev":
        scale_val = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scale_val = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

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
    ##    delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps

    if level == 0:
        hpi = (24 * np.sqrt(np.pi) / n) ** (1.0 / 3.0)
    elif level == 1:
        alpha = (2 / (3 * n)) ** (1.0 / 5.0) * np.sqrt(2)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 2:
        alpha = ((2 / (5 * n)) ** (1.0 / 7.0)) * np.sqrt(2)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 3:
        alpha = ((2 / (7 * n)) ** (1.0 / 9.0)) * np.sqrt(2)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 4:
        alpha = ((2 / (9 * n)) ** (1.0 / 11.0)) * np.sqrt(2)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)
    else:  # level == 5
        alpha = ((2 / (11 * n)) ** (1.0 / 13.0)) * np.sqrt(2)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * np.sqrt(2 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)

    return float(scale_val * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    ## match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
    kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in kernel_choices:
        matches = [c for c in kernel_choices if c.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError(
                f"'kernel' should be one of {kernel_choices}"
            )

    ## Set kernel constants
    if canonical:
        del0 = 1.0
    else:
        del0_map = {
            "normal": 1.0 / ((4.0 * np.pi) ** (1.0 / 10.0)),
            "box": (9.0 / 2.0) ** (1.0 / 5.0),
            "epanech": 15.0 ** (1.0 / 5.0),
            "biweight": 35.0 ** (1.0 / 5.0),
            "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
        }
        del0 = del0_map[kernel]

    ## Rename variables

    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (np.min(x), np.max(x))
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate

    scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in scalest_choices:
        matches = [c for c in scalest_choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError(
                f"'arg' should be one of {scalest_choices}"
            )

    if scalest == "stdev":
        scale_val = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_val = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scale_val = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

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
    ##    delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps:

    if level == 0:
        psi4hat = 3.0 / (8.0 * np.sqrt(np.pi))
    elif level == 1:
        alpha = (2 * (np.sqrt(2)) ** 7 / (5 * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2 * (np.sqrt(2)) ** 9 / (7 * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2 * (np.sqrt(2)) ** 11 / (9 * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2 * (np.sqrt(2)) ** 13 / (11 * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * np.sqrt(2 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:  # level == 5
        alpha = (2 * (np.sqrt(2)) ** 15 / (13 * n)) ** (1.0 / 15.0)  # bandwidth for psi_12
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = (945 * np.sqrt(2 / np.pi) / (psi12hat * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * np.sqrt(2 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return float(scale_val * del0 * (1 / (psi4hat * n)) ** (1.0 / 5.0))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Sort the (x, y) data with respect to the x's -- equivalent to
    # cbind(x, y) followed by xy[sort.list(xy[, 1L]), ]
    sort_idx = np.argsort(x, kind="stable")
    x = x[sort_idx]
    y = y[sort_idx]

    # Trim the 100(trim)% of the data from each end (in the x-direction).
    n_orig = len(x)
    floor_trim = int(math.floor(trim * n_orig))
    x = x[floor_trim: n_orig - floor_trim]
    y = y[floor_trim: n_orig - floor_trim]

    # Rename common parameters
    n = len(x)
    M = gridsize

    # R's default argument `range.x = range(x)` is a lazily-evaluated
    # promise: it is only evaluated when first accessed (at `a <- range.x[1L]`),
    # which occurs AFTER the sorting/trimming above have reassigned `x`.
    # Hence the default range must be computed from the trimmed x, not the
    # original (pre-trim) x.
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
                      range_x=(a, b), binned=True)["y"]

    floor_proptrun = int(math.floor(proptrun * M))
    th22kn = np.sum(
        (mddest[floor_proptrun: M - floor_proptrun] ** 2)
        * xcounts[floor_proptrun: M - floor_proptrun]
    ) / n

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / math.sqrt(2 * math.pi)) ** (1 / 9)
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
    return float((sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1 / 5))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[-1]

    # Allocate the output buffer -- equivalent to R's double(M)
    gcnts = np.zeros(M, dtype=np.float64)

    # .Fortran(F_linbin, ...) -- writes into gcnts by reference
    _KernSmooth.linbin(
        np.asarray(X, dtype=np.float64),
        n,
        float(a),
        float(b),
        M,
        trun,
        gcnts,
    )

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = X.shape[0]
    # X <- c(X[, 1L], X[, 2L]) -- column-major concatenation of the two columns
    Xflat = np.concatenate([X[:, 0], X[:, 1]]).astype(np.float64)
    M1 = len(gpoints1)
    M2 = len(gpoints2)
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Allocate the output buffer -- equivalent to R's double(M1*M2)
    gcnts = np.zeros(M1 * M2, dtype=np.float64)

    # .Fortran(F_lbtwod, ...) -- writes into gcnts by reference
    _KernSmooth.lbtwod(
        Xflat,
        n,
        float(a1),
        float(a2),
        float(b1),
        float(b2),
        M1,
        M2,
        gcnts,
    )

    # matrix(out[[9L]], M1, M2) fills column-major -- reshape with order='F'
    return gcnts.reshape(M1, M2, order='F')


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
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
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
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
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    curvest = np.zeros(M, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)
    Smat = np.zeros((pp, pp), dtype=np.float64)
    Tvec = np.zeros(pp, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int64)

    # Call FORTRAN routine "locpol" -- writes curvest (and the scratch
    # arrays fkap, midpts, ss, tt, Smat, Tvec, ipvt) by reference
    _KernSmooth.locpol(
        np.asarray(xcounts, dtype=np.float64),
        np.asarray(ycounts, dtype=np.float64),
        drv,
        float(delta),
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
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[-1]

    # Allocate the output buffers -- equivalent to R's double(M), double(M)
    xcounts = np.zeros(M, dtype=np.float64)
    ycounts = np.zeros(M, dtype=np.float64)

    # .Fortran(F_rlbin, ...) -- writes into xcounts and ycounts by reference
    _KernSmooth.rlbin(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        n,
        float(a),
        float(b),
        M,
        trun,
        xcounts,
        ycounts,
    )

    return {"xcounts": xcounts, "ycounts": ycounts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
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
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    Smat = np.zeros((pp, pp), dtype=np.float64)
    work = np.zeros(pp, dtype=np.float64)
    det = np.zeros(2, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int64)
    Sdg = np.zeros(M, dtype=np.float64)

    # Call FORTRAN routine "sdiag" -- writes Sdg (and the scratch
    # arrays fkap, midpts, ss, Smat, work, det, ipvt) by reference
    _KernSmooth.sdiag(
        np.asarray(xcounts, dtype=np.float64),
        float(delta),
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
    if range_x is None and not binned:
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
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    Smat = np.zeros((pp, pp), dtype=np.float64)
    Umat = np.zeros((pp, pp), dtype=np.float64)
    work = np.zeros(pp, dtype=np.float64)
    det = np.zeros(2, dtype=np.float64)
    ipvt = np.zeros(pp, dtype=np.int64)
    SSTd = np.zeros(M, dtype=np.float64)

    # Call FORTRAN routine "sstdg" -- writes SSTd (and the scratch
    # arrays fkap, midpts, ss, uu, Smat, Umat, work, det, ipvt) by reference
    _KernSmooth.sstdg(
        np.asarray(xcounts, dtype=np.float64),
        float(delta),
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
    # Mirrors R's .onAttach hook: packageStartupMessage writes an
    # informational, suppressible message to stderr when the package
    # is attached/imported.
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def on_unload(libpath: str) -> None:
    # R's .onUnload hook called library.dynam.unload("KernSmooth", libpath)
    # to explicitly unload the compiled Fortran/C shared library when the
    # package namespace was detached.
    #
    # Python's import system has no safe, supported mechanism for unloading
    # a compiled C/Fortran extension module (e.g. `_KernSmooth`) once it has
    # been imported: `importlib` does not support unloading extension
    # modules, and forcibly closing the shared library (e.g. via `dlclose`)
    # while live references to its symbols may still exist can crash the
    # interpreter. The extension module's lifetime is instead managed
    # automatically by `sys.modules` for as long as the Python process runs.
    #
    # Consequently there is nothing to translate from `library.dynam.unload`;
    # this function is kept only as a structural placeholder mirroring the
    # R package's `.onUnload` hook and intentionally performs no action.
    pass
