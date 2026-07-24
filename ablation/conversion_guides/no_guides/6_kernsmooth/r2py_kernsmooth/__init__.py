import math
import sys
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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError(
            "'kernel' should be one of " + ", ".join(repr(k) for k in valid_kernels)
        )

    # Rename common variables
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    M = gridsize

    # Set canonical scaling factors
    if kernel == "normal":
        del0 = (1 / (4 * math.pi)) ** (1 / 10)
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
        h = del0 * (243 / (35 * n)) ** (1 / 5) * math.sqrt(np.var(x, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values
    tau = 4 if kernel == "normal" else 1

    if range_x is None:
        range_x = (float(np.min(x)) - tau * h, float(np.max(x)) + tau * h)
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute kernel weights
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
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    # np.fft.ifft already normalizes by 1/P, matching R's fft(..., inverse=TRUE)/P
    y = np.real(np.fft.ifft(kappa * gcounts))[0:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | list[float] | tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]], gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: list[tuple[float, float]] | tuple[tuple[float, float], tuple[float, float]] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.min(h) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    if len(h) == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [
            (float(np.min(x[:, 0])) - 1.5 * h[0], float(np.max(x[:, 0])) + 1.5 * h[0]),
            (float(np.min(x[:, 1])) - 1.5 * h[1], float(np.max(x[:, 1])) + 1.5 * h[1]),
        ]

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [
        np.zeros((1, 1), dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
    ]
    for idx in range(2):
        L[idx] = min(
            int(np.floor(tau * h[idx] * (M[idx] - 1) / (b[idx] - a[idx]))),
            int(M[idx]) - 1,
        )
        lvecid = np.arange(0, int(L[idx]) + 1, dtype=np.float64)
        facid = (b[idx] - a[idx]) / (h[idx] * (M[idx] - 1))
        z = (norm.pdf(lvecid * facid) / h[idx]).reshape(-1, 1)
        tot = float(np.sum(np.concatenate([z.flatten(), z.flatten()[1:][::-1]]))) * facid * h[idx]
        kapid[idx] = z / tot

    kapp = (kapid[0] @ kapid[1].T) / n

    if int(np.min(L)) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    # smallest powers of 2 >= M+L
    P = (2 ** np.ceil(np.log(M.astype(np.float64) + L.astype(np.float64)) / np.log(2))).astype(
        np.int64
    )
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0 : L1 + 1, 0 : L2 + 1] = kapp
    if L1:
        rp[P1 - L1 : P1, 0 : L2 + 1] = kapp[L1:0:-1, :]
    if L2:
        rp[:, P2 - L2 : P2] = rp[:, L2:0:-1]
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    # np.fft.ifft2 already normalizes by 1/(P1*P2), matching
    # R's fft(rp * sp, inverse = TRUE) / (P1 * P2)
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> float:
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
            hmold0 = hmold1        # Compute mth degree Hermite polynomial
            hmold1 = hmnew         # by recurrence.
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    # np.fft.ifft already normalizes by 1/P, matching R's fft(..., inverse=TRUE)/P
    estimate = np.sum(gcounts * np.real(np.fft.ifft(kappam * Gcounts))[0:M]) / (n ** 2)

    return float(estimate)


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    qq = q + 1

    # Native Python/numpy re-implementation of the Fortran routine
    # F_blkest. The n sorted (x, y) points are split into Nval
    # contiguous, nearly equal-size blocks. Within each block a
    # least-squares polynomial fit of degree q is obtained (equivalent
    # to the QR-based fit performed by the Fortran code using Xmat,
    # wk and qraux). The pooled residual sum of squares over all
    # blocks gives sigsqe, while the fitted coefficients are used to
    # evaluate the 2nd and 4th derivatives of the local polynomial at
    # every data point; these are accumulated into th22e and th24e,
    # the plug-in estimates of theta22 = int (m'')^2 and
    # theta24 = int m'' m'''' used by the RSW (1995) bandwidth
    # selector.
    sigsqe = 0.0
    th22e = 0.0
    th24e = 0.0

    stride = n // Nval
    iseold = 0
    for l in range(1, Nval + 1):
        if l < Nval:
            ise = stride * l
        else:
            ise = n

        xj = x[iseold:ise]
        yj = y[iseold:ise]
        k = ise - iseold

        # Design (Vandermonde) matrix for the degree-q polynomial fit
        # of yj on xj: columns are xj**0, xj**1, ..., xj**q.
        Xmat = np.vander(xj, N=qq, increasing=True)
        coef, _, _, _ = np.linalg.lstsq(Xmat, yj, rcond=None)

        fitted = Xmat @ coef
        resid = yj - fitted
        sigsqe += float(np.sum(resid * resid))

        for i in range(k):
            xi = xj[i]

            # Second derivative of the fitted polynomial at xi:
            # d/dx^2 (coef[kpow] * x**kpow) = coef[kpow] * kpow * (kpow-1) * x**(kpow-2)
            d2 = 0.0
            for kpow in range(2, q + 1):
                d2 += coef[kpow] * kpow * (kpow - 1) * xi ** (kpow - 2)

            # Fourth derivative of the fitted polynomial at xi:
            # d/dx^4 (coef[kpow] * x**kpow) =
            #     coef[kpow] * kpow * (kpow-1) * (kpow-2) * (kpow-3) * x**(kpow-4)
            d4 = 0.0
            for kpow in range(4, q + 1):
                d4 += (
                    coef[kpow]
                    * kpow
                    * (kpow - 1)
                    * (kpow - 2)
                    * (kpow - 3)
                    * xi ** (kpow - 4)
                )

            th22e += d2 * d2
            th24e += d2 * d4

        iseold = ise

    sigsqe = sigsqe / (n - Nval * qq)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": sigsqe, "th22e": th22e, "th24e": th24e}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    order = np.argsort(X, kind="stable")
    X = X[order]
    Y = Y[order]

    qq = q + 1

    # Native Python/numpy re-implementation of the Fortran routine
    # F_cp. For each candidate number of blocks Nval = 1, ..., Nmax the
    # n sorted (X, Y) points are split into Nval contiguous, nearly
    # equal-size blocks (block j covers indices
    # (j-1)*floor(n/Nval)+1 .. j*floor(n/Nval), with the final block
    # extended to include any remainder points, exactly as in the
    # Fortran code). A least-squares polynomial fit of degree q is
    # obtained on each block (equivalent to the QR-based fit performed
    # by the Fortran code using Xmat, wk and qraux) and the residual
    # sums of squares are pooled across blocks to give RSS(Nval).
    RSS = np.zeros(Nmax)

    for Nval in range(1, Nmax + 1):
        idiv = n // Nval
        RSS_Nval = 0.0
        for j in range(1, Nval + 1):
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n

            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]

            # Design (Vandermonde) matrix for the degree-q polynomial
            # fit of Yj on Xj: columns are Xj**0, Xj**1, ..., Xj**q.
            Xmat = np.vander(Xj, N=qq, increasing=True)
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fitted = Xmat @ coef
            resid = Yj - fitted
            RSS_Nval += float(np.sum(resid * resid))

        RSS[Nval - 1] = RSS_Nval

    # Compute the array of Mallow's C_p values, using RSS(Nmax) (the
    # finest, least-biased partition) as the reference variance
    # estimate, exactly as in the Fortran code:
    #     Cp(i) = (n - qq*Nmax) * RSS(i) / RSS(Nmax) + 2*qq*i - n
    Cpvals = np.zeros(Nmax)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = (n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1] + 2 * qq * i - n

    # order(Cpvec)[1L] returns the (1-based) index of the smallest
    # C_p value; np.argmin returns the index of the first occurrence
    # of the minimum, matching R's stable ordering behaviour on ties.
    return int(np.argmin(Cpvals)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    # Rename variables
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate (mimic R's match.arg with partial matching)
    choices = ["minim", "stdev", "iqr"]
    if scalest not in choices:
        matches = [c for c in choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError("'scalest' should be one of " + ", ".join(f'\"{c}\"' for c in choices))

    if scalest == "stdev":
        scale_val = float(np.sqrt(np.var(x, ddof=1)))
    elif scalest == "iqr":
        scale_val = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    else:  # "minim"
        scale_val = float(min((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
                               np.sqrt(np.var(x, ddof=1))))

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    mean_x = float(np.mean(x))
    sx = (x - mean_x) / scale_val
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


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    # Resolve 'kernel' (mimic R's match.arg with partial matching)
    kernel_choices = ["normal", "box", "epanech", "biweight", "triweight"]
    if kernel not in kernel_choices:
        matches = [c for c in kernel_choices if c.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError("'kernel' should be one of " + ", ".join(f'\"{c}\"' for c in kernel_choices))

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
    else:  # "triweight"
        del0 = (9450 / 143) ** (1 / 5)

    # Rename variables
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate (mimic R's match.arg with partial matching)
    scalest_choices = ["minim", "stdev", "iqr"]
    if scalest not in scalest_choices:
        matches = [c for c in scalest_choices if c.startswith(scalest)]
        if len(matches) == 1:
            scalest = matches[0]
        else:
            raise ValueError("'scalest' should be one of " + ", ".join(f'\"{c}\"' for c in scalest_choices))

    if scalest == "stdev":
        scale_val = float(np.sqrt(np.var(x, ddof=1)))
    elif scalest == "iqr":
        scale_val = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    else:  # "minim"
        scale_val = float(min((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
                               np.sqrt(np.var(x, ddof=1))))

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    mean_x = float(np.mean(x))
    sx = (x - mean_x) / scale_val
    sa = (a - mean_x) / scale_val
    sb = (b - mean_x) / scale_val

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
    else:
        raise ValueError("Level should be between 0 and 5")

    return float(scale_val * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float] | None = None, truncate: bool = True) -> float:
    # Trim the 100(trim)% of the data from each end (in the x-direction).
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    indlow = int(np.floor(trim * len(x))) + 1
    indupp = len(x) - int(np.floor(trim * len(x)))

    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

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
                      range_x=range_x, binned=True)["y"]

    llow = int(np.floor(proptrun * M)) + 1
    lupp = M - int(np.floor(proptrun * M))
    th22kn = float(np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n)

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * np.sqrt(2) - (4 / 3) * np.sqrt(3)
    C3K = (4 * C3K / (np.sqrt(2 * np.pi))) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    # Now compute a local linear kernel estimate of
    # the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                    range_x=range_x, binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh,
                range_x=range_x, binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh,
                     range_x=range_x, binned=True)["y"]
    sigsqn = float(np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts))
    sigsqd = float(n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts))
    sigsqkn = sigsqn / sigsqd

    # Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * np.sqrt(np.pi) * th22kn * n)) ** (1 / 5))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = float(gpoints[0])
    b = float(gpoints[M - 1])

    # Native Python/numpy re-implementation of the Fortran routine
    # F_linbin, which performs linear binning of the univariate data
    # in X onto the M grid points defined by [a, b].
    X = np.asarray(X, dtype=np.float64)
    gcounts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)
    for i in range(n):
        lxi = ((X[i] - a) / delta) + 1

        # Find integer part of "lxi" (Fortran INT truncates toward zero)
        li = int(lxi)

        rem = lxi - li
        if li >= 1 and li < M:
            gcounts[li - 1] += (1 - rem)
            gcounts[li] += rem

        if li < 1 and trun == 0:
            gcounts[0] += 1

        if li >= M and trun == 0:
            gcounts[M - 1] += 1

    return gcounts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = X.shape[0]
    x1 = np.asarray(X[:, 0], dtype=np.float64)
    x2 = np.asarray(X[:, 1], dtype=np.float64)

    M1 = len(gpoints1)
    M2 = len(gpoints2)
    a1 = float(gpoints1[0])
    a2 = float(gpoints2[0])
    b1 = float(gpoints1[M1 - 1])
    b2 = float(gpoints2[M2 - 1])

    # Native Python/numpy re-implementation of the Fortran routine
    # F_lbtwod, which performs bivariate linear binning of the data
    # in X onto the M1 x M2 grid defined by [a1, b1] x [a2, b2].
    # Observations outside the mesh are ignored (no truncate argument
    # exists for the bivariate case, matching the Fortran comment).
    gcnts = np.zeros(M1 * M2, dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    for i in range(n):
        lxi1 = ((x1[i] - a1) / delta1) + 1
        lxi2 = ((x2[i] - a2) / delta2) + 1

        # Find the integer part of "lxi1" and "lxi2"
        # (Fortran INT truncates toward zero)
        li1 = int(lxi1)
        li2 = int(lxi2)
        rem1 = lxi1 - li1
        rem2 = lxi2 - li2

        if li1 >= 1 and li2 >= 1 and li1 < M1 and li2 < M2:
            ind1 = M1 * (li2 - 1) + li1
            ind2 = M1 * (li2 - 1) + li1 + 1
            ind3 = M1 * li2 + li1
            ind4 = M1 * li2 + li1 + 1
            gcnts[ind1 - 1] += (1 - rem1) * (1 - rem2)
            gcnts[ind2 - 1] += rem1 * (1 - rem2)
            gcnts[ind3 - 1] += (1 - rem1) * rem2
            gcnts[ind4 - 1] += rem1 * rem2

    # R's matrix(out, M1, M2) fills in column-major order, which matches
    # the flat Fortran storage: index = M1*(j-1) + i for entry (i, j).
    return gcnts.reshape(M2, M1).T


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths
    if bandwidth is not None:
        bandwidth_check = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
        if np.any(bandwidth_check <= 0):
            raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (float(np.max(x)) - float(np.min(x)))
            range_x = (float(np.min(x)) - extra, float(np.max(x)) + extra)
        else:
            range_x = (float(np.min(x)), float(np.max(x)))

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
            out = rlbin(x, y, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = x
            ycounts = y
            M = len(xcounts)
            gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    if bandwidth is None:
        raise ValueError('argument "bandwidth" is missing, with no default')
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(math.log(hlow), math.log(hupp), Q))

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
                    ((np.log(bandwidth_arr) - math.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, int(math.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if int(np.min(Lvec)) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Native Python/numpy re-implementation of the Fortran routine F_locpol.
    # (kernel is currently ignored, exactly as in the reference Fortran
    # implementation, which only supports the Gaussian kernel.)
    # For every grid point g, fit a local weighted polynomial regression of
    # the requested "degree" using a Gaussian kernel with the discretised
    # bandwidth hdisc[indic[g]] over the bin centres within Lvec[indic[g]]
    # bins of g, weighted by the bin counts "xcounts" (playing the role of
    # sample sizes) and the kernel weight dnorm(dist / h) / h. The (pp x pp)
    # moment matrix Smat and the (pp,) vector Tvec are built from a shared
    # vector of weighted moments (mirroring the Fortran "ss"/"tt" arrays),
    # solved for the local polynomial coefficients, and the drv-th raw
    # coefficient is kept as the (pre-gamma) curve estimate, matching the
    # Fortran output that R later scales by gamma(drv + 1).
    curvest = np.zeros(M, dtype=np.float64)
    sqrt_two_pi = math.sqrt(2.0 * math.pi)

    for g in range(M):
        qi = int(indic[g]) - 1
        h = float(hdisc[qi])
        L = int(Lvec[qi])

        lo = max(0, g - L)
        hi = min(M - 1, g + L)
        idx = np.arange(lo, hi + 1)

        dist = delta * (idx - g).astype(np.float64)
        kern_weight = np.exp(-0.5 * (dist / h) ** 2) / (h * sqrt_two_pi)

        moments = np.empty(ppp, dtype=np.float64)
        for s in range(ppp):
            moments[s] = np.sum(xcounts[idx] * dist ** s * kern_weight)

        Tvec = np.empty(pp, dtype=np.float64)
        for s in range(pp):
            Tvec[s] = np.sum(ycounts[idx] * dist ** s * kern_weight)

        Smat = np.empty((pp, pp), dtype=np.float64)
        for i in range(pp):
            for j in range(pp):
                Smat[i, j] = moments[i + j]

        try:
            beta = np.linalg.solve(Smat, Tvec)
        except np.linalg.LinAlgError:
            beta = np.full(pp, np.nan, dtype=np.float64)

        curvest[g] = beta[drv]

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = float(gpoints[0])
    b = float(gpoints[M - 1])

    # Native Python/numpy re-implementation of the Fortran routine
    # F_rlbin, which performs linear binning of the regression data
    # (X, Y) onto the M grid points defined by [a, b]. xcounts holds
    # the linear-binning weights (as in linbin) and ycounts holds the
    # same weights applied to the corresponding Y values.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    xcounts = np.zeros(M, dtype=np.float64)
    ycounts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)
    for i in range(n):
        lxi = ((X[i] - a) / delta) + 1

        # Find integer part of "lxi" (Fortran INT truncates toward zero)
        li = int(lxi)
        rem = lxi - li

        # Correction for right endpoint (not included if li == M)
        if X[i] == b:
            li = M - 1
            rem = 1.0

        if li >= 1 and li < M:
            xcounts[li - 1] += (1 - rem)
            xcounts[li] += rem
            ycounts[li - 1] += (1 - rem) * Y[i]
            ycounts[li] += rem * Y[i]

        if li < 1 and trun == 0:
            xcounts[0] += 1
            ycounts[0] += Y[i]

        if li >= M and trun == 0:
            xcounts[M - 1] += 1
            ycounts[M - 1] += Y[i]

    return {"xcounts": xcounts, "ycounts": ycounts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for interface parity with the R
    # function but, exactly as in the original R/Fortran implementation,
    # are never referenced by the computation below (only a Gaussian
    # ("normal") kernel is implemented and no derivative order is used).
    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M = gridsize
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
                indic = np.ones(M, dtype=np.int64)
            else:
                # np.round uses round-half-to-even, matching R's round()
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = 2 * int(np.sum(Lvec)) + Q

    # Native Python/numpy re-implementation of the Fortran routine
    # F_sdiag, which computes the binned diagonal ("self-influence")
    # entries of the local polynomial smoother/hat matrix. All working
    # arrays below are allocated one element larger than needed and
    # left-padded with an unused element at index 0 so that the
    # original 1-based Fortran indexing (mid, k, j, i, ...) can be
    # transliterated directly without further index shifting.
    xcnts = np.zeros(M + 1, dtype=np.float64)
    xcnts[1:M + 1] = xcounts

    hdisc_p = np.zeros(Q + 1, dtype=np.float64)
    hdisc_p[1:Q + 1] = hdisc

    Lvec_p = np.zeros(Q + 1, dtype=np.int64)
    Lvec_p[1:Q + 1] = Lvec

    indic_p = np.zeros(M + 1, dtype=np.int64)
    indic_p[1:M + 1] = indic

    midpts = np.zeros(Q + 1, dtype=np.int64)
    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    ss = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    Sdg_p = np.zeros(M + 1, dtype=np.float64)

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

    # Combine kernel weights and grid counts into the local weighted
    # moment sums "ss" (columns hold the 0th, 1st, ..., (ppp-1)th
    # weighted moments of the centered distances at each grid point).
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

    # At each grid point, assemble the (pp x pp) local weighted moment
    # matrix Smat from "ss" and invert it (equivalent to the original
    # dgefa/dgedi LINPACK calls with job=01, i.e. inverse only); the
    # (1,1) entry of the inverse is the diagonal smoother-matrix entry.
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k, indss]

        Smat_inv = np.linalg.inv(Smat)
        Sdg_p[k] = Smat_inv[0, 0]

    Sdg = Sdg_p[1:M + 1]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables
    M = gridsize
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
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = int(2 * np.sum(Lvec) + Q)

    # Native Python/numpy re-implementation of the Fortran routine
    # F_sstdg, which computes the binned diagonal entries of S*S^T,
    # where S is the (binned) local polynomial smoother matrix, i.e.
    # SSTd[k] = sum_j S[k, j]^2 for each grid point k. All of the
    # working arrays below are padded with an unused leading entry so
    # that they can be indexed exactly as in the original 1-based
    # Fortran arrays.
    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    midpts = np.zeros(Q + 1, dtype=np.int64)
    ss = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    uu = np.zeros((M + 1, ppp + 1), dtype=np.float64)
    SSTd = np.zeros(M, dtype=np.float64)

    Lvec1 = np.concatenate(([0], Lvec))
    hdisc1 = np.concatenate(([0.0], hdisc))
    indic1 = np.concatenate(([0], indic))
    xcnts1 = np.concatenate(([0.0], np.asarray(xcounts, dtype=np.float64)))

    # Obtain kernel weights
    mid = int(Lvec1[1]) + 1
    for i in range(1, Q):
        midpts[i] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec1[i]) + 1):
            fkap[mid + j] = np.exp(-((delta * j / hdisc1[i]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lvec1[i]) + int(Lvec1[i + 1]) + 1
    midpts[Q] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lvec1[Q]) + 1):
        fkap[mid + j] = np.exp(-((delta * j / hdisc1[Q]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcnts1[k] != 0:
            for i in range(1, Q + 1):
                lo = max(1, k - int(Lvec1[i]))
                hi = min(M, k + int(Lvec1[i]))
                for j in range(lo, hi + 1):
                    if indic1[j] == i:
                        fac = 1.0
                        w = fkap[k - j + midpts[i]]
                        ss[j, 1] += xcnts1[k] * w
                        uu[j, 1] += xcnts1[k] * w ** 2
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j, ii] += xcnts1[k] * w * fac
                            uu[j, ii] += xcnts1[k] * (w ** 2) * fac

    # For each grid point, build the local moment matrix Smat and the
    # squared-weight moment matrix Umat, invert Smat (dgefa + dgedi
    # with job = 01 computes the matrix inverse in place), and combine
    # them into SSTd[k] = e1' * inv(Smat) * Umat * inv(Smat) * e1, i.e.
    # the squared L2 norm of the k-th row of the smoother matrix S.
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Umat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k, indss]
                Umat[i - 1, j - 1] = uu[k, indss]

        Sinv = np.linalg.inv(Smat)

        val = 0.0
        for i in range(pp):
            for j in range(pp):
                val += Sinv[0, i] * Umat[i, j] * Sinv[j, 0]
        SSTd[k - 1] = val

    return {"x": gpoints, "y": SSTd}


def on_attach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def on_unload(libpath: str) -> None:
    # In R, .onUnload(libpath) calls library.dynam.unload("KernSmooth", libpath)
    # to unload the package's compiled Fortran/C shared library when the
    # package is detached/unloaded. Python has no equivalent dynamic-library-
    # unload-on-package-detach hook, and this port has no compiled shared
    # library that needs to be explicitly unloaded, so this is a no-op
    # placeholder retained only to mirror the original R package structure.
    pass
