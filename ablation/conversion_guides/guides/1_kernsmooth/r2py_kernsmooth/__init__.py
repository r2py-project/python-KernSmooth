import math
import sys
import warnings
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from . import _KernSmooth


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        _matches = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(_matches) == 1:
            kernel = _matches[0]
        else:
            raise ValueError(
                f"'kernel' should be one of {', '.join(repr(c) for c in _kernel_choices)}"
            )

    x_arr = np.asarray(x, dtype=np.float64)

    ## Rename common variables

    n = len(x_arr)
    M = gridsize

    ## Set canonical scaling factors

    _kernel_del0 = {
        "normal": (1.0 / (4.0 * np.pi)) ** (1.0 / 10.0),
        "box": (9.0 / 2.0) ** (1.0 / 5.0),
        "epanech": 15.0 ** (1.0 / 5.0),
        "biweight": 35.0 ** (1.0 / 5.0),
        "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
    }
    del0 = _kernel_del0[kernel]

    if not isinstance(canonical, bool):
        raise ValueError("'canonical' must be a length-1 logical vector")

    ## Set default bandwidth

    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.std(x_arr, ddof=1)
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    ## Set kernel support values

    tau = 4.0 if kernel == "normal" else 1.0

    if range_x is None:
        range_x = (float(np.min(x_arr)) - tau * h, float(np.max(x_arr)) + tau * h)
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    ## Compute kernel weights

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
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # kernel == "triweight"
        kappa = 0.5 * beta_dist.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    return {"x": gpoints, "y": np.fft.ifft(kappa * gcounts).real[:M]}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] = (51, 51), range_x: list[tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.min(np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    ## Rename common variables

    x_arr = np.asarray(x, dtype=np.float64)
    n = x_arr.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    ## Use same bandwidth in each direction
    ## if only a single bandwidth is given.

    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    ## If range_x is not specified then set it at its default value.

    if range_x is None:
        range_x = [(0.0, 0.0), (0.0, 0.0)]
        range_x = list(range_x)
        for idx in range(2):
            range_x[idx] = (
                float(np.min(x_arr[:, idx]) - 1.5 * h[idx]),
                float(np.max(x_arr[:, idx]) + 1.5 * h[idx]),
            )

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    ## Set up grid points and bin the data

    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x_arr, gpoints1, gpoints2)

    ## Compute kernel weights

    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.array([]), np.array([])]
    for idx in range(2):
        L[idx] = min(
            int(np.floor(tau * h[idx] * (M[idx] - 1) / (b[idx] - a[idx]))),
            int(M[idx]) - 1,
        )
        lvecid = np.arange(0, L[idx] + 1)
        facid = (b[idx] - a[idx]) / (h[idx] * (M[idx] - 1))
        z = norm.pdf(lvecid * facid) / h[idx]
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * facid * h[idx]
        kapid[idx] = z / tot
    kapp = np.outer(kapid[0], kapid[1]) / n

    if int(np.min(L)) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    ## Now combine weight and counts using the FFT to obtain estimate

    P = (2 ** np.ceil(np.log(M.astype(np.float64) + L.astype(np.float64)) / np.log(2))).astype(
        np.int64
    )  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0 : (L1 + 1), 0 : (L2 + 1)] = kapp
    if L1:
        rp[(P1 - L1) : P1, 0 : (L2 + 1)] = kapp[L1:0:-1, 0 : (L2 + 1)]
    if L2:
        rp[:, (P2 - L2) : P2] = rp[:, L2:0:-1]
    ## wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    ## zero-padded version of "gcounts"

    rp = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp = np.fft.fft2(sp)
    rp = np.fft.ifft2(rp * sp).real[0:M1, 0:M2]
    ## invert element-wise product of FFT's
    ## and truncate and normalise it

    ## Ensure that rp is non-negative

    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        range_x = np.array([np.min(x), np.max(x)], dtype=np.float64)

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
    # we need P >= 2L+1L, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    return float(np.sum(gcounts * np.fft.ifft(kappam * Gcounts).real[:M]) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # For obtaining preliminary estimates of quantities required for the
    # "direct plug-in" regression bandwidth selector based on blocked
    # q'th degree polynomial fits. This is a faithful, self-contained
    # numpy reimplementation of the R wrapper together with the
    # KernSmooth Fortran subroutine 'blkest' (src/blkest.f).
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    n = len(x_arr)

    # Sort the (x, y) data with respect to the x's.
    datmat = np.column_stack((x_arr, y_arr))
    sort_idx = np.argsort(datmat[:, 0])
    datmat = datmat[sort_idx, :]
    x_sorted = datmat[:, 0]
    y_sorted = datmat[:, 1]

    # Set up arrays for the FORTRAN programme "blkest".
    qq = q + 1

    # It is assumed that the (x, y) data are sorted with respect to x.
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    idiv = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition. Fortran 1-based inclusive
        # indices [ilow, iupp] become the 0-based Python slice
        # [ilow0:iupp0].
        ilow0 = (j - 1) * idiv
        iupp0 = j * idiv
        if j == Nval:
            iupp0 = n
        nj = iupp0 - ilow0

        Xj = x_sorted[ilow0:iupp0]
        Yj = y_sorted[ilow0:iupp0]

        # Obtain a q'th degree fit over the current member of the
        # partition. Set up the "X" matrix: columns are
        # 1, Xj, Xj^2, ..., Xj^q.
        Xmat = np.empty((nj, qq), dtype=np.float64)
        Xmat[:, 0] = 1.0
        for k in range(2, qq + 1):
            Xmat[:, k - 1] = Xj ** (k - 1)

        # Least-squares solution via QR decomposition (equivalent to
        # the LINPACK dqrdc/dqrsl calls used by the original Fortran
        # code, which with job = 00100 compute the least-squares
        # coefficient vector 'coef').
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        fiti = Xmat @ coef

        # ddm/ddddm accumulate the estimated 2nd and 4th derivatives
        # of the fitted polynomial, evaluated at each Xj(i).
        ddm = np.full(nj, 2.0 * coef[2]) if qq > 2 else np.zeros(nj)
        ddddm = np.full(nj, 24.0 * coef[4]) if qq > 4 else np.zeros(nj)

        for k in range(2, qq + 1):
            if k <= (q - 1):
                ddm = ddm + k * (k + 1) * coef[k + 1] * Xj ** (k - 1)
                if k <= (q - 3):
                    ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef[k + 3] * Xj ** (k - 1)

        th22e += float(np.sum(ddm ** 2))
        th24e += float(np.sum(ddm * ddddm))
        RSS = RSS + float(np.sum((Yj - fiti) ** 2))

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # Chooses the number of blocks for the preliminary step of a plug-in
    # rule using Mallows' C_p. This is a faithful, self-contained numpy
    # reimplementation of the R wrapper together with the KernSmooth
    # Fortran subroutine 'cp' (src/cp.f).
    #
    # NOTE on return value: the R function returns
    # 'order(Cpvec)[1L]', i.e. the (1-based) position of the smallest
    # Mallows' C_p value within the vector Cpvals[1..Nmax]. Because the
    # index 'i' into that vector *is* the candidate number of blocks
    # (Nval = i, for i = 1, ..., Nmax), this 1-based position is not an
    # array index to be consumed elsewhere -- it is itself the selected
    # block count 'Nval' that downstream callers (e.g. dpill) use
    # directly as a quantity (e.g. passed straight into blkest as
    # 'Nval'). Therefore this Python implementation intentionally
    # returns the same 1-based block count as R (i.e. argmin(Cpvals) + 1),
    # rather than a 0-based array index, so that downstream logic that
    # treats the result as "the number of blocks" continues to work
    # identically to the R code.
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)

    n = len(X_arr)

    # Sort the (X, Y) data with respect to the X's.
    datmat = np.column_stack((X_arr, Y_arr))
    sort_idx = np.argsort(datmat[:, 0])
    datmat = datmat[sort_idx, :]
    X_sorted = datmat[:, 0]
    Y_sorted = datmat[:, 1]

    # Set up arrays for the FORTRAN programme "cp".
    qq = q + 1

    # It is assumed that the (X, Y) data are sorted with respect to X.
    # Compute vector of RSS values, one per candidate number of blocks
    # Nval = 1, ..., Nmax.
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # For each number of partitions.
        idiv = n // Nval
        RSSNval = 0.0
        for j in range(1, Nval + 1):
            # For each member of the partition. Fortran 1-based
            # inclusive indices [ilow, iupp] become the 0-based Python
            # slice [ilow0:iupp0].
            ilow0 = (j - 1) * idiv
            iupp0 = j * idiv
            if j == Nval:
                iupp0 = n
            nj = iupp0 - ilow0

            Xj = X_sorted[ilow0:iupp0]
            Yj = Y_sorted[ilow0:iupp0]

            # Obtain a q'th degree fit over the current member of the
            # partition. Set up the "X" matrix: columns are
            # 1, Xj, Xj^2, ..., Xj^q.
            Xmat = np.empty((nj, qq), dtype=np.float64)
            Xmat[:, 0] = 1.0
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            # Least-squares solution via QR decomposition (equivalent
            # to the LINPACK dqrdc/dqrsl calls used by the original
            # Fortran code, which with job = 00100 compute the
            # least-squares coefficient vector 'coef').
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fiti = Xmat @ coef
            RSSNval += float(np.sum((Yj - fiti) ** 2))

        RSS[Nval - 1] = RSSNval

    # Now compute array of Mallow's C_p values.
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    # order(Cpvec)[1L] in R returns the 1-based position of the first
    # (smallest-index) occurrence of the minimum C_p value, which is
    # equivalent to (0-based argmin) + 1.
    best_index0 = int(np.argmin(Cpvals))
    Nval_selected = best_index0 + 1

    return Nval_selected


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    ## Rename variables

    n = len(x_arr)
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    ## Compute scale estimate

    # match.arg(scalest, c("minim", "stdev", "iqr"))
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        _matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(_matches) == 1:
            scalest = _matches[0]
        else:
            raise ValueError(
                f"'scalest' should be one of {', '.join(repr(c) for c in _scalest_choices)}"
            )

    if scalest == "stdev":
        scalest_val = float(np.std(x_arr, ddof=1))
    elif scalest == "iqr":
        scalest_val = float(np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349
    else:  # scalest == "minim"
        scalest_val = min(
            float(np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349,
            float(np.std(x_arr, ddof=1)),
        )

    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = float(np.mean(x_arr))
    sx = (x_arr - x_mean) / scalest_val
    sa = (a - x_mean) / scalest_val
    sb = (b - x_mean) / scalest_val

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps

    if level == 0:
        hpi = (24.0 * math.sqrt(math.pi) / n) ** (1.0 / 3.0)
    elif level == 1:
        alpha = (2.0 / (3.0 * n)) ** (1.0 / 5.0) * math.sqrt(2.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 2:
        alpha = ((2.0 / (5.0 * n)) ** (1.0 / 7.0)) * math.sqrt(2.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2.0 / math.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 3:
        alpha = ((2.0 / (7.0 * n)) ** (1.0 / 9.0)) * math.sqrt(2.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2.0 / math.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 4:
        alpha = ((2.0 / (9.0 * n)) ** (1.0 / 11.0)) * math.sqrt(2.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2.0 / math.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    else:  # level == 5
        alpha = ((2.0 / (11.0 * n)) ** (1.0 / 13.0)) * math.sqrt(2.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * math.sqrt(2.0 / math.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (math.sqrt(2.0 / math.pi) / (psi4hat * n)) ** (1.0 / 5.0)  # bandwidth for psi_2
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)

    return scalest_val * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    # match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        _matches = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(_matches) == 1:
            kernel = _matches[0]
        else:
            raise ValueError(
                f"'kernel' should be one of {', '.join(repr(c) for c in _kernel_choices)}"
            )

    ## Set kernel constants

    if canonical:
        del0 = 1.0
    else:
        _del0_map = {
            "normal": 1.0 / ((4.0 * math.pi) ** (1.0 / 10.0)),
            "box": (9.0 / 2.0) ** (1.0 / 5.0),
            "epanech": 15.0 ** (1.0 / 5.0),
            "biweight": 35.0 ** (1.0 / 5.0),
            "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
        }
        del0 = _del0_map[kernel]

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    ## Rename variables

    n = len(x_arr)
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data

    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    ## Compute scale estimate

    # match.arg(scalest, c("minim", "stdev", "iqr"))
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        _matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(_matches) == 1:
            scalest = _matches[0]
        else:
            raise ValueError(
                f"'scalest' should be one of {', '.join(repr(c) for c in _scalest_choices)}"
            )

    if scalest == "stdev":
        scalest_val = float(np.std(x_arr, ddof=1))
    elif scalest == "iqr":
        scalest_val = float(np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349
    else:  # scalest == "minim"
        scalest_val = min(
            float(np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349,
            float(np.std(x_arr, ddof=1)),
        )

    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = float(np.mean(x_arr))
    sx = (x_arr - x_mean) / scalest_val
    sa = (a - x_mean) / scalest_val
    sb = (b - x_mean) / scalest_val

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps:

    if level == 0:
        psi4hat = 3.0 / (8.0 * math.sqrt(math.pi))
    elif level == 1:
        alpha = (2.0 * (math.sqrt(2.0)) ** 7 / (5.0 * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2.0 * (math.sqrt(2.0)) ** 9 / (7.0 * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2.0 * (math.sqrt(2.0)) ** 11 / (9.0 * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2.0 * (math.sqrt(2.0)) ** 13 / (11.0 * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * math.sqrt(2.0 / math.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:  # level == 5
        alpha = (2.0 * (math.sqrt(2.0)) ** 15 / (13.0 * n)) ** (1.0 / 15.0)  # bandwidth for psi_12
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = (945.0 * math.sqrt(2.0 / math.pi) / (psi12hat * n)) ** (1.0 / 13.0)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * math.sqrt(2.0 / math.pi) / (psi10hat * n)) ** (1.0 / 11.0)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * math.sqrt(2.0 / math.pi) / (psi8hat * n)) ** (1.0 / 9.0)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * math.sqrt(2.0 / math.pi) / (psi6hat * n)) ** (1.0 / 7.0)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return scalest_val * del0 * (1.0 / (psi4hat * n)) ** (1.0 / 5.0)


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, truncate: bool = True) -> float:
    # Computes a direct plug-in selector of the bandwidth for local
    # linear regression as described in the 1996 J. Amer. Statist.
    # Assoc. paper by Ruppert, Sheather and Wand.
    #
    # Trim the 100(trim)% of the data from each end (in the x-direction).
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    sort_idx = np.argsort(x_arr)
    x_sorted = x_arr[sort_idx]
    y_sorted = y_arr[sort_idx]

    n_full = len(x_sorted)
    # R: indlow <- floor(trim*length(x)) + 1 ; indupp <- length(x) - floor(trim*length(x))
    # R: x <- x[indlow:indupp] (1-based, inclusive). In 0-based Python this is
    # x_sorted[indlow0:indupp0] with indlow0 = indlow - 1 = floor(trim*n_full)
    # and indupp0 (exclusive stop) = indupp = n_full - floor(trim*n_full).
    indlow0 = int(math.floor(trim * n_full))
    indupp0 = n_full - int(math.floor(trim * n_full))

    x_t = x_sorted[indlow0:indupp0]
    y_t = y_sorted[indlow0:indupp0]

    # NOTE: In the original R code, 'range.x' defaults to 'range(x)', but
    # because R default arguments are lazily evaluated *in the function's
    # own execution environment*, and 'x' is reassigned above (sorted and
    # trimmed) before 'range.x' is first used, the default actually
    # evaluates to the range of the *trimmed* x, not the original input.
    # We replicate that behaviour explicitly here.
    if range_x is None:
        range_x = (float(np.min(x_t)), float(np.max(x_t)))

    ## Rename common parameters
    n = len(x_t)
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])

    ## Bin the data
    gpoints = np.linspace(a, b, M)
    out = rlbin(x_t, y_t, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    ## Choose the value of N using Mallow's C_p
    Nmax = max(min(int(math.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x_t, y_t, Nmax, 4)

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.
    out = blkest(x_t, y_t, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    ## Estimate theta_22 using a local cubic fit
    ## with a "rule-of-thumb" bandwidth: "gamseh"
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1.0 / 7.0)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1.0 / 7.0)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=range_x, binned=True)["y"]

    # R: llow <- floor(proptrun*M) + 1 ; lupp <- M - floor(proptrun*M)
    # mddest[llow:lupp] (1-based inclusive) -> mddest[llow0:lupp0] (0-based)
    # with llow0 = llow - 1 = floor(proptrun*M) and lupp0 (exclusive) = lupp.
    llow0 = int(math.floor(proptrun * M))
    lupp0 = M - int(math.floor(proptrun * M))
    th22kn = float(np.sum((mddest[llow0:lupp0] ** 2) * xcounts[llow0:lupp0]) / n)

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    C3K = 0.5 + 2 * math.sqrt(2) - (4.0 / 3.0) * math.sqrt(3)
    C3K = (4 * C3K / math.sqrt(2 * math.pi)) ** (1.0 / 9.0)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1.0 / 9.0))

    ## Now compute a local linear kernel estimate of
    ## the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                    range_x=range_x, binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh,
                range_x=range_x, binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh,
                    range_x=range_x, binned=True)["y"]
    sigsqn = float(np.sum(y_t ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts))
    sigsqd = float(n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts))
    sigsqkn = sigsqn / sigsqd

    ## Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1.0 / 5.0))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    # For application of linear binning to a univariate data set.
    # This is a faithful, self-contained numpy reimplementation of the
    # KernSmooth Fortran subroutine 'linbin' (src/linbin.f).
    X_arr = np.asarray(X, dtype=np.float64)
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)

    n = len(X_arr)
    M = len(gpoints_arr)
    trun = 1 if truncate else 0
    a = float(gpoints_arr[0])
    b = float(gpoints_arr[M - 1])

    # Initialize grid counts to zero
    gcnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)

    for i in range(n):
        lxi = ((X_arr[i] - a) / delta) + 1.0

        # Find integer part of 'lxi' (Fortran INT truncates toward zero,
        # matching Python's int() truncation behaviour on floats).
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
    # Creates the grid counts from a bivariate data set X
    # over an equally-spaced set of grid points contained in
    # 'gpoints1' and 'gpoints2' using the linear binning strategy.
    # This is a faithful, self-contained numpy reimplementation of the
    # KernSmooth Fortran subroutine 'lbtwod' (src/linbin2D.f). Observations
    # outside the mesh are ignored.
    X_arr = np.asarray(X, dtype=np.float64)
    gpoints1_arr = np.asarray(gpoints1, dtype=np.float64)
    gpoints2_arr = np.asarray(gpoints2, dtype=np.float64)

    n = X_arr.shape[0]
    X1 = X_arr[:, 0]
    X2 = X_arr[:, 1]

    M1 = len(gpoints1_arr)
    M2 = len(gpoints2_arr)
    a1 = float(gpoints1_arr[0])
    a2 = float(gpoints2_arr[0])
    b1 = float(gpoints1_arr[M1 - 1])
    b2 = float(gpoints2_arr[M2 - 1])

    # Initialize grid counts to zero. gcnts[i, j] corresponds to the grid
    # point (gpoints1[i], gpoints2[j]).
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    for i in range(n):
        lxi1 = ((X1[i] - a1) / delta1) + 1.0
        lxi2 = ((X2[i] - a2) / delta2) + 1.0

        # Find integer part of 'lxi1' and 'lxi2' (Fortran INT truncates
        # toward zero, matching Python's int() truncation on floats).
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


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For computing a binned local polynomial regression estimator of a
    # univariate regression function or its derivative. The data are
    # discretised on an equally spaced grid. The bandwidths are
    # discretised on a logarithmically spaced grid.

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
            range_x = np.array([np.min(x) - extra, np.max(x) + extra], dtype=np.float64)
        else:
            range_x = np.array([np.min(x), np.max(x)], dtype=np.float64)

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Decide whether a density estimate or regression estimate is required.
    if y is None:
        # obtain density estimate
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
        bw_sorted = np.sort(bandwidth_arr)
        hlow = bw_sorted[0]
        hupp = bw_sorted[M - 1]
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
                    ((np.log(bandwidth_arr) - np.log(bw_sorted[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
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

    # --- Reimplementation of the KernSmooth Fortran subroutine 'locpol' ---
    # (src/locpoly.f). Builds a table of kernel weights ("fkap") shared
    # across all bandwidth-discretisation bins, accumulates local moment
    # matrices ("ss"/"tt") per grid point, and solves the resulting
    # weighted normal equations at each grid point (equivalent to the
    # LINPACK dgefa/dgesl LU-decomposition solve used by the original
    # Fortran code).
    def _locpol(
        xcnts: np.ndarray[Any, np.dtype[np.float64]],
        ycnts: np.ndarray[Any, np.dtype[np.float64]],
        idrv: int,
        delta_: float,
        hdisc_: np.ndarray[Any, np.dtype[np.float64]],
        Lvec_: np.ndarray[Any, np.dtype[np.int64]],
        indic_: np.ndarray[Any, np.dtype[np.int64]],
        M_: int,
        Q_: int,
        pp_: int,
        ppp_: int,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        # Build the kernel-weight table "fkap" and the per-bandwidth
        # midpoint offsets "midpts". Both use 1-based positions
        # internally, exactly mirroring the Fortran source, but are
        # stored in 0-based numpy arrays (index - 1).
        dimfkap = 2 * int(np.sum(Lvec_)) + Q_
        fkap = np.zeros(dimfkap, dtype=np.float64)
        midpts = np.zeros(Q_, dtype=np.int64)

        mid = int(Lvec_[0]) + 1
        for i0 in range(Q_ - 1):
            midpts[i0] = mid
            fkap[mid - 1] = 1.0
            for j in range(1, int(Lvec_[i0]) + 1):
                val = math.exp(-((delta_ * j / hdisc_[i0]) ** 2) / 2)
                fkap[mid + j - 1] = val
                fkap[mid - j - 1] = val
            mid = mid + int(Lvec_[i0]) + int(Lvec_[i0 + 1]) + 1
        midpts[Q_ - 1] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec_[Q_ - 1]) + 1):
            val = math.exp(-((delta_ * j / hdisc_[Q_ - 1]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val

        # Combine kernel weights and grid counts.
        ss = np.zeros((M_, ppp_), dtype=np.float64)
        tt = np.zeros((M_, pp_), dtype=np.float64)

        for k0 in range(M_):
            if xcnts[k0] != 0:
                for i0 in range(Q_):
                    lo = max(0, k0 - int(Lvec_[i0]))
                    hi = min(M_ - 1, k0 + int(Lvec_[i0]))
                    for j0 in range(lo, hi + 1):
                        if indic_[j0] == i0 + 1:
                            fkap_val = fkap[(k0 - j0) + midpts[i0] - 1]
                            fac = 1.0
                            ss[j0, 0] += xcnts[k0] * fkap_val
                            tt[j0, 0] += ycnts[k0] * fkap_val
                            for ii in range(2, ppp_ + 1):
                                fac = fac * delta_ * (k0 - j0)
                                ss[j0, ii - 1] += xcnts[k0] * fkap_val * fac
                                if ii <= pp_:
                                    tt[j0, ii - 1] += ycnts[k0] * fkap_val * fac

        # Solve the weighted local normal equations at each grid point,
        # equivalent to the LINPACK dgefa (LU factorise) + dgesl (solve)
        # pair applied to Smat * Tvec = Tvec in the original Fortran.
        cvest = np.zeros(M_, dtype=np.float64)
        for k0 in range(M_):
            Smat = np.zeros((pp_, pp_), dtype=np.float64)
            Tvec = np.zeros(pp_, dtype=np.float64)
            for i in range(1, pp_ + 1):
                for j in range(1, pp_ + 1):
                    indss = i + j - 1
                    Smat[i - 1, j - 1] = ss[k0, indss - 1]
                Tvec[i - 1] = tt[k0, i - 1]
            try:
                sol = np.linalg.solve(Smat, Tvec)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(Smat, Tvec, rcond=None)[0]
            cvest[k0] = sol[idrv]

        return cvest

    cvest = _locpol(
        np.asarray(xcounts, dtype=np.float64),
        np.asarray(ycounts, dtype=np.float64),
        drv,
        delta,
        hdisc,
        Lvec,
        indic,
        M,
        Q,
        pp,
        ppp,
    )

    curvest = math.gamma(drv + 1) * cvest

    return {"x": gpoints, "y": curvest}


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For application of linear binning to a regression data set.
    # This is a faithful, self-contained numpy reimplementation of the
    # KernSmooth Fortran subroutine 'rlbin' (src/rlbin.f).
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)

    n = len(X_arr)
    M = len(gpoints_arr)
    trun = 1 if truncate else 0
    a = float(gpoints_arr[0])
    b = float(gpoints_arr[M - 1])

    # Initialize grid counts to zero
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)

    for i in range(n):
        lxi = ((X_arr[i] - a) / delta) + 1.0

        # Find integer part of 'lxi' (Fortran INT truncates toward zero,
        # matching Python's int() truncation behaviour on floats).
        li = int(lxi)

        rem = lxi - li

        # Correction for right endpoint (not included if li == M)
        if X_arr[i] == b:
            li = M - 1
            rem = 1.0

        if li >= 1 and li < M:
            xcnts[li - 1] = xcnts[li - 1] + (1 - rem)
            xcnts[li] = xcnts[li] + rem
            ycnts[li - 1] = ycnts[li - 1] + (1 - rem) * Y_arr[i]
            ycnts[li] = ycnts[li] + rem * Y_arr[i]

        if li < 1 and trun == 0:
            xcnts[0] = xcnts[0] + 1
            ycnts[0] = ycnts[0] + Y_arr[i]

        if li >= M and trun == 0:
            xcnts[M - 1] = xcnts[M - 1] + 1
            ycnts[M - 1] = ycnts[M - 1] + Y_arr[i]

    return {"xcounts": xcnts, "ycounts": ycnts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For computing the binned diagonal entries of a smoother
    # matrix for local polynomial kernel regression.
    #
    # Note: as in the original R function, 'drv' and 'kernel' are accepted
    # for interface compatibility with 'locpoly' but are not used in the
    # computation below.
    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")

    if range_x is None and not binned:
        range_x = np.array([np.min(x), np.max(x)], dtype=np.float64)

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
        bw_sorted = np.sort(bandwidth_arr)
        hlow = bw_sorted[0]
        hupp = bw_sorted[M - 1]
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
                    ((np.log(bandwidth_arr) - np.log(bw_sorted[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    # --- Reimplementation of the KernSmooth Fortran subroutine 'sdiag' ---
    # (src/sdiag.f). Builds the kernel-weight table ("fkap") shared across
    # bandwidth-discretisation bins, accumulates local moment matrices
    # ("ss") per grid point (identical accumulation logic to 'locpoly',
    # but without the response-dependent "tt" moments since 'sdiag' does
    # not use y-values), then computes the diagonal entry of the smoother
    # (hat) matrix at each grid point as
    # Sdg[k] = e_1^T (X_local^T W X_local)^{-1} e_1, i.e. the (1,1) entry
    # of the inverse of the local moment matrix. This mirrors the
    # LINPACK dgefa (LU factorise) + dgedi (job=01, inverse only) pair
    # used by the original Fortran code.
    xcounts_arr = np.asarray(xcounts, dtype=np.float64)

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)

    mid = int(Lvec[0]) + 1
    for i0 in range(Q - 1):
        midpts[i0] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec[i0]) + 1):
            val = math.exp(-((delta * j / hdisc[i0]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i0]) + int(Lvec[i0 + 1]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        val = math.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    # Combine kernel weights and grid counts.
    ss = np.zeros((M, ppp), dtype=np.float64)

    for k0 in range(M):
        if xcounts_arr[k0] != 0:
            for i0 in range(Q):
                lo = max(0, k0 - int(Lvec[i0]))
                hi = min(M - 1, k0 + int(Lvec[i0]))
                for j0 in range(lo, hi + 1):
                    if indic[j0] == i0 + 1:
                        fkap_val = fkap[(k0 - j0) + midpts[i0] - 1]
                        fac = 1.0
                        ss[j0, 0] += xcounts_arr[k0] * fkap_val
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k0 - j0)
                            ss[j0, ii - 1] += xcounts_arr[k0] * fkap_val * fac

    # Solve for the diagonal entry of the smoother matrix at each grid
    # point: Sdg[k] = (inverse of the local moment matrix)[0, 0].
    Sdg = np.zeros(M, dtype=np.float64)
    for k0 in range(M):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k0, indss - 1]
        try:
            Sinv = np.linalg.inv(Smat)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(Smat)
        Sdg[k0] = Sinv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: np.ndarray[Any, np.dtype[np.float64]] | tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For computing the binned diagonal entries of SS^T
    # where S is a smoother matrix for local polynomial
    # kernel regression.
    #
    # Note: as in the original R function, 'drv' and 'kernel' are accepted
    # for interface compatibility with 'locpoly' but are not used in the
    # computation below.
    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")

    if range_x is None and not binned:
        range_x = np.array([np.min(x), np.max(x)], dtype=np.float64)

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
        bw_sorted = np.sort(bandwidth_arr)
        hlow = bw_sorted[0]
        hupp = bw_sorted[M - 1]
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
                    ((np.log(bandwidth_arr) - np.log(bw_sorted[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    # --- Reimplementation of the KernSmooth Fortran subroutine 'sstdg' ---
    # (src/sstdiag.f). Builds the kernel-weight table ("fkap") shared
    # across bandwidth-discretisation bins (identical to 'sdiag'/'locpoly'),
    # then accumulates two local moment arrays per grid point: "ss", the
    # ordinary weighted moments (as in 'sdiag'), and "uu", the same moments
    # but weighted by the *square* of the kernel weight (needed because we
    # are forming diag(S S^T) rather than diag(S)). For each grid point k
    # the local moment matrix Smat = [ss] and the squared-weight moment
    # matrix Umat = [uu] are assembled, Smat is inverted (mirroring the
    # LINPACK dgefa/dgedi pair used by the original Fortran code), and the
    # diagonal entry of S S^T at that grid point is obtained as
    # SSTd[k] = e_1^T Sinv Umat Sinv^T e_1, i.e. the (1,1) entry of
    # Sinv @ Umat @ Sinv.T.
    xcounts_arr = np.asarray(xcounts, dtype=np.float64)

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)

    mid = int(Lvec[0]) + 1
    for i0 in range(Q - 1):
        midpts[i0] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec[i0]) + 1):
            val = math.exp(-((delta * j / hdisc[i0]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i0]) + int(Lvec[i0 + 1]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        val = math.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    # Combine kernel weights and grid counts.
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)

    for k0 in range(M):
        if xcounts_arr[k0] != 0:
            for i0 in range(Q):
                lo = max(0, k0 - int(Lvec[i0]))
                hi = min(M - 1, k0 + int(Lvec[i0]))
                for j0 in range(lo, hi + 1):
                    if indic[j0] == i0 + 1:
                        fkap_val = fkap[(k0 - j0) + midpts[i0] - 1]
                        fac = 1.0
                        ss[j0, 0] += xcounts_arr[k0] * fkap_val
                        uu[j0, 0] += xcounts_arr[k0] * (fkap_val ** 2)
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k0 - j0)
                            ss[j0, ii - 1] += xcounts_arr[k0] * fkap_val * fac
                            uu[j0, ii - 1] += (
                                xcounts_arr[k0] * (fkap_val ** 2) * fac
                            )

    # Solve for the diagonal entry of S S^T at each grid point:
    # SSTd[k] = (Sinv @ Umat @ Sinv.T)[0, 0].
    SSTd = np.zeros(M, dtype=np.float64)
    for k0 in range(M):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Umat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k0, indss - 1]
                Umat[i - 1, j - 1] = uu[k0, indss - 1]
        try:
            Sinv = np.linalg.inv(Smat)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(Smat)
        val = 0.0
        for i in range(pp):
            for j in range(pp):
                val += Sinv[0, i] * Umat[i, j] * Sinv[j, 0]
        SSTd[k0] = val

    return {"x": gpoints, "y": SSTd}


def on_attach(libname: str, pkgname: str) -> None:
    # Mirrors R's .onAttach(libname, pkgname) hook, which is invoked
    # automatically by R when the package is loaded via library()/require().
    # Python has no direct equivalent of this lifecycle hook, so it is
    # translated as a plain function with the same signature that emits
    # the package startup message to stderr, matching
    # packageStartupMessage()'s behavior of writing to stderr.
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def on_unload(libpath: str) -> None:
    # R's `.onUnload` hook calls `library.dynam.unload("KernSmooth", libpath)`
    # to unload the package's compiled shared library (.so/.dll) when the
    # R package is detached.
    #
    # Python has no direct equivalent: CPython's import system does not
    # support safely unloading compiled C/Fortran extension modules (e.g.
    # `_KernSmooth`). The C runtime may retain global state, and calling
    # `dlclose()` on a module that Python objects still reference can crash
    # the interpreter. The Python import machinery manages the extension's
    # shared-library lifetime automatically for the lifetime of the process,
    # so there is nothing to do here.
    #
    # This function is kept only to preserve the `.onUnload(libpath)` entry
    # point/signature from the original R package; it intentionally performs
    # no action.
    pass


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
