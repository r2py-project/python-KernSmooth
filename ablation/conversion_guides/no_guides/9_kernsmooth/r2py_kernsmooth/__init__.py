import math
import sys
import warnings
from typing import Any

import numpy as np
from scipy.stats import beta as _beta

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
        raise ValueError("'kernel' must be one of " + str(valid_kernels))

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
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
    L = min(int(np.floor(tau / delta)), M)
    if L == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    lvec = np.arange(0, L + 1)
    if kernel == "normal":
        z = lvec * delta
        kappa = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:  # triweight
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P = int(2 ** (np.ceil(np.log(M + L + 1) / np.log(2))))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)

    conv = np.fft.ifft(kappa_fft * gcounts_fft) * P
    y = (np.real(conv) / P)[0:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] = (51, 51), range_x: tuple[tuple[float, float], tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction if only a single bandwidth is given.
    if h.shape[0] == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = (
            (float(np.min(x[:, 0])) - 1.5 * h[0], float(np.max(x[:, 0])) + 1.5 * h[0]),
            (float(np.min(x[:, 1])) - 1.5 * h[1], float(np.max(x[:, 1])) + 1.5 * h[1]),
        )

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)]
    for idx in range(2):
        L[idx] = min(int(np.floor(tau * h[idx] * (M[idx] - 1) / (b[idx] - a[idx]))), int(M[idx]) - 1)
        lvecid = np.arange(0, L[idx] + 1)
        facid = (b[idx] - a[idx]) / (h[idx] * (M[idx] - 1))
        z = np.exp(-0.5 * (lvecid * facid) ** 2) / np.sqrt(2.0 * np.pi) / h[idx]
        # c(z, rev(z[-1L])): concatenate z (lags 0..L) with the reverse of
        # z with its first element dropped, forming the full symmetric
        # (2L+1)-point kernel used to compute the normalising constant.
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * facid * h[idx]
        kapid[idx] = z / tot

    kapp = np.outer(kapid[0], kapid[1]) / n

    if int(np.min(L)) == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    # Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log2(M + L))).astype(np.int64)  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[0:(L1 + 1), 0:(L2 + 1)] = kapp
    if L1:
        # kapp[(L1+1):2, ...] (R, 1-based, descending) == kapp rows
        # L1, L1-1, ..., 1 (0-based) == kapp[L1:0:-1, ...] in NumPy.
        rp[(P1 - L1):P1, 0:(L2 + 1)] = kapp[L1:0:-1, 0:(L2 + 1)]
    if L2:
        # Mirror the columns of the already partially-filled rp (must run
        # after the row wrap-around above, since it reads from rp itself).
        rp[:, (P2 - L2):P2] = rp[:, L2:0:-1]
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    # R's fft(z, inverse = TRUE) is the unnormalised inverse transform
    # (sum-based, i.e. numpy's ifft2 result multiplied by P1*P2); the R
    # code then explicitly divides by (P1*P2), so the two operations
    # cancel out and this is exactly numpy.fft.ifft2's own (mean-based)
    # normalisation -- no extra scaling is required here.
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    # invert element-wise product of FFT's and truncate and normalise it

    # Ensure that rp is non-negative
    rp = np.where(rp > 0, rp, 0.0)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    if bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        x = np.asarray(x, dtype=np.float64)
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
        M = gcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width
    n = np.sum(gcounts)
    delta = (b - a) / (M - 1)

    # Obtain kernel weights
    tau = 4 + drv
    L = min(int(np.floor(tau * h / delta)), M)

    if L == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = np.exp(-0.5 * arg ** 2) / np.sqrt(2.0 * np.pi)
    kappam = kappam / (h ** (drv + 1))
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
    kappam_fft = np.fft.fft(kappam)
    Gcounts_fft = np.fft.fft(Gcounts)

    conv = np.fft.ifft(kappam_fft * Gcounts_fft) * P
    result = np.sum(gcounts * (np.real(conv) / P)[0:M]) / (n ** 2)

    return float(result)


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # Faithful translation of KernSmooth's blkest R wrapper together with
    # its compiled Fortran routine F_blkest (blkest.f). The Fortran source
    # partitions the (x, y) data -- sorted by x -- into Nval contiguous
    # blocks, fits a q'th degree polynomial by least squares (via
    # dqrdc/dqrsl, a QR decomposition) within each block, and accumulates
    # residual sums of squares plus derivative-based sums used to form
    # sigsqe, th22e and th24e.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]

    # R wrapper: datmat <- datmat[sort.list(datmat[, 1]), ]
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    qq = q + 1

    # Fortran: idiv = n/Nval  (integer division)
    idiv = n // Nval

    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    # Fortran: do j = 1, Nval  (1-based block index)
    for j in range(1, Nval + 1):
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        # Convert the 1-based inclusive Fortran range [ilow, iupp] into a
        # 0-based Python slice covering the same nj = iupp - ilow + 1
        # points: Xj(1:nj) = X(ilow:iupp), Yj(1:nj) = Y(ilow:iupp).
        xb = x[ilow - 1:iupp]
        yb = y[ilow - 1:iupp]
        nj = xb.shape[0]

        # Set up the design matrix: Xmat(i, k) = Xj(i)**(k-1) for
        # k = 1..qq, i.e. ascending powers 1, Xj, Xj^2, ..., Xj^q.
        Xmat_block = np.vander(xb, N=qq, increasing=True)

        # dqrdc/dqrsl perform a QR-decomposition least-squares fit of Yj
        # on Xmat; np.linalg.lstsq reproduces the same (up to numerical
        # precision) ordinary least-squares coefficients. coef[k] is the
        # coefficient of Xj**k for k = 0..q (coef[0] is the intercept).
        coef, _, _, _ = np.linalg.lstsq(Xmat_block, yb, rcond=None)

        # fiti = coef(1) + sum_{k=2}^{qq} coef(k)*Xj(i)**(k-1)
        #      = sum_{k=0}^{q} coef[k]*Xj(i)**k  (0-based)
        fiti = Xmat_block @ coef
        RSS += float(np.sum((yb - fiti) ** 2))

        # ddm is the second derivative of the fitted polynomial at Xj(i):
        # Fortran initializes ddm = 2*coef(3) (the k=2, i.e. 0-based
        # coef[2], term of f''), then the k = 2..qq loop (restricted to
        # k <= q-1) adds the remaining k*(k+1)*coef(k+2)*Xj(i)**(k-1)
        # terms, which after re-indexing (m = k+1) are exactly the
        # m*(m-1)*coef[m]*Xj(i)**(m-2) terms for m = 3..q. Altogether:
        # ddm = sum_{m=2}^{q} m*(m-1)*coef[m]*Xj(i)**(m-2).
        ddm = np.zeros(nj, dtype=np.float64)
        for k in range(2, q + 1):
            ddm += k * (k - 1) * coef[k] * xb ** (k - 2)

        # ddddm is the fourth derivative of the fitted polynomial at
        # Xj(i), analogously reconstructed from Fortran's initial
        # ddddm = 24*coef(5) term (0-based coef[4]) plus the k <= q-3
        # branch of the loop (re-indexed m = k+3):
        # ddddm = sum_{m=4}^{q} m*(m-1)*(m-2)*(m-3)*coef[m]*Xj(i)**(m-4).
        ddddm = np.zeros(nj, dtype=np.float64)
        for k in range(4, q + 1):
            ddddm += k * (k - 1) * (k - 2) * (k - 3) * coef[k] * xb ** (k - 4)

        th22e += float(np.sum(ddm ** 2))
        th24e += float(np.sum(ddm * ddddm))

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # Faithful translation of KernSmooth's cpblock R wrapper together with
    # its compiled Fortran routine F_cp (cp.f). For each candidate number
    # of blocks Nval = 1..Nmax, the (X, Y) data -- sorted by X -- are
    # partitioned into Nval contiguous blocks (same partitioning scheme
    # as blkest), a q'th degree polynomial is fit by least squares
    # (via dqrdc/dqrsl, a QR decomposition) within each block, and the
    # residual sums of squares are accumulated over all blocks to form
    # RSS[Nval]. Mallow's C_p values are then computed from the RSS
    # vector, and the number of blocks minimizing C_p is returned.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]

    # R wrapper: datmat <- datmat[sort.list(datmat[, 1]), ]
    order = np.argsort(X, kind="stable")
    X = X[order]
    Y = Y[order]

    qq = q + 1

    RSS = np.zeros(Nmax, dtype=np.float64)

    # Fortran: do Nval = 1, Nmax
    for Nval in range(1, Nmax + 1):
        # Fortran: idiv = n/Nval  (integer division)
        idiv = n // Nval
        RSS_Nval = 0.0

        # Fortran: do j = 1, Nval  (1-based block index)
        for j in range(1, Nval + 1):
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n
            # Convert the 1-based inclusive Fortran range [ilow, iupp]
            # into a 0-based Python slice covering the same
            # nj = iupp - ilow + 1 points: Xj(1:nj) = X(ilow:iupp),
            # Yj(1:nj) = Y(ilow:iupp).
            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]

            # Set up the design matrix: Xmat(i, k) = Xj(i)**(k-1) for
            # k = 1..qq, i.e. ascending powers 1, Xj, Xj^2, ..., Xj^q.
            Xmat = np.vander(Xj, N=qq, increasing=True)

            # dqrdc/dqrsl perform a QR-decomposition least-squares fit
            # of Yj on Xmat; np.linalg.lstsq reproduces the same (up
            # to numerical precision) ordinary least-squares
            # coefficients. coef[k] is the coefficient of Xj**k for
            # k = 0..q (coef[0] is the intercept).
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            # fiti = coef(1) + sum_{k=2}^{qq} coef(k)*Xj(i)**(k-1)
            #      = sum_{k=0}^{q} coef[k]*Xj(i)**k  (0-based)
            fiti = Xmat @ coef
            RSS_Nval += float(np.sum((Yj - fiti) ** 2))

        RSS[Nval - 1] = RSS_Nval

    # Now compute array of Mallow's C_p values.
    # Fortran: Cpvals(i) = ((n-qq*Nmax)*RSS(i)/RSS(Nmax)) + 2*qq*i - n
    # for i = 1..Nmax (1-based); RSS(Nmax) is the last element of RSS.
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    RSS_last = RSS[Nmax - 1]
    for idx in range(Nmax):
        i = idx + 1
        Cpvals[idx] = (n - qq * Nmax) * RSS[idx] / RSS_last + 2 * qq * i - n

    # R wrapper: order(Cpvec)[1L], i.e. the 1-based index (number of
    # blocks) achieving the minimum Cp value.
    return int(np.argmin(Cpvals)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    M = gridsize

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    # NOTE: the R source computes an initial `gpoints`/`gcounts` pair here
    # (via seq(a, b, length.out = M) and linbin(x, gpoints, truncate)), but
    # both variables are immediately overwritten below once the data has
    # been standardized. That computation's results are never used, so it
    # is dead code and has been omitted here with no behavioral difference.

    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'scalest' must be one of " + str(valid_scalest))

    var_x = np.var(x, ddof=1)
    q75 = np.quantile(x, 0.75)
    q25 = np.quantile(x, 0.25)

    if scalest == "stdev":
        scalest_value = np.sqrt(var_x)
    elif scalest == "iqr":
        scalest_value = (q75 - q25) / 1.349
    else:  # "minim"
        scalest_value = min((q75 - q25) / 1.349, np.sqrt(var_x))

    if scalest_value == 0:
        raise ValueError("scale estimate is zero for input data")

    mean_x = np.mean(x)
    sx = (x - mean_x) / scalest_value
    sa = (a - mean_x) / scalest_value
    sb = (b - mean_x) / scalest_value

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    if level == 0:
        hpi = (24 * np.sqrt(np.pi) / n) ** (1.0 / 3.0)
    elif level == 1:
        alpha = (2.0 / (3.0 * n)) ** (1.0 / 5.0) * np.sqrt(2.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 2:
        alpha = ((2.0 / (5.0 * n)) ** (1.0 / 7.0)) * np.sqrt(2.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2.0 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 3:
        alpha = ((2.0 / (7.0 * n)) ** (1.0 / 9.0)) * np.sqrt(2.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2.0 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 4:
        alpha = ((2.0 / (9.0 * n)) ** (1.0 / 11.0)) * np.sqrt(2.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2.0 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)
    else:  # level == 5
        alpha = ((2.0 / (11.0 * n)) ** (1.0 / 13.0)) * np.sqrt(2.0)
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2.0 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6.0 / (-psi2hat * n)) ** (1.0 / 3.0)

    return float(scalest_value * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'kernel' must be one of " + str(valid_kernels))

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

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    M = gridsize

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    # NOTE: the R source computes an initial `gpoints`/`gcounts` pair here
    # (via seq(a, b, length.out = M) and linbin(x, gpoints, truncate)), but
    # both variables are immediately overwritten below once the data has
    # been standardized. That computation's results are never used, so it
    # is dead code and has been omitted here with no behavioral difference.

    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'scalest' must be one of " + str(valid_scalest))

    var_x = np.var(x, ddof=1)
    q75 = np.quantile(x, 0.75)
    q25 = np.quantile(x, 0.25)

    if scalest == "stdev":
        scalest_value = np.sqrt(var_x)
    elif scalest == "iqr":
        scalest_value = (q75 - q25) / 1.349
    else:  # "minim"
        scalest_value = min((q75 - q25) / 1.349, np.sqrt(var_x))

    if scalest_value == 0:
        raise ValueError("scale estimate is zero for input data")

    mean_x = np.mean(x)
    sx = (x - mean_x) / scalest_value
    sa = (a - mean_x) / scalest_value
    sb = (b - mean_x) / scalest_value

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    if level == 0:
        psi4hat = 3.0 / (8.0 * np.sqrt(np.pi))
    elif level == 1:
        alpha = (2.0 * (np.sqrt(2.0)) ** 7 / (5.0 * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2.0 * (np.sqrt(2.0)) ** 9 / (7.0 * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2.0 * (np.sqrt(2.0)) ** 11 / (9.0 * n)) ** (1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2.0 * (np.sqrt(2.0)) ** 13 / (11.0 * n)) ** (1.0 / 13.0)
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:  # level == 5
        alpha = (2.0 * (np.sqrt(2.0)) ** 15 / (13.0 * n)) ** (1.0 / 15.0)
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = (945.0 * np.sqrt(2.0 / np.pi) / (psi12hat * n)) ** (1.0 / 13.0)
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return float(scalest_value * del0 * (1.0 / (psi4hat * n)) ** (1.0 / 5.0))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Trim the 100(trim)% of the data from each end (in the x-direction).
    # R: xy <- cbind(x, y); xy <- xy[sort.list(xy[, 1L]), ]
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    n0 = x.shape[0]
    # R: indlow <- floor(trim*length(x)) + 1  (1-based); indupp <- length(x) - floor(trim*length(x))
    # Converted to a 0-based Python slice [indlow_py:indupp_py] covering the
    # same 1-based inclusive range indlow..indupp.
    indlow = int(np.floor(trim * n0))
    indupp = n0 - int(np.floor(trim * n0))

    x = x[indlow:indupp]
    y = y[indlow:indupp]

    # Rename common parameters. Note: R's default argument `range.x = range(x)`
    # is a lazy promise evaluated (using whatever `x` currently is bound to)
    # only at its first use below -- by which point `x` has already been
    # reassigned to the TRIMMED data above. Replicate that by computing the
    # default range_x here, from the trimmed x, rather than from the
    # original untrimmed x.
    n = x.shape[0]
    M = gridsize
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
    # NOTE: these are two independent `if`s in the original R code (not
    # if/elif); replicated literally, including the (unreachable in
    # practice) case th24Q == 0 where gamseh is left untransformed.
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * np.sqrt(np.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * np.sqrt(np.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                     range_x=range_x, binned=True)["y"]

    # R: llow <- floor(proptrun*M) + 1 (1-based); lupp <- M - floor(proptrun*M)
    # (1-based inclusive end); converted to a 0-based Python slice.
    llow = int(np.floor(proptrun * M))
    lupp = M - int(np.floor(proptrun * M))
    th22kn = float(np.sum((mddest[llow:lupp] ** 2) * xcounts[llow:lupp]) / n)

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh"
    C3K = 0.5 + 2 * np.sqrt(2) - (4.0 / 3.0) * np.sqrt(3)
    C3K = (4 * C3K / np.sqrt(2 * np.pi)) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    # Now compute a local linear kernel estimate of the variance.
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
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    n = X.shape[0]
    M = gpoints.shape[0]
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    gcounts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    # lxi is the (1-based) real-valued grid position of each data point.
    lxi = ((X - a) / delta) + 1.0

    # Fortran's INT() truncates toward zero; NumPy's astype(int) on floats
    # does the same, so this preserves the original integer-part semantics
    # (including for negative lxi values arising from X below a).
    li = lxi.astype(np.int64)
    rem = lxi - li

    interior = (li >= 1) & (li < M)
    li_interior = li[interior]
    rem_interior = rem[interior]

    # Distribute each point's weight linearly between its two bracketing
    # grid points; np.add.at performs an unbuffered (duplicate-safe)
    # scatter-add, matching the Fortran loop's accumulation.
    np.add.at(gcounts, li_interior - 1, 1.0 - rem_interior)
    np.add.at(gcounts, li_interior, rem_interior)

    if trun == 0:
        below = li < 1
        above = li >= M
        gcounts[0] += np.count_nonzero(below)
        gcounts[M - 1] += np.count_nonzero(above)

    return gcounts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)

    x1 = X[:, 0]
    x2 = X[:, 1]

    M1 = gpoints1.shape[0]
    M2 = gpoints2.shape[0]
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    # lxi1/lxi2 are the (1-based) real-valued grid positions of each
    # data point along each dimension.
    lxi1 = ((x1 - a1) / delta1) + 1.0
    lxi2 = ((x2 - a2) / delta2) + 1.0

    # Fortran's INT() truncates toward zero; NumPy's astype(int) on floats
    # does the same, so this preserves the original integer-part semantics
    # (including for negative lxi values arising from points below a1/a2).
    li1 = lxi1.astype(np.int64)
    li2 = lxi2.astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    # Only points strictly interior to the grid in both dimensions
    # contribute; points outside the mesh (including exactly on the
    # right edges b1/b2) are silently ignored, matching the Fortran
    # loop's strict 'li1.lt.M1' / 'li2.lt.M2' checks (no truncate option
    # here, unlike linbin/rlbin).
    interior = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)
    li1_int = li1[interior]
    li2_int = li2[interior]
    rem1_int = rem1[interior]
    rem2_int = rem2[interior]

    # Distribute each point's weight bilinearly among the four
    # bracketing grid points; np.add.at performs an unbuffered
    # (duplicate-safe) scatter-add, matching the Fortran loop's
    # accumulation. gcnts is indexed (li1-1, li2-1) with 0-based
    # indices, which is equivalent to R's column-major
    # 'matrix(out, M1, M2)' reshape of the flat Fortran array.
    np.add.at(gcnts, (li1_int - 1, li2_int - 1), (1.0 - rem1_int) * (1.0 - rem2_int))
    np.add.at(gcnts, (li1_int, li2_int - 1), rem1_int * (1.0 - rem2_int))
    np.add.at(gcnts, (li1_int - 1, li2_int), (1.0 - rem1_int) * rem2_int)
    np.add.at(gcnts, (li1_int, li2_int), rem1_int * rem2_int)

    return gcnts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'kernel' is accepted for signature compatibility with the R function,
    # but (as in the original KernSmooth Fortran implementation) only the
    # normal (Gaussian) kernel is ever actually used/supported.
    del kernel

    if bandwidth is None:
        raise ValueError("'bandwidth' must be supplied")

    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.any(bandwidth_arr <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    degree = drv + 1 if degree is None else int(degree)

    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x) - np.min(x))
            range_x = (float(np.min(x) - extra), float(np.max(x) + extra))
        else:
            range_x = (float(np.min(x)), float(np.max(x)))

    M = gridsize
    Q = int(bwdisc)
    a, b = range_x
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    if y is None:
        # Obtain density estimate. Note: unlike the regression branch below,
        # the 'binned' flag is never consulted here -- this mirrors the
        # original R wrapper, which always treats x as raw (unbinned) data
        # when y is missing.
        n = x.shape[0]
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        # Quirk of the original R wrapper: xcounts is deliberately
        # overwritten with all ones after being used to build ycounts, so
        # the core computation below always uses a uniform weight of 1 per
        # grid point in density-estimation mode.
        xcounts = np.ones(M, dtype=np.float64)
    else:
        y = np.asarray(y, dtype=np.float64)
        if not binned:
            # Obtain regression estimate.
            gpoints = np.linspace(a, b, M)
            out = rlbin(x, y, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = np.asarray(x, dtype=np.float64)
            ycounts = np.asarray(y, dtype=np.float64)
            M = xcounts.shape[0]
            gpoints = np.linspace(a, b, M)

    delta = (b - a) / (M - 1)

    if bandwidth_arr.shape[0] == M:
        sorted_bandwidth = np.sort(bandwidth_arr)
        hlow = sorted_bandwidth[0]
        hupp = sorted_bandwidth[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bandwidth[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.array([np.floor(tau * bandwidth_arr[0] / delta)], dtype=np.int64)
        hdisc = np.array([bandwidth_arr[0]], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Convert the (1-based, R-style) discretized-bandwidth group index for
    # each grid point into a 0-based index usable for NumPy indexing.
    indic0 = indic - 1

    # Combine kernel weights and grid counts: for every grid point j, build
    # the local weighted moment sums
    #   ss[j, ii] = sum_k xcounts[k] * K_h(k - j) * (delta*(k - j))**ii
    #   tt[j, ii] = sum_k ycounts[k] * K_h(k - j) * (delta*(k - j))**ii
    # where K_h is the (unnormalized) Gaussian kernel using the discretized
    # bandwidth assigned to grid point j (K_h(0) == 1), and the sum runs over
    # k within the window [j - L, j + L] (L = Lvec for that bandwidth
    # group). Grid points sharing the same discretized bandwidth group are
    # processed together in a single vectorized (batched) computation.
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)

    for i in range(Q):
        j_idx = np.where(indic0 == i)[0]
        if j_idx.size == 0:
            continue

        L = int(Lvec[i])
        h_i = hdisc[i]
        offsets = np.arange(-L, L + 1)
        weights = np.exp(-0.5 * (delta * offsets / h_i) ** 2)
        lag = delta * offsets

        powers_ss = lag[:, None] ** np.arange(ppp)[None, :]
        powers_tt = powers_ss[:, :pp]

        k_idx = j_idx[:, None] + offsets[None, :]
        valid = (k_idx >= 0) & (k_idx < M)
        k_idx_clipped = np.clip(k_idx, 0, M - 1)

        xk = np.where(valid, xcounts[k_idx_clipped], 0.0)
        yk = np.where(valid, ycounts[k_idx_clipped], 0.0)

        ss[j_idx, :] += (xk * weights[None, :]) @ powers_ss
        tt[j_idx, :] += (yk * weights[None, :]) @ powers_tt

    # For every grid point k, assemble the local (pp x pp) moment (Hankel)
    # matrix and pp-length vector from ss/tt, then solve the local weighted
    # least squares system for the local polynomial coefficients. All M
    # grid points are solved together via NumPy's batched linear solve.
    hankel_idx = np.add.outer(np.arange(pp), np.arange(pp))
    Smat_all = ss[:, hankel_idx]
    Tvec_all = tt

    # NumPy's batched np.linalg.solve requires an explicit trailing
    # right-hand-side axis to disambiguate a batch of vector systems
    # (M, pp, pp) / (M, pp) from a single (pp, pp) system with an (M, pp)
    # matrix right-hand side; add and then drop that axis.
    coef_all = np.linalg.solve(Smat_all, Tvec_all[:, :, None])[:, :, 0]
    cvest = coef_all[:, drv]

    curvest = math.gamma(drv + 1) * cvest

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

    xcounts = np.zeros(M, dtype=np.float64)
    ycounts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    # lxi is the (1-based) real-valued grid position of each data point.
    lxi = ((X - a) / delta) + 1.0

    # Fortran's INT() truncates toward zero; NumPy's astype(int) on floats
    # does the same, so this preserves the original integer-part semantics
    # (including for negative lxi values arising from X below a).
    li = lxi.astype(np.int64)
    rem = lxi - li

    # Fortran's rlbin.f applies a special-case correction for points
    # landing exactly on the right boundary 'b', assigning them fully
    # to the last bin regardless of floating-point rounding in li/rem.
    at_b = X == b
    li = np.where(at_b, M - 1, li)
    rem = np.where(at_b, 1.0, rem)

    interior = (li >= 1) & (li < M)
    li_interior = li[interior]
    rem_interior = rem[interior]
    Y_interior = Y[interior]

    # Distribute each point's weight (and its Y value weighted the same
    # way) linearly between its two bracketing grid points; np.add.at
    # performs an unbuffered (duplicate-safe) scatter-add, matching the
    # Fortran loop's accumulation.
    np.add.at(xcounts, li_interior - 1, 1.0 - rem_interior)
    np.add.at(xcounts, li_interior, rem_interior)
    np.add.at(ycounts, li_interior - 1, (1.0 - rem_interior) * Y_interior)
    np.add.at(ycounts, li_interior, rem_interior * Y_interior)

    if trun == 0:
        below = li < 1
        above = li >= M
        xcounts[0] += np.count_nonzero(below)
        xcounts[M - 1] += np.count_nonzero(above)
        ycounts[0] += np.sum(Y[below])
        ycounts[M - 1] += np.sum(Y[above])

    return {"xcounts": xcounts, "ycounts": ycounts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'kernel' is accepted for signature compatibility with the R function,
    # but (as in the original KernSmooth Fortran implementation) only the
    # normal (Gaussian) kernel is ever actually used/supported.
    del kernel

    # 'drv' is accepted for signature compatibility with the R function,
    # but (unlike locpoly) it is never actually used anywhere in sdiag's
    # body: sdiag() always returns the [0, 0] entry of the inverted local
    # moment matrix at each grid point, regardless of 'drv'.
    del drv

    if bandwidth is None:
        raise ValueError("'bandwidth' must be supplied")

    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    M = gridsize
    Q = int(bwdisc)
    a, b = range_x
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    delta = (b - a) / (M - 1)

    if bandwidth_arr.shape[0] == M:
        sorted_bandwidth = np.sort(bandwidth_arr)
        hlow = sorted_bandwidth[0]
        hupp = sorted_bandwidth[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bandwidth[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.array([np.floor(tau * bandwidth_arr[0] / delta)], dtype=np.int64)
        hdisc = np.array([bandwidth_arr[0]], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Convert the (1-based, R-style) discretized-bandwidth group index for
    # each grid point into a 0-based index usable for NumPy indexing.
    indic0 = indic - 1

    # Combine kernel weights and grid counts: for every grid point j, build
    # the local weighted moment sums
    #   ss[j, ii] = sum_k xcounts[k] * K_h(k - j) * (delta*(k - j))**ii
    # where K_h is the (unnormalized) Gaussian kernel using the discretized
    # bandwidth assigned to grid point j (K_h(0) == 1), and the sum runs over
    # k within the window [j - L, j + L] (L = Lvec for that bandwidth
    # group). Grid points sharing the same discretized bandwidth group are
    # processed together in a single vectorized (batched) computation.
    ss = np.zeros((M, ppp), dtype=np.float64)

    for i in range(Q):
        j_idx = np.where(indic0 == i)[0]
        if j_idx.size == 0:
            continue

        L = int(Lvec[i])
        h_i = hdisc[i]
        offsets = np.arange(-L, L + 1)
        weights = np.exp(-0.5 * (delta * offsets / h_i) ** 2)
        lag = delta * offsets

        powers_ss = lag[:, None] ** np.arange(ppp)[None, :]

        k_idx = j_idx[:, None] + offsets[None, :]
        valid = (k_idx >= 0) & (k_idx < M)
        k_idx_clipped = np.clip(k_idx, 0, M - 1)

        xk = np.where(valid, xcounts[k_idx_clipped], 0.0)

        ss[j_idx, :] += (xk * weights[None, :]) @ powers_ss

    # For every grid point k, assemble the local (pp x pp) moment (Hankel)
    # matrix from ss (Smat[i, j] = ss[k, i + j], 0-based), invert it, and
    # take the [0, 0] entry of the inverse -- this is the diagonal entry
    # of the "binned" local polynomial smoother matrix at grid point k.
    # All M grid points are inverted together via NumPy's batched matrix
    # inverse.
    hankel_idx = np.add.outer(np.arange(pp), np.arange(pp))
    Smat_all = ss[:, hankel_idx]
    Smat_inv_all = np.linalg.inv(Smat_all)
    Sdg = Smat_inv_all[:, 0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'kernel' is accepted for signature compatibility with the R function,
    # but (as in the original KernSmooth Fortran implementation) only the
    # normal (Gaussian) kernel is ever actually used/supported.
    del kernel

    # 'drv' is accepted for signature compatibility with the R function,
    # but it is never actually used anywhere in sstdiag's body: sstdiag()
    # always returns the diagonal of S S^T (via the quadratic form below)
    # at each grid point, regardless of 'drv'.
    del drv

    if bandwidth is None:
        raise ValueError("'bandwidth' must be supplied")

    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    M = gridsize
    Q = int(bwdisc)
    a, b = range_x
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    delta = (b - a) / (M - 1)

    if bandwidth_arr.shape[0] == M:
        sorted_bandwidth = np.sort(bandwidth_arr)
        hlow = sorted_bandwidth[0]
        hupp = sorted_bandwidth[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bandwidth[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.array([np.floor(tau * bandwidth_arr[0] / delta)], dtype=np.int64)
        hdisc = np.array([bandwidth_arr[0]], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Convert the (1-based, R-style) discretized-bandwidth group index for
    # each grid point into a 0-based index usable for NumPy indexing.
    indic0 = indic - 1

    # Combine kernel weights and grid counts: for every grid point j, build
    # the local weighted moment sums
    #   ss[j, ii] = sum_k xcounts[k] * K_h(k - j)      * (delta*(k - j))**ii
    #   uu[j, ii] = sum_k xcounts[k] * K_h(k - j)**2   * (delta*(k - j))**ii
    # where K_h is the (unnormalized) Gaussian kernel using the discretized
    # bandwidth assigned to grid point j (K_h(0) == 1), and the sum runs over
    # k within the window [j - L, j + L] (L = Lvec for that bandwidth
    # group). Note that the (delta*(k - j))**ii power progression ('fac' in
    # the Fortran source) is shared identically between ss and uu -- only
    # the kernel-weight factor is squared for uu. Grid points sharing the
    # same discretized bandwidth group are processed together in a single
    # vectorized (batched) computation.
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)

    for i in range(Q):
        j_idx = np.where(indic0 == i)[0]
        if j_idx.size == 0:
            continue

        L = int(Lvec[i])
        h_i = hdisc[i]
        offsets = np.arange(-L, L + 1)
        weights = np.exp(-0.5 * (delta * offsets / h_i) ** 2)
        lag = delta * offsets

        powers = lag[:, None] ** np.arange(ppp)[None, :]

        k_idx = j_idx[:, None] + offsets[None, :]
        valid = (k_idx >= 0) & (k_idx < M)
        k_idx_clipped = np.clip(k_idx, 0, M - 1)

        xk = np.where(valid, xcounts[k_idx_clipped], 0.0)

        ss[j_idx, :] += (xk * weights[None, :]) @ powers
        uu[j_idx, :] += (xk * (weights ** 2)[None, :]) @ powers

    # For every grid point k, assemble the local (pp x pp) moment (Hankel)
    # matrices Smat (from ss) and Umat (from uu):
    #   Smat[i, j] = ss[k, i + j],  Umat[i, j] = uu[k, i + j]  (0-based)
    # Invert Smat (batched over all M grid points via NumPy's batched matrix
    # inverse), then compute the quadratic form using the first row/column
    # of the inverse, giving the diagonal entry of S S^T at grid point k:
    #   SSTd[k] = Smat_inv[0, :] @ Umat @ Smat_inv[:, 0]
    hankel_idx = np.add.outer(np.arange(pp), np.arange(pp))
    Smat_all = ss[:, hankel_idx]
    Umat_all = uu[:, hankel_idx]
    Smat_inv_all = np.linalg.inv(Smat_all)

    row0 = Smat_inv_all[:, 0, :]
    SSTd = np.einsum('mi,mij,mj->m', row0, Umat_all, row0)

    return {"x": gpoints, "y": SSTd}


def onAttach(libname: str, pkgname: str) -> None:
    # Equivalent of R's packageStartupMessage(): emit a non-fatal startup
    # notice on the message stream (stderr), not stdout, and not as a
    # return value or a hard warning.
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def onUnload(libpath: str) -> None:
    # In R, this unloads the compiled Fortran/C shared object via
    # library.dynam.unload(). This pure-Python/NumPy port reimplements all
    # of KernSmooth's Fortran routines directly (no .Fortran() calls remain
    # and no compiled extension is loaded), so there is nothing to unload.
    pass
