from . import _KernSmooth

import math
import warnings
from typing import Any

import numpy as np
from scipy.stats import beta as _beta, norm as _norm


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
    # Install safeguard against non-positive bandwidths (mirrors
    # `if (!missing(bandwidth) && bandwidth <= 0) stop(...)`).
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
    kernel_choices = ["normal", "box", "epanech", "biweight", "triweight"]
    if kernel in kernel_choices:
        pass
    else:
        matches = [choice for choice in kernel_choices if choice.startswith(kernel)]
        if len(matches) == 1:
            kernel = matches[0]
        else:
            raise ValueError("'kernel' should be one of " + str(kernel_choices))

    x_arr = np.asarray(x, dtype=np.float64)

    # Rename common variables
    n = x_arr.shape[0]
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
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.sqrt(np.var(x_arr, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values
    tau = 4.0 if kernel == "normal" else 1.0

    if range_x is None:
        a = float(np.min(x_arr) - tau * h)
        b = float(np.max(x_arr) + tau * h)
    else:
        a = float(range_x[0])
        b = float(range_x[1])

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    # Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1, dtype=np.float64)
    if kernel == "normal":
        kappa = _norm.pdf(lvec * delta) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1.0), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1.0), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1.0), 3, 3) / (n * h)
    else:  # kernel == "triweight"
        kappa = 0.5 * _beta.pdf(0.5 * (lvec * delta + 1.0), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts_padded = np.concatenate([gcounts, np.zeros(P - M)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts_padded)

    # R's fft(z, inverse = TRUE) is the *unnormalized* inverse DFT, i.e.
    # numpy.fft.ifft(z) * len(z); the explicit division by P below matches
    # R's convention of leaving the normalization to the caller.
    inv = np.fft.ifft(kappa_fft * gcounts_fft) * P
    y = (np.real(inv) / P)[0:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, ...], gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.integer[Any]]] = (51, 51), range_x: tuple[tuple[float, float], tuple[float, float]] | list[list[float]] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths (mirrors
    # `if (!missing(bandwidth) && min(bandwidth) <= 0) stop(...)`; `bandwidth`
    # has no default in the R signature so it is effectively always supplied).
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.min(h) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x_arr = np.asarray(x, dtype=np.float64)

    # Rename common variables
    n = x_arr.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x_list: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.zeros(2), np.zeros(2)]
        for id_ in range(2):
            col = x_arr[:, id_]
            range_x_list[id_] = np.array(
                [np.min(col) - 1.5 * h[id_], np.max(col) + 1.5 * h[id_]]
            )
    else:
        range_x_list = [np.asarray(range_x[0], dtype=np.float64), np.asarray(range_x[1], dtype=np.float64)]

    a = np.array([range_x_list[0][0], range_x_list[1][0]])
    b = np.array([range_x_list[0][1], range_x_list[1][1]])

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x_arr, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.zeros(1), np.zeros(1)]
    for id_ in range(2):
        L[id_] = min(
            int(np.floor(tau * h[id_] * (M[id_] - 1) / (b[id_] - a[id_]))),
            int(M[id_]) - 1,
        )
        lvecid = np.arange(0, L[id_] + 1, dtype=np.float64)
        facid = (b[id_] - a[id_]) / (h[id_] * (M[id_] - 1))
        z = _norm.pdf(lvecid * facid) / h[id_]
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * facid * h[id_]
        kapid[id_] = z / tot

    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log(M.astype(np.float64) + L) / np.log(2))).astype(np.int64)
    # smallest powers of 2 >= M+L
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
    # R's fft(z, inverse = TRUE) is the *unnormalized* inverse DFT, i.e.
    # numpy.fft.ifft2(z) * z.size; the explicit division by (P1*P2) below
    # matches R's convention of leaving the normalization to the caller.
    inv = np.fft.ifft2(rp_fft * sp_fft) * (P1 * P2)
    rp = (np.real(inv) / (P1 * P2))[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths (mirrors
    # `if (!missing(bandwidth) && bandwidth <= 0) stop(...)`; bandwidth is
    # effectively required here, so it is always validated when provided).
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x_arr = np.asarray(x, dtype=np.float64)

    # if (missing(range.x) && !binned) range.x <- c(min(x), max(x))
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
        M = gcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width
    n = np.sum(gcounts)
    delta = (b - a) / (M - 1)

    # Obtain kernel weights
    tau = 4 + drv
    L = min(int(np.floor(tau * h / delta)), M)

    if L == 0:
        import warnings
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1, dtype=np.float64)
    arg = lvec * delta / h

    kappam = (np.exp(-arg ** 2 / 2.0) / np.sqrt(2.0 * np.pi)) / (h ** (drv + 1))
    hmold0 = 1.0
    hmold1 = arg
    hmnew = 1.0
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0       # Compute mth degree Hermite polynomial
            hmold0 = hmold1                                 # by recurrence.
            hmold1 = hmnew
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam_fft = np.fft.fft(kappam)
    Gcounts_fft = np.fft.fft(Gcounts)

    # R's fft(z, inverse = TRUE) is the *unnormalized* inverse DFT, i.e.
    # numpy.fft.ifft(z) * len(z); the explicit division by P below matches
    # R's convention of leaving the normalization to the caller.
    inv = np.fft.ifft(kappam_fft * Gcounts_fft) * P
    smooth = (np.real(inv) / P)[:M]

    return float(np.sum(gcounts * smooth) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # Coerce inputs to double-precision numpy arrays, mirroring as.double() coercions in R.
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    n = x_arr.shape[0]

    # Sort the (x, y) data with respect to the x's, matching
    # datmat <- datmat[sort.list(datmat[, 1L]), ] in R (a stable ascending sort).
    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    qq = q + 1

    # Equivalent of the Fortran subroutine `blkest`: partitions the sorted
    # (x, y) data into Nval contiguous blocks, fits a q'th degree polynomial
    # by least squares within each block (replacing the dqrdc/dqrsl QR
    # decomposition used in the Fortran code), and pools the residuals and
    # derivative-based quantities across blocks to obtain sigsqe, th22e and
    # th24e for the Ruppert-Sheather-Wand direct plug-in bandwidth selector.
    idiv = n // Nval

    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    for j in range(Nval):
        ilow = j * idiv
        iupp = n if j == Nval - 1 else (j + 1) * idiv
        Xj = x_sorted[ilow:iupp]
        Yj = y_sorted[ilow:iupp]

        # Obtain a q'th degree fit over the current member of the partition.
        # Xmat is the design matrix with columns 1, Xj, Xj**2, ..., Xj**q,
        # matching the Xmat set up before the dqrdc/dqrsl calls in Fortran.
        Xmat = np.vander(Xj, N=qq, increasing=True)
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        fiti = Xmat @ coef

        # Second derivative of the fitted polynomial evaluated at each Xj:
        # p''(x) = sum_{k=2}^{q} k*(k-1)*coef[k]*x**(k-2), equivalent to the
        # ddm accumulation in the Fortran loop.
        powers = np.arange(qq)
        d2_coef = coef[2:] * (powers[2:] * (powers[2:] - 1))
        if d2_coef.size > 0:
            ddm = np.vander(Xj, N=d2_coef.size, increasing=True) @ d2_coef
        else:
            ddm = np.zeros_like(Xj)

        # Fourth derivative of the fitted polynomial evaluated at each Xj:
        # p''''(x) = sum_{k=4}^{q} k*(k-1)*(k-2)*(k-3)*coef[k]*x**(k-4),
        # equivalent to the ddddm accumulation in the Fortran loop.
        d4_coef = coef[4:] * (powers[4:] * (powers[4:] - 1) * (powers[4:] - 2) * (powers[4:] - 3))
        if d4_coef.size > 0:
            ddddm = np.vander(Xj, N=d4_coef.size, increasing=True) @ d4_coef
        else:
            ddddm = np.zeros_like(Xj)

        th22e += np.sum(ddm ** 2)
        th24e += np.sum(ddm * ddddm)
        RSS += np.sum((Yj - fiti) ** 2)

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # Coerce inputs to double-precision numpy arrays, mirroring as.double() coercions in R.
    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(Y, dtype=np.float64)

    n = x_arr.shape[0]

    # Sort the (X, Y) data with respect to the X's, matching
    # datmat <- datmat[sort.list(datmat[, 1L]), ] in R (a stable ascending sort).
    order_idx = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order_idx]
    y_sorted = y_arr[order_idx]

    qq = q + 1

    # Equivalent of the Fortran subroutine `cp`: for each candidate number of
    # blocks Nval = 1, ..., Nmax, partition the sorted (X, Y) data into Nval
    # contiguous blocks, fit a q'th degree polynomial by least squares within
    # each block (replacing the dqrdc/dqrsl QR decomposition used in the
    # Fortran code), and accumulate the residual sum of squares over all
    # blocks to obtain RSS(Nval).
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        idiv = n // Nval
        RSSj_total = 0.0

        for j in range(1, Nval + 1):
            ilow = (j - 1) * idiv + 1
            iupp = n if j == Nval else j * idiv
            Xj = x_sorted[ilow - 1:iupp]
            Yj = y_sorted[ilow - 1:iupp]

            # Obtain a q'th degree fit over the current member of the partition.
            # Xmat is the design matrix with columns 1, Xj, Xj**2, ..., Xj**q,
            # matching the Xmat set up before the dqrdc/dqrsl calls in Fortran.
            Xmat = np.vander(Xj, N=qq, increasing=True)
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fiti = Xmat @ coef
            RSSj_total += np.sum((Yj - fiti) ** 2)

        RSS[Nval - 1] = RSSj_total

    # Now compute array of Mallow's C_p values.
    i = np.arange(1, Nmax + 1, dtype=np.float64)
    Cpvals = ((n - qq * Nmax) * RSS / RSS[Nmax - 1]) + 2 * qq * i - n

    # order(Cpvec)[1L] returns the (1-based) index of the smallest C_p value.
    return int(np.argmin(Cpvals)) + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x_arr = np.asarray(x, dtype=np.float64)

    ## Rename variables

    n = x_arr.shape[0]
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data
    ## (This first, unstandardised binning is computed by the original R
    ## implementation but immediately overwritten below; it is omitted here
    ## as dead code with no effect on the result.)

    ## Compute scale estimate

    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError("'scalest' should be one of 'minim', 'stdev', 'iqr'")

    if scalest == "stdev":
        scale_val = math.sqrt(np.var(x_arr, ddof=1))
    elif scalest == "iqr":
        q75 = np.quantile(x_arr, 0.75)
        q25 = np.quantile(x_arr, 0.25)
        scale_val = (q75 - q25) / 1.349
    else:
        q75 = np.quantile(x_arr, 0.75)
        q25 = np.quantile(x_arr, 0.25)
        scale_val = min((q75 - q25) / 1.349, math.sqrt(np.var(x_arr, ddof=1)))

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = np.mean(x_arr)
    sx = (x_arr - x_mean) / scale_val
    sa = (a - x_mean) / scale_val
    sb = (b - x_mean) / scale_val

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps

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
    else:  # level == 5
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

    return scale_val * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    if kernel not in ("normal", "box", "epanech", "biweight", "triweight"):
        raise ValueError(
            "'kernel' should be one of 'normal', 'box', 'epanech', 'biweight', 'triweight'"
        )

    x_arr = np.asarray(x, dtype=np.float64)

    ## Set kernel constants

    if canonical:
        del0 = 1.0
    elif kernel == "normal":
        del0 = 1.0 / ((4 * math.pi) ** (1 / 10))
    elif kernel == "box":
        del0 = (9 / 2) ** (1 / 5)
    elif kernel == "epanech":
        del0 = 15 ** (1 / 5)
    elif kernel == "biweight":
        del0 = 35 ** (1 / 5)
    else:  # kernel == "triweight"
        del0 = (9450 / 143) ** (1 / 5)

    ## Rename variables

    n = x_arr.shape[0]
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data
    ## (This first, unstandardised binning is computed by the original R
    ## implementation but immediately overwritten below; it is omitted here
    ## as dead code with no effect on the result.)

    ## Compute scale estimate

    if scalest not in ("minim", "stdev", "iqr"):
        raise ValueError("'scalest' should be one of 'minim', 'stdev', 'iqr'")

    if scalest == "stdev":
        scale_val = math.sqrt(np.var(x_arr, ddof=1))
    elif scalest == "iqr":
        q75 = np.quantile(x_arr, 0.75)
        q25 = np.quantile(x_arr, 0.25)
        scale_val = (q75 - q25) / 1.349
    else:
        q75 = np.quantile(x_arr, 0.75)
        q25 = np.quantile(x_arr, 0.25)
        scale_val = min((q75 - q25) / 1.349, math.sqrt(np.var(x_arr, ddof=1)))

    if scale_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = np.mean(x_arr)
    sx = (x_arr - x_mean) / scale_val
    sa = (a - x_mean) / scale_val
    sb = (b - x_mean) / scale_val

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M - 1)

    ## Perform plug-in steps:

    if level == 0:
        psi4hat = 3 / (8 * math.sqrt(math.pi))
    elif level == 1:
        alpha = (2 * (math.sqrt(2)) ** 7 / (5 * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        alpha = (2 * (math.sqrt(2)) ** 9 / (7 * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        alpha = (2 * (math.sqrt(2)) ** 11 / (9 * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * math.sqrt(2 / math.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        alpha = (2 * (math.sqrt(2)) ** 13 / (11 * n)) ** (1 / 13)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * math.sqrt(2 / math.pi) / (psi10hat * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * math.sqrt(2 / math.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    else:  # level == 5
        alpha = (2 * (math.sqrt(2)) ** 15 / (13 * n)) ** (1 / 15)  # bandwidth for psi_12
        psi12hat = bkfe(gcounts, 12, alpha, range_x=(sa, sb), binned=True)
        alpha = (945 * math.sqrt(2 / math.pi) / (psi12hat * n)) ** (1 / 13)  # bandwidth for psi_10
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * math.sqrt(2 / math.pi) / (psi10hat * n)) ** (1 / 11)  # bandwidth for psi_8
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * math.sqrt(2 / math.pi) / (psi8hat * n)) ** (1 / 9)  # bandwidth for psi_6
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * math.sqrt(2 / math.pi) / (psi6hat * n)) ** (1 / 7)  # bandwidth for psi_4
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)

    return scale_val * del0 * (1 / (psi4hat * n)) ** (1 / 5)


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    # Coerce inputs to double-precision numpy arrays, mirroring as.double() coercions in R.
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # xy <- cbind(x, y); xy <- xy[sort.list(xy[, 1L]), ]
    # sort.list() performs a stable sort on the x column, so use a stable
    # argsort to reproduce identical tie-breaking behaviour.
    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    # Trim the 100(trim)% of the data from each end (in the x-direction).
    # indlow/indupp below are computed using R's 1-based convention and then
    # translated to a 0-based Python slice: R's x[indlow:indupp] (inclusive)
    # becomes x_sorted[indlow-1:indupp] in Python.
    n_full = x_sorted.shape[0]
    indlow = int(math.floor(trim * n_full)) + 1
    indupp = n_full - int(math.floor(trim * n_full))

    x = x_sorted[indlow - 1:indupp]
    y = y_sorted[indlow - 1:indupp]

    # range.x = range(x) is a default argument in R, which is evaluated
    # lazily (as a promise) the first time it is used inside the function
    # body -- by which point 'x' has already been reassigned to the sorted
    # and trimmed vector above. Reproduce that by computing the default
    # range from the trimmed x, not the original input.
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common parameters
    n = x.shape[0]
    M = int(gridsize)
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

    # llow/lupp are R 1-based inclusive bounds; the inclusive R slice
    # mddest[llow:lupp] becomes mddest[llow-1:lupp] in 0-based Python.
    llow = int(math.floor(proptrun * M)) + 1
    lupp = M - int(math.floor(proptrun * M))
    th22kn = np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n

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
    # Coerce inputs to double-precision numpy arrays, mirroring as.double()/as.integer() coercions in R.
    X_arr = np.asarray(X, dtype=np.float64)
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)

    M = gpoints_arr.shape[0]
    trun = 1 if truncate else 0
    a = gpoints_arr[0]
    b = gpoints_arr[M - 1]

    # Equivalent of the Fortran subroutine `linbin`: obtains bin counts for
    # univariate data via the linear binning strategy.
    gcnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    # 1-based fractional grid position of each observation (kept 1-based to
    # mirror the Fortran indexing before converting to 0-based numpy indices).
    lxi = (X_arr - a) / delta + 1.0

    # Fortran's int() truncates toward zero; np.trunc followed by casting
    # reproduces that behaviour exactly (unlike floor, which differs for
    # negative values).
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Observations whose grid position falls strictly inside [1, M) split
    # their weight between the two bracketing grid points.
    mid_mask = (li >= 1) & (li < M)
    li_mid = li[mid_mask]
    rem_mid = rem[mid_mask]
    np.add.at(gcnts, li_mid - 1, 1.0 - rem_mid)
    np.add.at(gcnts, li_mid, rem_mid)

    if trun == 0:
        # Give full weight of out-of-range observations to the corresponding
        # end grid point instead of truncating them.
        low_mask = li < 1
        gcnts[0] += np.count_nonzero(low_mask)

        high_mask = li >= M
        gcnts[M - 1] += np.count_nonzero(high_mask)

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    # Coerce inputs to double-precision numpy arrays, mirroring as.double()/as.integer() coercions in R.
    X_arr = np.asarray(X, dtype=np.float64)
    gpoints1_arr = np.asarray(gpoints1, dtype=np.float64)
    gpoints2_arr = np.asarray(gpoints2, dtype=np.float64)

    n = X_arr.shape[0]
    x1 = X_arr[:, 0]
    x2 = X_arr[:, 1]

    M1 = gpoints1_arr.shape[0]
    M2 = gpoints2_arr.shape[0]
    a1 = gpoints1_arr[0]
    a2 = gpoints2_arr[0]
    b1 = gpoints1_arr[M1 - 1]
    b2 = gpoints2_arr[M2 - 1]

    # Equivalent of the Fortran subroutine `lbtwod`: obtains bin counts for
    # bivariate data via the linear binning strategy. Observations outside
    # the mesh are ignored (no truncate/no-truncate option in this version).
    gcnts = np.zeros((M1, M2), dtype=np.float64)
    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    # 1-based fractional grid position of each observation along each
    # dimension (kept 1-based to mirror the Fortran indexing before
    # converting to 0-based numpy indices).
    lxi1 = (x1 - a1) / delta1 + 1.0
    lxi2 = (x2 - a2) / delta2 + 1.0

    # Fortran's int() truncates toward zero; np.trunc followed by casting
    # reproduces that behaviour exactly (unlike floor, which differs for
    # negative values).
    li1 = np.trunc(lxi1).astype(np.int64)
    li2 = np.trunc(lxi2).astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    # Only observations whose grid position falls strictly inside
    # [1, M1) x [1, M2) contribute; observations outside the mesh are
    # ignored, matching the Fortran subroutine's nested range checks.
    mask = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)
    li1_m = li1[mask] - 1
    li2_m = li2[mask] - 1
    rem1_m = rem1[mask]
    rem2_m = rem2[mask]

    # Bilinear splitting of each observation's unit weight across the four
    # grid points surrounding it.
    np.add.at(gcnts, (li1_m, li2_m), (1.0 - rem1_m) * (1.0 - rem2_m))
    np.add.at(gcnts, (li1_m + 1, li2_m), rem1_m * (1.0 - rem2_m))
    np.add.at(gcnts, (li1_m, li2_m + 1), (1.0 - rem1_m) * rem2_m)
    np.add.at(gcnts, (li1_m + 1, li2_m + 1), rem1_m * rem2_m)

    return gcnts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    # if (!missing(bandwidth) && any(bandwidth <= 0)) stop(...)
    if bandwidth is not None:
        bandwidth_check = np.asarray(bandwidth, dtype=np.float64)
        if np.any(bandwidth_check <= 0):
            raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    # Coerce x to a double-precision numpy array, mirroring as.double() coercions in R.
    x_arr = np.asarray(x, dtype=np.float64)

    # if (missing(range.x) && !binned) ...
    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x_arr) - np.min(x_arr))
            range_x = (np.min(x_arr) - extra, np.max(x_arr) + extra)
        else:
            range_x = (np.min(x_arr), np.max(x_arr))

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
        # Obtain density estimate.
        n = x_arr.shape[0]
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.full(M, 1.0, dtype=np.float64)
    else:
        # Obtain regression estimate.
        y_arr = np.asarray(y, dtype=np.float64)
        if not binned:
            # Bin the data if not already binned.
            gpoints = np.linspace(a, b, M)
            out = rlbin(x_arr, y_arr, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = x_arr
            ycounts = y_arr
            M = xcounts.shape[0]
            gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # 'bandwidth' has no default in R, so a missing value only surfaces as an
    # error once it is actually used below.
    if bandwidth is None:
        raise TypeError("argument 'bandwidth' is missing, with no default")

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    bandwidth_len = bandwidth_arr.shape[0]

    if bandwidth_len == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of 'hdisc' to each member of 'bandwidth'
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.full(M, 1, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.full(M, 1, dtype=np.int64)
    elif bandwidth_len == 1:
        indic = np.full(M, 1, dtype=np.int64)
        Q = 1
        bw_scalar = float(bandwidth_arr[0])
        Lvec = np.full(Q, int(np.floor(tau * bw_scalar / delta)), dtype=np.int64)
        hdisc = np.full(Q, bw_scalar, dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Obtain kernel weights: for each discretised bandwidth level i = 0, ..., Q-1
    # build the (normal-kernel) weight table for offsets -Lvec[i], ..., Lvec[i],
    # equivalent to the flat 'fkap' array indexed via 'midpts' in the Fortran
    # subroutine 'locpol', but kept here as one array per discretised bandwidth.
    kernel_tables: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for i in range(Q):
        Li = int(Lvec[i])
        offsets = np.arange(-Li, Li + 1)
        weights = np.exp(-((delta * np.abs(offsets)) / hdisc[i]) ** 2 / 2.0)
        kernel_tables.append(weights)

    # Combine kernel weights and grid counts: for each grid point j (using its
    # own discretised bandwidth level indic[j]) accumulate weighted sums of
    # powers of (delta * (k - j)) over the bin counts k within its window,
    # matching the ss/tt accumulation loop over k, i, j in the Fortran code
    # (reordered here to loop directly over the target grid point j, which is
    # equivalent since only i == indic(j) ever contributes at j).
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)
    powers_ss = np.arange(ppp)
    powers_tt = np.arange(pp)

    for j in range(M):
        i = int(indic[j]) - 1
        Li = int(Lvec[i])
        k_lo = max(0, j - Li)
        k_hi = min(M - 1, j + Li)
        k_range = np.arange(k_lo, k_hi + 1)
        xk = xcounts[k_range]
        nz_mask = xk != 0
        if not np.any(nz_mask):
            continue
        k_sel = k_range[nz_mask]
        xk_sel = xk[nz_mask]
        yk_sel = ycounts[k_sel]
        offsets = k_sel - j
        weights = kernel_tables[i][offsets + Li]
        diffs = delta * offsets

        diff_powers_ss = diffs[:, None] ** powers_ss[None, :]
        ss[j, :] += np.sum((xk_sel * weights)[:, None] * diff_powers_ss, axis=0)

        diff_powers_tt = diffs[:, None] ** powers_tt[None, :]
        tt[j, :] += np.sum((yk_sel * weights)[:, None] * diff_powers_tt, axis=0)

    # For each grid point, build the (pp x pp) weighted design matrix Smat from
    # 'ss' (Smat[i, j] = ss[k, i + j]) and the right-hand side Tvec = tt[k, :pp],
    # then solve the local weighted least-squares system, replacing the
    # Fortran LINPACK dgefa/dgesl call pair with np.linalg.solve.
    idx = np.arange(pp)
    indss_matrix = idx[:, None] + idx[None, :]

    raw_curvest = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = ss[k][indss_matrix]
        Tvec = tt[k, :pp]
        solution = np.linalg.solve(Smat, Tvec)
        raw_curvest[k] = solution[drv]

    curvest = math.gamma(drv + 1) * raw_curvest

    return {"x": gpoints, "y": curvest}


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Coerce inputs to double-precision numpy arrays, mirroring as.double()/as.integer() coercions in R.
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)

    M = gpoints_arr.shape[0]
    trun = 1 if truncate else 0
    a = gpoints_arr[0]
    b = gpoints_arr[M - 1]

    # Equivalent of the Fortran subroutine `rlbin`: obtains bin counts for
    # univariate regression data (x, y) via the linear binning strategy,
    # producing both xcounts and ycounts on the grid.
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    # 1-based fractional grid position of each observation (kept 1-based to
    # mirror the Fortran indexing before converting to 0-based numpy indices).
    lxi = (X_arr - a) / delta + 1.0

    # Fortran's int() truncates toward zero; np.trunc followed by casting
    # reproduces that behaviour exactly (unlike floor, which differs for
    # negative values).
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Correction for the right endpoint: an observation exactly equal to `b`
    # is not otherwise included (li would equal M), so it is forced into the
    # last bin interval with full weight on the final grid point.
    eq_b_mask = X_arr == b
    li = np.where(eq_b_mask, M - 1, li)
    rem = np.where(eq_b_mask, 1.0, rem)

    # Observations whose grid position falls strictly inside [1, M) split
    # their weight (and their y-contribution) between the two bracketing
    # grid points.
    mid_mask = (li >= 1) & (li < M)
    li_mid = li[mid_mask]
    rem_mid = rem[mid_mask]
    y_mid = Y_arr[mid_mask]
    np.add.at(xcnts, li_mid - 1, 1.0 - rem_mid)
    np.add.at(xcnts, li_mid, rem_mid)
    np.add.at(ycnts, li_mid - 1, (1.0 - rem_mid) * y_mid)
    np.add.at(ycnts, li_mid, rem_mid * y_mid)

    if trun == 0:
        # Give full weight (and full y-contribution) of out-of-range
        # observations to the corresponding end grid point instead of
        # truncating them.
        low_mask = li < 1
        xcnts[0] += np.count_nonzero(low_mask)
        ycnts[0] += np.sum(Y_arr[low_mask])

        high_mask = li >= M
        xcnts[M - 1] += np.count_nonzero(high_mask)
        ycnts[M - 1] += np.sum(Y_arr[high_mask])

    return {"xcounts": xcnts, "ycounts": ycnts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for interface compatibility with the
    # original R function, which never references 'drv' in its body and only
    # ever exercises the normal-kernel branch of the underlying Fortran code.
    _ = drv
    _ = kernel

    x_arr = np.asarray(x, dtype=np.float64)

    # if (missing(range.x) && !binned) range.x <- c(min(x), max(x))
    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    degree_int = int(degree)
    pp = degree_int + 1
    ppp = 2 * degree_int + 1
    tau = 4

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
    else:
        xcounts = np.asarray(x_arr, dtype=np.float64)
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # 'bandwidth' has no default in R, so a missing value only surfaces as an
    # error once it is actually used below.
    if bandwidth is None:
        raise TypeError("argument 'bandwidth' is missing, with no default")

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    bandwidth_len = bandwidth_arr.shape[0]

    if bandwidth_len == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of 'hdisc' to each member of 'bandwidth'
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.full(M, 1, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.full(M, 1, dtype=np.int64)
    elif bandwidth_len == 1:
        indic = np.full(M, 1, dtype=np.int64)
        Q = 1
        bw_scalar = float(bandwidth_arr[0])
        Lvec = np.full(Q, int(np.floor(tau * bw_scalar / delta)), dtype=np.int64)
        hdisc = np.full(Q, bw_scalar, dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Obtain kernel weights: for each discretised bandwidth level i = 0, ..., Q-1
    # build the (normal-kernel) weight table for offsets -Lvec[i], ..., Lvec[i],
    # equivalent to the flat 'fkap' array indexed via 'midpts' in the Fortran
    # subroutine 'sdiag' (reusing the identical kernel-table construction used
    # for 'locpoly', since both derive from the same locpoly.f Fortran family).
    kernel_tables: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for i in range(Q):
        Li = int(Lvec[i])
        offsets = np.arange(-Li, Li + 1)
        weights = np.exp(-((delta * np.abs(offsets)) / hdisc[i]) ** 2 / 2.0)
        kernel_tables.append(weights)

    # Combine kernel weights and grid counts: for each grid point j (using its
    # own discretised bandwidth level indic[j]) accumulate weighted sums of
    # powers of (delta * (k - j)) over the bin counts k within its window,
    # matching the ss(j, ii) accumulation loop over k, i, j in the Fortran
    # code (reordered here to loop directly over the target grid point j,
    # which is equivalent since only i == indic(j) ever contributes at j).
    ss = np.zeros((M, ppp), dtype=np.float64)
    powers_ss = np.arange(ppp)

    for j in range(M):
        i = int(indic[j]) - 1
        Li = int(Lvec[i])
        k_lo = max(0, j - Li)
        k_hi = min(M - 1, j + Li)
        k_range = np.arange(k_lo, k_hi + 1)
        xk = xcounts[k_range]
        nz_mask = xk != 0
        if not np.any(nz_mask):
            continue
        k_sel = k_range[nz_mask]
        xk_sel = xk[nz_mask]
        offsets = k_sel - j
        weights = kernel_tables[i][offsets + Li]
        diffs = delta * offsets

        diff_powers_ss = diffs[:, None] ** powers_ss[None, :]
        ss[j, :] += np.sum((xk_sel * weights)[:, None] * diff_powers_ss, axis=0)

    # For each grid point, build the (pp x pp) weighted design matrix Smat from
    # 'ss' (Smat[i, j] = ss[k, i + j]), then invert it, replacing the Fortran
    # LINPACK dgefa/dgedi call pair (job=01, which overwrites Smat with its own
    # inverse) with np.linalg.inv. The diagonal 'hat' matrix entry for grid
    # point k is the (1, 1) element of that inverse (Smat(1,1) in Fortran, the
    # self-weight e_1^T (X_k^T W_k X_k)^{-1} placed by the local fit at k on
    # its own observation, since the design row at zero offset is [1, 0, ..., 0]
    # for the normal kernel with unit weight at distance zero).
    idx = np.arange(pp)
    indss_matrix = idx[:, None] + idx[None, :]

    Sdg = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = ss[k][indss_matrix]
        Smat_inv = np.linalg.inv(Smat)
        Sdg[k] = Smat_inv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for interface compatibility with the
    # original R function, which never references 'drv' in its body and only
    # ever exercises the normal-kernel branch of the underlying Fortran code.
    _ = drv
    _ = kernel

    x_arr = np.asarray(x, dtype=np.float64)

    # if (missing(range.x) && !binned) range.x <- c(min(x), max(x))
    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename common variables
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    degree_int = int(degree)
    pp = degree_int + 1
    ppp = 2 * degree_int + 1
    tau = 4

    # Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
    else:
        xcounts = np.asarray(x_arr, dtype=np.float64)
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # 'bandwidth' has no default in R, so a missing value only surfaces as an
    # error once it is actually used below.
    if bandwidth is None:
        raise TypeError("argument 'bandwidth' is missing, with no default")

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    bandwidth_len = bandwidth_arr.shape[0]

    if bandwidth_len == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of 'hdisc' to each member of 'bandwidth'
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.full(M, 1, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.full(M, 1, dtype=np.int64)
    elif bandwidth_len == 1:
        indic = np.full(M, 1, dtype=np.int64)
        Q = 1
        bw_scalar = float(bandwidth_arr[0])
        Lvec = np.full(Q, int(np.floor(tau * bw_scalar / delta)), dtype=np.int64)
        hdisc = np.full(Q, bw_scalar, dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Obtain kernel weights: for each discretised bandwidth level i = 0, ..., Q-1
    # build the (normal-kernel) weight table for offsets -Lvec[i], ..., Lvec[i],
    # equivalent to the flat 'fkap' array indexed via 'midpts' in the Fortran
    # subroutine 'sstdg' (reusing the identical kernel-table construction used
    # for 'sdiag'/'locpoly', since all three derive from the same locpoly.f
    # Fortran family).
    kernel_tables: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for i in range(Q):
        Li = int(Lvec[i])
        offsets = np.arange(-Li, Li + 1)
        weights = np.exp(-((delta * np.abs(offsets)) / hdisc[i]) ** 2 / 2.0)
        kernel_tables.append(weights)

    # Combine kernel weights and grid counts: for each grid point j (using its
    # own discretised bandwidth level indic[j]) accumulate weighted sums of
    # powers of (delta * (k - j)) over the bin counts k within its window,
    # matching the ss(j, ii) accumulation loop over k, i, j in the Fortran
    # code (reordered here to loop directly over the target grid point j,
    # which is equivalent since only i == indic(j) ever contributes at j).
    # 'uu' is accumulated identically but using the *squared* kernel weight,
    # since (SS^T) relies on the second weighted-moment array built from
    # fkap(...)**2 in the Fortran source.
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    powers = np.arange(ppp)

    for j in range(M):
        i = int(indic[j]) - 1
        Li = int(Lvec[i])
        k_lo = max(0, j - Li)
        k_hi = min(M - 1, j + Li)
        k_range = np.arange(k_lo, k_hi + 1)
        xk = xcounts[k_range]
        nz_mask = xk != 0
        if not np.any(nz_mask):
            continue
        k_sel = k_range[nz_mask]
        xk_sel = xk[nz_mask]
        offsets = k_sel - j
        weights = kernel_tables[i][offsets + Li]
        diffs = delta * offsets

        diff_powers = diffs[:, None] ** powers[None, :]
        ss[j, :] += np.sum((xk_sel * weights)[:, None] * diff_powers, axis=0)
        uu[j, :] += np.sum((xk_sel * (weights ** 2))[:, None] * diff_powers, axis=0)

    # For each grid point, build the (pp x pp) weighted design matrices Smat
    # and Umat from 'ss'/'uu' (Smat[i, j] = ss[k, i + j], Umat[i, j] = uu[k, i + j]),
    # then invert Smat, replacing the Fortran LINPACK dgefa/dgedi call pair
    # (job=01, which overwrites Smat with its own inverse) with np.linalg.inv.
    # SSTd(k) is then the 'sandwich' quadratic form e_1^T Smat^{-1} Umat Smat^{-1} e_1,
    # computed in the Fortran source as the double sum
    # sum_i sum_j Smat(1,i)*Umat(i,j)*Smat(j,1) after Smat has been overwritten
    # by its inverse.
    idx = np.arange(pp)
    indss_matrix = idx[:, None] + idx[None, :]

    SSTd = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = ss[k][indss_matrix]
        Umat = uu[k][indss_matrix]
        Smat_inv = np.linalg.inv(Smat)
        v = Smat_inv[0, :]
        SSTd[k] = v @ Umat @ v

    return {"x": gpoints, "y": SSTd}


def on_attach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")


def _on_unload(libpath: str) -> None:
    # R's .onUnload(libpath) hook calls library.dynam.unload("KernSmooth", libpath)
    # to unload the package's compiled Fortran/C shared library (DLL/.so) when
    # the package is detached/unloaded.
    #
    # This pure-Python port reimplements all former Fortran routines natively
    # in Python/NumPy (see linbin, rlbin, linbin2D, blkest, cpblock) rather than
    # calling into a compiled shared library via R's .Fortran()/.Call() FFI.
    # Consequently there is no dynamic library handle to release, and this hook
    # has no meaningful equivalent action here. It is retained as a no-op stub
    # only for structural parity with the original R package's onUnload hook.
    return None
