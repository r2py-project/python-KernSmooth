import logging
import math
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


def _on_attach() -> None:
    # R's .onAttach() is invoked automatically by the R runtime when the
    # package is loaded via library()/require(); packageStartupMessage()
    # writes an informational, suppressible message to stderr. Python has
    # no equivalent load hook, so this is exposed as a plain function that
    # the package's __init__.py can call on import to reproduce the same
    # behavior, using the logging module as the idiomatic counterpart to
    # packageStartupMessage (informational, stream-based, suppressible).
    logger = logging.getLogger("KernSmooth")
    logger.info("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # kernel <- match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
    kernel_choices = ["normal", "box", "epanech", "biweight", "triweight"]
    if kernel not in kernel_choices:
        matched = [k for k in kernel_choices if k.startswith(kernel)]
        if len(matched) == 1:
            kernel = matched[0]
        else:
            raise ValueError(
                "'kernel' should be one of " + ", ".join(f'\"{k}\"' for k in kernel_choices)
            )

    # Rename common variables
    n = len(x)
    M = gridsize

    # Set canonical scaling factors
    del0_map: dict[str, float] = {
        "normal": (1.0 / (4.0 * np.pi)) ** (1.0 / 10.0),
        "box": (9.0 / 2.0) ** (1.0 / 5.0),
        "epanech": 15.0 ** (1.0 / 5.0),
        "biweight": 35.0 ** (1.0 / 5.0),
        "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
    }
    del0 = del0_map[kernel]

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
        range_x = (float(np.min(x)) - tau * h, float(np.max(x)) + tau * h)
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

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1, dtype=np.float64), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M, dtype=np.float64)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)
    # R's fft(z, inverse = TRUE) is unnormalised (sums without dividing by P),
    # so dividing the R result by P is equivalent to numpy's normalised ifft,
    # which already divides by the number of elements.
    y = np.real(np.fft.ifft(kappa_fft * gcounts_fft))[0:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] = (51, 51), range_x: list[tuple[float, float]] | list[np.ndarray[Any, np.dtype[np.float64]]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x = np.asarray(x, dtype=np.float64)

    # Rename common variables
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = [None, None]
        for id_ in range(2):
            range_x[id_] = (np.min(x[:, id_]) - 1.5 * h[id_], np.max(x[:, id_]) + 1.5 * h[id_])

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.empty((0, 0)), np.empty((0, 0))]
    for id_ in range(2):
        L[id_] = min(int(np.floor(tau * h[id_] * (M[id_] - 1) / (b[id_] - a[id_]))), int(M[id_]) - 1)
        lvecid = np.arange(0, L[id_] + 1)
        facid = (b[id_] - a[id_]) / (h[id_] * (M[id_] - 1))
        z = (norm.pdf(lvecid * facid) / h[id_]).reshape(-1, 1)
        tot = np.sum(np.concatenate([z.flatten(), z.flatten()[1:][::-1]])) * facid * h[id_]
        kapid[id_] = z / tot

    kapp = (kapid[0] @ kapid[1].T) / n

    if np.min(L) == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    # Now combine weight and counts using the FFT to obtain estimate
    P = (2.0 ** np.ceil(np.log(M + L) / np.log(2.0))).astype(np.int64)  # smallest powers of 2 >= M+L
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
    # R's fft(..., inverse = TRUE) is unnormalised, so the manual division
    # by (P1*P2) in the original code is equivalent to numpy's normalised
    # ifft2, which already divides by the number of elements.
    rp_out = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative
    rp_out = rp_out * (rp_out > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp_out}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

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
        gcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
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

    return float(
        np.sum(gcounts * np.real(np.fft.ifft(kappam * Gcounts))[:M]) / (n ** 2)
    )


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    # Equivalent to the Fortran subroutine blkest: computes blocked
    # q'th degree polynomial fit estimates required for the direct
    # plug-in bandwidth selector of Ruppert, Sheather and Wand.
    qq = q + 1

    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    idiv = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition (1-based bounds as in Fortran).
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        nj = iupp - ilow + 1

        Xj = x[ilow - 1:iupp]
        Yj = y[ilow - 1:iupp]

        # Obtain a q'th degree fit over the current member of the partition.
        # Set up the design matrix (Vandermonde-like matrix).
        Xmat = np.zeros((nj, qq), dtype=np.float64)
        Xmat[:, 0] = 1.0
        for k in range(2, qq + 1):
            Xmat[:, k - 1] = Xj ** (k - 1)

        # Least-squares solve (equivalent to the QR-based dqrdc/dqrsl calls).
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        for i in range(nj):
            xi = Xj[i]
            fiti = coef[0]
            ddm = 2 * coef[2]
            ddddm = 24 * coef[4]
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

    return {"sigsqe": sigsqe, "th22e": th22e, "th24e": th24e}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # NOTE: No .Fortran(F_cp) conversion guide is available for this build, so
    # the Fortran "cp" routine (KernSmooth src/cp.f) is reimplemented directly
    # in NumPy from first principles, following Ruppert, Sheather & Wand's
    # Mallows' C_p block-selection procedure used in the preliminary step of
    # the KernSmooth plug-in bandwidth rules.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    order_idx = np.argsort(X, kind="stable")
    Xs = X[order_idx]
    Ys = Y[order_idx]

    qq = q + 1

    # RSS[N - 1] holds the total residual sum of squares obtained by
    # partitioning the sorted data into N contiguous, near-equal-sized
    # blocks and fitting a degree-q polynomial least-squares regression
    # within each block.
    RSS = np.zeros(Nmax, dtype=np.float64)

    for N in range(1, Nmax + 1):
        base_size, remainder = divmod(n, N)
        start = 0
        rss_total = 0.0
        for j in range(1, N + 1):
            # Distribute the remainder points among the first blocks so
            # that block sizes differ by at most one observation.
            block_size = base_size + (1 if j <= remainder else 0)
            end = start + block_size
            Xj = Xs[start:end]
            Yj = Ys[start:end]

            if block_size >= qq:
                # Design matrix with columns X^0, X^1, ..., X^q
                Xmat = np.vander(Xj, N=qq, increasing=True)
                coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)
                resid = Yj - Xmat @ coef
                rss_total += float(np.dot(resid, resid))

            start = end

        RSS[N - 1] = rss_total

    # Estimate the error variance from the finest partition (N = Nmax),
    # which is assumed to have negligible bias.
    denom = n - Nmax * qq
    sig2 = RSS[Nmax - 1] / denom if denom != 0 else np.nan

    Nvals = np.arange(1, Nmax + 1, dtype=np.float64)
    Cpvec = RSS / sig2 - n + 2.0 * Nvals * qq

    # Index (0-based) of the smallest C_p value.
    best_index = int(np.argmin(Cpvec))

    # The R original returns order(Cpvec)[1L], i.e. the 1-based position of
    # the minimum, which here coincides with the optimal number of blocks N
    # (since Cpvec is indexed by N = 1, ..., Nmax). That count is a
    # meaningful quantity (the chosen number of blocks), not merely a
    # Python array index, so it is returned unchanged as a 1-based count
    # (best_index + 1) to keep its value consistent with the rest of the
    # converted codebase.
    return best_index + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

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

    choices = ("minim", "stdev", "iqr")
    matches = [ch for ch in choices if ch.startswith(scalest)]
    if len(matches) != 1:
        raise ValueError(
            "'scalest' should be one of " + ", ".join('"%s"' % c for c in choices)
        )
    scalest_choice = matches[0]

    if scalest_choice == "stdev":
        scalest_value = np.sqrt(np.var(x, ddof=1))
    elif scalest_choice == "iqr":
        scalest_value = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scalest_value = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

    if scalest_value == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = np.mean(x)
    sx = (x - x_mean) / scalest_value
    sa = (a - x_mean) / scalest_value
    sb = (b - x_mean) / scalest_value

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ##    delta <- (sb-sa)/(M - 1)

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

    return float(scalest_value * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    kernel_matches = [ch for ch in kernel_choices if ch.startswith(kernel)]
    if len(kernel_matches) != 1:
        raise ValueError(
            "'kernel' should be one of " + ", ".join('"%s"' % c for c in kernel_choices)
        )
    kernel_choice = kernel_matches[0]

    ## Set kernel constants

    if canonical:
        del0 = 1.0
    elif kernel_choice == "normal":
        del0 = 1 / ((4 * np.pi) ** (1 / 10))
    elif kernel_choice == "box":
        del0 = (9 / 2) ** (1 / 5)
    elif kernel_choice == "epanech":
        del0 = 15 ** (1 / 5)
    elif kernel_choice == "biweight":
        del0 = 35 ** (1 / 5)
    else:  # "triweight"
        del0 = (9450 / 143) ** (1 / 5)

    ## Rename variables

    x = np.asarray(x, dtype=np.float64)
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

    scalest_choices = ("minim", "stdev", "iqr")
    scalest_matches = [ch for ch in scalest_choices if ch.startswith(scalest)]
    if len(scalest_matches) != 1:
        raise ValueError(
            "'scalest' should be one of " + ", ".join('"%s"' % c for c in scalest_choices)
        )
    scalest_choice = scalest_matches[0]

    if scalest_choice == "stdev":
        scalest_value = np.sqrt(np.var(x, ddof=1))
    elif scalest_choice == "iqr":
        scalest_value = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # "minim"
        scalest_value = min(
            (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
            np.sqrt(np.var(x, ddof=1)),
        )

    if scalest_value == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = np.mean(x)
    sx = (x - x_mean) / scalest_value
    sa = (a - x_mean) / scalest_value
    sb = (b - x_mean) / scalest_value

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ##    delta <- (sb-sa)/(M-1)

    ## Perform plug-in steps:

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

    return float(scalest_value * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Trim the 100(trim)% of the data from each end (in the x-direction).
    # Equivalent to: xy <- cbind(x, y); xy <- xy[sort.list(xy[, 1L]), ]
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    indlow = int(np.floor(trim * len(x))) + 1
    indupp = len(x) - int(np.floor(trim * len(x)))

    # R's x[indlow:indupp] is a 1-based, inclusive slice; the equivalent
    # 0-based, end-exclusive Python slice is x[indlow - 1:indupp].
    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    # Rename common parameters.
    n = len(x)
    M = int(gridsize)

    # NOTE: in the original R code, the formal argument default
    # `range.x = range(x)` is a *lazy* promise. It is only forced later,
    # at the point `a <- range.x[1L]` is first evaluated -- by which time
    # `x` has already been reassigned (sorted and trimmed) above. As a
    # result, when `range.x` is not supplied by the caller, R's default
    # actually uses the *trimmed* `x`, not the `x` originally passed in.
    # That behaviour is reproduced here by computing the default range
    # from the (already sorted/trimmed) local `x` at this point.
    if range_x is None:
        a = float(np.min(x))
        b = float(np.max(x))
    else:
        a = float(range_x[0])
        b = float(range_x[1])

    # Bin the data.
    gpoints = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    # Choose the value of N using Mallow's C_p.
    Nmax = max(min(int(np.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x, y, Nmax, 4)

    # Estimate sig^2, theta_22 and theta_24 using quartic fits
    # on "Nval" blocks.
    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    # Estimate theta_22 using a local cubic fit
    # with a "rule-of-thumb" bandwidth: "gamseh".
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=(a, b), binned=True)["y"]

    llow = int(np.floor(proptrun * M)) + 1
    lupp = M - int(np.floor(proptrun * M))
    th22kn = np.sum((mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]) / n

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh".
    C3K = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / math.sqrt(2 * math.pi)) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    # Now compute a local linear kernel estimate of the variance.
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
    a = float(gpoints[0])
    b = float(gpoints[M - 1])

    gcounts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    for i in range(n):
        # 1-based grid position, matching the Fortran F_linbin routine
        lxi = ((X[i] - a) / delta) + 1
        # Integer part of "lxi" (Fortran INT truncates toward zero, same as Python int())
        li = int(lxi)
        rem = lxi - li
        if li >= 1 and li < M:
            gcounts[li - 1] += (1 - rem)
            gcounts[li] += rem
        elif li < 1 and trun == 0:
            gcounts[0] += 1
        elif li >= M and trun == 0:
            gcounts[M - 1] += 1

    return gcounts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    # Creates the grid counts from a bivariate data set X
    # over an equally-spaced set of grid points contained in
    # (gpoints1, gpoints2) using the linear binning strategy.
    # This reproduces the numerical logic of the FORTRAN subroutine
    # "lbtwod" without calling out to compiled code.
    x1 = np.asarray(X, dtype=np.float64)[:, 0]
    x2 = np.asarray(X, dtype=np.float64)[:, 1]

    M1 = len(gpoints1)
    M2 = len(gpoints2)

    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    # 0-based grid position (equivalent to the FORTRAN 1-based
    # "lxi1"/"lxi2" shifted down by one)
    lxi1 = (x1 - a1) / delta1
    lxi2 = (x2 - a2) / delta2

    # Integer part of the grid position (0-based bin index)
    li1 = np.floor(lxi1).astype(np.int64)
    li2 = np.floor(lxi2).astype(np.int64)

    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    # Observations outside the mesh are ignored, mirroring the
    # bounds checks "li1.ge.1", "li2.ge.1", "li1.lt.M1", "li2.lt.M2"
    # (translated to 0-based indexing)
    valid = (li1 >= 0) & (li2 >= 0) & (li1 < M1 - 1) & (li2 < M2 - 1)

    li1_v = li1[valid]
    li2_v = li2[valid]
    rem1_v = rem1[valid]
    rem2_v = rem2[valid]

    # Distribute each observation's weight across the four
    # surrounding grid points, accumulating contributions from
    # observations that fall into the same bin
    np.add.at(gcnts, (li1_v, li2_v), (1 - rem1_v) * (1 - rem2_v))
    np.add.at(gcnts, (li1_v + 1, li2_v), rem1_v * (1 - rem2_v))
    np.add.at(gcnts, (li1_v, li2_v + 1), (1 - rem1_v) * rem2_v)
    np.add.at(gcnts, (li1_v + 1, li2_v + 1), rem1_v * rem2_v)

    return gcnts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None:
        bw_check = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
        if np.any(bw_check <= 0):
            raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x) - np.min(x))
            range_x = (float(np.min(x) - extra), float(np.max(x) + extra))
        else:
            range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables.
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Decide whether a density estimate or regression estimate is required.
    # Note: the "kernel" argument is retained for interface compatibility
    # only; as in the original Fortran routine, a (truncated) Gaussian
    # kernel is always used.
    if y is None:  # obtain density estimate
        n = len(x)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        y = np.asarray(y, dtype=np.float64)
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

    if bandwidth is None:
        raise ValueError('argument "bandwidth" is missing, with no default')
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    # Discretise the bandwidths.
    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc".
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc" to each member of
        # "bandwidth".
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
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Allocate space for the kernel vector and final estimate.
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    curvest = np.zeros(M, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)

    # ---- Equivalent to the Fortran subroutine "locpol" (src/locpoly.f) ----
    # No .Fortran(F_locpol) conversion guide is available for this build, so
    # the Fortran "locpol" routine is reimplemented directly in NumPy from
    # the original source. "mid", "i", "j", "k" and "ii" below are kept as
    # 1-based Fortran-style loop/bookkeeping indices (as in the original
    # routine) and are only converted to 0-based numpy indices at the point
    # of array access.

    # Obtain kernel weights.
    mid = int(Lvec[0]) + 1
    for i in range(1, Q):  # Fortran: do i = 1, (Q - 1)
        midpts[i - 1] = mid
        fkap[mid - 1] = 1.0
        Li = int(Lvec[i - 1])
        hi = hdisc[i - 1]
        for j in range(1, Li + 1):
            val = np.exp(-((delta * j / hi) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    LQ = int(Lvec[Q - 1])
    hQ = hdisc[Q - 1]
    for j in range(1, LQ + 1):
        val = np.exp(-((delta * j / hQ) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    # Combine kernel weights and grid counts.
    for k in range(1, M + 1):
        xk = xcounts[k - 1]
        if xk != 0:
            yk = ycounts[k - 1]
            for i in range(1, Q + 1):
                Li = int(Lvec[i - 1])
                mp = int(midpts[i - 1])
                jlo = max(1, k - Li)
                jhi = min(M, k + Li)
                for j in range(jlo, jhi + 1):
                    if indic[j - 1] == i:
                        fkval = fkap[k - j + mp - 1]
                        ss[j - 1, 0] += xk * fkval
                        tt[j - 1, 0] += yk * fkval
                        fac = 1.0
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xk * fkval * fac
                            if ii <= pp:
                                tt[j - 1, ii - 1] += yk * fkval * fac

    # For each grid point, build the (Hankel) system matrix "Smat" from
    # "ss" and the vector "Tvec" from "tt", solve the weighted
    # least-squares normal equations (replacing the original LINPACK
    # dgefa/dgesl calls), and extract the drv'th derivative coefficient.
    idx_pp = np.arange(pp)
    hankel_idx = idx_pp[:, None] + idx_pp[None, :]

    for k in range(M):
        Smat = ss[k][hankel_idx]
        Tvec = tt[k, :pp]
        try:
            sol = np.linalg.solve(Smat, Tvec)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(Smat, Tvec, rcond=None)[0]
        curvest[k] = sol[drv]

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def on_unload(libpath: str) -> None:
    # In the R package, `.onUnload` calls `library.dynam.unload("KernSmooth", libpath)`
    # to unload the compiled Fortran/C shared library backing the package when it is
    # detached/unloaded. This Python port does not load a separate compiled dynamic
    # library that needs explicit unloading (any compiled extension module is managed
    # by Python's own import system), so there is no equivalent teardown action to
    # perform here. This function is kept as a no-op stub purely for API parity with
    # the original R package hook.
    return None


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)

    n = len(X)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Equivalent to the Fortran subroutine rlbin: obtains bin counts
    # for univariate regression data via the linear binning strategy.
    xcounts = np.zeros(M, dtype=np.float64)
    ycounts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)
    for i in range(n):
        lxi = ((X[i] - a) / delta) + 1

        # Find integer part of "lxi" (Fortran int() truncates toward zero,
        # which matches Python's int() truncation behavior for floats).
        li = int(lxi)
        rem = lxi - li

        # Correction for right endpoint (not included if li == M)
        if X[i] == b:
            li = M - 1
            rem = 1.0

        if 1 <= li < M:
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


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for interface consistency with the
    # other KernSmooth routines but, exactly as in the original R code,
    # are not actually used anywhere in the body of this function.
    x = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename common variables.
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned.
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    # Set the bin width.
    delta = (b - a) / (M - 1)

    if bandwidth is None:
        raise ValueError('argument "bandwidth" is missing, with no default')
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    # Discretise the bandwidths.
    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc".
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of "hdisc" to each member of
        # "bandwidth".
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
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Allocate space for the kernel vector and final estimate.
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    Sdg = np.zeros(M, dtype=np.float64)

    # ---- Equivalent to the Fortran subroutine "sdiag" (src/sdiag.f) ----
    # No .Fortran(F_sdiag) conversion guide is available for this build, so
    # the Fortran "sdiag" routine is reimplemented directly in NumPy from
    # the original source. It computes the diagonal (self-influence)
    # entries of the local polynomial "hat"/smoother matrix. "mid", "i",
    # "j", "k" and "ii" below are kept as 1-based Fortran-style
    # loop/bookkeeping indices (as in the original routine) and are only
    # converted to 0-based numpy indices at the point of array access.

    # Obtain kernel weights.
    mid = int(Lvec[0]) + 1
    for i in range(1, Q):  # Fortran: do i = 1, (Q - 1)
        midpts[i - 1] = mid
        fkap[mid - 1] = 1.0
        Li = int(Lvec[i - 1])
        hi = hdisc[i - 1]
        for j in range(1, Li + 1):
            val = np.exp(-((delta * j / hi) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    LQ = int(Lvec[Q - 1])
    hQ = hdisc[Q - 1]
    for j in range(1, LQ + 1):
        val = np.exp(-((delta * j / hQ) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    # Combine kernel weights and grid counts.
    for k in range(1, M + 1):
        xk = xcounts[k - 1]
        if xk != 0:
            for i in range(1, Q + 1):
                Li = int(Lvec[i - 1])
                mp = int(midpts[i - 1])
                jlo = max(1, k - Li)
                jhi = min(M, k + Li)
                for j in range(jlo, jhi + 1):
                    if indic[j - 1] == i:
                        fkval = fkap[k - j + mp - 1]
                        ss[j - 1, 0] += xk * fkval
                        fac = 1.0
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xk * fkval * fac

    # For each grid point, build the (Hankel) system matrix "Smat" from
    # "ss", invert it (replacing the original LINPACK dgefa/dgedi calls,
    # which factor "Smat" in place and then overwrite it with its
    # inverse), and take the (1, 1) entry as the diagonal smoother value.
    idx_pp = np.arange(pp)
    hankel_idx = idx_pp[:, None] + idx_pp[None, :]

    for k in range(M):
        Smat = ss[k][hankel_idx]
        try:
            Sinv = np.linalg.inv(Smat)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(Smat)
        Sdg[k] = Sinv[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", *, bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For computing the binned diagonal entries of SS^T where S is a
    # smoother matrix for local polynomial kernel regression.
    #
    # NOTE: No .Fortran(F_sstdg) conversion guide is available for this
    # build, so the Fortran "sstdg" routine (KernSmooth src/sstdiag.f) is
    # reimplemented directly in NumPy from the original Fortran source.
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
    bw_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bw_arr.size == M:
        sorted_bw = np.sort(bw_arr)
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
                indic = np.round(((np.log(bw_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bw_arr.size == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bw_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bw_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    SSTd = np.zeros(M, dtype=np.float64)

    # Obtain kernel weights.
    # "mid" is kept as a 0-based index into "fkap" throughout (the Fortran
    # routine uses a 1-based index "mid = Lvec(1) + 1"; subtracting 1 gives
    # the 0-based baseline used below, and all offsets "+j"/"-j" carry
    # over unchanged since they are index-shift invariant).
    mid = int(Lvec[0])
    for i_f in range(1, Q):
        i_p = i_f - 1
        midpts[i_p] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec[i_p]) + 1):
            fkap[mid + j] = np.exp(-((delta * j / hdisc[i_p]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lvec[i_p]) + int(Lvec[i_p + 1]) + 1
    midpts[Q - 1] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        fkap[mid + j] = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts.
    # (k_p, j_p, i_p are the 0-based counterparts of the Fortran 1-based
    # loop indices k, j, i; "k - j" is invariant under this shift.)
    for k_p in range(M):
        if xcounts[k_p] != 0:
            for i_p in range(Q):
                Li = int(Lvec[i_p])
                lo = max(0, k_p - Li)
                hi = min(M - 1, k_p + Li)
                for j_p in range(lo, hi + 1):
                    if int(indic[j_p]) == i_p + 1:
                        fkap_val = fkap[k_p - j_p + int(midpts[i_p])]
                        fac = 1.0
                        ss[j_p, 0] += xcounts[k_p] * fkap_val
                        uu[j_p, 0] += xcounts[k_p] * fkap_val ** 2
                        for ii_p in range(1, ppp):
                            fac = fac * delta * (k_p - j_p)
                            ss[j_p, ii_p] += xcounts[k_p] * fkap_val * fac
                            uu[j_p, ii_p] += xcounts[k_p] * (fkap_val ** 2) * fac

    # For each grid point, assemble the local "Smat"/"Umat" moment
    # matrices from "ss"/"uu", invert "Smat" (equivalent to the
    # LINPACK dgefa/dgedi calls with job = 01, which factor and then
    # invert Smat in place), and form the SS^T diagonal entry as the
    # quadratic form e1^T Smat^{-1} Umat Smat^{-1} e1.
    for k_p in range(M):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Umat = np.zeros((pp, pp), dtype=np.float64)
        for i_p in range(pp):
            for j_p in range(pp):
                col = i_p + j_p
                Smat[i_p, j_p] = ss[k_p, col]
                Umat[i_p, j_p] = uu[k_p, col]

        Sinv = np.linalg.inv(Smat)
        SSTd[k_p] = float(Sinv[0, :] @ Umat @ Sinv[:, 0])

    return {"x": gpoints, "y": SSTd}


_on_attach()
