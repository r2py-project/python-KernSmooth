import math
import sys
from typing import Any
import warnings

import numpy as np
from numpy.polynomial import polynomial as P
from scipy.stats import beta, norm

from . import _KernSmooth


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    # n <- length(X); M <- length(gpoints)
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    n = len(X)
    M = len(gpoints)

    # trun <- if (truncate) 1L else 0L
    trun = 1 if truncate else 0

    # a <- gpoints[1L]; b <- gpoints[M]
    a = gpoints[0]
    b = gpoints[M - 1]

    # Pure Python/NumPy re-implementation of the Fortran `linbin` subroutine
    # (KernSmooth/src/linbin.f), replacing the .Fortran(F_linbin, ...) call.
    gcnts = np.zeros(M, dtype=np.float64)

    if n == 0:
        return gcnts

    delta = (b - a) / (M - 1)

    # lxi <- ((X(i)-a)/delta) + 1  (1-based grid coordinate)
    lxi = (X - a) / delta + 1.0

    # li <- int(lxi)  -- Fortran INT truncates toward zero, matching np.trunc
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # if (li.ge.1.and.li.lt.M) then
    #    gcnts(li) = gcnts(li) + (1-rem)
    #    gcnts(li+1) = gcnts(li+1) + rem
    # endif
    mask_mid = (li >= 1) & (li < M)
    idx_mid = li[mask_mid] - 1  # convert 1-based Fortran index to 0-based
    rem_mid = rem[mask_mid]
    np.add.at(gcnts, idx_mid, 1.0 - rem_mid)
    np.add.at(gcnts, idx_mid + 1, rem_mid)

    if trun == 0:
        # if (li.lt.1.and.trun.eq.0) then gcnts(1) = gcnts(1) + 1
        gcnts[0] += np.count_nonzero(li < 1)
        # if (li.ge.M.and.trun.eq.0) then gcnts(M) = gcnts(M) + 1
        gcnts[M - 1] += np.count_nonzero(li >= M)

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    # n <- nrow(X); X <- c(X[, 1L], X[, 2L])
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)
    n = X.shape[0]
    x1 = X[:, 0]
    x2 = X[:, 1]

    # M1 <- length(gpoints1); M2 <- length(gpoints2)
    M1 = len(gpoints1)
    M2 = len(gpoints2)

    # a1 <- gpoints1[1L]; a2 <- gpoints2[1L]
    # b1 <- gpoints1[M1]; b2 <- gpoints2[M2]
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Pure Python/NumPy re-implementation of the Fortran `lbtwod` subroutine
    # (KernSmooth/src/linbin2D.f), replacing the .Fortran(F_lbtwod, ...) call.
    # This version always ignores (truncates) observations outside the mesh,
    # matching the Fortran comment: 'observations outside the mesh are ignored'.
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    if n == 0:
        return gcnts

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    # lxi1 <- ((X(i)-a1)/delta1) + 1 ; lxi2 <- ((X(n+i)-a2)/delta2) + 1
    lxi1 = (x1 - a1) / delta1 + 1.0
    lxi2 = (x2 - a2) / delta2 + 1.0

    # li1 <- int(lxi1); li2 <- int(lxi2)  -- Fortran INT truncates toward zero,
    # matching np.trunc
    li1 = np.trunc(lxi1).astype(np.int64)
    li2 = np.trunc(lxi2).astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    # if (li1.ge.1.and.li2.ge.1.and.li1.lt.M1.and.li2.lt.M2) then
    #    ind1 = (li1,   li2)   -> gcnts(ind1) += (1-rem1)*(1-rem2)
    #    ind2 = (li1+1, li2)   -> gcnts(ind2) += rem1*(1-rem2)
    #    ind3 = (li1,   li2+1) -> gcnts(ind3) += (1-rem1)*rem2
    #    ind4 = (li1+1, li2+1) -> gcnts(ind4) += rem1*rem2
    # endif
    mask = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)
    i1 = li1[mask] - 1  # convert 1-based Fortran index to 0-based
    i2 = li2[mask] - 1
    r1 = rem1[mask]
    r2 = rem2[mask]

    np.add.at(gcnts, (i1, i2), (1.0 - r1) * (1.0 - r2))
    np.add.at(gcnts, (i1 + 1, i2), r1 * (1.0 - r2))
    np.add.at(gcnts, (i1, i2 + 1), (1.0 - r1) * r2)
    np.add.at(gcnts, (i1 + 1, i2 + 1), r1 * r2)

    return gcnts


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # n <- length(X); M <- length(gpoints)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    n = len(X)
    M = len(gpoints)

    # trun <- if (truncate) 1L else 0L
    trun = 1 if truncate else 0

    # a <- gpoints[1L]; b <- gpoints[M]
    a = gpoints[0]
    b = gpoints[M - 1]

    # Pure Python/NumPy re-implementation of the Fortran `rlbin` subroutine
    # (KernSmooth/src/rlbin.f), replacing the .Fortran(F_rlbin, ...) call.
    # It performs linear binning of a bivariate (X, Y) regression data set:
    # xcounts are the linearly-binned grid counts of X (identical to `linbin`),
    # and ycounts are the linearly-binned grid sums of the corresponding Y values,
    # using the same fractional grid weights derived from X.
    xcounts = np.zeros(M, dtype=np.float64)
    ycounts = np.zeros(M, dtype=np.float64)

    if n == 0:
        return {"xcounts": xcounts, "ycounts": ycounts}

    delta = (b - a) / (M - 1)

    # lxi <- ((X(i)-a)/delta) + 1  (1-based grid coordinate)
    lxi = (X - a) / delta + 1.0

    # li <- int(lxi)  -- Fortran INT truncates toward zero, matching np.trunc
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Correction for right endpoint (not included if li.eq.M):
    # if (X(i).eq.b) then li = M - 1; rem = 1; endif
    mask_b = (X == b)
    li = np.where(mask_b, M - 1, li)
    rem = np.where(mask_b, 1.0, rem)

    # if (li.ge.1.and.li.lt.M) then
    #    xcounts(li) = xcounts(li) + (1-rem);   xcounts(li+1) = xcounts(li+1) + rem
    #    ycounts(li) = ycounts(li) + (1-rem)*Y(i); ycounts(li+1) = ycounts(li+1) + rem*Y(i)
    # endif
    mask_mid = (li >= 1) & (li < M)
    idx_mid = li[mask_mid] - 1  # convert 1-based Fortran index to 0-based
    rem_mid = rem[mask_mid]
    Y_mid = Y[mask_mid]
    np.add.at(xcounts, idx_mid, 1.0 - rem_mid)
    np.add.at(xcounts, idx_mid + 1, rem_mid)
    np.add.at(ycounts, idx_mid, (1.0 - rem_mid) * Y_mid)
    np.add.at(ycounts, idx_mid + 1, rem_mid * Y_mid)

    if trun == 0:
        # elseif (li.lt.1.and.trun.eq.0) then
        #    xcounts(1) = xcounts(1) + 1; ycounts(1) = ycounts(1) + Y(i)
        mask_lo = li < 1
        xcounts[0] += np.count_nonzero(mask_lo)
        ycounts[0] += np.sum(Y[mask_lo])

        # elseif (li.ge.M.and.trun.eq.0) then
        #    xcounts(M) = xcounts(M) + 1; ycounts(M) = ycounts(M) + Y(i)
        mask_hi = li >= M
        xcounts[M - 1] += np.count_nonzero(mask_hi)
        ycounts[M - 1] += np.sum(Y[mask_hi])

    return {"xcounts": xcounts, "ycounts": ycounts}


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    # if (!missing(bandwidth) && bandwidth <= 0)
    #     stop("'bandwidth' must be strictly positive")
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # kernel <- match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        _matches = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(_matches) == 1:
            kernel = _matches[0]
        else:
            raise ValueError(
                f"'kernel' should be one of {_kernel_choices}"
            )

    x = np.asarray(x, dtype=np.float64)

    ## Rename common variables
    # n <- length(x); M <- gridsize
    n = len(x)
    M = gridsize

    ## Set canonical scaling factors
    # del0 <- switch(kernel,
    #                "normal" = (1/(4*pi))^(1/10),
    #                "box" = (9/2)^(1/5),
    #                "epanech" = 15^(1/5),
    #                "biweight" = 35^(1/5),
    #                "triweight" = (9450/143)^(1/5))
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

    # if (length(canonical) != 1L || !is.logical(canonical))
    #     stop("'canonical' must be a length-1 logical vector")
    if not isinstance(canonical, bool):
        raise ValueError("'canonical' must be a length-1 logical vector")

    ## Set default bandwidth
    # h <- if (missing(bandwidth)) del0 * (243/(35*n))^(1/5)*sqrt(var(x))
    # else if(canonical) del0 * bandwidth else bandwidth
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.sqrt(np.var(x, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    ## Set kernel support values
    # tau <-  if (kernel == "normal") 4 else 1
    tau = 4 if kernel == "normal" else 1

    # if (missing(range.x)) range.x <- c(min(x)-tau*h, max(x)+tau*h)
    if range_x is None:
        range_x = (np.min(x) - tau * h, np.max(x) + tau * h)
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data
    # gpoints <- seq(a, b, length.out = M)
    # gcounts <- linbin(x, gpoints, truncate)
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute kernel weights
    # delta  <- (b - a)/(h * (M-1L))
    delta = (b - a) / (h * (M - 1))

    # L <- min(floor(tau/delta), M)
    L = min(int(np.floor(tau / delta)), M)

    # if (L == 0)
    #     warning("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # lvec <- 0L:L
    lvec = np.arange(0, L + 1)

    ## Compute kernel weights
    # kappa <- if (kernel == "normal") dnorm(lvec*delta)/(n*h)
    #          else if (kernel == "box") 0.5*dbeta(0.5*(lvec*delta+1), 1, 1)/(n*h)
    #          else if (kernel == "epanech") 0.5*dbeta(0.5*(lvec*delta+1), 2, 2)/(n*h)
    #          else if (kernel == "biweight") 0.5*dbeta(0.5*(lvec*delta+1), 3, 3)/(n*h)
    #          else if (kernel == "triweight") 0.5*dbeta(0.5*(lvec*delta+1), 4, 4)/(n*h)
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
    # P <- 2^(ceiling(log(M+L+1L)/log(2)))
    P = int(2 ** np.ceil(np.log2(M + L + 1)))

    # kappa <- c(kappa, rep(0, P-2L*L-1L), rev(kappa[-1L]))
    kappa = np.concatenate([
        kappa,
        np.zeros(P - 2 * L - 1, dtype=np.float64),
        kappa[1:][::-1],
    ])

    # tot <- sum(kappa) * (b-a)/(M-1L) * n # should have total weight one
    tot = np.sum(kappa) * (b - a) / (M - 1) * n

    # gcounts <- c(gcounts, rep(0L, P-M))
    gcounts = np.concatenate([gcounts, np.zeros(P - M, dtype=np.float64)])

    # kappa <- fft(kappa/tot)
    # gcounts <- fft(gcounts)
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    # list(x = gpoints, y = (Re(fft(kappa*gcounts, TRUE))/P)[1L:M])
    y = np.fft.ifft(kappa * gcounts).real[:M]
    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: list[tuple[float, float]] | tuple[tuple[float, float], tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    ## Install safeguard against non-positive bandwidths:
    # if (!missing(bandwidth) && min(bandwidth) <= 0)
    #     stop("'bandwidth' must be strictly positive")
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x = np.asarray(x, dtype=np.float64)

    ## Rename common variables
    # n <- nrow(x); M <- gridsize; h <- bandwidth
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    ## Use same bandwidth in each direction
    ## if only a single bandwidth is given.
    # if (length(h) == 1L) h <- c(h, h)
    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    ## If range.x is not specified then set it at its default value.
    # if (missing(range.x)) {
    #     range.x <- list(0, 0)
    #     for (id in (1L:2L))
    #         range.x[[id]] <- c(min(x[, id])-1.5*h[id], max(x[, id])+1.5*h[id])
    # }
    if range_x is None:
        range_x = [None, None]
        for id_ in range(2):
            range_x[id_] = (
                float(np.min(x[:, id_]) - 1.5 * h[id_]),
                float(np.max(x[:, id_]) + 1.5 * h[id_]),
            )

    # a <- c(range.x[[1L]][1L], range.x[[2L]][1L])
    # b <- c(range.x[[1L]][2L], range.x[[2L]][2L])
    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    ## Set up grid points and bin the data
    # gpoints1 <- seq(a[1L], b[1L], length.out = M[1L])
    # gpoints2 <- seq(a[2L], b[2L], length.out = M[2L])
    # gcounts <- linbin2D(x, gpoints1, gpoints2)
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    ## Compute kernel weights
    # L <- numeric(2L); kapid <- list(0, 0)
    # for (id in 1L:2L) {
    #     L[id] <- min(floor(tau*h[id]*(M[id]-1)/(b[id]-a[id])), M[id] - 1L)
    #     lvecid <- 0:L[id]
    #     facid <- (b[id] - a[id])/(h[id]*(M[id]-1L))
    #     z <- matrix(dnorm(lvecid*facid)/h[id])
    #     tot <- sum(c(z, rev(z[-1L]))) * facid * h[id]
    #     kapid[[id]] <- z/tot
    # }
    # kapp <- kapid[[1L]] %*% (t(kapid[[2L]]))/n
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [None, None]  # type: ignore[list-item]
    for id_ in range(2):
        Lid = min(
            int(np.floor(tau * h[id_] * (M[id_] - 1) / (b[id_] - a[id_]))),
            int(M[id_]) - 1,
        )
        L[id_] = Lid
        lvecid = np.arange(0, Lid + 1)
        facid = (b[id_] - a[id_]) / (h[id_] * (M[id_] - 1))
        z = norm.pdf(lvecid * facid) / h[id_]
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * facid * h[id_]
        kapid[id_] = z / tot

    # kapid[[id]] is a column vector in R and kapp is the outer product
    # kapid[[1L]] %*% t(kapid[[2L]]) / n; using 1-D vectors here is
    # numerically equivalent and simpler.
    kapp = np.outer(kapid[0], kapid[1]) / n

    # if (min(L) == 0)
    #     warning("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    ## Now combine weight and counts using the FFT to obtain estimate
    # P <- 2^(ceiling(log(M+L)/log(2)))   # smallest powers of 2 >= M+L
    P = (2 ** np.ceil(np.log2(M.astype(np.float64) + L.astype(np.float64)))).astype(
        np.int64
    )
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    # rp <- matrix(0, P1, P2)
    # rp[1L:(L1+1), 1L:(L2+1)] <- kapp
    # if (L1) rp[(P1-L1+1):P1, 1L:(L2+1)] <- kapp[(L1+1):2, 1L:(L2+1)]
    # if (L2) rp[, (P2-L2+1):P2] <- rp[, (L2+1):2]
    ## wrap-around version of "kapp"
    rp = np.zeros((P1, P2), dtype=np.float64)
    rp[: L1 + 1, : L2 + 1] = kapp
    if L1:
        rp[P1 - L1 : P1, : L2 + 1] = kapp[L1:0:-1, : L2 + 1]
    if L2:
        rp[:, P2 - L2 : P2] = rp[:, L2:0:-1]

    # sp <- matrix(0, P1, P2)
    # sp[1L:M1, 1L:M2] <- gcounts
    ## zero-padded version of "gcounts"
    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[:M1, :M2] = gcounts

    # rp <- fft(rp)                       # Obtain FFT's of r and s
    # sp <- fft(sp)
    # rp <- Re(fft(rp*sp, inverse = TRUE)/(P1*P2))[1L:M1, 1L:M2]
    ## invert element-wise product of FFT's
    ## and truncate and normalise it
    rp = np.fft.fft2(rp)
    sp = np.fft.fft2(sp)
    rp = np.fft.ifft2(rp * sp).real[:M1, :M2]

    ## Ensure that rp is non-negative
    # rp <- rp * matrix(as.numeric(rp>0), nrow(rp), ncol(rp))
    rp = np.where(rp > 0, rp, 0.0)

    # list(x1 = gpoints1, x2 = gpoints2, fhat = rp)
    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    # if (!missing(bandwidth) && bandwidth <= 0)
    #     stop("'bandwidth' must be strictly positive")
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # if (missing(range.x) && !binned) range.x <- c(min(x), max(x))
    x = np.asarray(x, dtype=np.float64)
    if range_x is None and not binned:
        range_x = np.array([np.min(x), np.max(x)])

    # Rename variables
    # M <- gridsize; a <- range.x[1L]; b <- range.x[2L]; h <- bandwidth
    M = gridsize
    a = range_x[0]
    b = range_x[1]
    h = bandwidth

    # Bin the data if not already binned
    if not binned:
        # gpoints <- seq(a, b, length.out = gridsize)
        # gcounts <- linbin(x, gpoints, truncate)
        gpoints = np.linspace(a, b, M)
        gcounts = linbin(x, gpoints, truncate)
    else:
        # gcounts <- x; M <- length(gcounts); gpoints <- seq(a, b, length.out = M)
        gcounts = np.asarray(x, dtype=np.float64)
        M = len(gcounts)
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width
    # n <- sum(gcounts); delta <- (b-a)/(M-1)
    n = np.sum(gcounts)
    delta = (b - a) / (M - 1)

    # Obtain kernel weights
    # tau <- 4 + drv; L <- min(floor(tau*h/delta), M)
    tau = 4 + drv
    L = min(int(np.floor(tau * h / delta)), M)

    # if (L == 0)
    #     warning("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # lvec <- 0L:L; arg <- lvec*delta/h
    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    # kappam <- dnorm(arg)/(h^(drv+1))
    kappam = norm.pdf(arg) / (h ** (drv + 1))

    # hmold0 <- 1; hmold1 <- arg; hmnew <- 1
    hmold0 = 1.0
    hmold1 = arg
    hmnew = 1.0

    # if (drv >= 2L)
    #     for (i in (2L:drv)) {
    #         hmnew <- arg*hmold1 - (i-1)*hmold0
    #         hmold0 <- hmold1       # Compute mth degree Hermite polynomial
    #         hmold1 <- hmnew        # by recurrence.
    #     }
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0
            hmold0 = hmold1
            hmold1 = hmnew

    # kappam <- hmnew * kappam
    kappam = hmnew * kappam

    ## Now combine weights and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    # P <- 2^(ceiling(log(M+L+1L)/log(2)))
    P = int(2 ** np.ceil(np.log2(M + L + 1)))

    # kappam <- c(kappam, rep(0,  P-2L*L-1L), rev(kappam[-1L]))
    # Gcounts <- c(gcounts, rep(0, P-M))
    kappam = np.concatenate([
        kappam,
        np.zeros(P - 2 * L - 1, dtype=np.float64),
        kappam[1:][::-1],
    ])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M, dtype=np.float64)])

    # kappam <- fft(kappam); Gcounts <- fft(Gcounts)
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    # sum(gcounts * (Re(fft(kappam*Gcounts, TRUE))/P)[1L:M] )/(n^2)
    conv = np.real(np.fft.ifft(kappam * Gcounts))[:M]
    return float(np.sum(gcounts * conv) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # n <- length(x)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    # datmat <- cbind(x, y); datmat <- datmat[sort.list(datmat[, 1L]), ]
    # x <- datmat[, 1L]; y <- datmat[, 2L]
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Set up arrays for FORTRAN programme "blkest"
    # qq <- q + 1L
    qq = q + 1

    # Pure Python/NumPy re-implementation of the Fortran `blkest` subroutine
    # (KernSmooth/src/blkest.f), replacing the .Fortran(F_blkest, ...) call.
    # It divides the sorted (x, y) data into `Nval` roughly-equal contiguous
    # blocks (block sizes given by integer division idiv = n %/% Nval, with
    # the final block absorbing any remainder), fits a q'th degree polynomial
    # by least squares within each block (equivalent to the Fortran QR-based
    # fit via dqrdc/dqrsl), and accumulates the residual sum of squares plus
    # the 2nd- and 4th-derivative-based quantities (th22e, th24e) used by the
    # direct plug-in bandwidth selector of Ruppert, Sheather and Wand.
    #
    # NOTE: The Fortran code's per-point derivative accumulation
    #   ddm   = 2*coef(3)  + sum_{k=2}^{q-1} k*(k+1)*coef(k+2)*Xj(i)**(k-1)
    #   ddddm = 24*coef(5) + sum_{k=2}^{q-3} k*(k+1)*(k+2)*(k+3)*coef(k+4)*Xj(i)**(k-1)
    # is algebraically identical to evaluating the exact 2nd and 4th
    # derivatives of the fitted degree-q polynomial (with coefficients
    # coef[0..q] in increasing power order) at Xj(i); this is computed
    # below via numpy.polynomial.polynomial.polyder/polyval instead of the
    # explicit index-shifted Fortran loop.
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0
    idiv = n // Nval

    for j in range(Nval):
        # ilow <- (j-1)*idiv + 1; iupp <- j*idiv (1-based Fortran indices)
        # if (j.eq.Nval) iupp <- n
        ilow0 = j * idiv
        iupp0 = (j + 1) * idiv - 1
        if j == Nval - 1:
            iupp0 = n - 1

        # Xj(k) <- X(ilow+k-1); Yj(k) <- Y(ilow+k-1)
        Xj = x[ilow0:iupp0 + 1]
        Yj = y[ilow0:iupp0 + 1]

        # Obtain a q'th degree fit over the current member of the partition.
        # Set up "X" matrix: Xmat(i,1) = 1; Xmat(i,k) = Xj(i)**(k-1), k=2..qq
        Xmat = np.vander(Xj, N=qq, increasing=True)

        # QR-based least-squares fit (call dqrdc; call dqrsl), equivalent to
        # ordinary least-squares regression of Yj on Xmat.
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        # fiti is the fitted polynomial value; ddm/ddddm are its 2nd- and
        # 4th-derivative values, evaluated at each Xj(i).
        fiti = P.polyval(Xj, coef)
        ddm = P.polyval(Xj, P.polyder(coef, m=2))
        ddddm = P.polyval(Xj, P.polyder(coef, m=4))

        # th22e <- th22e + ddm**2; th24e <- th24e + ddm*ddddm
        # RSS <- RSS + (Yj(i)-fiti)**2
        th22e += float(np.sum(ddm ** 2))
        th24e += float(np.sum(ddm * ddddm))
        RSS += float(np.sum((Yj - fiti) ** 2))

    # sigsqe <- RSS/(n-qq*Nval); th22e <- th22e/n; th24e <- th24e/n
    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    # list(sigsqe = out[[13]], th22e = out[[14]], th24e = out[[15]])
    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # n <- length(X)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    # Sort the (X, Y) data with respect to the X's.
    # datmat <- cbind(X, Y); datmat <- datmat[sort.list(datmat[, 1L]), ]
    # X <- datmat[, 1L]; Y <- datmat[, 2L]
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Set up arrays for FORTRAN subroutine "cp"
    # qq <- q + 1L
    qq = q + 1

    # Pure Python/NumPy re-implementation of the Fortran `cp` subroutine
    # (KernSmooth/src/cp.f), replacing the .Fortran(F_cp, ...) call.
    #
    # For each candidate number of blocks Nval = 1, ..., Nmax, the sorted
    # (x, y) data are divided into Nval contiguous blocks (block sizes given
    # by integer division idiv = n // Nval, with the final block absorbing
    # any remainder), a q'th degree polynomial is fit by least squares within
    # each block (equivalent to the Fortran QR-based fit via dqrdc/dqrsl),
    # and the residual sums of squares are accumulated across blocks into
    # RSS[Nval - 1]. Afterwards, Mallow's C_p statistic is computed for each
    # Nval using RSS at Nval == Nmax as the estimate of the true residual
    # variance, per Mallow's C_p methodology.
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # idiv <- n %/% Nval
        idiv = n // Nval
        RSS_Nval = 0.0

        for j in range(1, Nval + 1):
            # For each member of the partition (1-based Fortran indices)
            # ilow <- (j-1)*idiv + 1; iupp <- j*idiv
            # if (j.eq.Nval) iupp <- n
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n

            # Xj(k) <- X(ilow+k-1); Yj(k) <- Y(ilow+k-1)
            Xj = x[ilow - 1:iupp]
            Yj = y[ilow - 1:iupp]

            # Obtain a q'th degree fit over the current member of the partition.
            # Set up "X" matrix: Xmat(i,1) = 1; Xmat(i,k) = Xj(i)**(k-1), k=2..qq
            Xmat = np.vander(Xj, N=qq, increasing=True)

            # QR-based least-squares fit (call dqrdc; call dqrsl), equivalent to
            # ordinary least-squares regression of Yj on Xmat.
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            # fiti <- coef(1) + sum_{k=2}^{qq} coef(k)*Xj(i)**(k-1)
            # RSSj <- RSSj + (Yj(i)-fiti)**2
            fiti = Xmat @ coef
            RSSj = float(np.sum((Yj - fiti) ** 2))

            # RSS(Nval) <- RSS(Nval) + RSSj
            RSS_Nval += RSSj

        RSS[Nval - 1] = RSS_Nval

    # Now compute array of Mallow's C_p values.
    # Cpvals(i) <- ((n-qq*Nmax)*RSS(i)/RSS(Nmax)) + 2*qq*i - n , i = 1, ..., Nmax
    i_vals = np.arange(1, Nmax + 1, dtype=np.float64)
    Cpvec = ((n - qq * Nmax) * RSS / RSS[Nmax - 1]) + 2 * qq * i_vals - n

    # order(Cpvec)[1L]: R's order() returns the 1-based position of the
    # minimum element in Cpvec. Since Cpvec is indexed 1..Nmax by block
    # count Nval, that 1-based position is itself the chosen block count.
    # np.argmin gives the equivalent 0-based index, so add 1 to recover the
    # same numeric block-count value that R returns.
    best_idx = int(np.argmin(Cpvec))
    return best_idx + 1


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    # if (level > 5L) stop("Level should be between 0 and 5")
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    ## Rename variables
    # n <- length(x); M <- gridsize; a <- range.x[1L]; b <- range.x[2L]
    n = len(x)
    M = gridsize
    # range.x = range(x) default: c(min(x), max(x))
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data
    # gpoints <- seq(a, b, length.out = M)
    # gcounts <- linbin(x, gpoints, truncate)
    # (This first binning result is never used below -- it is immediately
    #  overwritten by the standardised-data binning that follows -- so it is
    #  omitted here as dead code with no observable effect on the output.)

    ## Compute scale estimate
    # scalest <- match.arg(scalest, c("minim", "stdev", "iqr"))
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        _matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(_matches) == 1:
            scalest = _matches[0]
        else:
            raise ValueError(
                f"'scalest' should be one of {_scalest_choices}"
            )

    # scalest <- switch(scalest,
    #                   "stdev" = sqrt(var(x)),
    #                   "iqr"= (quantile(x, 3/4)-quantile(x, 1/4))/1.349,
    #                   "minim" = min((quantile(x, 3/4)-quantile(x, 1/4))/1.349, sqrt(var(x))) )
    if scalest == "stdev":
        scalest_val = float(np.std(x, ddof=1))
    elif scalest == "iqr":
        scalest_val = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    else:  # "minim"
        scalest_val = float(
            min(
                (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
                np.std(x, ddof=1),
            )
        )

    # if (scalest == 0) stop("scale estimate is zero for input data")
    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:
    # sx <- (x-mean(x))/scalest
    # sa <- (a-mean(x))/scalest ; sb <- (b-mean(x))/scalest
    mean_x = np.mean(x)
    sx = (x - mean_x) / scalest_val
    sa = (a - mean_x) / scalest_val
    sb = (b - mean_x) / scalest_val

    ## Set up grid points and bin the data:
    # gpoints <- seq(sa, sb, length.out = M)
    # gcounts <- linbin(sx, gpoints, truncate)
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    ## Perform plug-in steps
    range_sasb = np.array([sa, sb], dtype=np.float64)

    if level == 0:
        # hpi <- (24*sqrt(pi)/n)^(1/3)
        hpi = np.power(24.0 * np.sqrt(np.pi) / n, 1.0 / 3.0)
    elif level == 1:
        # alpha <- (2/(3*n))^(1/5)*sqrt(2)
        alpha = np.power(2.0 / (3.0 * n), 1.0 / 5.0) * np.sqrt(2.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=range_sasb, binned=True)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        hpi = np.power(6.0 / (-psi2hat * n), 1.0 / 3.0)
    elif level == 2:
        # alpha <- ((2/(5*n))^(1/7))*sqrt(2) # bandwidth for psi_4
        alpha = np.power(2.0 / (5.0 * n), 1.0 / 7.0) * np.sqrt(2.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=range_sasb, binned=True)
        # alpha <- (sqrt(2/pi)/(psi4hat*n))^(1/5) # bandwidth for psi_2
        alpha = np.power(np.sqrt(2.0 / np.pi) / (psi4hat * n), 1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=range_sasb, binned=True)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        hpi = np.power(6.0 / (-psi2hat * n), 1.0 / 3.0)
    elif level == 3:
        # alpha <- ((2/(7*n))^(1/9))*sqrt(2) # bandwidth for psi_6
        alpha = np.power(2.0 / (7.0 * n), 1.0 / 9.0) * np.sqrt(2.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        alpha = np.power(-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n), 1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=range_sasb, binned=True)
        # alpha <- (sqrt(2/pi)/(psi4hat*n))^(1/5) # bandwidth for psi_2
        alpha = np.power(np.sqrt(2.0 / np.pi) / (psi4hat * n), 1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=range_sasb, binned=True)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        hpi = np.power(6.0 / (-psi2hat * n), 1.0 / 3.0)
    elif level == 4:
        # alpha <- ((2/(9*n))^(1/11))*sqrt(2) # bandwidth for psi_8
        alpha = np.power(2.0 / (9.0 * n), 1.0 / 11.0) * np.sqrt(2.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=range_sasb, binned=True)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9) # bandwidth for psi_6
        alpha = np.power(15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n), 1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        alpha = np.power(-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n), 1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=range_sasb, binned=True)
        # alpha <- (sqrt(2/pi)/(psi4hat*n))^(1/5) # bandwidth for psi_2
        alpha = np.power(np.sqrt(2.0 / np.pi) / (psi4hat * n), 1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=range_sasb, binned=True)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        hpi = np.power(6.0 / (-psi2hat * n), 1.0 / 3.0)
    else:  # level == 5
        # alpha <- ((2/(11*n))^(1/13))*sqrt(2) # bandwidth for psi_10
        alpha = np.power(2.0 / (11.0 * n), 1.0 / 13.0) * np.sqrt(2.0)
        psi10hat = bkfe(gcounts, 10, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-105*sqrt(2/pi)/(psi10hat*n))^(1/11) # bandwidth for psi_8
        alpha = np.power(-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n), 1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=range_sasb, binned=True)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9) # bandwidth for psi_6
        alpha = np.power(15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n), 1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        alpha = np.power(-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n), 1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=range_sasb, binned=True)
        # alpha <- (sqrt(2/pi)/(psi4hat*n))^(1/5) # bandwidth for psi_2
        alpha = np.power(np.sqrt(2.0 / np.pi) / (psi4hat * n), 1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=range_sasb, binned=True)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        hpi = np.power(6.0 / (-psi2hat * n), 1.0 / 3.0)

    # scalest * hpi
    return float(scalest_val * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    # if (level > 5L) stop("Level should be between 0 and 5")
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    # kernel <- match.arg(kernel, c("normal", "box", "epanech", "biweight", "triweight"))
    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        _matches = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(_matches) == 1:
            kernel = _matches[0]
        else:
            raise ValueError(
                f"'kernel' should be one of {_kernel_choices}"
            )

    ## Set kernel constants
    # del0 <- if (canonical) 1 else switch(kernel,
    #                                      "normal" = 1/((4*pi)^(1/10)),
    #                                      "box" = (9/2)^(1/5),
    #                                      "epanech" = 15^(1/5),
    #                                      "biweight" = 35^(1/5),
    #                                      "triweight" = (9450/143)^(1/5))
    if canonical:
        del0 = 1.0
    elif kernel == "normal":
        del0 = 1.0 / np.power(4.0 * np.pi, 1.0 / 10.0)
    elif kernel == "box":
        del0 = np.power(9.0 / 2.0, 1.0 / 5.0)
    elif kernel == "epanech":
        del0 = np.power(15.0, 1.0 / 5.0)
    elif kernel == "biweight":
        del0 = np.power(35.0, 1.0 / 5.0)
    else:  # "triweight"
        del0 = np.power(9450.0 / 143.0, 1.0 / 5.0)

    x = np.asarray(x, dtype=np.float64)

    ## Rename variables
    # n <- length(x); M <- gridsize; a <- range.x[1L]; b <- range.x[2L]
    n = len(x)
    M = gridsize
    # range.x = range(x) default: c(min(x), max(x))
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data
    # gpoints <- seq(a, b, length.out = M)
    # gcounts <- linbin(x, gpoints, truncate)
    # (This first binning result is never used below -- it is immediately
    #  overwritten by the standardised-data binning that follows -- so it is
    #  omitted here as dead code with no observable effect on the output.)

    ## Compute scale estimate
    # scalest <- match.arg(scalest, c("minim", "stdev", "iqr"))
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        _matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(_matches) == 1:
            scalest = _matches[0]
        else:
            raise ValueError(
                f"'scalest' should be one of {_scalest_choices}"
            )

    # scalest <- switch(scalest,
    #                   "stdev" = sqrt(var(x)),
    #                   "iqr"= (quantile(x, 3/4)-quantile(x, 1/4))/1.349,
    #                   "minim" = min((quantile(x, 3/4)-quantile(x, 1/4))/1.349, sqrt(var(x))) )
    if scalest == "stdev":
        scalest_val = float(np.std(x, ddof=1))
    elif scalest == "iqr":
        scalest_val = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
    else:  # "minim"
        scalest_val = float(
            min(
                (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
                np.std(x, ddof=1),
            )
        )

    # if (scalest == 0) stop("scale estimate is zero for input data")
    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:
    # sx <- (x-mean(x))/scalest
    # sa <- (a-mean(x))/scalest ; sb <- (b-mean(x))/scalest
    mean_x = np.mean(x)
    sx = (x - mean_x) / scalest_val
    sa = (a - mean_x) / scalest_val
    sb = (b - mean_x) / scalest_val

    ## Set up grid points and bin the data:
    # gpoints <- seq(sa, sb, length.out = M)
    # gcounts <- linbin(sx, gpoints, truncate)
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    ## Perform plug-in steps
    range_sasb = np.array([sa, sb], dtype=np.float64)

    if level == 0:
        # psi4hat <- 3/(8*sqrt(pi))
        psi4hat = 3.0 / (8.0 * np.sqrt(np.pi))
    elif level == 1:
        # alpha <- (2*(sqrt(2))^7/(5*n))^(1/7) # bandwidth for psi_4
        alpha = np.power(2.0 * np.power(np.sqrt(2.0), 7) / (5.0 * n), 1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=range_sasb, binned=True)
    elif level == 2:
        # alpha <- (2*(sqrt(2))^9/(7*n))^(1/9) # bandwidth for psi_6
        alpha = np.power(2.0 * np.power(np.sqrt(2.0), 9) / (7.0 * n), 1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        alpha = np.power(-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n), 1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=range_sasb, binned=True)
    elif level == 3:
        # alpha <- (2*(sqrt(2))^11/(9*n))^(1/11) # bandwidth for psi_8
        alpha = np.power(2.0 * np.power(np.sqrt(2.0), 11) / (9.0 * n), 1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=range_sasb, binned=True)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9) # bandwidth for psi_6
        alpha = np.power(15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n), 1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        alpha = np.power(-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n), 1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=range_sasb, binned=True)
    elif level == 4:
        # alpha <- (2*(sqrt(2))^13/(11*n))^(1/13) # bandwidth for psi_10
        alpha = np.power(2.0 * np.power(np.sqrt(2.0), 13) / (11.0 * n), 1.0 / 13.0)
        psi10hat = bkfe(gcounts, 10, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-105*sqrt(2/pi)/(psi10hat*n))^(1/11) # bandwidth for psi_8
        alpha = np.power(-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n), 1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=range_sasb, binned=True)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9) # bandwidth for psi_6
        alpha = np.power(15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n), 1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        alpha = np.power(-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n), 1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=range_sasb, binned=True)
    else:  # level == 5
        # alpha <- (2*(sqrt(2))^15/(13*n))^(1/15) # bandwidth for psi_12
        alpha = np.power(2.0 * np.power(np.sqrt(2.0), 15) / (13.0 * n), 1.0 / 15.0)
        psi12hat = bkfe(gcounts, 12, alpha, range_x=range_sasb, binned=True)
        # alpha <- (945*sqrt(2/pi)/(psi12hat*n))^(1/13) # bandwidth for psi_10
        alpha = np.power(945.0 * np.sqrt(2.0 / np.pi) / (psi12hat * n), 1.0 / 13.0)
        psi10hat = bkfe(gcounts, 10, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-105*sqrt(2/pi)/(psi10hat*n))^(1/11) # bandwidth for psi_8
        alpha = np.power(-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n), 1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=range_sasb, binned=True)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9) # bandwidth for psi_6
        alpha = np.power(15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n), 1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=range_sasb, binned=True)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        alpha = np.power(-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n), 1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=range_sasb, binned=True)

    # scalest * del0 * (1/(psi4hat*n))^(1/5)
    return float(scalest_val * del0 * np.power(1.0 / (psi4hat * n), 1.0 / 5.0))


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    # if (!missing(bandwidth) && any(bandwidth <= 0))
    #     stop("'bandwidth' must be strictly positive")
    if bandwidth is not None and np.any(np.asarray(bandwidth, dtype=np.float64) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    # drv <- as.integer(drv)
    # if (missing(degree)) degree <- drv + 1L else degree <- as.integer(degree)
    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x = np.asarray(x, dtype=np.float64)
    if y is not None:
        y = np.asarray(y, dtype=np.float64)

    # if (missing(range.x) && !binned)
    #     if (missing(y)) {
    #         extra <- 0.05*(max(x) - min(x))
    #         range.x <- c(min(x)-extra,  max(x)+extra)
    #     } else range.x <- c(min(x), max(x))
    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x) - np.min(x))
            range_x = (np.min(x) - extra, np.max(x) + extra)
        else:
            range_x = (np.min(x), np.max(x))

    ## Rename common variables
    # M <- gridsize; Q <- as.integer(bwdisc)
    # a <- range.x[1L]; b <- range.x[2L]
    # pp <- degree + 1L; ppp <- 2L*degree + 1L; tau <- 4
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Decide whether a density estimate or regression estimate is required.
    if y is None:
        # obtain density estimate
        # n <- length(x); gpoints <- seq(a, b, length.out = M)
        # xcounts <- linbin(x, gpoints, truncate)
        # ycounts <- (M-1)*xcounts/(n*(b-a))
        # xcounts <- rep(1, M)
        n = len(x)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:
        # obtain regression estimate
        if not binned:
            # gpoints <- seq(a, b, length.out = M)
            # out <- rlbin(x, y, gpoints, truncate)
            # xcounts <- out$xcounts; ycounts <- out$ycounts
            gpoints = np.linspace(a, b, M)
            out_bin = rlbin(x, y, gpoints, truncate)
            xcounts = out_bin["xcounts"]
            ycounts = out_bin["ycounts"]
        else:
            # xcounts <- x; ycounts <- y
            # M <- length(xcounts); gpoints <- seq(a, b, length.out = M)
            xcounts = x
            ycounts = y
            M = len(xcounts)
            gpoints = np.linspace(a, b, M)

    ## Set the bin width
    # delta <- (b-a)/(M-1L)
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if len(bandwidth_arr) == M:
        # hlow <- sort(bandwidth)[1L]; hupp <- sort(bandwidth)[M]
        # hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        # Lvec <- floor(tau*hdisc/delta)
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
        # indic <- if (Q > 1L) {
        #     lhdisc <- log(hdisc)
        #     gap <- (lhdisc[Q]-lhdisc[1L])/(Q-1)
        #     if (gap == 0) rep(1, M)
        #     else round(((log(bandwidth) - log(sort(bandwidth)[1L]))/gap) + 1)
        # } else rep(1, M)
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(hlow)) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        # indic <- rep(1, M); Q <- 1L
        # Lvec <- rep(floor(tau*bandwidth/delta), Q)
        # hdisc <- rep(bandwidth, Q)
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, math.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    # if (min(Lvec) == 0)
    #     stop("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    ## Direct re-implementation of the FORTRAN routine "locpol" (KernSmooth/src/locpoly.f).
    ## For each grid point j (0-based) with its discretised bandwidth level
    ## i = indic[j]-1 (Lvec[i], hdisc[i]), fit a weighted local polynomial of
    ## degree `degree` over the neighbouring bins k in [j-Lvec[i], j+Lvec[i]],
    ## using Gaussian kernel weights exp(-(delta*(k-j)/hdisc[i])**2/2), then
    ## extract the drv-th coefficient of the fit (equivalent to solving the
    ## weighted normal equations Smat %*% coefs = Tvec via LU decomposition,
    ## as done in Fortran via dgefa/dgesl).
    curvest = np.zeros(M, dtype=np.float64)
    powers = np.arange(ppp)
    idx_sum = np.add.outer(np.arange(pp), np.arange(pp))  # a+b for Smat lookup

    for j in range(M):
        i = indic[j] - 1
        L = int(Lvec[i])
        h = hdisc[i]

        k_lo = max(0, j - L)
        k_hi = min(M - 1, j + L)
        ks = np.arange(k_lo, k_hi + 1)

        dvals = (ks - j).astype(np.float64)
        u = delta * dvals
        weight = np.exp(-((u / h) ** 2) / 2.0)

        xw = xcounts[ks] * weight
        yw = ycounts[ks] * weight

        u_powers = u[:, None] ** powers[None, :]  # shape (len(ks), ppp)
        ss_j = (xw[:, None] * u_powers).sum(axis=0)  # length ppp, powers 0..2*degree
        tt_j = (yw[:, None] * u_powers[:, :pp]).sum(axis=0)  # length pp, powers 0..degree

        Smat = ss_j[idx_sum]
        Tvec = tt_j

        coefs = np.linalg.solve(Smat, Tvec)
        curvest[j] = coefs[drv]

    # curvest <- gamma(drv+1) * out[[19L]]
    curvest = math.gamma(drv + 1) * curvest

    # list(x = gpoints, y = curvest)
    return {"x": gpoints, "y": curvest}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # if (missing(range.x) && !binned) range.x <- c(min(x), max(x))
    x = np.asarray(x, dtype=np.float64)
    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename common variables
    # M <- gridsize; Q <- as.integer(bwdisc)
    # a <- range.x[1L]; b <- range.x[2L]
    # pp <- degree + 1L; ppp <- 2L*degree + 1L; tau <- 4
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    degree = int(degree)
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Bin the data if not already binned
    if not binned:
        # gpoints <- seq(a, b, length.out = M)
        # xcounts <- linbin(x, gpoints, truncate)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        # xcounts <- x; M <- length(xcounts)
        # gpoints <- seq(a, b, length.out = M)
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    # delta <- (b-a)/(M-1L)
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if len(bandwidth_arr) == M:
        # hlow <- sort(bandwidth)[1L]; hupp <- sort(bandwidth)[M]
        # hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        # Lvec <- floor(tau*hdisc/delta)
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
        # indic <- if (Q > 1L) {
        #     lhdisc <- log(hdisc)
        #     gap <- (lhdisc[Q]-lhdisc[1L])/(Q-1)
        #     if (gap == 0) rep(1, M)
        #     else round(((log(bandwidth) - log(sort(bandwidth)[1L]))/gap) + 1)
        # } else rep(1, M)
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(hlow)) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        # indic <- rep(1, M); Q <- 1L
        # Lvec <- rep(floor(tau*bandwidth/delta), Q)
        # hdisc <- rep(bandwidth, Q)
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    ## Direct re-implementation of the FORTRAN routine "sdiag" (KernSmooth/src/sdiag.f).
    ## For each grid point j (0-based) with its discretised bandwidth level
    ## i = indic[j]-1 (Lvec[i], hdisc[i]), accumulate the weighted moment vector
    ## ss_j[p] = sum_k xcounts[k] * exp(-(delta*(k-j)/hdisc[i])**2/2) * (delta*(k-j))**p
    ## for p = 0 .. 2*degree, over neighbouring bins k in [j-Lvec[i], j+Lvec[i]].
    ## Smat is then the (degree+1)x(degree+1) matrix built from ss_j, and the
    ## diagonal entry of the smoother matrix at j is Sdg[j] = (Smat^{-1})[0,0],
    ## i.e. the (1,1) Fortran entry after inverting via dgefa/dgedi -- note this
    ## does NOT depend on 'drv': the Fortran code always takes Smat(1,1), since
    ## the local design row evaluated at the point itself (u=0) is (1,0,...,0).
    Sdg = np.zeros(M, dtype=np.float64)
    powers = np.arange(ppp)
    idx_sum = np.add.outer(np.arange(pp), np.arange(pp))  # a+b for Smat lookup

    for j in range(M):
        i = indic[j] - 1
        L = int(Lvec[i])
        h = hdisc[i]

        k_lo = max(0, j - L)
        k_hi = min(M - 1, j + L)
        ks = np.arange(k_lo, k_hi + 1)

        dvals = (ks - j).astype(np.float64)
        u = delta * dvals
        weight = np.exp(-((u / h) ** 2) / 2.0)

        xw = xcounts[ks] * weight

        u_powers = u[:, None] ** powers[None, :]  # shape (len(ks), ppp)
        ss_j = (xw[:, None] * u_powers).sum(axis=0)  # length ppp, powers 0..2*degree

        Smat = ss_j[idx_sum]
        Smat_inv = np.linalg.inv(Smat)
        Sdg[j] = Smat_inv[0, 0]

    # list(x = gpoints, y = out[[17L]])
    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # if (missing(range.x) && !binned) range.x <- c(min(x), max(x))
    x = np.asarray(x, dtype=np.float64)
    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename common variables
    # M <- gridsize; Q <- as.integer(bwdisc)
    # a <- range.x[1L]; b <- range.x[2L]
    # pp <- degree + 1L; ppp <- 2L*degree + 1L; tau <- 4
    M = int(gridsize)
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    degree = int(degree)
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Bin the data if not already binned
    if not binned:
        # gpoints <- seq(a, b, length.out = M)
        # xcounts <- linbin(x, gpoints, truncate)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        # xcounts <- x; M <- length(xcounts)
        # gpoints <- seq(a, b, length.out = M)
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    # delta <- (b-a)/(M-1L)
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if len(bandwidth_arr) == M:
        # hlow <- sort(bandwidth)[1L]; hupp <- sort(bandwidth)[M]
        # hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        # Lvec <- floor(tau*hdisc/delta)
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
        # indic <- if (Q > 1L) {
        #     lhdisc <- log(hdisc)
        #     gap <- (lhdisc[Q]-lhdisc[1L])/(Q-1)
        #     if (gap == 0) rep(1, M)
        #     else round(((log(bandwidth) - log(sort(bandwidth)[1L]))/gap) + 1)
        # } else rep(1, M)
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(
                    ((np.log(bandwidth_arr) - np.log(hlow)) / gap) + 1
                ).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif len(bandwidth_arr) == 1:
        # indic <- rep(1, M); Q <- 1L
        # Lvec <- rep(floor(tau*bandwidth/delta), Q)
        # hdisc <- rep(bandwidth, Q)
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    ## Direct re-implementation of the FORTRAN routine "sstdg" (KernSmooth/src/sstdiag.f).
    ## For each grid point j (0-based) with its discretised bandwidth level
    ## i = indic[j]-1 (Lvec[i], hdisc[i]), accumulate TWO weighted moment vectors
    ## over neighbouring bins k in [j-Lvec[i], j+Lvec[i]]:
    ##   ss_j[p] = sum_k xcounts[k] * w(k,j)    * (delta*(k-j))**p
    ##   uu_j[p] = sum_k xcounts[k] * w(k,j)**2 * (delta*(k-j))**p
    ## for p = 0 .. 2*degree, where w(k,j) = exp(-(delta*(k-j)/hdisc[i])**2/2).
    ## Smat and Umat are the (degree+1)x(degree+1) matrices built from ss_j/uu_j
    ## respectively (Smat[a,b] = ss_j[a+b], Umat[a,b] = uu_j[a+b]). Smat is then
    ## inverted (dgefa/dgedi computes the full inverse, not just a linear solve),
    ## and the diagonal entry of S*S^T at j is
    ##   SSTd[j] = sum_a sum_b Smat_inv[0,a] * Umat[a,b] * Smat_inv[b,0]
    ## -- note this does NOT depend on 'drv': the Fortran code always uses row/
    ## column 0 of Smat_inv, since the local design row evaluated at the point
    ## itself (u=0) is (1,0,...,0).
    SSTd = np.zeros(M, dtype=np.float64)
    powers = np.arange(ppp)
    idx_sum = np.add.outer(np.arange(pp), np.arange(pp))  # a+b for Smat/Umat lookup

    for j in range(M):
        i = indic[j] - 1
        L = int(Lvec[i])
        h = hdisc[i]

        k_lo = max(0, j - L)
        k_hi = min(M - 1, j + L)
        ks = np.arange(k_lo, k_hi + 1)

        dvals = (ks - j).astype(np.float64)
        u = delta * dvals
        weight = np.exp(-((u / h) ** 2) / 2.0)

        xw = xcounts[ks] * weight
        xw2 = xcounts[ks] * (weight ** 2)

        u_powers = u[:, None] ** powers[None, :]  # shape (len(ks), ppp)
        ss_j = (xw[:, None] * u_powers).sum(axis=0)  # length ppp, powers 0..2*degree
        uu_j = (xw2[:, None] * u_powers).sum(axis=0)  # length ppp, powers 0..2*degree

        Smat = ss_j[idx_sum]
        Umat = uu_j[idx_sum]
        Smat_inv = np.linalg.inv(Smat)

        SSTd[j] = Smat_inv[0, :] @ Umat @ Smat_inv[:, 0]

    # list(x = gpoints, y = SSTd)
    return {"x": gpoints, "y": SSTd}


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # xy <- cbind(x, y); xy <- xy[sort.list(xy[, 1L]), ]
    # x <- xy[, 1L]; y <- xy[, 2L]
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # indlow <- floor(trim*length(x)) + 1
    # indupp <- length(x) - floor(trim*length(x))
    n_full = len(x)
    indlow = int(np.floor(trim * n_full)) + 1
    indupp = n_full - int(np.floor(trim * n_full))

    # x <- x[indlow:indupp]; y <- y[indlow:indupp]
    # (R's x[indlow:indupp] is 1-based and INCLUSIVE of both endpoints;
    #  convert to a 0-based half-open slice.)
    x = x[indlow - 1:indupp]
    y = y[indlow - 1:indupp]

    ## Rename common parameters
    # n <- length(x); M <- gridsize
    n = len(x)
    M = gridsize

    # a <- range.x[1L]; b <- range.x[2L]
    # NOTE: R's default argument `range.x = range(x)` is evaluated lazily, the
    # first time `range.x` is referenced in the body -- which happens AFTER
    # `x` has already been reassigned above to the sorted+trimmed data. So if
    # the caller does not supply range_x explicitly, the default here must be
    # computed from the TRIMMED `x`, not the original untrimmed argument.
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    ## Bin the data
    # gpoints <- seq(a, b, length.out = M)
    # out <- rlbin(x, y, gpoints, truncate)
    gpoints = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    ## Choose the value of N using Mallow's C_p
    # Nmax <- max(min(floor(n/divisor), blockmax), 1)
    Nmax = max(min(int(np.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x, y, Nmax, 4)

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.
    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    ## Estimate theta_22 using a local cubic fit
    ## with a "rule-of-thumb" bandwidth: "gamseh"
    # gamseh <- (sigsqQ*(b-a)/(abs(th24Q)*n))
    # if (th24Q < 0) gamseh <- (3*gamseh/(8*sqrt(pi)))^(1/7)
    # if (th24Q > 0) gamseh <- (15*gamseh/(16*sqrt(pi)))^(1/7)
    # (Note: if th24Q == 0 exactly, gamseh is left as the un-transformed
    #  value from the first line -- which involves division by zero -- this
    #  matches the original R behaviour exactly, including any resulting
    #  inf/nan, with no defensive handling added.)
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = np.power(3.0 * gamseh / (8.0 * np.sqrt(np.pi)), 1.0 / 7.0)
    if th24Q > 0:
        gamseh = np.power(15.0 * gamseh / (16.0 * np.sqrt(np.pi)), 1.0 / 7.0)

    mddest = locpoly(
        xcounts, ycounts, drv=2, bandwidth=gamseh,
        range_x=range_x, binned=True
    )["y"]

    # llow <- floor(proptrun*M) + 1; lupp <- M - floor(proptrun*M)
    # th22kn <- sum((mddest[llow:lupp]^2)*xcounts[llow:lupp])/n
    # (again, R's 1-based inclusive slice llow:lupp converts to the 0-based
    #  half-open slice [llow-1:lupp])
    llow = int(np.floor(proptrun * M)) + 1
    lupp = M - int(np.floor(proptrun * M))
    th22kn = np.sum(
        (mddest[llow - 1:lupp] ** 2) * xcounts[llow - 1:lupp]
    ) / n

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    # C3K <- (1/2) + 2*sqrt(2) - (4/3)*sqrt(3)
    # C3K <- (4*C3K/(sqrt(2*pi)))^(1/9)
    C3K = 0.5 + 2.0 * np.sqrt(2.0) - (4.0 / 3.0) * np.sqrt(3.0)
    C3K = np.power(4.0 * C3K / np.sqrt(2.0 * np.pi), 1.0 / 9.0)
    # lamseh <- C3K*(((sigsqQ^2)*(b-a)/((th22kn*n)^2))^(1/9))
    lamseh = C3K * np.power(
        (sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2), 1.0 / 9.0
    )

    ## Now compute a local linear kernel estimate of
    ## the variance.
    mest = locpoly(
        xcounts, ycounts, bandwidth=lamseh,
        range_x=range_x, binned=True
    )["y"]
    Sdg = sdiag(
        xcounts, bandwidth=lamseh,
        range_x=range_x, binned=True
    )["y"]
    SSTdg = sstdiag(
        xcounts, bandwidth=lamseh,
        range_x=range_x, binned=True
    )["y"]
    sigsqn = np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    ## Combine to obtain final answer.
    # (sigsqkn*(b-a)/(2*sqrt(pi)*th22kn*n))^(1/5)
    return float(
        np.power(
            sigsqkn * (b - a) / (2.0 * np.sqrt(np.pi) * th22kn * n), 1.0 / 5.0
        )
    )


def _on_attach(libname: str, pkgname: str) -> None:
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def _on_unload(libpath: str) -> None:
    # R's .onUnload calls library.dynam.unload("KernSmooth", libpath) to unload
    # the compiled shared library when the package namespace is detached.
    # CPython does not support safely unloading native C extension modules
    # (dlclose()-ing a still-referenced extension can segfault), so there is
    # no direct Python equivalent. The import system manages the lifetime of
    # the compiled _KernSmooth extension automatically, so no action is needed.
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
