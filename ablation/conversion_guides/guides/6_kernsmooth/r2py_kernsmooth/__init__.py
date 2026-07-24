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


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    # Install safeguard against non-positive bandwidths:
    # if (!missing(bandwidth) && bandwidth <= 0)
    #     stop("'bandwidth' must be strictly positive")
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # kernel <- match.arg(kernel,
    #                     c("normal", "box", "epanech", "biweight", "triweight"))
    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        raise ValueError(
            f"'kernel' = '{kernel}' is not a valid choice; must be one of {_kernel_choices}."
        )

    # Rename common variables
    # n <- length(x); M <- gridsize
    n = len(x)
    M = gridsize

    # Set canonical scaling factors
    # del0 <- switch(kernel,
    #                "normal" = (1/(4*pi))^(1/10),
    #                "box" = (9/2)^(1/5),
    #                "epanech" = 15^(1/5),
    #                "biweight" = 35^(1/5),
    #                "triweight" = (9450/143)^(1/5))
    _del0_map = {
        "normal": (1.0 / (4.0 * np.pi)) ** (1.0 / 10.0),
        "box": (9.0 / 2.0) ** (1.0 / 5.0),
        "epanech": 15.0 ** (1.0 / 5.0),
        "biweight": 35.0 ** (1.0 / 5.0),
        "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
    }
    del0 = _del0_map[kernel]

    # if (length(canonical) != 1L || !is.logical(canonical))
    #     stop("'canonical' must be a length-1 logical vector")
    if not isinstance(canonical, (bool, np.bool_)):
        raise TypeError("'canonical' must be a length-1 logical vector")

    # Set default bandwidth
    # h <- if (missing(bandwidth)) del0 * (243/(35*n))^(1/5)*sqrt(var(x))
    #      else if(canonical) del0 * bandwidth else bandwidth
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.std(x, ddof=1)
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values
    # tau <-  if (kernel == "normal") 4 else 1
    tau = 4 if kernel == "normal" else 1

    # if (missing(range.x)) range.x <- c(min(x)-tau*h, max(x)+tau*h)
    # a <- range.x[1L]; b <- range.x[2L]
    if range_x is None:
        a = float(np.min(x)) - tau * h
        b = float(np.max(x)) + tau * h
    else:
        a = float(range_x[0])
        b = float(range_x[1])

    # Set up grid points and bin the data
    # gpoints <- seq(a, b, length.out = M)
    # gcounts <- linbin(x, gpoints, truncate)
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute kernel weights
    # delta  <- (b - a)/(h * (M-1L))
    delta = (b - a) / (h * (M - 1))

    # L <- min(floor(tau/delta), M)
    L = int(min(np.floor(tau / delta), M))

    # if (L == 0)
    #     warning("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # lvec <- 0L:L
    lvec = np.arange(0, L + 1)

    # kappa <- if (kernel == "normal")
    #     dnorm(lvec*delta)/(n*h)
    # else if (kernel == "box")
    #     0.5*dbeta(0.5*(lvec*delta+1), 1, 1)/(n*h)
    # else if (kernel == "epanech")
    #     0.5*dbeta(0.5*(lvec*delta+1), 2, 2)/(n*h)
    # else if (kernel == "biweight")
    #     0.5*dbeta(0.5*(lvec*delta+1), 3, 3)/(n*h)
    # else if (kernel == "triweight")
    #     0.5*dbeta(0.5*(lvec*delta+1), 4, 4)/(n*h)
    if kernel == "normal":
        kappa = norm.pdf(lvec * delta) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 3, 3) / (n * h)
    else:
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    # P <- 2^(ceiling(log(M+L+1L)/log(2)))
    P = int(2 ** (np.ceil(np.log(M + L + 1) / np.log(2))))

    # kappa <- c(kappa, rep(0, P-2L*L-1L), rev(kappa[-1L]))
    kappa = np.concatenate((kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]))

    # tot <- sum(kappa) * (b-a)/(M-1L) * n # should have total weight one
    tot = np.sum(kappa) * (b - a) / (M - 1) * n

    # gcounts <- c(gcounts, rep(0L, P-M))
    gcounts = np.concatenate((gcounts, np.zeros(P - M)))

    # kappa <- fft(kappa/tot)
    # gcounts <- fft(gcounts)
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    # list(x = gpoints, y = (Re(fft(kappa*gcounts, TRUE))/P)[1L:M])
    # R's fft(z, inverse=TRUE) is unnormalized (== P * np.fft.ifft(z)), so
    # Re(fft(kappa*gcounts, TRUE))/P == np.real(np.fft.ifft(kappa*gcounts)).
    y = np.real(np.fft.ifft(kappa * gcounts))[0:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: list[np.ndarray[Any, np.dtype[np.float64]] | list[float] | tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    # if (!missing(bandwidth) && min(bandwidth) <= 0)
    #     stop("'bandwidth' must be strictly positive")
    if bandwidth is not None and np.min(bandwidth) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables
    # n <- nrow(x)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]

    # M <- gridsize
    M = np.asarray(gridsize, dtype=np.int64)

    # h <- bandwidth
    # tau <- 3.4                          # For bivariate normal kernel.
    h = np.asarray(bandwidth, dtype=np.float64).ravel()
    tau = 3.4

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    # if (length(h) == 1L) h <- c(h, h)
    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range.x is not specified then set it at its default value.
    # if (missing(range.x)) {
    #     range.x <- list(0, 0)
    #     for (id in (1L:2L))
    #         range.x[[id]] <- c(min(x[, id])-1.5*h[id], max(x[, id])+1.5*h[id])
    # }
    if range_x is None:
        range_x = [None, None]
        for idx in range(2):
            range_x[idx] = (
                float(np.min(x[:, idx]) - 1.5 * h[idx]),
                float(np.max(x[:, idx]) + 1.5 * h[idx]),
            )

    # a <- c(range.x[[1L]][1L], range.x[[2L]][1L])
    # b <- c(range.x[[1L]][2L], range.x[[2L]][2L])
    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    # gpoints1 <- seq(a[1L], b[1L], length.out = M[1L])
    # gpoints2 <- seq(a[2L], b[2L], length.out = M[2L])
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    # gcounts <- linbin2D(x, gpoints1, gpoints2)
    gcounts = linbin2D(x, gpoints1, gpoints2)

    # Compute kernel weights
    # L <- numeric(2L)
    # kapid <- list(0, 0)
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]] | None] = [None, None]
    for idx in range(2):
        # L[id] <- min(floor(tau*h[id]*(M[id]-1)/(b[id]-a[id])), M[id] - 1L)
        L[idx] = int(
            min(
                np.floor(tau * h[idx] * (M[idx] - 1) / (b[idx] - a[idx])),
                M[idx] - 1,
            )
        )

        # lvecid <- 0:L[id]
        lvecid = np.arange(0, int(L[idx]) + 1)

        # facid <- (b[id] - a[id])/(h[id]*(M[id]-1L))
        facid = (b[idx] - a[idx]) / (h[idx] * (M[idx] - 1))

        # z <- matrix(dnorm(lvecid*facid)/h[id])
        z = (norm.pdf(lvecid * facid) / h[idx]).reshape(-1, 1)

        # tot <- sum(c(z, rev(z[-1L]))) * facid * h[id]
        tot = (np.sum(z) + np.sum(z[1:][::-1])) * facid * h[idx]

        # kapid[[id]] <- z/tot
        kapid[idx] = z / tot

    # kapp <- kapid[[1L]] %*% (t(kapid[[2L]]))/n
    kapp = (kapid[0] @ kapid[1].T) / n

    # if (min(L) == 0)
    #     warning("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
    if int(np.min(L)) == 0:
        import warnings

        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    # P <- 2^(ceiling(log(M+L)/log(2)))   # smallest powers of 2 >= M+L
    P = (2 ** np.ceil(np.log(M + L) / np.log(2))).astype(np.int64)

    # L1 <- L[1L] ; L2 <- L[2L]
    # M1 <- M[1L] ; M2 <- M[2L]
    # P1 <- P[1L] ; P2 <- P[2L]
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    # rp <- matrix(0, P1, P2)
    rp = np.zeros((P1, P2), dtype=np.float64)

    # rp[1L:(L1+1), 1L:(L2+1)] <- kapp
    rp[0:L1 + 1, 0:L2 + 1] = kapp

    # if (L1) rp[(P1-L1+1):P1, 1L:(L2+1)] <- kapp[(L1+1):2, 1L:(L2+1)]
    if L1:
        rp[P1 - L1:P1, 0:L2 + 1] = kapp[L1:0:-1, 0:L2 + 1]

    # if (L2) rp[, (P2-L2+1):P2] <- rp[, (L2+1):2]
    # wrap-around version of "kapp"
    if L2:
        rp[:, P2 - L2:P2] = rp[:, L2:0:-1]

    # sp <- matrix(0, P1, P2)
    # sp[1L:M1, 1L:M2] <- gcounts
    # zero-padded version of "gcounts"
    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts

    # rp <- fft(rp)                       # Obtain FFT's of r and s
    # sp <- fft(sp)
    rp = np.fft.fft2(rp)
    sp = np.fft.fft2(sp)

    # rp <- Re(fft(rp*sp, inverse = TRUE)/(P1*P2))[1L:M1, 1L:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it
    # R's fft(z, inverse=TRUE) is unnormalized, so Re(fft(rp*sp, TRUE))/(P1*P2)
    # is equivalent to np.fft.ifft2(rp*sp).real, since NumPy's ifft2 already
    # divides by the total number of elements (P1*P2).
    rp = np.real(np.fft.ifft2(rp * sp))[0:M1, 0:M2]

    # Ensure that rp is non-negative
    # rp <- rp * matrix(as.numeric(rp>0), nrow(rp), ncol(rp))
    rp = rp * (rp > 0).astype(np.float64)

    # list(x1 = gpoints1, x2 = gpoints2, fhat = rp)
    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
    # if (!missing(bandwidth) && bandwidth <= 0)
    #     stop("'bandwidth' must be strictly positive")
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # if (missing(range.x) && !binned) range.x <- c(min(x), max(x))
    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename variables
    # M <- gridsize; a <- range.x[1L]; b <- range.x[2L]; h <- bandwidth
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])
    h = bandwidth

    # Bin the data if not already binned
    if not binned:
        # gpoints <- seq(a, b, length.out = gridsize)
        # gcounts <- linbin(x, gpoints, truncate)
        gpoints = np.linspace(a, b, gridsize)
        gcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
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
    L = int(min(np.floor(tau * h / delta), M))

    # if (L == 0)
    #     warning("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
    if L == 0:
        import warnings
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    # lvec <- 0L:L; arg <- lvec*delta/h
    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    # kappam <- dnorm(arg)/(h^(drv+1))
    kappam = norm.pdf(arg) / (h ** (drv + 1))
    hmold0 = np.ones_like(arg)
    hmold1 = arg
    hmnew = 1

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

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    # P <- 2^(ceiling(log(M+L+1L)/log(2)))
    P = int(2 ** (np.ceil(np.log(M + L + 1) / np.log(2))))

    # kappam <- c(kappam, rep(0,  P-2L*L-1L), rev(kappam[-1L]))
    kappam = np.concatenate((kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]))

    # Gcounts <- c(gcounts, rep(0, P-M))
    Gcounts = np.concatenate((gcounts, np.zeros(P - M)))

    # kappam <- fft(kappam); Gcounts <- fft(Gcounts)
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    # sum(gcounts * (Re(fft(kappam*Gcounts, TRUE))/P)[1L:M] )/(n^2)
    # R's fft(z, inverse=TRUE) is unnormalized (== P * np.fft.ifft(z)), so
    # Re(fft(kappam*Gcounts, TRUE))/P == np.real(np.fft.ifft(kappam*Gcounts)).
    conv = np.real(np.fft.ifft(kappam * Gcounts))[0:M]
    return float(np.sum(gcounts * conv) / (n ** 2))


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # n <- length(x)
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    # datmat <- cbind(x, y); datmat <- datmat[sort.list(datmat[, 1L]), ]
    # x <- datmat[, 1L]; y <- datmat[, 2L]
    sort_idx = np.argsort(np.asarray(x, dtype=np.float64), kind="stable")
    x = np.asarray(x, dtype=np.float64)[sort_idx]
    y = np.asarray(y, dtype=np.float64)[sort_idx]

    # qq <- q + 1L
    qq = q + 1

    # Faithful port of the KernSmooth Fortran subroutine `blkest` (src/blkest.f),
    # since this blocked q'th-degree-polynomial-fit estimator has no reusable
    # internal dependency or generic conversion-guide equivalent. It performs,
    # for each of the Nval blocks of the (sorted) data, a q'th degree polynomial
    # least-squares fit (equivalent to the Fortran dqrdc/dqrsl QR least-squares
    # solve), then accumulates the residual sum of squares (RSS) and the
    # second/fourth-derivative-based quantities th22e/th24e evaluated at each
    # data point of the block.
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    # idiv = n/Nval  (Fortran integer division)
    idiv = n // Nval

    # do j = 1,Nval
    for j in range(1, Nval + 1):

        # For each member of the partition
        # ilow = (j-1)*idiv + 1 ; iupp = j*idiv ; if (j.eq.Nval) iupp = n
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        nj = iupp - ilow + 1

        # do k = 1,nj : Xj(k) = X(ilow+k-1) ; Yj(k) = Y(ilow+k-1)
        # (1-based Fortran slice X(ilow:iupp) -> 0-based Python x[ilow-1:iupp])
        Xj = x[ilow - 1:iupp]
        Yj = y[ilow - 1:iupp]

        # Obtain a q'th degree fit over the current member of the partition.
        # Set up "X" matrix:
        # do i = 1,nj : Xmat(i,1) = 1.0d0 ; do k = 2,qq : Xmat(i,k) = Xj(i)**(k-1)
        Xmat_j = np.empty((nj, qq), dtype=np.float64)
        Xmat_j[:, 0] = 1.0
        for k in range(2, qq + 1):
            Xmat_j[:, k - 1] = Xj ** (k - 1)

        # call dqrdc(...) ; call dqrsl(..., 00100, info)
        # dqrsl with job digit requesting the least-squares coefficients is
        # equivalent to the (minimum-norm) least-squares solution obtained
        # from a QR decomposition of Xmat_j against Yj.
        coef, _, _, _ = np.linalg.lstsq(Xmat_j, Yj, rcond=None)

        # coef has length qq; coef(3) and coef(5) (1-based) are referenced
        # unconditionally below (they correspond to the x^2 and x^4 term
        # coefficients). Pad with zeros up to at least 5 elements to mirror
        # the zero-initialized R/Fortran coef array and avoid out-of-range
        # access when qq < 5 (in practice q = 4, qq = 5, so no padding occurs).
        coef_ext = np.zeros(max(qq, 5), dtype=np.float64)
        coef_ext[:qq] = coef

        # do i = 1,nj
        for i in range(nj):
            xi = Xj[i]

            # fiti = coef(1) ; ddm = 2*coef(3) ; ddddm = 24*coef(5)
            fiti = coef_ext[0]
            ddm = 2.0 * coef_ext[2]
            ddddm = 24.0 * coef_ext[4]

            # do k = 2,qq
            for k in range(2, qq + 1):
                # fiti = fiti + coef(k)*Xj(i)**(k-1)
                fiti = fiti + coef_ext[k - 1] * xi ** (k - 1)

                # if (k.le.(q-1)) then
                if k <= (q - 1):
                    # ddm = ddm + k*(k+1)*coef(k+2)*Xj(i)**(k-1)
                    ddm = ddm + k * (k + 1) * coef_ext[k + 2 - 1] * xi ** (k - 1)

                    # if (k.le.(q-3)) then
                    if k <= (q - 3):
                        # ddddm = ddddm + k*(k+1)*(k+2)*(k+3)*coef(k+4)*Xj(i)**(k-1)
                        ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef_ext[k + 4 - 1] * xi ** (k - 1)

            # th22e = th22e + ddm**2 ; th24e = th24e + ddm*ddddm ; RSS = RSS + (Yj(i)-fiti)**2
            th22e = th22e + ddm ** 2
            th24e = th24e + ddm * ddddm
            RSS = RSS + (Yj[i] - fiti) ** 2

    # sigsqe = RSS/(n-qq*Nval) ; th22e = th22e/n ; th24e = th24e/n
    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    # list(sigsqe = out[[13]], th22e = out[[14]], th24e = out[[15]])
    return {"sigsqe": sigsqe, "th22e": th22e, "th24e": th24e}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # n <- length(X)
    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    # datmat <- cbind(X, Y); datmat <- datmat[sort.list(datmat[, 1L]), ]
    # X <- datmat[, 1L]; Y <- datmat[, 2L]
    sort_idx = np.argsort(np.asarray(X, dtype=np.float64))
    X = np.asarray(X, dtype=np.float64)[sort_idx]
    Y = np.asarray(Y, dtype=np.float64)[sort_idx]

    # qq <- q + 1L
    qq = q + 1

    # Faithful port of the KernSmooth Fortran subroutine `cp` (src/cp.f), since
    # this blocked-polynomial-fit / Mallows' C_p routine has no reusable
    # internal dependency or generic conversion-guide equivalent. It computes,
    # for each candidate number of blocks Nval = 1..Nmax, the total residual
    # sum of squares (RSS) from fitting a degree-q polynomial (qq = q+1
    # coefficients) separately on each of the Nval blocks of the sorted
    # (X, Y) data, and then Mallows' C_p statistic for each Nval.
    #
    # RSS and Cpvals are 0-based arrays of length Nmax here: RSS[i - 1] and
    # Cpvals[i - 1] correspond to Fortran's RSS(i) and Cpvals(i), i.e. to a
    # block count of i.
    RSS = np.zeros(Nmax, dtype=np.float64)
    Cpvals = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # idiv = n/Nval : integer division, as in Fortran's INTEGER arithmetic
        idiv = n // Nval

        for j in range(1, Nval + 1):
            # For each member of the partition
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n
            nj = iupp - ilow + 1

            # Xj(k) <- X(ilow+k-1), Yj(k) <- Y(ilow+k-1) for k = 1..nj
            # (ilow, iupp are 1-based Fortran bounds -> 0-based slice [ilow-1:iupp])
            Xj = X[ilow - 1:iupp]
            Yj = Y[ilow - 1:iupp]

            # Obtain a q'th degree fit over the current member of the partition.
            # Set up "X" matrix: Xmat(i,1) = 1; Xmat(i,k) = Xj(i)**(k-1), k = 2..qq
            Xmat = np.empty((nj, qq), dtype=np.float64)
            Xmat[:, 0] = 1.0
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            # call dqrdc(...); call dqrsl(...): QR-decomposition least-squares
            # fit of Yj on Xmat, producing the polynomial coefficients "coef".
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            # fiti(i) = coef(1) + sum_{k=2}^{qq} coef(k) * Xj(i)**(k-1)
            fiti = Xmat @ coef

            RSSj = np.sum((Yj - fiti) ** 2)
            RSS[Nval - 1] = RSS[Nval - 1] + RSSj

    # Now compute array of Mallow's C_p values.
    # Cpvals(i) = ((n - qq*Nmax) * RSS(i) / RSS(Nmax)) + 2*qq*i - n
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    Cpvec = Cpvals

    # order(Cpvec)[1L]: in R, Cpvec is indexed 1..Nmax by block count, so the
    # 1-based array index of the minimum is numerically identical to the
    # chosen number of blocks (Nval). Here Cpvec is 0-based, so the block
    # count Nval is the argmin index plus one; this is what dpill expects
    # when it calls cpblock() to obtain Nval (an actual block count, not a
    # raw array index).
    Nval_chosen = int(np.argmin(Cpvec)) + 1

    return Nval_chosen


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    # if (level > 5L) stop("Level should be between 0 and 5")
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x = np.asarray(x, dtype=np.float64)

    # if (missing(range.x)) range.x <- range(x)  (default argument, evaluated per-call)
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    # Rename variables
    # n <- length(x); M <- gridsize; a <- range.x[1L]; b <- range.x[2L]
    n = len(x)
    M = gridsize
    a = float(range_x[0])
    b = float(range_x[1])

    # Set up grid points and bin the data
    # gpoints <- seq(a, b, length.out = M)
    # gcounts <- linbin(x, gpoints, truncate)
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    # Compute scale estimate
    # scalest <- match.arg(scalest, c("minim", "stdev", "iqr"))
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        raise ValueError(
            f"'scalest' = '{scalest}' is not a valid choice; must be one of {_scalest_choices}."
        )

    # scalest <- switch(scalest,
    #                   "stdev" = sqrt(var(x)),
    #                   "iqr"= (quantile(x, 3/4)-quantile(x, 1/4))/1.349,
    #                   "minim" = min((quantile(x, 3/4)-quantile(x, 1/4))/1.349, sqrt(var(x))) )
    if scalest == "stdev":
        scale_value = float(np.sqrt(np.var(x, ddof=1)))
    elif scalest == "iqr":
        q75 = np.quantile(x, 0.75, method="linear")
        q25 = np.quantile(x, 0.25, method="linear")
        scale_value = float((q75 - q25) / 1.349)
    else:  # "minim"
        q75 = np.quantile(x, 0.75, method="linear")
        q25 = np.quantile(x, 0.25, method="linear")
        iqr_scale = (q75 - q25) / 1.349
        stdev_scale = np.sqrt(np.var(x, ddof=1))
        scale_value = float(min(iqr_scale, stdev_scale))

    # if (scalest == 0) stop("scale estimate is zero for input data")
    if scale_value == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    # sx <- (x-mean(x))/scalest
    # sa <- (a-mean(x))/scalest ; sb <- (b-mean(x))/scalest
    x_mean = float(np.mean(x))
    sx = (x - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    # Set up grid points and bin the data:
    # gpoints <- seq(sa, sb, length.out = M)
    # gcounts <- linbin(sx, gpoints, truncate)
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    # Perform plug-in steps
    if level == 0:
        # hpi <- (24*sqrt(pi)/n)^(1/3)
        hpi = (24 * np.sqrt(np.pi) / n) ** (1.0 / 3.0)
    elif level == 1:
        # alpha <- (2/(3*n))^(1/5)*sqrt(2)                 # bandwidth for psi_2
        # psi2hat <- bkfe(gcounts, 2L, alpha, range.x = c(sa, sb), binned = TRUE)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        alpha = (2 / (3 * n)) ** (1.0 / 5.0) * np.sqrt(2)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 2:
        # alpha <- ((2/(5*n))^(1/7))*sqrt(2)               # bandwidth for psi_4
        # psi4hat <- bkfe(gcounts, 4L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (sqrt(2/pi)/(psi4hat*n))^(1/5)          # bandwidth for psi_2
        # psi2hat <- bkfe(gcounts, 2L, alpha, range.x = c(sa, sb), binned = TRUE)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        alpha = ((2 / (5 * n)) ** (1.0 / 7.0)) * np.sqrt(2)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 3:
        # alpha <- ((2/(7*n))^(1/9))*sqrt(2)               # bandwidth for psi_6
        # psi6hat <- bkfe(gcounts, 6L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7)       # bandwidth for psi_4
        # psi4hat <- bkfe(gcounts, 4L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (sqrt(2/pi)/(psi4hat*n))^(1/5)          # bandwidth for psi_2
        # psi2hat <- bkfe(gcounts, 2L, alpha, range.x = c(sa, sb), binned = TRUE)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        alpha = ((2 / (7 * n)) ** (1.0 / 9.0)) * np.sqrt(2)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)
    elif level == 4:
        # alpha <- ((2/(9*n))^(1/11))*sqrt(2)              # bandwidth for psi_8
        # psi8hat <- bkfe(gcounts, 8L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9)       # bandwidth for psi_6
        # psi6hat <- bkfe(gcounts, 6L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7)       # bandwidth for psi_4
        # psi4hat <- bkfe(gcounts, 4L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (sqrt(2/pi)/(psi4hat*n))^(1/5)          # bandwidth for psi_2
        # psi2hat <- bkfe(gcounts, 2L, alpha, range.x = c(sa, sb), binned = TRUE)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        alpha = ((2 / (9 * n)) ** (1.0 / 11.0)) * np.sqrt(2)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)
    else:  # level == 5
        # alpha <- ((2/(11*n))^(1/13))*sqrt(2)             # bandwidth for psi_10
        # psi10hat <- bkfe(gcounts, 10L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-105*sqrt(2/pi)/(psi10hat*n))^(1/11)   # bandwidth for psi_8
        # psi8hat <- bkfe(gcounts, 8L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9)       # bandwidth for psi_6
        # psi6hat <- bkfe(gcounts, 6L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7)       # bandwidth for psi_4
        # psi4hat <- bkfe(gcounts, 4L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (sqrt(2/pi)/(psi4hat*n))^(1/5)          # bandwidth for psi_2
        # psi2hat <- bkfe(gcounts, 2L, alpha, range.x = c(sa, sb), binned = TRUE)
        # hpi <- (6/(-psi2hat*n))^(1/3)
        alpha = ((2 / (11 * n)) ** (1.0 / 13.0)) * np.sqrt(2)
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105 * np.sqrt(2 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15 * np.sqrt(2 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3 * np.sqrt(2 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
        alpha = (np.sqrt(2 / np.pi) / (psi4hat * n)) ** (1.0 / 5.0)
        psi2hat = bkfe(gcounts, 2, alpha, range_x=(sa, sb), binned=True)
        hpi = (6 / (-psi2hat * n)) ** (1.0 / 3.0)

    # scalest * hpi
    return float(scale_value * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)

    # if (level > 5L) stop("Level should be between 0 and 5")
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    # kernel <- match.arg(kernel,
    #                     c("normal", "box", "epanech", "biweight", "triweight"))
    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in _kernel_choices:
        raise ValueError(
            f"'kernel' = '{kernel}' is not a valid choice; must be one of {_kernel_choices}."
        )

    ## Set kernel constants
    # del0 <- if (canonical) 1 else switch(kernel,
    #                                      "normal" = 1/((4*pi)^(1/10)),
    #                                      "box" = (9/2)^(1/5),
    #                                      "epanech" = 15^(1/5),
    #                                      "biweight" = 35^(1/5),
    #                                      "triweight" = (9450/143)^(1/5))
    _del0_map = {
        "normal": 1.0 / ((4.0 * np.pi) ** (1.0 / 10.0)),
        "box": (9.0 / 2.0) ** (1.0 / 5.0),
        "epanech": 15.0 ** (1.0 / 5.0),
        "biweight": 35.0 ** (1.0 / 5.0),
        "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
    }
    del0 = 1.0 if canonical else _del0_map[kernel]

    ## Rename variables
    # n <- length(x); M <- gridsize; a <- range.x[1L]; b <- range.x[2L]
    n = len(x)
    M = gridsize
    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))
    a = float(range_x[0])
    b = float(range_x[1])

    ## Set up grid points and bin the data
    # gpoints <- seq(a, b, length.out = M)
    # gcounts <- linbin(x, gpoints, truncate)
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute scale estimate
    # scalest <- match.arg(scalest, c("minim", "stdev", "iqr"))
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest not in _scalest_choices:
        raise ValueError(
            f"'scalest' = '{scalest}' is not a valid choice; must be one of {_scalest_choices}."
        )

    # scalest <- switch(scalest,
    #                   "stdev" = sqrt(var(x)),
    #                   "iqr"= (quantile(x, 3/4)-quantile(x, 1/4))/1.349,
    #                   "minim" = min((quantile(x, 3/4)-quantile(x, 1/4))/1.349, sqrt(var(x))) )
    std_x = float(np.std(x, ddof=1))
    iqr_x = float(np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    _scalest_map = {
        "stdev": std_x,
        "iqr": iqr_x,
        "minim": min(iqr_x, std_x),
    }
    scalest_val = _scalest_map[scalest]

    # if (scalest == 0) stop("scale estimate is zero for input data")
    if scalest_val == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:
    # sx <- (x-mean(x))/scalest
    # sa <- (a-mean(x))/scalest ; sb <- (b-mean(x))/scalest
    mean_x = float(np.mean(x))
    sx = (x - mean_x) / scalest_val
    sa = (a - mean_x) / scalest_val
    sb = (b - mean_x) / scalest_val

    ## Set up grid points and bin the data:
    # gpoints <- seq(sa, sb, length.out = M)
    # gcounts <- linbin(sx, gpoints, truncate)
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

    ## Perform plug-in steps:
    if level == 0:
        # psi4hat <- 3/(8*sqrt(pi))
        psi4hat = 3.0 / (8.0 * np.sqrt(np.pi))
    elif level == 1:
        # alpha <- (2*(sqrt(2))^7/(5*n))^(1/7) # bandwidth for psi_4
        # bkfe(gcounts, 4L, alpha, range.x = c(sa, sb), binned = TRUE)
        alpha = (2.0 * (np.sqrt(2.0)) ** 7 / (5.0 * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 2:
        # alpha <- (2*(sqrt(2))^9/(7*n))^(1/9) # bandwidth for psi_6
        # psi6hat <- bkfe(gcounts, 6L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        # bkfe(gcounts, 4L, alpha, range.x = c(sa, sb), binned = TRUE)
        alpha = (2.0 * (np.sqrt(2.0)) ** 9 / (7.0 * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 3:
        # alpha <- (2*(sqrt(2))^11/(9*n))^(1/11) # bandwidth for psi_8
        # psi8hat <- bkfe(gcounts, 8L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9) # bandwidth for psi_6
        # psi6hat <- bkfe(gcounts, 6L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        # bkfe(gcounts, 4L, alpha, range.x = c(sa, sb), binned = TRUE)
        alpha = (2.0 * (np.sqrt(2.0)) ** 11 / (9.0 * n)) ** (1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 4:
        # alpha <- (2*(sqrt(2))^13/(11*n))^(1/13) # bandwidth for psi_10
        # psi10hat <- bkfe(gcounts, 10L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-105*sqrt(2/pi)/(psi10hat*n))^(1/11) # bandwidth for psi_8
        # psi8hat <- bkfe(gcounts, 8L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9) # bandwidth for psi_6
        # psi6hat <- bkfe(gcounts, 6L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        # bkfe(gcounts, 4L, alpha, range.x = c(sa, sb), binned = TRUE)
        alpha = (2.0 * (np.sqrt(2.0)) ** 13 / (11.0 * n)) ** (1.0 / 13.0)
        psi10hat = bkfe(gcounts, 10, alpha, range_x=(sa, sb), binned=True)
        alpha = (-105.0 * np.sqrt(2.0 / np.pi) / (psi10hat * n)) ** (1.0 / 11.0)
        psi8hat = bkfe(gcounts, 8, alpha, range_x=(sa, sb), binned=True)
        alpha = (15.0 * np.sqrt(2.0 / np.pi) / (psi8hat * n)) ** (1.0 / 9.0)
        psi6hat = bkfe(gcounts, 6, alpha, range_x=(sa, sb), binned=True)
        alpha = (-3.0 * np.sqrt(2.0 / np.pi) / (psi6hat * n)) ** (1.0 / 7.0)
        psi4hat = bkfe(gcounts, 4, alpha, range_x=(sa, sb), binned=True)
    elif level == 5:
        # alpha <- (2*(sqrt(2))^15/(13*n))^(1/15) # bandwidth for psi_12
        # psi12hat <- bkfe(gcounts, 12L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (945*sqrt(2/pi)/(psi12hat*n))^(1/13) # bandwidth for psi_10
        # psi10hat <- bkfe(gcounts, 10L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-105*sqrt(2/pi)/(psi10hat*n))^(1/11) # bandwidth for psi_8
        # psi8hat <- bkfe(gcounts, 8L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (15*sqrt(2/pi)/(psi8hat*n))^(1/9) # bandwidth for psi_6
        # psi6hat <- bkfe(gcounts, 6L, alpha, range.x = c(sa, sb), binned = TRUE)
        # alpha <- (-3*sqrt(2/pi)/(psi6hat*n))^(1/7) # bandwidth for psi_4
        # bkfe(gcounts, 4L, alpha, range.x = c(sa, sb), binned = TRUE)
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

    # scalest * del0 * (1/(psi4hat*n))^(1/5)
    return float(scalest_val * del0 * (1.0 / (psi4hat * n)) ** (1.0 / 5.0))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | list[float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    # xy <- cbind(x, y); xy <- xy[sort.list(xy[, 1L]), ]
    # x <- xy[, 1L]; y <- xy[, 2L]
    sort_idx = np.argsort(x, kind="stable")
    x = x[sort_idx]
    y = y[sort_idx]

    # indlow <- floor(trim*length(x)) + 1
    # indupp <- length(x) - floor(trim*length(x))
    # x <- x[indlow:indupp] ; y <- y[indlow:indupp]
    n_before_trim = len(x)
    ntrim = int(np.floor(trim * n_before_trim))
    # R's 1-based indlow = ntrim + 1 corresponds to 0-based start index `ntrim`;
    # R's 1-based indupp = n_before_trim - ntrim is used as an inclusive upper
    # bound in R, which equals the exclusive Python upper bound below.
    indlow0 = ntrim
    indupp0 = n_before_trim - ntrim
    x = x[indlow0:indupp0]
    y = y[indlow0:indupp0]

    ## Rename common parameters
    # n <- length(x); M <- gridsize
    n = len(x)
    M = gridsize

    # a <- range.x[1L]; b <- range.x[2L]
    # NOTE: R's default `range.x = range(x)` is a lazily-evaluated promise; it
    # is first forced here (i.e. AFTER the sort + trim above), so the default
    # range must be computed from the TRIMMED x, not the original input x.
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
    # Nval <- cpblock(x, y, Nmax, 4)
    Nmax = max(min(int(np.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x, y, Nmax, 4)

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.
    # out <- blkest(x, y, Nval, 4)
    # sigsqQ <- out$sigsqe ; th24Q <- out$th24e
    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    ## Estimate theta_22 using a local cubic fit
    ## with a "rule-of-thumb" bandwidth: "gamseh"
    # gamseh <- (sigsqQ*(b-a)/(abs(th24Q)*n))
    # if (th24Q < 0) gamseh <- (3*gamseh/(8*sqrt(pi)))^(1/7)
    # if (th24Q > 0) gamseh <- (15*gamseh/(16*sqrt(pi)))^(1/7)
    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3.0 * gamseh / (8.0 * np.sqrt(np.pi))) ** (1.0 / 7.0)
    if th24Q > 0:
        gamseh = (15.0 * gamseh / (16.0 * np.sqrt(np.pi))) ** (1.0 / 7.0)
    # (If th24Q == 0, neither branch fires and gamseh keeps its unmodified value.)

    # mddest <- locpoly(xcounts, ycounts, drv=2L, bandwidth=gamseh,
    #                   range.x=range.x, binned=TRUE)$y
    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=range_x, binned=True)["y"]

    # llow <- floor(proptrun*M) + 1
    # lupp <- M - floor(proptrun*M)
    # th22kn <- sum((mddest[llow:lupp]^2)*xcounts[llow:lupp])/n
    Mtrim = int(np.floor(proptrun * M))
    # R's 1-based llow = Mtrim + 1 corresponds to 0-based start index `Mtrim`;
    # R's 1-based lupp = M - Mtrim is used as an inclusive upper bound in R,
    # which equals the exclusive Python upper bound below.
    llow0 = Mtrim
    lupp0 = M - Mtrim
    th22kn = np.sum((mddest[llow0:lupp0] ** 2) * xcounts[llow0:lupp0]) / n

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    # C3K <- (1/2) + 2*sqrt(2) - (4/3)*sqrt(3)
    # C3K <- (4*C3K/(sqrt(2*pi)))^(1/9)
    # lamseh <- C3K*(((sigsqQ^2)*(b-a)/((th22kn*n)^2))^(1/9))
    C3K = 0.5 + 2.0 * np.sqrt(2.0) - (4.0 / 3.0) * np.sqrt(3.0)
    C3K = (4.0 * C3K / np.sqrt(2.0 * np.pi)) ** (1.0 / 9.0)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1.0 / 9.0))

    ## Now compute a local linear kernel estimate of
    ## the variance.
    # mest <- locpoly(xcounts, ycounts, bandwidth=lamseh,
    #                 range.x=range.x, binned=TRUE)$y
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                    range_x=range_x, binned=True)["y"]

    # Sdg <- sdiag(xcounts, bandwidth=lamseh,
    #              range.x=range.x, binned=TRUE)$y
    Sdg = sdiag(xcounts, bandwidth=lamseh, range_x=range_x, binned=True)["y"]

    # SSTdg <- sstdiag(xcounts, bandwidth=lamseh,
    #                  range.x=range.x, binned=TRUE)$y
    SSTdg = sstdiag(xcounts, bandwidth=lamseh, range_x=range_x, binned=True)["y"]

    # sigsqn <- sum(y^2) - 2*sum(mest*ycounts) + sum((mest^2)*xcounts)
    # sigsqd <- n - 2*sum(Sdg*xcounts) + sum(SSTdg*xcounts)
    # sigsqkn <- sigsqn/sigsqd
    sigsqn = np.sum(y ** 2) - 2.0 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2.0 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    ## Combine to obtain final answer.
    # (sigsqkn*(b-a)/(2*sqrt(pi)*th22kn*n))^(1/5)
    return float((sigsqkn * (b - a) / (2.0 * np.sqrt(np.pi) * th22kn * n)) ** (1.0 / 5.0))


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    # n <- length(X); M <- length(gpoints)
    n = len(X)
    M = len(gpoints)

    # trun <- if (truncate) 1L else 0L
    trun = 1 if truncate else 0

    # a <- gpoints[1L]; b <- gpoints[M]
    a = float(gpoints[0])
    b = float(gpoints[M - 1])

    # double(M): pre-allocated output buffer for the bin counts
    gcnts = np.zeros(M, dtype=np.float64)

    # Faithful port of the KernSmooth Fortran subroutine `linbin` (src/linbin.f),
    # since this univariate linear-binning routine has no reusable internal
    # dependency or generic conversion-guide equivalent.
    delta = (b - a) / (M - 1)
    for i in range(n):
        lxi = ((float(X[i]) - a) / delta) + 1.0

        # Find integer part of "lxi". Fortran's INT() truncates toward zero,
        # which matches Python's int() truncation behavior on floats.
        li = int(lxi)

        rem = lxi - li

        # li and lxi are 1-based (Fortran convention); gcnts is 0-based here,
        # so gcnts(li) -> gcnts[li - 1] and gcnts(li+1) -> gcnts[li].
        if li >= 1 and li < M:
            gcnts[li - 1] = gcnts[li - 1] + (1 - rem)
            gcnts[li] = gcnts[li] + rem

        if li < 1 and trun == 0:
            gcnts[0] = gcnts[0] + 1

        if li >= M and trun == 0:
            gcnts[M - 1] = gcnts[M - 1] + 1

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    # n <- nrow(X)
    n = X.shape[0]

    # X <- c(X[, 1L], X[, 2L]): flatten so the x-coordinates come first,
    # followed by the y-coordinates. We keep X as the original (n, 2) array
    # and index the two columns directly instead of building the
    # concatenated 1-D vector, which is equivalent for this port.
    # M1 <- length(gpoints1); M2 <- length(gpoints2)
    M1 = len(gpoints1)
    M2 = len(gpoints2)

    # a1 <- gpoints1[1L]; a2 <- gpoints2[1L]
    # b1 <- gpoints1[M1]; b2 <- gpoints2[M2]
    a1 = float(gpoints1[0])
    a2 = float(gpoints2[0])
    b1 = float(gpoints1[M1 - 1])
    b2 = float(gpoints2[M2 - 1])

    # double(M1*M2): pre-allocated output buffer for the 2-D bin counts,
    # kept here as an (M1, M2) array (row index <-> gpoints1, column index
    # <-> gpoints2), which matches R's column-major
    # `matrix(out[[9L]], M1, M2)` reshape of the Fortran linear buffer,
    # since the Fortran routine addresses gcnts with li1 varying fastest
    # (ind = M1*(li2-1) + li1), i.e. column-major with li1 as the row index.
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    # Faithful port of the KernSmooth Fortran subroutine `lbtwod`
    # (src/linbin2D.f), since this bivariate linear-binning routine has no
    # reusable internal dependency or generic conversion-guide equivalent.
    # Observations outside the mesh are ignored (no truncate argument is
    # exposed by the R wrapper, and the Fortran routine silently drops any
    # point whose li1/li2 falls outside [1, M1-1] x [1, M2-1]).
    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)
    for i in range(n):
        xi = float(X[i, 0])
        yi = float(X[i, 1])
        lxi1 = ((xi - a1) / delta1) + 1.0
        lxi2 = ((yi - a2) / delta2) + 1.0

        # Find integer part of "lxi1" and "lxi2". Fortran's INT() truncates
        # toward zero, which matches Python's int() truncation behavior on
        # floats.
        li1 = int(lxi1)
        li2 = int(lxi2)

        rem1 = lxi1 - li1
        rem2 = lxi2 - li2

        # li1, li2 are 1-based (Fortran convention); gcnts is 0-based here,
        # so gcnts(li1, li2) -> gcnts[li1 - 1, li2 - 1],
        # gcnts(li1+1, li2) -> gcnts[li1, li2 - 1],
        # gcnts(li1, li2+1) -> gcnts[li1 - 1, li2],
        # gcnts(li1+1, li2+1) -> gcnts[li1, li2].
        if li1 >= 1 and li2 >= 1 and li1 < M1 and li2 < M2:
            gcnts[li1 - 1, li2 - 1] = gcnts[li1 - 1, li2 - 1] + (1 - rem1) * (1 - rem2)
            gcnts[li1, li2 - 1] = gcnts[li1, li2 - 1] + rem1 * (1 - rem2)
            gcnts[li1 - 1, li2] = gcnts[li1 - 1, li2] + (1 - rem1) * rem2
            gcnts[li1, li2] = gcnts[li1, li2] + rem1 * rem2

    return gcnts


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    # if (!missing(bandwidth) && any(bandwidth <= 0))
    #     stop("'bandwidth' must be strictly positive")
    if bandwidth is not None and np.any(np.asarray(bandwidth, dtype=np.float64) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    # drv <- as.integer(drv)
    drv = int(drv)

    # if (missing(degree)) degree <- drv + 1L
    # else degree <- as.integer(degree)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x = np.asarray(x, dtype=np.float64)

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
    # M <- gridsize
    M = int(gridsize)
    # Q <- as.integer(bwdisc)
    Q = int(bwdisc)
    # a <- range.x[1L]; b <- range.x[2L]
    a = float(range_x[0])
    b = float(range_x[1])
    # pp <- degree + 1L
    pp = degree + 1
    # ppp <- 2L*degree + 1L
    ppp = 2 * degree + 1
    # tau <- 4
    tau = 4

    ## Decide whether a density estimate or regression estimate is required.
    # if (missing(y)) {  # obtain density estimate
    #     y <- NULL
    #     n <- length(x)
    #     gpoints <- seq(a, b, length.out = M)
    #     xcounts <- linbin(x, gpoints, truncate)
    #     ycounts <- (M-1)*xcounts/(n*(b-a))
    #     xcounts <- rep(1, M)
    # } else {            # obtain regression estimate
    #     ## Bin the data if not already binned
    #     if (!binned) {
    #         gpoints <- seq(a, b, length.out = M)
    #         out <- rlbin(x, y, gpoints, truncate)
    #         xcounts <- out$xcounts
    #         ycounts <- out$ycounts
    #     } else {
    #         xcounts <- x
    #         ycounts <- y
    #         M <- length(xcounts)
    #         gpoints <- seq(a, b, length.out = M)
    #     }
    # }
    if y is None:
        n = len(x)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:
        y = np.asarray(y, dtype=np.float64)
        if not binned:
            gpoints = np.linspace(a, b, M)
            out = rlbin(x, y, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = np.asarray(x, dtype=np.float64)
            ycounts = y
            M = len(xcounts)
            gpoints = np.linspace(a, b, M)

    ## Set the bin width
    # delta <- (b-a)/(M-1L)
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    # if (length(bandwidth) == M) { ... }
    # else if (length(bandwidth) == 1L) { ... }
    # else stop("'bandwidth' must be a scalar or an array of length 'gridsize'")
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.size == M:
        # hlow <- sort(bandwidth)[1L]; hupp <- sort(bandwidth)[M]
        hlow = np.min(bandwidth_arr)
        hupp = np.max(bandwidth_arr)
        # hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
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
                raw = ((np.log(bandwidth_arr) - np.log(hlow)) / gap) + 1
                indic = np.round(raw).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.size == 1:
        # indic <- rep(1, M); Q <- 1L
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        # Lvec <- rep(floor(tau*bandwidth/delta), Q)
        # hdisc <- rep(bandwidth, Q)
        scalar_bw = float(bandwidth_arr[0])
        Lvec = np.full(Q, int(math.floor(tau * scalar_bw / delta)), dtype=np.int64)
        hdisc = np.full(Q, scalar_bw, dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # if (min(Lvec) == 0)
    #     stop("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
    if np.min(Lvec) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'"
        )

    ## Allocate space for the kernel vector and final estimate
    # dimfkap <- 2L * sum(Lvec) + Q
    dimfkap = int(2 * np.sum(Lvec) + Q)
    fkap = np.zeros(dimfkap, dtype=np.float64)
    curvest = np.zeros(M, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)

    ## Call FORTRAN routine "locpol" -- faithful line-by-line NumPy port of
    ## src/locpoly.f below, since this is the core (and most complex) binned
    ## local-polynomial kernel-regression routine, with no reusable internal
    ## dependency or generic conversion-guide equivalent. Fortran's 1-based
    ## loop/array indices are kept as 1-based Python loop counters below (to
    ## mirror the original algebra exactly), with an explicit "-1" applied
    ## only at the point of indexing into the 0-based NumPy arrays.

    # Obtain kernel weights
    # mid = Lvec(1) + 1
    mid = int(Lvec[0]) + 1
    # do i=1,(iQ-1)
    for i in range(1, Q):
        # midpts(i) = mid
        midpts[i - 1] = mid
        # fkap(mid) = 1.0d0
        fkap[mid - 1] = 1.0
        # do j=1,Lvec(i)
        for j in range(1, int(Lvec[i - 1]) + 1):
            # fkap(mid+j) = exp(-(delta*j/hdisc(i))**2/2)
            fkap[mid + j - 1] = math.exp(-((delta * j / hdisc[i - 1]) ** 2) / 2)
            # fkap(mid-j) = fkap(mid+j)
            fkap[mid - j - 1] = fkap[mid + j - 1]
        # mid = mid + Lvec(i) + Lvec(i+1) + 1
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    # midpts(iQ) = mid; fkap(mid) = 1.0d0
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    # do j=1,Lvec(iQ)
    for j in range(1, int(Lvec[Q - 1]) + 1):
        # fkap(mid+j) = exp(-(delta*j/hdisc(iQ))**2/2)
        fkap[mid + j - 1] = math.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        # fkap(mid-j) = fkap(mid+j)
        fkap[mid - j - 1] = fkap[mid + j - 1]

    # Combine kernel weights and grid counts
    # do k = 1,M
    for k in range(1, M + 1):
        # if (xcnts(k).ne.0) then
        if xcounts[k - 1] != 0:
            # do i = 1,iQ
            for i in range(1, Q + 1):
                # do j = max(1,k-Lvec(i)),min(M,k+Lvec(i))
                lo = max(1, k - int(Lvec[i - 1]))
                hi = min(M, k + int(Lvec[i - 1]))
                for j in range(lo, hi + 1):
                    # if (indic(j).eq.i) then
                    if indic[j - 1] == i:
                        # fac = 1.0d0
                        fac = 1.0
                        # (shared kernel weight term fkap(k-j+midpts(i)))
                        kap = fkap[k - j + midpts[i - 1] - 1]
                        # ss(j,1) = ss(j,1) + xcnts(k)*fkap(k-j+midpts(i))
                        ss[j - 1, 0] += xcounts[k - 1] * kap
                        # tt(j,1) = tt(j,1) + ycnts(k)*fkap(k-j+midpts(i))
                        tt[j - 1, 0] += ycounts[k - 1] * kap
                        # do ii = 2,ippp
                        for ii in range(2, ppp + 1):
                            # fac = fac*delta*(k-j)
                            fac = fac * delta * (k - j)
                            # ss(j,ii) = ss(j,ii) + xcnts(k)*fkap(k-j+midpts(i))*fac
                            ss[j - 1, ii - 1] += xcounts[k - 1] * kap * fac
                            # if (ii.le.ipp) then
                            #     tt(j,ii) = tt(j,ii) + ycnts(k)*fkap(k-j+midpts(i))*fac
                            if ii <= pp:
                                tt[j - 1, ii - 1] += ycounts[k - 1] * kap * fac

    # do k = 1,M
    #    do i = 1,ipp
    #       do j = 1,ipp
    #          indss = i + j - 1
    #          Smat(i,j) = ss(k,indss)
    #       end do
    #       Tvec(i) = tt(k,i)
    #    end do
    #   call dgefa(Smat,ipp,ipp,ipvt,info)
    #   call dgesl(Smat,ipp,ipp,ipvt,Tvec,0)
    #   cvest(k) = Tvec(idrv+1)
    # end do
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Tvec = np.zeros(pp, dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k - 1, indss - 1]
            Tvec[i - 1] = tt[k - 1, i - 1]

        # dgefa/dgesl (LINPACK LU factorization + solve, job=0, i.e. solve
        # Smat @ coef = Tvec without transposing) is equivalent to a direct
        # linear solve of the local weighted least-squares normal equations.
        coef = np.linalg.solve(Smat, Tvec)

        # cvest(k) = Tvec(idrv+1) -- Tvec is overwritten in place by dgesl to
        # hold the solution, so this is the (idrv+1)-th (1-based) solution
        # coefficient, i.e. coef[drv] here (drv is already 0-based).
        curvest[k - 1] = coef[drv]

    # curvest <- gamma(drv+1) * out[[19L]]
    curvest = math.gamma(drv + 1) * curvest

    # list(x = gpoints, y = curvest)
    return {"x": gpoints, "y": curvest}


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # n <- length(X); M <- length(gpoints)
    n = len(X)
    M = len(gpoints)

    # trun <- if (truncate) 1L else 0L
    trun = 1 if truncate else 0

    # a <- gpoints[1L]; b <- gpoints[M]
    a = float(gpoints[0])
    b = float(gpoints[M - 1])

    # double(M), double(M): pre-allocated output buffers for the x-counts and y-counts
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)

    # Faithful port of the KernSmooth Fortran subroutine `rlbin` (src/rlbin.f),
    # since this bivariate (regression) linear-binning routine has no reusable
    # internal dependency or generic conversion-guide equivalent. It mirrors
    # `linbin` but additionally accumulates ycnts weighted by the Y values.
    delta = (b - a) / (M - 1)
    for i in range(n):
        xi = float(X[i])
        yi = float(Y[i])
        lxi = ((xi - a) / delta) + 1.0

        # Find integer part of "lxi". Fortran's INT() truncates toward zero,
        # which matches Python's int() truncation behavior on floats.
        li = int(lxi)
        rem = lxi - li

        # Correction for right endpoint (not included if li == M)
        if xi == b:
            li = M - 1
            rem = 1.0

        # li and lxi are 1-based (Fortran convention); xcnts/ycnts are 0-based
        # here, so xcnts(li) -> xcnts[li - 1] and xcnts(li+1) -> xcnts[li].
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
    # `drv` and `kernel` are accepted for signature compatibility with the
    # rest of the KernSmooth API (as in the R original) but, exactly as in
    # the R body of `sdiag`, they are never referenced below: only a normal
    # (Gaussian) kernel is used, and no derivative order is needed because
    # `sdiag` only returns the diagonal of the smoother/hat matrix S, not a
    # fitted curve.
    del drv, kernel

    # if (missing(range.x) && !binned) range.x <- c(min(x), max(x))
    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename common variables

    # M <- gridsize
    M = int(gridsize)
    # Q <- as.integer(bwdisc)
    Q = int(bwdisc)
    # a <- range.x[1L]; b <- range.x[2L]
    a = float(range_x[0])
    b = float(range_x[1])
    # pp <- degree + 1L; ppp <- 2L*degree + 1L
    pp = degree + 1
    ppp = 2 * degree + 1
    # tau <- 4
    tau = 4.0

    ## Bin the data if not already binned

    # if (!binned) { gpoints <- seq(a, b, length.out = M); xcounts <- linbin(x, gpoints, truncate) }
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
    else:
        # xcounts <- x; M <- length(xcounts); gpoints <- seq(a, b, length.out = M)
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    # delta <- (b-a)/(M-1L)
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths

    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    # if (length(bandwidth) == M) {
    if bandwidth_arr.size == M:
        # hlow <- sort(bandwidth)[1L]; hupp <- sort(bandwidth)[M]
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        # hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        # Lvec <- floor(tau*hdisc/delta)
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc" to each member of "bandwidth"
        # indic <- if (Q > 1L) { ... } else rep(1, M)
        if Q > 1:
            # lhdisc <- log(hdisc); gap <- (lhdisc[Q]-lhdisc[1L])/(Q-1)
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            # if (gap == 0) rep(1, M)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                # else round(((log(bandwidth) - log(sort(bandwidth)[1L]))/gap) + 1)
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    # } else if (length(bandwidth) == 1L) {
    elif bandwidth_arr.size == 1:
        # indic <- rep(1, M); Q <- 1L
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        # Lvec <- rep(floor(tau*bandwidth/delta), Q)
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        # hdisc <- rep(bandwidth, Q)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    # } else stop("'bandwidth' must be a scalar or an array of length 'gridsize'")
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # dimfkap <- 2L * sum(Lvec) + Q; fkap <- rep(0, dimfkap); midpts <- rep(0, Q)
    # ss <- matrix(0, M, ppp); Sdg <- rep(0, M)
    # (Smat, work, det, ipvt are LINPACK (dgefa/dgedi) scratch arrays in the
    # Fortran source; they are not needed here since Smat's inverse below is
    # obtained directly via numpy's linear-algebra routines.)
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    Sdg = np.zeros(M, dtype=np.float64)

    # Faithful port of the KernSmooth Fortran subroutine `sdiag` (src/sdiag.f),
    # since this diagonal-of-the-smoother-matrix computation has no reusable
    # internal dependency or generic conversion-guide equivalent. Fortran's
    # 1-based array/loop indices are preserved as 1-based Python loop
    # variables below (with an explicit "-1" applied at each point of actual
    # array access) to keep the port line-for-line traceable to sdiag.f.

    # c Obtain kernel weights
    # mid = Lvec(1) + 1
    mid = int(Lvec[0]) + 1
    # do i=1,(iQ-1)
    for i in range(1, Q):
        # midpts(i) = mid
        midpts[i - 1] = mid
        # fkap(mid) = 1.0d0
        fkap[mid - 1] = 1.0
        # do j=1,Lvec(i)
        for j in range(1, int(Lvec[i - 1]) + 1):
            # fkap(mid+j) = exp(-(delta*j/hdisc(i))**2/2)
            fkap[mid - 1 + j] = np.exp(-((delta * j / hdisc[i - 1]) ** 2) / 2)
            # fkap(mid-j) = fkap(mid+j)
            fkap[mid - 1 - j] = fkap[mid - 1 + j]
        # mid = mid + Lvec(i) + Lvec(i+1) + 1
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    # midpts(iQ) = mid
    midpts[Q - 1] = mid
    # fkap(mid) = 1.0d0
    fkap[mid - 1] = 1.0
    # do j=1,Lvec(iQ)
    for j in range(1, int(Lvec[Q - 1]) + 1):
        # fkap(mid+j) = exp(-(delta*j/hdisc(iQ))**2/2)
        fkap[mid - 1 + j] = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        # fkap(mid-j) = fkap(mid+j)
        fkap[mid - 1 - j] = fkap[mid - 1 + j]

    # c Combine kernel weights and grid counts
    # do k = 1,M
    for k in range(1, M + 1):
        # if (xcnts(k).ne.0) then
        if xcounts[k - 1] != 0:
            # do i = 1,iQ
            for i in range(1, Q + 1):
                # do j = max(1,k-Lvec(i)),min(M,k+Lvec(i))
                lo = max(1, k - int(Lvec[i - 1]))
                hi = min(M, k + int(Lvec[i - 1]))
                for j in range(lo, hi + 1):
                    # if (indic(j).eq.i) then
                    if indic[j - 1] == i:
                        # fac = 1.0d0
                        fac = 1.0
                        # ss(j,1) = ss(j,1) + xcnts(k)*fkap(k-j+midpts(i))
                        ss[j - 1, 0] = ss[j - 1, 0] + xcounts[k - 1] * fkap[k - j + midpts[i - 1] - 1]
                        # do ii = 2,ippp
                        for ii in range(2, ppp + 1):
                            # fac = fac*delta*(k-j)
                            fac = fac * delta * (k - j)
                            # ss(j,ii) = ss(j,ii) + xcnts(k)*fkap(k-j+midpts(i))*fac
                            ss[j - 1, ii - 1] = ss[j - 1, ii - 1] + xcounts[k - 1] * fkap[k - j + midpts[i - 1] - 1] * fac

    # do k = 1,M
    for k in range(1, M + 1):
        # do i = 1,ipp; do j = 1,ipp; indss = i + j - 1; Smat(i,j) = ss(k,indss)
        Smat = np.empty((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k - 1, indss - 1]

        # call dgefa(Smat,ipp,ipp,ipvt,info); call dgedi(Smat,ipp,ipp,ipvt,det,work,01)
        # (LINPACK LU-decompose Smat then invert it in place, since job=01
        # requests the inverse only, no determinant; np.linalg.inv is the
        # direct NumPy equivalent of this LU-based matrix inversion.)
        Smat_inv = np.linalg.inv(Smat)

        # Sdg(k) = Smat(1,1)
        Sdg[k - 1] = Smat_inv[0, 0]

    # list(x = gpoints, y = out[[17L]])
    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Note on argument order: in R, 'bandwidth' has no default and is a
    # required argument (positioned after several defaulted arguments,
    # which R allows because arguments are matched/supplied by name at the
    # call sites, e.g. from dpill()). Python requires non-default parameters
    # to precede default ones, so 'bandwidth' is moved directly after 'x'
    # here; all other parameters keep their original names/defaults.
    #
    # 'drv' and 'kernel' are accepted (as in the R signature) but are not
    # otherwise used in the body, exactly mirroring the original R function.

    # if (missing(range.x) && !binned) range.x <- c(min(x), max(x))
    x = np.asarray(x, dtype=np.float64)
    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Rename common variables
    # M <- gridsize ; Q <- as.integer(bwdisc) ; a <- range.x[1L] ; b <- range.x[2L]
    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    # pp <- degree + 1L ; ppp <- 2L*degree + 1L ; tau <- 4L
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Bin the data if not already binned
    if not binned:
        # gpoints <- seq(a, b, length.out = M) ; xcounts <- linbin(x, gpoints, truncate)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
    else:
        # xcounts <- x ; M <- length(xcounts) ; gpoints <- seq(a, b, length.out = M)
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    # delta <- (b-a)/(M-1L)
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.size == M:
        # hlow <- sort(bandwidth)[1L] ; hupp <- sort(bandwidth)[M]
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        # hdisc <- exp(seq(log(hlow), log(hupp), length.out = Q))
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        # Lvec <- floor(tau*hdisc/delta)
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc" to each member of "bandwidth"
        if Q > 1:
            # lhdisc <- log(hdisc) ; gap <- (lhdisc[Q]-lhdisc[1L])/(Q-1)
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                # round(((log(bandwidth) - log(sort(bandwidth)[1L]))/gap) + 1)
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.size == 1:
        # indic <- rep(1, M) ; Q <- 1L
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        # Lvec <- rep(floor(tau*bandwidth/delta), Q) ; hdisc <- rep(bandwidth, Q)
        Lvec = np.full(Q, int(np.floor(tau * bandwidth_arr[0] / delta)), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # dimfkap <- 2L * sum(Lvec) + Q
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    # fkap <- rep(0, dimfkap); padded with an unused index 0 so that the
    # 1-based Fortran offset arithmetic (mid +/- j) below can be ported
    # verbatim without re-deriving 0-based offsets.
    fkap = np.zeros(dimfkap + 1, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    SSTd = np.zeros(M, dtype=np.float64)

    # Faithful port of the KernSmooth Fortran subroutine `sstdg` (src/sstdiag.f),
    # since this routine (computing the binned diagonal of S*S^T for local
    # polynomial regression) has no reusable internal dependency or generic
    # conversion-guide equivalent. It is structurally a twin of the `sdiag`
    # Fortran routine (which computes the diagonal of S alone), extended with
    # a second "uu"/"Umat" moment accumulator (holding squared-kernel-weighted
    # binned moments) needed to assemble the SS^T diagonal.

    # Obtain kernel weights (Fortran: mid = Lvec(1) + 1)
    mid = int(Lvec[0]) + 1
    for i in range(1, Q):
        midpts[i - 1] = mid
        fkap[mid] = 1.0
        for j in range(1, int(Lvec[i - 1]) + 1):
            fkap[mid + j] = np.exp(-((delta * j / hdisc[i - 1]) ** 2) / 2)
            fkap[mid - j] = fkap[mid + j]
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    midpts[Q - 1] = mid
    fkap[mid] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        fkap[mid + j] = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid - j] = fkap[mid + j]

    # Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i in range(1, Q + 1):
                lo = max(1, k - int(Lvec[i - 1]))
                hi = min(M, k + int(Lvec[i - 1]))
                for j in range(lo, hi + 1):
                    if indic[j - 1] == i:
                        fac = 1.0
                        fkidx = k - j + int(midpts[i - 1])
                        ss[j - 1, 0] += xcounts[k - 1] * fkap[fkidx]
                        uu[j - 1, 0] += xcounts[k - 1] * (fkap[fkidx] ** 2)
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xcounts[k - 1] * fkap[fkidx] * fac
                            uu[j - 1, ii - 1] += xcounts[k - 1] * (fkap[fkidx] ** 2) * fac

    # Faithful port of the LINPACK routines `dgefa` (Gaussian elimination with
    # partial pivoting) and `dgedi` (job=01: inverse only, no determinant),
    # as used by the Fortran `sstdg` subroutine (src/dgefa.f, src/dgedi.f).
    # A hand-rolled port (rather than np.linalg.inv) is used deliberately:
    # dgefa/dgedi never raise on an exactly-singular pivot (they silently
    # produce Inf/NaN/garbage via a division by (near-)zero, per the LINPACK
    # documentation), whereas np.linalg.inv raises LinAlgError in that case;
    # matching the Fortran behaviour (including its numerical instability at
    # grid points with too little local data) requires replicating the exact
    # elimination/back-substitution steps rather than using a robust solver.
    def _linpack_inverse(a_in: np.ndarray[Any, np.dtype[np.float64]], n: int) -> np.ndarray[Any, np.dtype[np.float64]]:
        a = a_in.copy()
        ipvt = np.zeros(n, dtype=np.int64)

        # dgefa: gaussian elimination with partial pivoting
        nm1 = n - 1
        if nm1 >= 1:
            for k in range(1, nm1 + 1):
                kp1 = k + 1
                # l = idamax(n-k+1, a(k,k), 1) + k - 1
                col = a[k - 1:n, k - 1]
                l = int(np.argmax(np.abs(col))) + k
                ipvt[k - 1] = l
                if a[l - 1, k - 1] != 0.0:
                    if l != k:
                        t = a[l - 1, k - 1]
                        a[l - 1, k - 1] = a[k - 1, k - 1]
                        a[k - 1, k - 1] = t
                    t = -1.0 / a[k - 1, k - 1]
                    a[k:n, k - 1] = a[k:n, k - 1] * t
                    for j in range(kp1, n + 1):
                        t = a[l - 1, j - 1]
                        if l != k:
                            a[l - 1, j - 1] = a[k - 1, j - 1]
                            a[k - 1, j - 1] = t
                        a[k:n, j - 1] = a[k:n, j - 1] + t * a[k:n, k - 1]
                # else: zero pivot -- column already triangularized, nothing to do
        ipvt[n - 1] = n

        # dgedi(job=01): compute inverse(U) in place
        for k in range(1, n + 1):
            a[k - 1, k - 1] = 1.0 / a[k - 1, k - 1]
            t = -a[k - 1, k - 1]
            if k - 1 >= 1:
                a[0:k - 1, k - 1] = a[0:k - 1, k - 1] * t
            kp1 = k + 1
            if n >= kp1:
                for j in range(kp1, n + 1):
                    t = a[k - 1, j - 1]
                    a[k - 1, j - 1] = 0.0
                    a[0:k, j - 1] = a[0:k, j - 1] + t * a[0:k, k - 1]

        # form inverse(U) * inverse(L)
        nm1 = n - 1
        if nm1 >= 1:
            work = np.zeros(n, dtype=np.float64)
            for kb in range(1, nm1 + 1):
                k = n - kb
                kp1 = k + 1
                for i in range(kp1, n + 1):
                    work[i - 1] = a[i - 1, k - 1]
                    a[i - 1, k - 1] = 0.0
                for j in range(kp1, n + 1):
                    t = work[j - 1]
                    a[:, k - 1] = a[:, k - 1] + t * a[:, j - 1]
                l = int(ipvt[k - 1])
                if l != k:
                    tmp_col = a[:, k - 1].copy()
                    a[:, k - 1] = a[:, l - 1]
                    a[:, l - 1] = tmp_col
        return a

    # For each grid point, assemble Smat/Umat from the binned moments, invert
    # Smat via the LINPACK port above, and combine with Umat to get the
    # SS^T diagonal entry.
    for k in range(1, M + 1):
        Smat = np.zeros((pp, pp), dtype=np.float64)
        Umat = np.zeros((pp, pp), dtype=np.float64)
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k - 1, indss - 1]
                Umat[i - 1, j - 1] = uu[k - 1, indss - 1]

        Smat_inv = _linpack_inverse(Smat, pp)

        sstd_k = 0.0
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                sstd_k += Smat_inv[0, i - 1] * Umat[i - 1, j - 1] * Smat_inv[j - 1, 0]
        SSTd[k - 1] = sstd_k

    # list(x = gpoints, y = SSTd)
    return {"x": gpoints, "y": SSTd}


def on_attach(libname: str, pkgname: str) -> None:
    # R's packageStartupMessage() writes a message to the "message" connection,
    # which by default prints to stderr. We mirror that behavior here using
    # print(..., file=sys.stderr) since Python has no direct equivalent of a
    # package-attach hook or packageStartupMessage().
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def _on_unload(libpath: str) -> None:
    # R's .onUnload hook calls library.dynam.unload("KernSmooth", libpath) to
    # explicitly unload the compiled shared library when the package is detached.
    #
    # CPython provides no safe, stable API to unload a native/compiled extension
    # module (there is no equivalent of R's dyn.unload() for extension modules);
    # the import system manages the extension's lifetime automatically for the
    # lifetime of the interpreter. Therefore this hook intentionally performs no
    # action -- omitting the unload call is the correct behavior, not a gap.
    pass
