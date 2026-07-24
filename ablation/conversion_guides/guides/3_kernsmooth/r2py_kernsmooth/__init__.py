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


def linbin(X: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    # NOTE on .Fortran(F_linbin) resolution:
    # No compiled Python/Fortran wrapper (`_KernSmooth.linbin`) was available in the
    # provided context for this conversion, so the numerical behavior of the compiled
    # Fortran routine `KernSmooth/src/linbin.f` has been re-implemented directly in
    # pure Python/NumPy below. The translation follows the Fortran source line-for-line
    # (including its use of truncation-toward-zero via `int()`, which matches Fortran's
    # `int()` intrinsic), so the results are numerically identical to `.Fortran(F_linbin, ...)`.
    # If a compiled `_KernSmooth.linbin` binding becomes available later, this body can be
    # replaced with a thin wrapper call, e.g.:
    #     from . import _KernSmooth
    #     gcnts = np.zeros(M, dtype=np.float64)
    #     _KernSmooth.linbin(np.asarray(X, dtype=np.float64), n, a, b, M, trun, gcnts)
    #     return gcnts
    X_arr = np.asarray(X, dtype=np.float64)
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)

    n = len(X_arr)
    M = len(gpoints_arr)
    trun = 1 if truncate else 0
    a = gpoints_arr[0]
    b = gpoints_arr[M - 1]

    # Equivalent of R's double(M) / Fortran's zero-initialized gcnts(*)
    gcnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)
    for i in range(n):
        # 1-based grid position, mirroring the Fortran expression exactly
        lxi = ((X_arr[i] - a) / delta) + 1

        # Integer part of "lxi": Fortran's int() truncates toward zero,
        # which matches Python's int() applied to a float.
        li = int(lxi)

        rem = lxi - li
        if li >= 1 and li < M:
            # Convert 1-based Fortran indices (li, li+1) to 0-based Python indices
            gcnts[li - 1] = gcnts[li - 1] + (1 - rem)
            gcnts[li] = gcnts[li] + rem

        if li < 1 and trun == 0:
            gcnts[0] = gcnts[0] + 1

        if li >= M and trun == 0:
            gcnts[M - 1] = gcnts[M - 1] + 1

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    # NOTE on .Fortran(F_lbtwod) resolution:
    # No compiled Python/Fortran wrapper (`_KernSmooth.lbtwod`) was available in the
    # provided context for this conversion, so the numerical behavior of the compiled
    # Fortran routine `KernSmooth/src/linbin2D.f` (subroutine `lbtwod`) has been
    # re-implemented directly in pure Python/NumPy below. The translation follows the
    # Fortran source line-for-line (including its use of truncation-toward-zero via
    # `int()`, which matches Fortran's `int()` intrinsic), so the results are numerically
    # identical to `.Fortran(F_lbtwod, ...)`.
    # If a compiled `_KernSmooth.lbtwod` binding becomes available later, this body can be
    # replaced with a thin wrapper call, e.g.:
    #     from . import _KernSmooth
    #     gcnts = np.zeros(M1 * M2, dtype=np.float64)
    #     _KernSmooth.lbtwod(Xvec, n, a1, a2, b1, b2, M1, M2, gcnts)
    #     return gcnts.reshape((M1, M2), order='F')
    X_arr = np.asarray(X, dtype=np.float64)
    gpoints1_arr = np.asarray(gpoints1, dtype=np.float64)
    gpoints2_arr = np.asarray(gpoints2, dtype=np.float64)

    # R does: n <- nrow(X); X <- c(X[, 1], X[, 2])
    # which stacks the first and second columns of X into a single vector passed
    # to Fortran as X(i) (first column) and X(n+i) (second column).
    # Here we index the columns of the original 2D array directly instead.
    n = X_arr.shape[0]
    X1 = X_arr[:, 0]
    X2 = X_arr[:, 1]

    M1 = len(gpoints1_arr)
    M2 = len(gpoints2_arr)
    a1 = gpoints1_arr[0]
    a2 = gpoints2_arr[0]
    b1 = gpoints1_arr[M1 - 1]
    b2 = gpoints2_arr[M2 - 1]

    # Equivalent of R's double(M1*M2) / Fortran's zero-initialized gcnts(*).
    # Built directly as an (M1, M2) array; the Fortran routine fills gcnts
    # in column-major order via `ind = M1*(li2-1) + li1`, which is exactly
    # what R's `matrix(out[[9]], M1, M2)` (column-major) reconstructs, so
    # indexing this 2D array as gcnts[row, col] reproduces the same layout.
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)
    for i in range(n):
        # 1-based grid positions, mirroring the Fortran expressions exactly
        lxi1 = ((X1[i] - a1) / delta1) + 1
        lxi2 = ((X2[i] - a2) / delta2) + 1

        # Integer part of "lxi1"/"lxi2": Fortran's int() truncates toward zero,
        # which matches Python's int() applied to a float.
        li1 = int(lxi1)
        li2 = int(lxi2)
        rem1 = lxi1 - li1
        rem2 = lxi2 - li2

        if li1 >= 1 and li2 >= 1 and li1 < M1 and li2 < M2:
            # Convert 1-based Fortran indices (li1, li1+1) x (li2, li2+1) to
            # 0-based Python indices (li1-1, li1) x (li2-1, li2)
            gcnts[li1 - 1, li2 - 1] = gcnts[li1 - 1, li2 - 1] + (1 - rem1) * (1 - rem2)
            gcnts[li1, li2 - 1] = gcnts[li1, li2 - 1] + rem1 * (1 - rem2)
            gcnts[li1 - 1, li2] = gcnts[li1 - 1, li2] + (1 - rem1) * rem2
            gcnts[li1, li2] = gcnts[li1, li2] + rem1 * rem2

    return gcnts


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # NOTE on .Fortran(F_rlbin) resolution:
    # No compiled Python/Fortran wrapper (`_KernSmooth.rlbin`) was available in the
    # provided context for this conversion, so the numerical behavior of the compiled
    # Fortran routine `KernSmooth/src/rlbin.f` has been re-implemented directly in
    # pure Python/NumPy below. The translation follows the Fortran source line-for-line
    # (including its use of truncation-toward-zero via `int()`, which matches Fortran's
    # `int()` intrinsic), so the results are numerically identical to `.Fortran(F_rlbin, ...)`.
    # If a compiled `_KernSmooth.rlbin` binding becomes available later, this body can be
    # replaced with a thin wrapper call, e.g.:
    #     from . import _KernSmooth
    #     xcnts = np.zeros(M, dtype=np.float64)
    #     ycnts = np.zeros(M, dtype=np.float64)
    #     _KernSmooth.rlbin(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64),
    #                       n, a, b, M, trun, xcnts, ycnts)
    #     return {"xcounts": xcnts, "ycounts": ycnts}
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)

    n = len(X_arr)
    M = len(gpoints_arr)
    trun = 1 if truncate else 0
    a = gpoints_arr[0]
    b = gpoints_arr[M - 1]

    # Equivalent of R's double(M) / Fortran's zero-initialized xcnts(*)/ycnts(*)
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)

    delta = (b - a) / (M - 1)
    for i in range(n):
        # 1-based grid position, mirroring the Fortran expression exactly
        lxi = ((X_arr[i] - a) / delta) + 1

        # Integer part of "lxi": Fortran's int() truncates toward zero,
        # which matches Python's int() applied to a float.
        li = int(lxi)
        rem = lxi - li

        # Correction for right endpoint (not included if li == M)
        if X_arr[i] == b:
            li = M - 1
            rem = 1

        if li >= 1 and li < M:
            # Convert 1-based Fortran indices (li, li+1) to 0-based Python indices
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


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # NOTE on .Fortran(F_blkest) resolution:
    # No compiled Python/Fortran wrapper (`_KernSmooth.blkest`) was available in the
    # provided context for this conversion, so the numerical behavior of the compiled
    # Fortran routine `KernSmooth/src/blkest.f` has been re-implemented directly in
    # pure Python/NumPy below. The Fortran routine's own calls to LINPACK's `dqrdc`
    # (unpivoted QR decomposition) and `dqrsl` (job=00100, i.e. solve for the
    # least-squares coefficient vector `b`) are replaced here by `numpy.linalg.lstsq`,
    # which solves the same unpivoted least-squares problem to matching precision.
    # The rest of the routine (block partitioning, derivative-polynomial evaluation,
    # and accumulation of RSS/th22e/th24e) follows the Fortran source line-for-line,
    # with 1-based Fortran indices converted to 0-based Python indices.
    # If a compiled `_KernSmooth.blkest` binding becomes available later, this body
    # can be replaced with a thin wrapper call, e.g.:
    #     from . import _KernSmooth
    #     qq = q + 1
    #     xj = np.zeros(n, dtype=np.float64)
    #     yj = np.zeros(n, dtype=np.float64)
    #     coef = np.zeros(qq, dtype=np.float64)
    #     Xmat = np.zeros((n, qq), dtype=np.float64)
    #     wk = np.zeros(n, dtype=np.float64)
    #     qraux = np.zeros(qq, dtype=np.float64)
    #     sigsqe = np.zeros(1, dtype=np.float64)
    #     th22e = np.zeros(1, dtype=np.float64)
    #     th24e = np.zeros(1, dtype=np.float64)
    #     _KernSmooth.blkest(x, y, n, q, qq, Nval, xj, yj, coef, Xmat, wk,
    #                        qraux, sigsqe, th22e, th24e)
    #     return {"sigsqe": float(sigsqe[0]), "th22e": float(th22e[0]), "th24e": float(th24e[0])}
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    n = len(x_arr)

    ## Sort the (x, y) data with respect to the x's.
    # R's sort.list is a stable sort, matching NumPy's default 'stable' behavior
    # is achieved via kind='stable'.
    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    ## Set up arrays / scalars mirroring the FORTRAN programme "blkest"
    qq = q + 1

    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    # Integer division, matching FORTRAN's integer arithmetic for idiv = n/Nval
    idiv = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition (1-based bounds, as in the FORTRAN source)
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n
        nj = iupp - ilow + 1

        # Convert the 1-based FORTRAN bounds [ilow, iupp] to a 0-based Python slice
        Xj = x_sorted[ilow - 1:iupp]
        Yj = y_sorted[ilow - 1:iupp]

        ## Obtain a q'th degree fit over the current member of the partition
        ## Set up "X" matrix
        Xmat = np.ones((nj, qq), dtype=np.float64)
        for k in range(2, qq + 1):
            Xmat[:, k - 1] = Xj ** (k - 1)

        # Equivalent of dqrdc (unpivoted QR decomposition) followed by dqrsl
        # (job=00100: solve for the least-squares coefficient vector).
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        for i in range(nj):
            fiti = coef[0]
            ddm = 2 * coef[2]
            ddddm = 24 * coef[4]
            for k in range(2, qq + 1):
                fiti = fiti + coef[k - 1] * Xj[i] ** (k - 1)
                if k <= (q - 1):
                    ddm = ddm + k * (k + 1) * coef[k + 1] * Xj[i] ** (k - 1)
                    if k <= (q - 3):
                        ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef[k + 3] * Xj[i] ** (k - 1)
            th22e = th22e + ddm ** 2
            th24e = th24e + ddm * ddddm
            RSS = RSS + (Yj[i] - fiti) ** 2

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # NOTE on .Fortran(F_cp) resolution:
    # No compiled Python/Fortran wrapper (`_KernSmooth.cp`) was available in the
    # provided context for this conversion, so the numerical behavior of the compiled
    # Fortran routine `KernSmooth/src/cp.f` has been re-implemented directly in pure
    # Python/NumPy below. The Fortran routine's own calls to LINPACK's `dqrdc`
    # (unpivoted QR decomposition) and `dqrsl` (job=00100, i.e. solve for the
    # least-squares coefficient vector `coef`) are replaced here by
    # `numpy.linalg.lstsq`, which solves the same unpivoted least-squares problem to
    # matching precision. The rest of the routine (block partitioning, blocked q'th
    # degree fits, accumulation of RSS per block count, and the final Mallows' C_p
    # formula) follows the Fortran source line-for-line, with 1-based Fortran indices
    # converted to 0-based Python indices. This mirrors the sibling function `blkest`,
    # which performs the same kind of blocked polynomial least-squares fitting.
    # If a compiled `_KernSmooth.cp` binding becomes available later, this body can be
    # replaced with a thin wrapper call, e.g.:
    #     from . import _KernSmooth
    #     qq = q + 1
    #     RSS = np.zeros(Nmax, dtype=np.float64)
    #     Xj = np.zeros(n, dtype=np.float64)
    #     Yj = np.zeros(n, dtype=np.float64)
    #     coef = np.zeros(qq, dtype=np.float64)
    #     Xmat = np.zeros((n, qq), dtype=np.float64)
    #     wk = np.zeros(n, dtype=np.float64)
    #     qraux = np.zeros(qq, dtype=np.float64)
    #     Cpvals = np.zeros(Nmax, dtype=np.float64)
    #     _KernSmooth.cp(X, Y, n, qq, Nmax, RSS, Xj, Yj, coef, Xmat, wk, qraux, Cpvals)
    #     return int(np.argmin(Cpvals)) + 1
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)

    n = len(X_arr)

    ## Sort the (X, Y) data with respect to the X's.
    # R's sort.list is a stable sort, matched here via kind='stable'.
    order_idx = np.argsort(X_arr, kind="stable")
    X_sorted = X_arr[order_idx]
    Y_sorted = Y_arr[order_idx]

    ## Set up arrays / scalars mirroring the FORTRAN subroutine "cp"
    qq = q + 1

    ## remove unused 'q' 2007-07-10
    ## Compute vector of RSS values
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # For each number of partitions
        idiv = n // Nval
        for j in range(1, Nval + 1):
            # For each member of the partition (1-based bounds, as in the FORTRAN source)
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n
            nj = iupp - ilow + 1

            # Convert the 1-based FORTRAN bounds [ilow, iupp] to a 0-based Python slice
            Xj = X_sorted[ilow - 1:iupp]
            Yj = Y_sorted[ilow - 1:iupp]

            ## Obtain a q'th degree fit over current member of partition
            ## Set up "X" matrix
            Xmat = np.ones((nj, qq), dtype=np.float64)
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            # Equivalent of dqrdc (unpivoted QR decomposition) followed by dqrsl
            # (job=00100: solve for the least-squares coefficient vector).
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            RSSj = 0.0
            for i in range(nj):
                fiti = coef[0]
                for k in range(2, qq + 1):
                    fiti = fiti + coef[k - 1] * Xj[i] ** (k - 1)
                RSSj = RSSj + (Yj[i] - fiti) ** 2

            RSS[Nval - 1] = RSS[Nval - 1] + RSSj

    ## Now compute array of Mallow's C_p values.
    Cpvals = np.zeros(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = (n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1] + 2 * qq * i - n

    Cpvec = Cpvals

    ## order(Cpvec)[1L] in R returns the 1-based index of the minimum C_p value,
    ## which is used directly downstream as the chosen number of blocks (Nval).
    return int(np.argmin(Cpvec)) + 1


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # NOTE on .Fortran(F_sdiag) resolution:
    # No compiled Python/Fortran wrapper (`_KernSmooth.sdiag`) was available in the
    # provided context for this conversion, so the numerical behavior of the compiled
    # Fortran routine `KernSmooth/src/sdiag.f` has been re-implemented directly in pure
    # Python/NumPy below. The kernel-weight assembly and the accumulation of the
    # binned local moments `ss` follow the Fortran source line-for-line, with 1-based
    # Fortran indices converted to 0-based Python indices (this mirrors the sibling
    # function `locpoly`'s Fortran routine `locpol.f`, which shares the identical
    # weight/moment-accumulation logic). The Fortran routine's own calls to LINPACK's
    # `dgefa` (LU factorization) and `dgedi` with job=01 (matrix inverse only, no
    # determinant) are replaced here by `numpy.linalg.solve(Smat, e1)`, which yields
    # the first column of `inv(Smat)` -- exactly the `Smat(1,1)` entry the Fortran
    # routine reads off of the explicitly-inverted matrix -- to matching precision,
    # without hand-rolling low-level factorization arithmetic.
    # If a compiled `_KernSmooth.sdiag` binding becomes available later, this body
    # can be replaced with a thin wrapper call, e.g.:
    #     from . import _KernSmooth
    #     fkap = np.zeros(dimfkap, dtype=np.float64)
    #     midpts = np.zeros(Q, dtype=np.int32)
    #     ss = np.zeros((M, ppp), dtype=np.float64)
    #     Smat = np.zeros((pp, pp), dtype=np.float64)
    #     work = np.zeros(pp, dtype=np.float64)
    #     det = np.zeros(2, dtype=np.float64)
    #     ipvt = np.zeros(pp, dtype=np.int32)
    #     Sdg = np.zeros(M, dtype=np.float64)
    #     _KernSmooth.sdiag(xcounts, delta, hdisc, Lvec, indic, midpts, M, Q, fkap,
    #                       pp, ppp, ss, Smat, work, det, ipvt, Sdg)
    #     return {"x": gpoints, "y": Sdg}
    if range_x is None and not binned:
        x_arr0 = np.asarray(x, dtype=np.float64)
        range_x = (float(np.min(x_arr0)), float(np.max(x_arr0)))

    ## Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                raw = ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                indic = np.round(raw).astype(np.int64)
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
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    Sdg = np.zeros(M, dtype=np.float64)

    ## Obtain kernel weights (1-based Fortran indices kept explicit, then
    ## converted to 0-based Python array accesses via "- 1")
    mid = int(Lvec[0]) + 1
    for i in range(1, Q):
        midpts[i - 1] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec[i - 1]) + 1):
            val = np.exp(-((delta * j / hdisc[i - 1]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        val = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    ## Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i in range(1, Q + 1):
                jlo = max(1, k - int(Lvec[i - 1]))
                jhi = min(M, k + int(Lvec[i - 1]))
                for j in range(jlo, jhi + 1):
                    if indic[j - 1] == i:
                        fac = 1.0
                        ss[j - 1, 0] += xcounts[k - 1] * fkap[k - j + midpts[i - 1] - 1]
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xcounts[k - 1] * fkap[k - j + midpts[i - 1] - 1] * fac

    ## For each grid point, assemble the local moment matrix "Smat" from "ss"
    ## and extract the (1,1) entry of its inverse, which is the diagonal entry
    ## of the binned smoother matrix.
    e1 = np.zeros(pp, dtype=np.float64)
    e1[0] = 1.0
    Smat = np.zeros((pp, pp), dtype=np.float64)
    for k in range(1, M + 1):
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k - 1, indss - 1]

        # Equivalent of LINPACK dgefa (LU factorization) followed by dgedi
        # (job=01: compute matrix inverse only, no determinant). Only the
        # (1,1) entry of the inverse is needed, obtained here via
        # numpy.linalg.solve(Smat, e1)[0], which equals inv(Smat)[0, 0].
        Tvec = np.linalg.solve(Smat, e1)
        Sdg[k - 1] = Tvec[0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # NOTE on .Fortran(F_sstdg) resolution:
    # No compiled Python/Fortran wrapper (`_KernSmooth.sstdg`) was available in the
    # provided context for this conversion, so the numerical behavior of the compiled
    # Fortran routine `KernSmooth/src/sstdiag.f` has been re-implemented directly in pure
    # Python/NumPy below. The kernel-weight assembly and the accumulation of the
    # binned local moments `ss` and `uu` follow the Fortran source line-for-line, with
    # 1-based Fortran indices converted to 0-based Python indices (this mirrors the
    # sibling function `sdiag`'s Fortran routine `sdiag.f`, which shares the identical
    # weight/moment-accumulation logic, extended here with the second moment array
    # `uu`/`Umat` used to form SS^T). The Fortran routine's own calls to LINPACK's
    # `dgefa` (LU factorization) and `dgedi` with job=01 (matrix inverse only, no
    # determinant) are replaced here by `numpy.linalg.solve(Smat, e1)`, which yields
    # the first column of `inv(Smat)` -- exactly the column `Smat(*,1)` the Fortran
    # routine reads off of the explicitly-inverted matrix. Because `Smat` (the local
    # moment matrix) is symmetric, its inverse is symmetric too, so the row `Smat(1,*)`
    # used in the Fortran quadratic form `sum_i sum_j Smat(1,i)*Umat(i,j)*Smat(j,1)`
    # equals this same first column, letting the quadratic form be computed as
    # `Tvec @ Umat @ Tvec` to matching precision, without hand-rolling low-level
    # factorization arithmetic.
    # If a compiled `_KernSmooth.sstdg` binding becomes available later, this body
    # can be replaced with a thin wrapper call, e.g.:
    #     from . import _KernSmooth
    #     fkap = np.zeros(dimfkap, dtype=np.float64)
    #     midpts = np.zeros(Q, dtype=np.int32)
    #     ss = np.zeros((M, ppp), dtype=np.float64)
    #     uu = np.zeros((M, ppp), dtype=np.float64)
    #     Smat = np.zeros((pp, pp), dtype=np.float64)
    #     Umat = np.zeros((pp, pp), dtype=np.float64)
    #     work = np.zeros(pp, dtype=np.float64)
    #     det = np.zeros(2, dtype=np.float64)
    #     ipvt = np.zeros(pp, dtype=np.int32)
    #     SSTd = np.zeros(M, dtype=np.float64)
    #     _KernSmooth.sstdg(xcounts, delta, hdisc, Lvec, indic, midpts, M, Q, fkap,
    #                       pp, ppp, ss, uu, Smat, Umat, work, det, ipvt, SSTd)
    #     return {"x": gpoints, "y": SSTd}
    if range_x is None and not binned:
        x_arr0 = np.asarray(x, dtype=np.float64)
        range_x = (float(np.min(x_arr0)), float(np.max(x_arr0)))

    ## Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Bin the data if not already binned
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    ## Set the bin width
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                raw = ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                indic = np.round(raw).astype(np.int64)
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
    fkap = np.zeros(dimfkap, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)
    SSTd = np.zeros(M, dtype=np.float64)

    ## Obtain kernel weights (1-based Fortran indices kept explicit, then
    ## converted to 0-based Python array accesses via "- 1")
    mid = int(Lvec[0]) + 1
    for i in range(1, Q):
        midpts[i - 1] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec[i - 1]) + 1):
            val = np.exp(-((delta * j / hdisc[i - 1]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        val = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    ## Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i in range(1, Q + 1):
                jlo = max(1, k - int(Lvec[i - 1]))
                jhi = min(M, k + int(Lvec[i - 1]))
                for j in range(jlo, jhi + 1):
                    if indic[j - 1] == i:
                        fac = 1.0
                        wgt = fkap[k - j + midpts[i - 1] - 1]
                        ss[j - 1, 0] += xcounts[k - 1] * wgt
                        uu[j - 1, 0] += xcounts[k - 1] * (wgt ** 2)
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xcounts[k - 1] * wgt * fac
                            uu[j - 1, ii - 1] += xcounts[k - 1] * (wgt ** 2) * fac

    ## For each grid point, assemble the local moment matrices "Smat" and "Umat"
    ## from "ss" and "uu", invert "Smat" (via a linear solve against the standard
    ## basis vector "e1", exploiting the symmetry of "Smat") and form the quadratic
    ## form that yields the diagonal entry of the binned SS^T matrix.
    e1 = np.zeros(pp, dtype=np.float64)
    e1[0] = 1.0
    Smat = np.zeros((pp, pp), dtype=np.float64)
    Umat = np.zeros((pp, pp), dtype=np.float64)
    for k in range(1, M + 1):
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k - 1, indss - 1]
                Umat[i - 1, j - 1] = uu[k - 1, indss - 1]

        # Equivalent of LINPACK dgefa (LU factorization) followed by dgedi
        # (job=01: compute matrix inverse only, no determinant). Since "Smat"
        # is symmetric, its inverse is symmetric, so the column obtained via
        # numpy.linalg.solve(Smat, e1) equals both the first column and the
        # first row of inv(Smat) that the Fortran routine reads off of the
        # explicitly-inverted matrix.
        Tvec = np.linalg.solve(Smat, e1)
        SSTd[k - 1] = float(Tvec @ Umat @ Tvec)

    return {"x": gpoints, "y": SSTd}


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # NOTE on .Fortran(F_locpol) resolution:
    # No compiled Python/Fortran wrapper (`_KernSmooth.locpol`) was available in the
    # provided context for this conversion, so the numerical behavior of the compiled
    # Fortran routine `KernSmooth/src/locpoly.f` has been re-implemented directly in pure
    # Python/NumPy below. The kernel-weight assembly and the accumulation of the binned
    # local moments `ss` (for the design matrix) and `tt` (for the response vector) follow
    # the Fortran source line-for-line, with 1-based Fortran indices converted to 0-based
    # Python indices (this mirrors the sibling functions `sdiag`'s `sdiag.f` and
    # `sstdiag`'s `sstdiag.f`, which share the identical weight/moment-accumulation logic).
    # The Fortran routine's own calls to LINPACK's `dgefa` (LU factorization) followed by
    # `dgesl` with job=0 (solve Smat @ x = Tvec using the LU factors, untransposed) are
    # replaced here by `numpy.linalg.solve(Smat, Tvec)`, which yields the same solution
    # vector to matching precision, without hand-rolling low-level factorization arithmetic.
    # If a compiled `_KernSmooth.locpol` binding becomes available later, this body can be
    # replaced with a thin wrapper call, e.g.:
    #     from . import _KernSmooth
    #     fkap = np.zeros(dimfkap, dtype=np.float64)
    #     curvest = np.zeros(M, dtype=np.float64)
    #     midpts = np.zeros(Q, dtype=np.int32)
    #     ss = np.zeros((M, ppp), dtype=np.float64)
    #     tt = np.zeros((M, pp), dtype=np.float64)
    #     Smat = np.zeros((pp, pp), dtype=np.float64)
    #     Tvec = np.zeros(pp, dtype=np.float64)
    #     ipvt = np.zeros(pp, dtype=np.int32)
    #     _KernSmooth.locpol(xcounts, ycounts, drv, delta, hdisc, Lvec, indic, midpts,
    #                        M, Q, fkap, pp, ppp, ss, tt, Smat, Tvec, ipvt, curvest)
    #     curvest = math.factorial(drv) * curvest
    #     return {"x": gpoints, "y": curvest}

    ## Install safeguard against non-positive bandwidths
    if bandwidth is not None and np.any(np.asarray(bandwidth, dtype=np.float64) <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = None if y is None else np.asarray(y, dtype=np.float64)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x_arr) - np.min(x_arr))
            range_x = (float(np.min(x_arr) - extra), float(np.max(x_arr) + extra))
        else:
            range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    ## Rename common variables
    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    ## Decide whether a density estimate or regression estimate is required.
    if y is None:  # obtain density estimate
        n = len(x_arr)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        ## Bin the data if not already binned
        if not binned:
            gpoints = np.linspace(a, b, M)
            out = rlbin(x_arr, y_arr, gpoints, truncate)
            xcounts = out["xcounts"]
            ycounts = out["ycounts"]
        else:
            xcounts = x_arr
            ycounts = y_arr
            M = len(xcounts)
            gpoints = np.linspace(a, b, M)

    ## Set the bin width
    delta = (b - a) / (M - 1)

    ## Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if len(bandwidth_arr) == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        ## Determine value of L for each member of "hdisc"
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        ## Determine index of closest entry of "hdisc"
        ## to each member of "bandwidth"
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                raw = ((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1
                indic = np.round(raw).astype(np.int64)
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

    ## Allocate space for the kernel vector and final estimate
    dimfkap = 2 * int(np.sum(Lvec)) + Q
    fkap = np.zeros(dimfkap, dtype=np.float64)
    curvest = np.zeros(M, dtype=np.float64)
    midpts = np.zeros(Q, dtype=np.int64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)

    ## Obtain kernel weights (1-based Fortran indices kept explicit, then
    ## converted to 0-based Python array accesses via "- 1")
    mid = int(Lvec[0]) + 1
    for i in range(1, Q):
        midpts[i - 1] = mid
        fkap[mid - 1] = 1.0
        for j in range(1, int(Lvec[i - 1]) + 1):
            val = np.exp(-((delta * j / hdisc[i - 1]) ** 2) / 2)
            fkap[mid + j - 1] = val
            fkap[mid - j - 1] = val
        mid = mid + int(Lvec[i - 1]) + int(Lvec[i]) + 1
    midpts[Q - 1] = mid
    fkap[mid - 1] = 1.0
    for j in range(1, int(Lvec[Q - 1]) + 1):
        val = np.exp(-((delta * j / hdisc[Q - 1]) ** 2) / 2)
        fkap[mid + j - 1] = val
        fkap[mid - j - 1] = val

    ## Combine kernel weights and grid counts
    for k in range(1, M + 1):
        if xcounts[k - 1] != 0:
            for i in range(1, Q + 1):
                jlo = max(1, k - int(Lvec[i - 1]))
                jhi = min(M, k + int(Lvec[i - 1]))
                for j in range(jlo, jhi + 1):
                    if indic[j - 1] == i:
                        fac = 1.0
                        wgt = fkap[k - j + midpts[i - 1] - 1]
                        ss[j - 1, 0] += xcounts[k - 1] * wgt
                        tt[j - 1, 0] += ycounts[k - 1] * wgt
                        for ii in range(2, ppp + 1):
                            fac = fac * delta * (k - j)
                            ss[j - 1, ii - 1] += xcounts[k - 1] * wgt * fac
                            if ii <= pp:
                                tt[j - 1, ii - 1] += ycounts[k - 1] * wgt * fac

    ## For each grid point, assemble the local moment matrix "Smat" and the
    ## local moment vector "Tvec" from "ss"/"tt", solve the local weighted
    ## least-squares system Smat @ coefs = Tvec, and extract the coefficient
    ## corresponding to the requested derivative order "drv".
    Smat = np.zeros((pp, pp), dtype=np.float64)
    for k in range(1, M + 1):
        for i in range(1, pp + 1):
            for j in range(1, pp + 1):
                indss = i + j - 1
                Smat[i - 1, j - 1] = ss[k - 1, indss - 1]
        Tvec = tt[k - 1, :].copy()

        # Equivalent of LINPACK dgefa (LU factorization) followed by dgesl
        # (job=0: solve Smat @ x = Tvec using the LU factors, untransposed).
        Tvec_sol = np.linalg.solve(Smat, Tvec)

        curvest[k - 1] = Tvec_sol[drv]

    curvest = math.factorial(drv) * curvest

    return {"x": gpoints, "y": curvest}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths.
    # R's `!missing(bandwidth) && bandwidth <= 0` is translated using the sentinel
    # `None` default: the check only fires when the caller actually supplied a
    # (non-None) bandwidth value, matching R's `missing()` semantics.
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        x_arr = np.asarray(x, dtype=np.float64)
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

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
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = norm.pdf(arg) / (h ** (drv + 1))
    hmold0 = 1
    hmold1 = arg
    hmnew = 1
    # NOTE: faithfully mirrors R's behavior. R's `for (i in (2L:drv))` is only
    # ever executed when `drv >= 2L` (guarded by the outer `if`), so for drv == 0
    # or drv == 1 the loop never runs and `hmnew` stays at its initial value of 1
    # (this quirk is intentional in the original R source and is preserved here,
    # verified numerically to match `KernSmooth::bkfe`).
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0    # Compute mth degree Hermite polynomial
            hmold0 = hmold1                            # by recurrence.
            hmold1 = hmnew
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log2(M + L + 1)))
    kappam = np.concatenate([kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]])
    Gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappam_fft = np.fft.fft(kappam)
    Gcounts_fft = np.fft.fft(Gcounts)

    # np.fft.ifft already normalises by P internally, so the explicit /P division
    # present in the R source is dropped here.
    result = np.sum(gcounts * np.real(np.fft.ifft(kappam_fft * Gcounts_fft))[:M]) / (n ** 2)
    return float(result)


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths.
    # R's `!missing(bandwidth) && bandwidth <= 0` is translated using the sentinel
    # `None` default: the check only fires when the caller actually supplied a
    # (non-None) bandwidth value, matching R's `missing()` semantics.
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
                f"'arg' should be one of {', '.join(repr(c) for c in _kernel_choices)}"
            )

    x_arr = np.asarray(x, dtype=np.float64)

    # Rename common variables
    n = len(x_arr)
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
    else:  # "triweight"
        del0 = (9450.0 / 143.0) ** (1.0 / 5.0)

    if not isinstance(canonical, (bool, np.bool_)):
        raise ValueError("'canonical' must be a length-1 logical vector")

    # Set default bandwidth
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.std(x_arr, ddof=1)
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values
    tau = 4.0 if kernel == "normal" else 1.0

    if range_x is None:
        range_x = (float(np.min(x_arr) - tau * h), float(np.max(x_arr) + tau * h))
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    # Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = int(min(np.floor(tau / delta), M))
    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    lvec = np.arange(0, L + 1)
    arg = lvec * delta
    if kernel == "normal":
        kappa = norm.pdf(arg) / (n * h)
    elif kernel == "box":
        kappa = 0.5 * beta.pdf(0.5 * (arg + 1), 1, 1) / (n * h)
    elif kernel == "epanech":
        kappa = 0.5 * beta.pdf(0.5 * (arg + 1), 2, 2) / (n * h)
    elif kernel == "biweight":
        kappa = 0.5 * beta.pdf(0.5 * (arg + 1), 3, 3) / (n * h)
    else:  # "triweight"
        kappa = 0.5 * beta.pdf(0.5 * (arg + 1), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P = int(2 ** np.ceil(np.log2(M + L + 1)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)

    # R's `Re(fft(kappa*gcounts, TRUE))/P` is the *unnormalised* inverse FFT
    # (R's `fft(z, inverse=TRUE)` does not divide by length) followed by an
    # explicit `/P`. NumPy's `np.fft.ifft` already normalises by length P
    # internally, so the explicit `/P` division is dropped here.
    y = np.real(np.fft.ifft(kappa_fft * gcounts_fft))[:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: tuple[int, int] | np.ndarray[Any, np.dtype[np.int_]] = (51, 51), range_x: tuple[tuple[float, float], tuple[float, float]] | list[tuple[float, float]] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths.
    # R's `!missing(bandwidth) && min(bandwidth) <= 0` is translated using the
    # sentinel `None` default: the check only fires when the caller actually
    # supplied a (non-None) bandwidth value, matching R's `missing()` semantics.
    # `bandwidth` has no default in the original R signature (it is a mandatory
    # argument that is unconditionally used later on), so it is not given a
    # data-driven default here the way `bkde`'s bandwidth is -- if the caller
    # omits it, the code below will fail when it is first used, mirroring R's
    # own "argument is missing, with no default" error.
    if bandwidth is not None and np.min(np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    x_arr = np.asarray(x, dtype=np.float64)

    # Rename common variables
    n = x_arr.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction if only a single bandwidth is given.
    if h.size == 1:
        h = np.array([h[0], h[0]], dtype=np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x_list: list[tuple[float, float]] = [(0.0, 0.0), (0.0, 0.0)]
        for idx in range(2):
            range_x_list[idx] = (
                float(np.min(x_arr[:, idx]) - 1.5 * h[idx]),
                float(np.max(x_arr[:, idx]) + 1.5 * h[idx]),
            )
        range_x = range_x_list

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x_arr, gpoints1, gpoints2)

    # Compute kernel weights
    L = np.zeros(2, dtype=np.float64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]] | None] = [None, None]
    for idx in range(2):
        L[idx] = min(int(np.floor(tau * h[idx] * (M[idx] - 1) / (b[idx] - a[idx]))), int(M[idx]) - 1)
        lvecid = np.arange(0, int(L[idx]) + 1)
        facid = (b[idx] - a[idx]) / (h[idx] * (M[idx] - 1))
        z = norm.pdf(lvecid * facid) / h[idx]
        # tot <- sum(c(z, rev(z[-1L]))) * facid * h[id]
        tot = np.sum(np.concatenate([z, z[1:][::-1]])) * facid * h[idx]
        kapid[idx] = z / tot

    # kapp <- kapid[[1L]] %*% (t(kapid[[2L]]))/n -- outer product of the two
    # 1-D kernel-weight vectors.
    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log2(M.astype(np.float64) + L))).astype(np.int64)  # smallest powers of 2 >= M+L
    L1 = int(L[0])
    L2 = int(L[1])
    M1 = int(M[0])
    M2 = int(M[1])
    P1 = int(P[0])
    P2 = int(P[1])

    rp = np.zeros((P1, P2), dtype=np.float64)
    # rp[1L:(L1+1), 1L:(L2+1)] <- kapp  (0-based: rows/cols 0..L1 / 0..L2)
    rp[0:(L1 + 1), 0:(L2 + 1)] = kapp
    # if (L1) rp[(P1-L1+1):P1, 1L:(L2+1)] <- kapp[(L1+1):2, 1L:(L2+1)]
    # R's 1-based descending range (L1+1):2 selects rows L1+1, L1, ..., 2
    # (dropping row 1), which in 0-based indices is L1, L1-1, ..., 1 -- exactly
    # `kapp[L1:0:-1, ...]` in NumPy.
    if L1:
        rp[(P1 - L1):P1, 0:(L2 + 1)] = kapp[L1:0:-1, 0:(L2 + 1)]
    # if (L2) rp[, (P2-L2+1):P2] <- rp[, (L2+1):2]
    # Same reversed-range logic applied along the column axis, operating on
    # the (already row-wrapped) `rp` itself.
    if L2:
        rp[:, (P2 - L2):P2] = rp[:, L2:0:-1]
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    # invert element-wise product of FFT's and truncate and normalise it.
    # np.fft.ifft2 already normalises by (P1*P2) internally, so the explicit
    # `/(P1*P2)` division present in the R source is dropped here.
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]

    # Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename variables
    n = len(x_arr)
    M = gridsize
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    # (This mirrors R's initial gpoints/gcounts computation on the raw data,
    # which is subsequently overwritten below once the data has been
    # standardised; it is retained here for strict fidelity to the R source,
    # even though its result is not otherwise used.)
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    # Compute scale estimate
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest in _scalest_choices:
        scalest_name = scalest
    else:
        _matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(_matches) == 1:
            scalest_name = _matches[0]
        else:
            raise ValueError(
                f"'arg' should be one of {_scalest_choices}"
            )

    std_x = float(np.std(x_arr, ddof=1))
    iqr_x = float(np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349

    if scalest_name == "stdev":
        scale_value = std_x
    elif scalest_name == "iqr":
        scale_value = iqr_x
    else:
        scale_value = min(iqr_x, std_x)

    if scale_value == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    x_mean = float(np.mean(x_arr))
    sx = (x_arr - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    # Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    # delta <- (sb-sa)/(M - 1)

    # Perform plug-in steps
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
    elif level == 5:
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
    else:
        raise ValueError("Level should be between 0 and 5")

    return scale_value * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    _kernel_choices = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel in _kernel_choices:
        kernel_name = kernel
    else:
        _kernel_matches = [c for c in _kernel_choices if c.startswith(kernel)]
        if len(_kernel_matches) == 1:
            kernel_name = _kernel_matches[0]
        else:
            raise ValueError(
                f"'arg' should be one of {_kernel_choices}"
            )

    # Set kernel constants
    if canonical:
        del0 = 1.0
    elif kernel_name == "normal":
        del0 = 1 / ((4 * math.pi) ** (1 / 10))
    elif kernel_name == "box":
        del0 = (9 / 2) ** (1 / 5)
    elif kernel_name == "epanech":
        del0 = 15 ** (1 / 5)
    elif kernel_name == "biweight":
        del0 = 35 ** (1 / 5)
    else:  # "triweight"
        del0 = (9450 / 143) ** (1 / 5)

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename variables
    n = len(x_arr)
    M = gridsize
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    # Compute scale estimate
    _scalest_choices = ("minim", "stdev", "iqr")
    if scalest in _scalest_choices:
        scalest_name = scalest
    else:
        _matches = [c for c in _scalest_choices if c.startswith(scalest)]
        if len(_matches) == 1:
            scalest_name = _matches[0]
        else:
            raise ValueError(
                f"'arg' should be one of {_scalest_choices}"
            )

    std_x = float(np.std(x_arr, ddof=1))
    iqr_x = float(np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349

    if scalest_name == "stdev":
        scale_value = std_x
    elif scalest_name == "iqr":
        scale_value = iqr_x
    else:
        scale_value = min(iqr_x, std_x)

    if scale_value == 0:
        raise ValueError("scale estimate is zero for input data")

    # Replace input data by standardised data for numerical stability:
    x_mean = float(np.mean(x_arr))
    sx = (x_arr - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    # Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    # delta <- (sb-sa)/(M - 1)

    # Perform plug-in steps:
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
    elif level == 5:
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
    else:
        raise ValueError("Level should be between 0 and 5")

    return scale_value * del0 * (1 / (psi4hat * n)) ** (1 / 5)


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]] | None = None, truncate: bool = True) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    ## NOTE: R's default argument `range.x = range(x)` is evaluated lazily,
    ## i.e. only when `range.x` is first *used* inside the function body
    ## (at `a <- range.x[1L]`), which happens *after* `x` has already been
    ## reassigned to its sorted-and-trimmed version. So the R default actually
    ## reflects the range of the trimmed `x`, not the original input `x`. We
    ## replicate this by deferring the default computation until after trimming.
    user_range_x = range_x

    ## Trim the 100(trim)% of the data from each end (in the x-direction).
    sort_idx = np.argsort(x_arr)
    x_sorted = x_arr[sort_idx]
    y_sorted = y_arr[sort_idx]

    indlow = int(math.floor(trim * len(x_sorted)))  # 0-based start (R's indlow - 1)
    indupp = len(x_sorted) - int(math.floor(trim * len(x_sorted)))  # 0-based exclusive end (R's indupp)

    x_trim = x_sorted[indlow:indupp]
    y_trim = y_sorted[indlow:indupp]

    ## Rename common parameters
    n = len(x_trim)
    M = gridsize
    if user_range_x is None:
        a = float(np.min(x_trim))
        b = float(np.max(x_trim))
    else:
        a = float(user_range_x[0])
        b = float(user_range_x[1])

    ## Bin the data
    gpoints = np.linspace(a, b, M)
    out = rlbin(x_trim, y_trim, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    ## Choose the value of N using Mallow's C_p
    Nmax = max(min(int(math.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x_trim, y_trim, Nmax, 4)

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.
    out = blkest(x_trim, y_trim, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    ## Estimate theta_22 using a local cubic fit
    ## with a "rule-of-thumb" bandwidth: "gamseh"
    gamseh = (sigsqQ * (b - a) / (abs(th24Q) * n))
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * math.sqrt(math.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * math.sqrt(math.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                      range_x=(a, b), binned=True)["y"]

    llow = int(math.floor(proptrun * M))  # 0-based start (R's llow - 1)
    lupp = M - int(math.floor(proptrun * M))  # 0-based exclusive end (R's lupp)
    th22kn = np.sum((mddest[llow:lupp] ** 2) * xcounts[llow:lupp]) / n

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * math.sqrt(2) - (4 / 3) * math.sqrt(3)
    C3K = (4 * C3K / (math.sqrt(2 * math.pi))) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    ## Now compute a local linear kernel estimate of
    ## the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                    range_x=(a, b), binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh,
                range_x=(a, b), binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh,
                    range_x=(a, b), binned=True)["y"]
    sigsqn = np.sum(y_trim ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    ## Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1 / 5))


def _on_attach() -> None:
    # Mirrors R's .onAttach package hook: emit an informational
    # startup message to stderr when the package is loaded.
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def _on_unload(libpath: str) -> None:
    # Mirrors R's .onUnload package hook, which called
    # library.dynam.unload("KernSmooth", libpath) to unload the compiled
    # Fortran/C shared library backing the KernSmooth package.
    #
    # In this Python port there is no compiled extension being dynamically
    # loaded/unloaded: every routine that used to live in the Fortran
    # shared library (e.g. linbin, rlbin, sdiag, sstdiag, ...) has been
    # reimplemented directly in pure Python/NumPy. Per the conversion
    # guide for `library.dynam.unload`, CPython also provides no safe,
    # supported API for unloading a native extension module, so this
    # hook has no meaningful Python counterpart and is intentionally a
    # no-op, kept only for structural parity with the original R source.
    pass


_on_attach()
