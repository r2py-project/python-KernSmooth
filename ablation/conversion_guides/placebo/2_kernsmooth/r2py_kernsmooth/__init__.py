import sys
import warnings
from typing import Any

import numpy as np
from scipy.special import gamma
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
    X = np.asarray(X, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Fortran subroutine linbin: obtains bin counts for univariate data
    # via the linear binning strategy. If trun == 0, weight from end
    # observations is given to the corresponding end grid points; if
    # trun == 1, end observations are truncated.
    gcnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    lxi = (X - a) / delta + 1.0
    # Fortran's int() truncates toward zero, matching numpy's astype(int64)
    li = lxi.astype(np.int64)
    rem = lxi - li

    in_range = (li >= 1) & (li < M)
    li_in_range = li[in_range]
    rem_in_range = rem[in_range]
    np.add.at(gcnts, li_in_range - 1, 1.0 - rem_in_range)
    np.add.at(gcnts, li_in_range, rem_in_range)

    if trun == 0:
        gcnts[0] += np.count_nonzero(li < 1)
        gcnts[M - 1] += np.count_nonzero(li >= M)

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    gpoints1 = np.asarray(gpoints1, dtype=np.float64)
    gpoints2 = np.asarray(gpoints2, dtype=np.float64)

    X1 = X[:, 0]
    X2 = X[:, 1]
    M1 = len(gpoints1)
    M2 = len(gpoints2)
    a1 = gpoints1[0]
    a2 = gpoints2[0]
    b1 = gpoints1[M1 - 1]
    b2 = gpoints2[M2 - 1]

    # Fortran subroutine lbtwod: obtains bin counts for bivariate data via
    # the linear binning strategy. Observations outside the mesh are
    # ignored (no truncate/end-weighting option, unlike the univariate case).
    gcnts = np.zeros((M1, M2), dtype=np.float64)
    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    lxi1 = (X1 - a1) / delta1 + 1.0
    lxi2 = (X2 - a2) / delta2 + 1.0

    # Fortran's int() truncates toward zero, matching numpy's astype(int64)
    li1 = lxi1.astype(np.int64)
    li2 = lxi2.astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    in_range = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)
    li1_in = li1[in_range]
    li2_in = li2[in_range]
    rem1_in = rem1[in_range]
    rem2_in = rem2[in_range]

    # Convert Fortran's 1-based grid indices to 0-based row/col indices
    r1 = li1_in - 1
    r2 = li2_in - 1

    np.add.at(gcnts, (r1, r2), (1.0 - rem1_in) * (1.0 - rem2_in))
    np.add.at(gcnts, (r1 + 1, r2), rem1_in * (1.0 - rem2_in))
    np.add.at(gcnts, (r1, r2 + 1), (1.0 - rem1_in) * rem2_in)
    np.add.at(gcnts, (r1 + 1, r2 + 1), rem1_in * rem2_in)

    return gcnts


def rlbin(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gpoints = np.asarray(gpoints, dtype=np.float64)
    M = len(gpoints)
    trun = 1 if truncate else 0
    a = gpoints[0]
    b = gpoints[M - 1]

    # Fortran subroutine rlbin: obtains bin counts for univariate regression
    # data via the linear binning strategy. If trun == 0, weight from end
    # observations is given to the corresponding end grid points; if
    # trun == 1, end observations are truncated.
    xcnts = np.zeros(M, dtype=np.float64)
    ycnts = np.zeros(M, dtype=np.float64)
    delta = (b - a) / (M - 1)

    lxi = (X - a) / delta + 1.0
    # Fortran's int() truncates toward zero, matching numpy's astype(int64)
    li = lxi.astype(np.int64)
    rem = lxi - li

    # Correction for right endpoint (not included if li == M)
    at_b = X == b
    li = np.where(at_b, M - 1, li)
    rem = np.where(at_b, 1.0, rem)

    in_range = (li >= 1) & (li < M)
    li_in_range = li[in_range]
    rem_in_range = rem[in_range]
    y_in_range = Y[in_range]
    np.add.at(xcnts, li_in_range - 1, 1.0 - rem_in_range)
    np.add.at(xcnts, li_in_range, rem_in_range)
    np.add.at(ycnts, li_in_range - 1, (1.0 - rem_in_range) * y_in_range)
    np.add.at(ycnts, li_in_range, rem_in_range * y_in_range)

    if trun == 0:
        below = li < 1
        xcnts[0] += np.count_nonzero(below)
        ycnts[0] += np.sum(Y[below])

        above = li >= M
        xcnts[M - 1] += np.count_nonzero(above)
        ycnts[M - 1] += np.sum(Y[above])

    return {"xcounts": xcnts, "ycounts": ycnts}


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for signature parity with the R
    # function but, exactly as in the original R/Fortran implementation,
    # are never actually used (only the Gaussian kernel is supported and
    # only the diagonal of the *unweighted* smoother matrix is returned).
    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

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
        xcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.size == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of 'hdisc' to each member of
        # 'bandwidth' (kept 1-based, as in R, since it is only ever
        # compared against the 1-based group index 'i' below).
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.size == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Fortran subroutine sdiag: computes the diagonal entries of the
    # binned local polynomial smoother matrix. The Fortran source builds
    # an explicit lookup array of kernel weights ('fkap') indexed via
    # per-group offsets ('midpts'); since every weight it stores is just
    # a Gaussian kernel evaluation exp(-(delta*dist/hdisc[i])**2/2) for
    # the group 'i' and lag 'dist' = k - j, that bookkeeping is skipped
    # here in favor of evaluating the same closed-form expression
    # directly wherever it is needed.
    indic0 = indic - 1  # 0-based group index per grid point
    ss = np.zeros((M, ppp), dtype=np.float64)

    # Combine kernel weights and grid counts
    for k in range(M):
        if xcounts[k] != 0:
            for i in range(Q):
                L = int(Lvec[i])
                j_lo = max(0, k - L)
                j_hi = min(M - 1, k + L)
                for j in range(j_lo, j_hi + 1):
                    if indic0[j] == i:
                        dist = k - j
                        w = np.exp(-((delta * dist / hdisc[i]) ** 2) / 2)
                        fac = 1.0
                        ss[j, 0] += xcounts[k] * w
                        for ii in range(1, ppp):
                            fac *= delta * dist
                            ss[j, ii] += xcounts[k] * w * fac

    # At each grid point, assemble the local moment matrix 'Smat' from
    # the accumulated sums in 'ss' and take the (1,1) entry of its
    # inverse (equivalent to the Fortran dgefa/dgedi LU-based inversion
    # via 'job = 01') as the diagonal entry of the smoother matrix.
    Sdg = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = np.empty((pp, pp), dtype=np.float64)
        for ii in range(pp):
            for jj in range(pp):
                Smat[ii, jj] = ss[k, ii + jj]
        Sdg[k] = np.linalg.inv(Smat)[0, 0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'drv' and 'kernel' are accepted for signature parity with the R
    # function but, exactly as in the original R/Fortran implementation,
    # are never actually used (only the Gaussian kernel is supported).
    if bandwidth is None:
        raise ValueError("argument 'bandwidth' is missing, with no default")

    if range_x is None and not binned:
        range_x = (float(np.min(x)), float(np.max(x)))

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
        xcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
    else:
        xcounts = np.asarray(x, dtype=np.float64)
        M = len(xcounts)
        gpoints = np.linspace(a, b, M)

    # Set the bin width
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if bandwidth_arr.size == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of 'hdisc' to each member of
        # 'bandwidth' (kept 1-based, as in R, since it is only ever
        # compared against the 1-based group index 'i' below).
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.size == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    # Fortran subroutine sstdg: computes the diagonal entries of the
    # binned SS^T matrix, where S is the local polynomial smoother
    # matrix. As in 'sdiag', the explicit kernel-weight lookup table
    # ('fkap') built by the Fortran code is bypassed in favor of
    # evaluating the equivalent closed-form Gaussian weight directly
    # wherever it is needed. Unlike 'sdiag', a second moment
    # accumulator 'uu' (built from the squared kernel weights) is also
    # tracked alongside 'ss', so that the diagonal of S*S^T, rather
    # than just the diagonal of S, can be recovered.
    indic0 = indic - 1  # 0-based group index per grid point
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)

    # Combine kernel weights and grid counts
    for k in range(M):
        if xcounts[k] != 0:
            for i in range(Q):
                L = int(Lvec[i])
                j_lo = max(0, k - L)
                j_hi = min(M - 1, k + L)
                for j in range(j_lo, j_hi + 1):
                    if indic0[j] == i:
                        dist = k - j
                        w = np.exp(-((delta * dist / hdisc[i]) ** 2) / 2)
                        fac = 1.0
                        ss[j, 0] += xcounts[k] * w
                        uu[j, 0] += xcounts[k] * (w ** 2)
                        for ii in range(1, ppp):
                            fac *= delta * dist
                            ss[j, ii] += xcounts[k] * w * fac
                            uu[j, ii] += xcounts[k] * (w ** 2) * fac

    # At each grid point, assemble the local moment matrices 'Smat' and
    # 'Umat' from the accumulated sums in 'ss' and 'uu', invert 'Smat'
    # (equivalent to the Fortran dgefa/dgedi LU-based inversion via
    # 'job = 01'), and form the quadratic expression giving the SS^T
    # diagonal entry: SSTd[k] = Sinv[0, :] @ Umat @ Sinv[0, :], which is
    # exactly the double sum Smat(1,i)*Umat(i,j)*Smat(j,1) computed by
    # the Fortran routine (Smat there holding the inverse after the
    # dgedi call).
    SSTd = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = np.empty((pp, pp), dtype=np.float64)
        Umat = np.empty((pp, pp), dtype=np.float64)
        for ii in range(pp):
            for jj in range(pp):
                Smat[ii, jj] = ss[k, ii + jj]
                Umat[ii, jj] = uu[k, ii + jj]
        Sinv = np.linalg.inv(Smat)
        v = Sinv[0, :]
        SSTd[k] = v @ Umat @ v

    return {"x": gpoints, "y": SSTd}


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]

    qq = q + 1

    # Fortran subroutine blkest: computes blocked q'th degree polynomial
    # (least-squares) fits over Nval contiguous blocks of the sorted
    # data, accumulating the residual sum of squares (for sigsqe) and
    # the theta_22 / theta_24 curvature estimates required for the
    # direct plug-in bandwidth selector. It is assumed that the (x, y)
    # data are sorted with respect to the x's.
    idiv = n // Nval
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0

    for j in range(Nval):
        # 0-based [start, stop) block boundaries; the last block absorbs
        # any remainder so that it ends exactly at n.
        start = j * idiv
        stop = (j + 1) * idiv if j != Nval - 1 else n
        Xj = x[start:stop]
        Yj = y[start:stop]

        # Obtain a q'th degree fit over the current block: build the
        # design (Vandermonde) matrix with columns x^0, x^1, ..., x^q
        # and solve the least-squares problem (equivalent to the
        # Fortran QR decomposition solve via dqrdc/dqrsl).
        Xmat = Xj[:, np.newaxis] ** np.arange(qq)[np.newaxis, :]
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        fiti = np.full_like(Xj, coef[0])
        ddm = np.full_like(Xj, 2.0 * coef[2])
        ddddm = np.full_like(Xj, 24.0 * coef[4])
        for k in range(2, qq + 1):
            fiti = fiti + coef[k - 1] * Xj ** (k - 1)
            if k <= q - 1:
                ddm = ddm + k * (k + 1) * coef[k + 1] * Xj ** (k - 1)
                if k <= q - 3:
                    ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef[k + 3] * Xj ** (k - 1)

        th22e += np.sum(ddm ** 2)
        th24e += np.sum(ddm * ddddm)
        RSS += np.sum((Yj - fiti) ** 2)

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": float(sigsqe), "th22e": float(th22e), "th24e": float(th24e)}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    order_idx = np.argsort(X, kind="stable")
    X = X[order_idx]
    Y = Y[order_idx]

    qq = q + 1

    # Fortran subroutine cp: computes Mallow's C_p values for a set of
    # "Nmax" blocked q'th degree fits. It is assumed that the (X, Y)
    # data are sorted with respect to the X's.
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # For each number of partitions.
        idiv = n // Nval
        RSS_Nval = 0.0
        for j in range(1, Nval + 1):
            # For each member of the partition; 0-based [ilow, iupp)
            # block boundaries, with the last block absorbing any
            # remainder so that it ends exactly at n.
            ilow = (j - 1) * idiv
            iupp = j * idiv if j != Nval else n
            Xj = X[ilow:iupp]
            Yj = Y[ilow:iupp]

            # Obtain a q'th degree fit over the current member of the
            # partition: build the design (Vandermonde) matrix with
            # columns Xj^0, Xj^1, ..., Xj^q and solve the least-squares
            # problem (equivalent to the Fortran QR decomposition solve
            # via dqrdc/dqrsl).
            Xmat = Xj[:, np.newaxis] ** np.arange(qq)[np.newaxis, :]
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fiti = np.full_like(Xj, coef[0])
            for k in range(2, qq + 1):
                fiti = fiti + coef[k - 1] * Xj ** (k - 1)

            RSS_Nval += np.sum((Yj - fiti) ** 2)

        RSS[Nval - 1] = RSS_Nval

    # Now compute array of Mallow's C_p values.
    Cpvec = np.empty(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvec[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    # R's order(Cpvec)[1L] returns the 1-based index of the minimum Cp
    # value. Since block counts run 1..Nmax and align directly with the
    # (1-based) index into Cpvec, this is equivalent to the (0-based)
    # argmin plus one, which preserves the semantic meaning of the
    # return value as the chosen number of blocks.
    return int(np.argmin(Cpvec) + 1)


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Install safeguard against non-positive bandwidths:
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
        gpoints = np.linspace(a, b, M)
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
    L = min(int(np.floor(tau * h / delta)), M)

    if L == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    lvec = np.arange(0, L + 1)
    arg = lvec * delta / h

    kappam = norm.pdf(arg) / (h ** (drv + 1))
    hmold0 = 1.0
    hmold1 = arg
    hmnew = 1.0
    if drv >= 2:
        for i in range(2, drv + 1):
            hmnew = arg * hmold1 - (i - 1) * hmold0       # Compute mth degree Hermite polynomial
            hmold0 = hmold1                                # by recurrence.
            hmold1 = hmnew
    kappam = hmnew * kappam

    # Now combine weights and counts to obtain estimate
    # we need P >= 2L+1L, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappam = np.concatenate((kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]))
    Gcounts = np.concatenate((gcounts, np.zeros(P - M)))
    kappam = np.fft.fft(kappam)
    Gcounts = np.fft.fft(Gcounts)

    # R's fft(x, inverse=TRUE) does not normalize by length, and the
    # result is divided by P explicitly afterward; numpy's ifft already
    # normalizes by the transform length, so it directly matches that
    # combined R expression without an extra division by P.
    conv = np.fft.ifft(kappam * Gcounts)

    return float(np.sum(gcounts * np.real(conv)[:M]) / (n ** 2))


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # 'kernel' is accepted for signature parity with the R function but,
    # exactly as in the original R/Fortran implementation, is never
    # actually used (only the Gaussian kernel is supported).

    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None:
        bw_check = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
        if np.any(bw_check <= 0):
            raise ValueError("'bandwidth' must be strictly positive")
    else:
        raise ValueError("argument 'bandwidth' is missing, with no default")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (float(np.max(x)) - float(np.min(x)))
            range_x = (float(np.min(x)) - extra, float(np.max(x)) + extra)
        else:
            range_x = (float(np.min(x)), float(np.max(x)))

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
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
    else:  # obtain regression estimate
        # Bin the data if not already binned
        if not binned:
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
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
    if bandwidth_arr.size == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of 'hdisc'
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index of closest entry of 'hdisc' to each member of
        # 'bandwidth' (kept 1-based, as in R, since it is only ever
        # compared against the 1-based group index 'i' below).
        if Q > 1:
            lhdisc = np.log(hdisc)
            gap = (lhdisc[Q - 1] - lhdisc[0]) / (Q - 1)
            if gap == 0:
                indic = np.ones(M, dtype=np.int64)
            else:
                indic = np.round(((np.log(bandwidth_arr) - np.log(sorted_bw[0])) / gap) + 1).astype(np.int64)
        else:
            indic = np.ones(M, dtype=np.int64)
    elif bandwidth_arr.size == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError("'bandwidth' must be a scalar or an array of length 'gridsize'")

    if int(np.min(Lvec)) == 0:
        raise ValueError("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    # Fortran subroutine locpol: computes the binned local polynomial
    # kernel regression (or density derivative) estimate. As in 'sdiag'
    # and 'sstdiag', the explicit kernel-weight lookup table ('fkap')
    # built by the Fortran code is bypassed in favor of evaluating the
    # equivalent closed-form Gaussian weight directly wherever it is
    # needed. Unlike 'sdiag', a second moment accumulator 'tt' (built
    # from the kernel weights multiplied by 'ycounts') is also tracked
    # alongside 'ss', so that at each grid point the local weighted
    # least-squares system Smat * Tvec_full = tt can be solved and the
    # 'drv'-th coefficient of the fit ('drv' + 1'-th entry, 1-based, of
    # the solution vector) recovered as the raw curve estimate.
    indic0 = indic - 1  # 0-based group index per grid point
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)

    # Combine kernel weights and grid counts
    for k in range(M):
        if xcounts[k] != 0:
            for i in range(Q):
                L = int(Lvec[i])
                j_lo = max(0, k - L)
                j_hi = min(M - 1, k + L)
                for j in range(j_lo, j_hi + 1):
                    if indic0[j] == i:
                        dist = k - j
                        w = np.exp(-((delta * dist / hdisc[i]) ** 2) / 2)
                        fac = 1.0
                        ss[j, 0] += xcounts[k] * w
                        tt[j, 0] += ycounts[k] * w
                        for ii in range(1, ppp):
                            fac *= delta * dist
                            ss[j, ii] += xcounts[k] * w * fac
                            if ii < pp:
                                tt[j, ii] += ycounts[k] * w * fac

    # At each grid point, assemble the local moment matrix 'Smat' and
    # right-hand side 'Tvec' from the accumulated sums in 'ss' and 'tt'
    # (equivalent to the Fortran dgefa/dgesl LU-based linear solve), and
    # take the ('drv'+1)-th (1-based) entry of the solution as the raw
    # curve estimate at that grid point.
    curvest = np.zeros(M, dtype=np.float64)
    for k in range(M):
        Smat = np.empty((pp, pp), dtype=np.float64)
        for ii in range(pp):
            for jj in range(pp):
                Smat[ii, jj] = ss[k, ii + jj]
        Tvec = tt[k, :pp].copy()
        Tsol = np.linalg.solve(Smat, Tvec)
        curvest[k] = Tsol[drv]

    curvest = gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Install safeguard against non-positive bandwidths:
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'kernel' should be one of " + ", ".join(f'"{k}"' for k in valid_kernels))

    ## Rename common variables
    x = np.asarray(x, dtype=np.float64)
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
        h = del0 * (243 / (35 * n)) ** (1 / 5) * np.sqrt(np.var(x, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    ## Set kernel support values
    tau = 4 if kernel == "normal" else 1

    if range_x is None:
        range_x = (float(np.min(x) - tau * h), float(np.max(x) + tau * h))
    a = range_x[0]
    b = range_x[1]

    ## Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x, gpoints, truncate)

    ## Compute kernel weights
    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
    if L == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

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

    ## Now combine weight and counts to obtain estimate
    ## we need P >= 2L+1L, M: L <= M.
    P = 2 ** int(np.ceil(np.log(M + L + 1) / np.log(2)))
    kappa = np.concatenate((kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]))
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate((gcounts, np.zeros(P - M)))
    kappa = np.fft.fft(kappa / tot)
    gcounts = np.fft.fft(gcounts)

    # R's fft(x, inverse=TRUE) does not normalize by length, and the
    # result is divided by P explicitly afterward; numpy's ifft already
    # normalizes by the transform length, so it directly matches that
    # combined R expression without an extra division by P.
    y = np.real(np.fft.ifft(kappa * gcounts))[:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | np.ndarray[Any, np.dtype[np.float64]], gridsize: tuple[int, int] | list[int] | np.ndarray[Any, np.dtype[np.integer[Any]]] = (51, 51), range_x: tuple[tuple[float, float], tuple[float, float]] | list[tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    x = np.asarray(x, dtype=np.float64)

    ## Install safeguard against non-positive bandwidths:
    if np.min(np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    ## Rename common variables
    n = x.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    h = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    tau = 3.4  # For bivariate normal kernel.

    ## Use same bandwidth in each direction
    ## if only a single bandwidth is given.
    if len(h) == 1:
        h = np.array([h[0], h[0]])

    ## If range.x is not specified then set it at its default value.
    if range_x is None:
        range_x = [
            (float(np.min(x[:, 0]) - 1.5 * h[0]), float(np.max(x[:, 0]) + 1.5 * h[0])),
            (float(np.min(x[:, 1]) - 1.5 * h[1]), float(np.max(x[:, 1]) + 1.5 * h[1])),
        ]

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    ## Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x, gpoints1, gpoints2)

    ## Compute kernel weights
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [np.empty(0), np.empty(0)]
    for idx in range(2):
        L[idx] = min(int(np.floor(tau * h[idx] * (M[idx] - 1) / (b[idx] - a[idx]))), int(M[idx]) - 1)
        lvecid = np.arange(0, L[idx] + 1)
        facid = (b[idx] - a[idx]) / (h[idx] * (M[idx] - 1))
        z = norm.pdf(lvecid * facid) / h[idx]
        tot = np.sum(np.concatenate((z, z[1:][::-1]))) * facid * h[idx]
        kapid[idx] = z / tot
    kapp = np.outer(kapid[0], kapid[1]) / n

    if np.min(L) == 0:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")

    ## Now combine weight and counts using the FFT to obtain estimate
    P = (2 ** np.ceil(np.log((M + L).astype(np.float64)) / np.log(2))).astype(np.int64)  # smallest powers of 2 >= M+L
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
    # R's fft(z, inverse=TRUE) does not normalize by length, and the result
    # is divided by (P1*P2) explicitly afterward; numpy's ifft2 already
    # normalizes by the transform length, so it directly matches that
    # combined R expression without an extra division by P1*P2.
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    ## invert element-wise product of FFT's
    ## and truncate and normalise it

    ## Ensure that rp is non-negative
    rp = rp * (rp > 0).astype(np.float64)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

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

    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'scalest' should be one of " + ", ".join(f'"{s}"' for s in valid_scalest))

    if scalest == "stdev":
        scale_value = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_value = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # minim
        scale_value = min((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349, np.sqrt(np.var(x, ddof=1)))

    if scale_value == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = np.mean(x)
    sx = (x - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M - 1)

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

    return scale_value * hpi


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    valid_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in valid_kernels:
        raise ValueError("'kernel' should be one of " + ", ".join(f'"{k}"' for k in valid_kernels))

    ## Set kernel constants
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
    else:  # triweight
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

    valid_scalest = ("minim", "stdev", "iqr")
    if scalest not in valid_scalest:
        raise ValueError("'scalest' should be one of " + ", ".join(f'"{s}"' for s in valid_scalest))

    if scalest == "stdev":
        scale_value = np.sqrt(np.var(x, ddof=1))
    elif scalest == "iqr":
        scale_value = (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349
    else:  # minim
        scale_value = min((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349, np.sqrt(np.var(x, ddof=1)))

    if scale_value == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:

    x_mean = np.mean(x)
    sx = (x - x_mean) / scale_value
    sa = (a - x_mean) / scale_value
    sb = (b - x_mean) / scale_value

    ## Set up grid points and bin the data:

    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)
    ## delta <- (sb-sa)/(M-1)

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

    return scale_value * del0 * (1 / (psi4hat * n)) ** (1 / 5)


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if range_x is None:
        range_x = (float(np.min(x)), float(np.max(x)))

    ## Trim the 100(trim)% of the data from each end (in the x-direction).

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    indlow = int(np.floor(trim * len(x)) + 1)
    indupp = int(len(x) - np.floor(trim * len(x)))

    x = x[(indlow - 1):indupp]
    y = y[(indlow - 1):indupp]

    ## Rename common parameters
    n = len(x)
    M = gridsize
    a = range_x[0]
    b = range_x[1]

    ## Bin the data

    gpoints = np.linspace(a, b, M)
    out = rlbin(x, y, gpoints, truncate)
    xcounts = out["xcounts"]
    ycounts = out["ycounts"]

    ## Choose the value of N using Mallow's C_p
    Nmax = max(min(int(np.floor(n / divisor)), blockmax), 1)
    Nval = cpblock(x, y, Nmax, 4)

    ## Estimate sig^2, theta_22 and theta_24 using quartic fits
    ## on "Nval" blocks.

    out = blkest(x, y, Nval, 4)
    sigsqQ = out["sigsqe"]
    th24Q = out["th24e"]

    ## Estimate theta_22 using a local cubic fit
    ## with a "rule-of-thumb" bandwidth: "gamseh"

    gamseh = sigsqQ * (b - a) / (abs(th24Q) * n)
    if th24Q < 0:
        gamseh = (3 * gamseh / (8 * np.sqrt(np.pi))) ** (1 / 7)
    if th24Q > 0:
        gamseh = (15 * gamseh / (16 * np.sqrt(np.pi))) ** (1 / 7)

    mddest = locpoly(xcounts, ycounts, drv=2, bandwidth=gamseh,
                     range_x=range_x, binned=True)["y"]

    llow = int(np.floor(proptrun * M) + 1)
    lupp = int(M - np.floor(proptrun * M))
    th22kn = np.sum((mddest[(llow - 1):lupp] ** 2) * xcounts[(llow - 1):lupp]) / n

    ## Estimate sigma^2 using a local linear fit
    ## with a "direct plug-in" bandwidth: "lamseh"
    C3K = (1 / 2) + 2 * np.sqrt(2) - (4 / 3) * np.sqrt(3)
    C3K = (4 * C3K / (np.sqrt(2 * np.pi))) ** (1 / 9)
    lamseh = C3K * (((sigsqQ ** 2) * (b - a) / ((th22kn * n) ** 2)) ** (1 / 9))

    ## Now compute a local linear kernel estimate of
    ## the variance.
    mest = locpoly(xcounts, ycounts, bandwidth=lamseh,
                   range_x=range_x, binned=True)["y"]
    Sdg = sdiag(xcounts, bandwidth=lamseh,
                range_x=range_x, binned=True)["y"]
    SSTdg = sstdiag(xcounts, bandwidth=lamseh,
                    range_x=range_x, binned=True)["y"]
    sigsqn = np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts)
    sigsqd = n - 2 * np.sum(Sdg * xcounts) + np.sum(SSTdg * xcounts)
    sigsqkn = sigsqn / sigsqd

    ## Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * np.sqrt(np.pi) * th22kn * n)) ** (1 / 5))


def on_attach(libname: str, pkgname: str) -> None:
    # R's .onAttach package-load hook has no direct Python equivalent;
    # translated here as a simple function that prints the startup message.
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009")


def on_unload(libpath: str) -> None:
    # R's .onUnload package-unload hook (library.dynam.unload) has no direct
    # Python equivalent; translated here as a function that performs the
    # analogous action of unloading the compiled native extension module
    # (the compiled counterpart of the "KernSmooth" shared library) by
    # removing it from the module cache so a subsequent import reloads it.
    module_name = "_KernSmooth"
    sys.modules.pop(module_name, None)
