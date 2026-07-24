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


def on_attach(libname: str, pkgname: str) -> None:
    # R's .onAttach is a package-load hook automatically invoked by R's
    # library()/require() machinery when the package is attached; it has
    # no direct structural equivalent in Python's import system. The
    # closest analogue is a message emitted when this module is imported,
    # which is what is reproduced here so the informational banner is not
    # silently lost in translation.
    #
    # packageStartupMessage() writes its argument (here, a single string
    # containing an embedded newline) as a 'message' condition, which by
    # default is displayed on stderr and can be suppressed by the caller
    # (e.g. via suppressPackageStartupMessages()); printing to sys.stderr
    # is the closest Python equivalent of that behavior.
    print("KernSmooth 2.23 loaded\nCopyright M. P. Wand 1997-2009", file=sys.stderr)


def on_unload(libpath: str) -> None:
    # R's .onUnload(libpath) is a package-lifecycle hook that R invokes
    # automatically when the KernSmooth package is detached/unloaded; its
    # sole purpose is to call library.dynam.unload("KernSmooth", libpath) to
    # release the compiled Fortran/C shared library that was loaded via
    # library.dynam() in .onLoad.
    #
    # This Python port reimplements the former Fortran routines directly in
    # pure Python/NumPy rather than loading a compiled shared library through
    # an FFI layer, so there is no dynamic library handle to release here.
    # Python's own module import/unload lifecycle (module objects garbage
    # collected, no dlopen handle to close) makes this hook unnecessary.
    #
    # The function is retained only as a structural placeholder mirroring
    # the original R API; it intentionally performs no action.
    return None


def linbin(x: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> np.ndarray[Any, np.dtype[np.float64]]:
    # For application of linear binning to a univariate data set.
    #
    # This reimplements the Fortran routine F_linbin (KernSmooth/src/linbin.f)
    # in pure NumPy, since no prior Python conversion of that routine exists.
    # Obtains bin counts for univariate data via the linear binning strategy.
    # If truncate is False, weight from end observations is given to the
    # corresponding end grid points. If truncate is True, end observations
    # are truncated.
    x_arr = np.asarray(x, dtype=np.float64)
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)

    n = x_arr.shape[0]
    m = gpoints_arr.shape[0]
    trun = 1 if truncate else 0
    a = gpoints_arr[0]
    b = gpoints_arr[m - 1]

    # Initialize grid counts to zero
    gcnts = np.zeros(m, dtype=np.float64)

    if n == 0 or m < 2:
        return gcnts

    delta = (b - a) / (m - 1)

    # 1-based fractional grid position (mirrors Fortran variable 'lxi')
    lxi = ((x_arr - a) / delta) + 1.0

    # Fortran's int() truncates toward zero, matching np.trunc
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Interior points: 1 <= li < M (still 1-based, as in the Fortran code)
    mid_mask = (li >= 1) & (li < m)
    li_mid = li[mid_mask]
    rem_mid = rem[mid_mask]

    # Convert li (1-based left grid index) to 0-based indexing for gcnts.
    # np.add.at is used instead of plain fancy-index += so that repeated
    # indices accumulate correctly, matching the Fortran do-loop semantics.
    np.add.at(gcnts, li_mid - 1, 1.0 - rem_mid)
    np.add.at(gcnts, li_mid, rem_mid)

    if trun == 0:
        left_mask = li < 1
        gcnts[0] += np.count_nonzero(left_mask)

        right_mask = li >= m
        gcnts[m - 1] += np.count_nonzero(right_mask)

    return gcnts


def linbin2D(X: np.ndarray[Any, np.dtype[np.float64]], gpoints1: np.ndarray[Any, np.dtype[np.float64]], gpoints2: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    # Creates the grid counts from a bivariate data set X
    # over an equally-spaced set of grid points contained in
    # "gpoints1" and "gpoints2" using the linear binning strategy.
    #
    # This reimplements the Fortran routine F_lbtwod
    # (KernSmooth/src/linbin2D.f) in pure NumPy, since no prior Python
    # conversion of that routine exists. Observations outside the mesh
    # are ignored (no truncation weight is added back to the boundary,
    # matching the Fortran source).
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

    # Initialize grid counts to zero. Built directly as an (M1, M2) array
    # indexed by (row = grid1 position, col = grid2 position), which is
    # equivalent to R's matrix(out, M1, M2) column-major reshape of the
    # flat Fortran output, since we never flatten/reshape here.
    gcnts = np.zeros((M1, M2), dtype=np.float64)

    if n == 0 or M1 < 2 or M2 < 2:
        return gcnts

    delta1 = (b1 - a1) / (M1 - 1)
    delta2 = (b2 - a2) / (M2 - 1)

    # 1-based fractional grid positions (mirrors Fortran variables
    # 'lxi1' and 'lxi2')
    lxi1 = ((x1 - a1) / delta1) + 1.0
    lxi2 = ((x2 - a2) / delta2) + 1.0

    # Fortran's int() truncates toward zero, matching np.trunc
    li1 = np.trunc(lxi1).astype(np.int64)
    li2 = np.trunc(lxi2).astype(np.int64)
    rem1 = lxi1 - li1
    rem2 = lxi2 - li2

    # Only points whose bottom-left grid corner falls strictly inside the
    # mesh (1 <= li1 < M1, 1 <= li2 < M2, still 1-based as in the Fortran
    # code) contribute; observations outside the mesh are ignored.
    mask = (li1 >= 1) & (li2 >= 1) & (li1 < M1) & (li2 < M2)

    li1_m = li1[mask]
    li2_m = li2[mask]
    rem1_m = rem1[mask]
    rem2_m = rem2[mask]

    # Convert 1-based (li1, li2) bottom-left grid corner indices to
    # 0-based row/column indices.
    r0 = li1_m - 1
    c0 = li2_m - 1

    # np.add.at is used instead of plain fancy-index += so that repeated
    # indices accumulate correctly, matching the Fortran do-loop semantics.
    np.add.at(gcnts, (r0, c0), (1.0 - rem1_m) * (1.0 - rem2_m))
    np.add.at(gcnts, (r0 + 1, c0), rem1_m * (1.0 - rem2_m))
    np.add.at(gcnts, (r0, c0 + 1), (1.0 - rem1_m) * rem2_m)
    np.add.at(gcnts, (r0 + 1, c0 + 1), rem1_m * rem2_m)

    return gcnts


def rlbin(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], gpoints: np.ndarray[Any, np.dtype[np.float64]], truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For application of linear binning to a regression data set.
    #
    # This reimplements the Fortran routine F_rlbin (KernSmooth/src/rlbin.f)
    # in pure NumPy, since no prior Python conversion of that routine exists.
    # Obtains bin counts (xcounts) and weighted response sums (ycounts) for
    # bivariate regression data via the linear binning strategy. If truncate
    # is False, weight from end observations is given to the corresponding
    # end grid points. If truncate is True, end observations are truncated.
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    gpoints_arr = np.asarray(gpoints, dtype=np.float64)

    n = x_arr.shape[0]
    m = gpoints_arr.shape[0]
    trun = 1 if truncate else 0
    a = gpoints_arr[0]
    b = gpoints_arr[m - 1]

    # Initialize grid counts to zero
    xcnts = np.zeros(m, dtype=np.float64)
    ycnts = np.zeros(m, dtype=np.float64)

    if n == 0 or m < 2:
        return {"xcounts": xcnts, "ycounts": ycnts}

    delta = (b - a) / (m - 1)

    # 1-based fractional grid position (mirrors Fortran variable 'lxi')
    lxi = ((x_arr - a) / delta) + 1.0

    # Fortran's int() truncates toward zero, matching np.trunc
    li = np.trunc(lxi).astype(np.int64)
    rem = lxi - li

    # Correction for right endpoint (not included if li == M, since the
    # loop below only handles 1 <= li < M for the linear-interpolation case)
    right_endpoint_mask = x_arr == b
    li = np.where(right_endpoint_mask, m - 1, li)
    rem = np.where(right_endpoint_mask, 1.0, rem)

    # Interior points: 1 <= li < M (still 1-based, as in the Fortran code)
    mid_mask = (li >= 1) & (li < m)
    li_mid = li[mid_mask]
    rem_mid = rem[mid_mask]
    y_mid = y_arr[mid_mask]

    # Convert li (1-based left grid index) to 0-based indexing for xcnts/ycnts.
    # np.add.at is used instead of plain fancy-index += so that repeated
    # indices accumulate correctly, matching the Fortran do-loop semantics.
    np.add.at(xcnts, li_mid - 1, 1.0 - rem_mid)
    np.add.at(xcnts, li_mid, rem_mid)
    np.add.at(ycnts, li_mid - 1, (1.0 - rem_mid) * y_mid)
    np.add.at(ycnts, li_mid, rem_mid * y_mid)

    if trun == 0:
        left_mask = li < 1
        xcnts[0] += np.count_nonzero(left_mask)
        ycnts[0] += np.sum(y_arr[left_mask])

        right_mask = li >= m
        xcnts[m - 1] += np.count_nonzero(right_mask)
        ycnts[m - 1] += np.sum(y_arr[right_mask])

    return {"xcounts": xcnts, "ycounts": ycnts}


def blkest(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], Nval: int, q: int) -> dict[str, float]:
    # For obtaining preliminary estimates of quantities required for the
    # "direct plug-in" regression bandwidth selector based on blocked
    # qth degree polynomial fits.
    #
    # This reimplements the Fortran routine F_blkest (KernSmooth/src/blkest.f)
    # in pure NumPy, since no prior Python conversion of that routine exists.
    # The data are split into Nval contiguous blocks (after sorting by x);
    # a qth degree polynomial is least-squares fitted within each block, and
    # the fitted values/derivatives are used to accumulate an estimate of the
    # residual variance (sigsqe) and of theta_22 / theta_24, the integrated
    # squared second derivative and the analogous mixed second/fourth
    # derivative functional, respectively.
    n = len(x)

    # Sort the (x, y) data with respect to the x's.
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    qq = q + 1

    # It is assumed that the (x, y) data are sorted with respect to the x's.
    RSS = 0.0
    th22e = 0.0
    th24e = 0.0
    idiv = n // Nval

    for j in range(1, Nval + 1):
        # For each member of the partition (kept 1-based to mirror the
        # Fortran indexing, then converted to 0-based slices below).
        ilow = (j - 1) * idiv + 1
        iupp = j * idiv
        if j == Nval:
            iupp = n

        Xj = x_sorted[ilow - 1:iupp]
        Yj = y_sorted[ilow - 1:iupp]
        nj = iupp - ilow + 1

        # Obtain a qth degree fit over the current member of the partition.
        # Set up the design matrix (columns are powers 0, 1, ..., q of Xj).
        Xmat = np.empty((nj, qq), dtype=np.float64)
        Xmat[:, 0] = 1.0
        for k in range(2, qq + 1):
            Xmat[:, k - 1] = Xj ** (k - 1)

        # Least-squares solve, equivalent to the Fortran dqrdc/dqrsl QR fit.
        coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

        fiti = np.full(nj, coef[0], dtype=np.float64)
        ddm = np.full(nj, 2.0 * coef[2], dtype=np.float64)
        ddddm = np.full(nj, 24.0 * coef[4], dtype=np.float64)

        for k in range(2, qq + 1):
            fiti = fiti + coef[k - 1] * Xj ** (k - 1)
            if k <= q - 1:
                ddm = ddm + k * (k + 1) * coef[k + 1] * Xj ** (k - 1)
                if k <= q - 3:
                    ddddm = ddddm + k * (k + 1) * (k + 2) * (k + 3) * coef[k + 3] * Xj ** (k - 1)

        th22e += float(np.sum(ddm ** 2))
        th24e += float(np.sum(ddm * ddddm))
        RSS += float(np.sum((Yj - fiti) ** 2))

    sigsqe = RSS / (n - qq * Nval)
    th22e = th22e / n
    th24e = th24e / n

    return {"sigsqe": sigsqe, "th22e": th22e, "th24e": th24e}


def cpblock(X: np.ndarray[Any, np.dtype[np.float64]], Y: np.ndarray[Any, np.dtype[np.float64]], Nmax: int, q: int) -> int:
    # Chooses the number of blocks for the preliminary step of a plug-in
    # rule using Mallows' C_p.
    #
    # This reimplements the Fortran routine F_cp (KernSmooth/src/cp.f) in
    # pure NumPy, in the same style as the sibling function blkest, since no
    # prior Python conversion of that routine exists. For each candidate
    # number of blocks Nval = 1, ..., Nmax, the sorted (X, Y) data are split
    # into Nval contiguous blocks, a qq = q + 1 degree polynomial is
    # least-squares fitted within each block, and the residual sums of
    # squares are accumulated into RSS[Nval]. Mallow's C_p statistic is
    # then computed for every Nval from 1 to Nmax, and the Nval attaining
    # the minimum C_p value is returned.
    #
    # Note on return convention: R's `order(Cpvec)[1L]` returns the
    # 1-based *position* of the smallest element of Cpvec, where position i
    # (1-based) corresponds to a block count of i. Because the position and
    # the block count coincide, that same 1-based value is also the actual
    # number of blocks Nval, which is what the caller (dpill) uses directly
    # (e.g. `Nval <- cpblock(x, y, Nmax, 4)` is later passed straight into
    # blkest as the block count). This Python translation therefore returns
    # `int(np.argmin(Cpvals)) + 1`: a 0-based argmin converted to the
    # 1-based block count, so that the returned value can be used directly
    # (with no further adjustment) anywhere the R code used `Nval`.
    n = len(X)

    # Sort the (X, Y) data with respect to the X's.
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    order = np.argsort(X_arr, kind="stable")
    X_sorted = X_arr[order]
    Y_sorted = Y_arr[order]

    qq = q + 1

    # It is assumed that the (X, Y) data are sorted with respect to the X's.
    # Compute vector of RSS values.
    RSS = np.zeros(Nmax, dtype=np.float64)

    for Nval in range(1, Nmax + 1):
        # For each number of partitions.
        idiv = n // Nval
        RSS_Nval = 0.0

        for j in range(1, Nval + 1):
            # For each member of the partition (kept 1-based to mirror the
            # Fortran indexing, then converted to 0-based slices below).
            ilow = (j - 1) * idiv + 1
            iupp = j * idiv
            if j == Nval:
                iupp = n
            nj = iupp - ilow + 1

            Xj = X_sorted[ilow - 1:iupp]
            Yj = Y_sorted[ilow - 1:iupp]

            # Obtain a qq'th degree fit over the current member of the
            # partition. Set up the design matrix (columns are powers
            # 0, 1, ..., qq - 1 of Xj).
            Xmat = np.empty((nj, qq), dtype=np.float64)
            Xmat[:, 0] = 1.0
            for k in range(2, qq + 1):
                Xmat[:, k - 1] = Xj ** (k - 1)

            # Least-squares solve, equivalent to the Fortran dqrdc/dqrsl
            # QR fit.
            coef, _, _, _ = np.linalg.lstsq(Xmat, Yj, rcond=None)

            fiti = np.full(nj, coef[0], dtype=np.float64)
            for k in range(2, qq + 1):
                fiti = fiti + coef[k - 1] * Xj ** (k - 1)

            RSS_Nval += float(np.sum((Yj - fiti) ** 2))

        RSS[Nval - 1] += RSS_Nval

    # Now compute array of Mallow's C_p values.
    Cpvals = np.empty(Nmax, dtype=np.float64)
    for i in range(1, Nmax + 1):
        Cpvals[i - 1] = ((n - qq * Nmax) * RSS[i - 1] / RSS[Nmax - 1]) + 2 * qq * i - n

    return int(np.argmin(Cpvals)) + 1


def sdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For computing the binned diagonal entries of a smoother matrix for
    # local polynomial kernel regression.
    #
    # This reimplements the Fortran routine F_sdiag (KernSmooth/src/sdiag.f)
    # in pure NumPy, since no prior Python conversion of that routine exists.
    #
    # Note: 'kernel' is accepted for API compatibility only, matching the R
    # function signature; the underlying Fortran routine implements only
    # the "normal" kernel, so this argument has no effect on the
    # computation (exactly as in the original R code, which never
    # references it inside the function body).
    #
    # Note: 'drv' is likewise accepted for API compatibility only; it is a
    # parameter of the R function's signature but is never referenced in
    # its body. The diagonal entries computed here always correspond to
    # the fitted regression function itself (the (1,1) entry of the local
    # normal-equation matrix's inverse), controlled solely by 'degree'.
    if bandwidth is None:
        raise ValueError("argument \"bandwidth\" is missing, with no default")
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename common variables.
    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned.
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
    else:
        xcounts = np.asarray(x_arr, dtype=np.float64)
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width.
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths.
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc".
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index (1-based, as in R) of closest entry of "hdisc"
        # to each member of "bandwidth"; converted to 0-based just below.
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
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    # Convert the 1-based bandwidth-level index (computed above, mirroring
    # R) to a 0-based index for use below.
    indic0 = indic - 1

    # Allocate space for the local-polynomial normal-equation components
    # and the diagonal smoother-matrix entries. Note: the Fortran routine
    # precomputes a flat kernel-weight lookup table "fkap" purely for
    # efficiency reasons; the value it stores at offset (k - j) for
    # bandwidth level i is always exactly
    # exp(-(delta*(k-j)/hdisc[i])**2 / 2), so that table is not needed here
    # and the kernel weight is instead computed directly, in closed form,
    # in the loop below (following the same pattern established for
    # F_locpol).
    Sdg = np.zeros(M, dtype=np.float64)
    ss = np.zeros((M, ppp), dtype=np.float64)

    powers_exp = np.arange(ppp)

    # Combine kernel weights and grid counts. The Fortran routine loops
    # over data bins k, then over bandwidth levels i, then over grid
    # points j within L(i) of k subject to indic(j) == i; since indic(j)
    # selects a single bandwidth level per grid point j, this triple loop
    # is mathematically equivalent to, for each grid point j, summing
    # over data bins k within L(indic(j)) of j using j's own discretised
    # bandwidth level -- exactly the per-grid-point windowing pattern
    # used by F_locpol, which is what is implemented directly below.
    # (Bins with a zero count contribute nothing to the sum either way,
    # so it is immaterial whether the Fortran routine's 'xcnts(k).ne.0'
    # guard is reproduced here.)
    for jz in range(M):
        i0 = int(indic0[jz])
        L = int(Lvec[i0])
        h = hdisc[i0]

        k_lo = max(0, jz - L)
        k_hi = min(M - 1, jz + L)
        k_idx = np.arange(k_lo, k_hi + 1)

        offset = k_idx - jz
        w = np.exp(-((delta * offset / h) ** 2) / 2)
        powers = np.power((delta * offset)[:, None], powers_exp[None, :])

        ss[jz, :] = np.sum((xcounts[k_idx] * w)[:, None] * powers, axis=0)

    # Form the local-polynomial normal-equation matrix at each grid point
    # and extract the (1,1) entry of its inverse -- equivalent to the
    # Fortran dgefa/dgedi(job=01) full-matrix inversion followed by
    # 'Sdg(k) = Smat(1,1)' -- which is the diagonal entry of the binned
    # smoother matrix at that grid point. Solving against the first
    # standard basis vector is algebraically identical to, but cheaper
    # than, forming the full inverse.
    indss = np.arange(pp)[:, None] + np.arange(pp)[None, :]
    e1 = np.zeros(pp, dtype=np.float64)
    e1[0] = 1.0
    for kz in range(M):
        Smat = ss[kz, indss]
        Sdg[kz] = np.linalg.solve(Smat, e1)[0]

    return {"x": gpoints, "y": Sdg}


def sstdiag(x: np.ndarray[Any, np.dtype[np.float64]], drv: int = 0, degree: int = 1, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For computing the binned diagonal entries of SS^T where S is a
    # smoother matrix for local polynomial kernel regression.
    #
    # This reimplements the Fortran routine F_sstdg (KernSmooth/src/sstdiag.f)
    # in pure NumPy, since no prior Python conversion of that routine exists.
    # It closely parallels F_sdiag (as translated for 'sdiag' above), but
    # additionally accumulates a second local-polynomial moment matrix
    # ('uu'/'Umat') built from *squared* kernel weights, which is combined
    # with the (symmetric) inverse of the ordinary normal-equation matrix
    # to obtain diag(S S^T)[j] = sum_k S[j,k]**2, rather than diag(S)[j].
    #
    # Note: 'kernel' is accepted for API compatibility only, matching the R
    # function signature; the underlying Fortran routine implements only
    # the "normal" kernel, so this argument has no effect on the
    # computation (exactly as in the original R code, which never
    # references it inside the function body).
    #
    # Note: 'drv' is likewise accepted for API compatibility only; it is a
    # parameter of the R function's signature but is never referenced in
    # its body. The diagonal entries computed here always correspond to
    # the fitted regression function itself (the (1,1) entry of the
    # sandwich Ainv @ Umat @ Ainv at each grid point), controlled solely
    # by 'degree'.
    if bandwidth is None:
        raise ValueError("argument \"bandwidth\" is missing, with no default")
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename common variables.
    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Bin the data if not already binned.
    if not binned:
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
    else:
        xcounts = np.asarray(x_arr, dtype=np.float64)
        M = xcounts.shape[0]
        gpoints = np.linspace(a, b, M)

    # Set the bin width.
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths.
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc".
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index (1-based, as in R) of closest entry of "hdisc"
        # to each member of "bandwidth"; converted to 0-based just below.
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
    elif bandwidth_arr.shape[0] == 1:
        indic = np.ones(M, dtype=np.int64)
        Q = 1
        Lvec = np.full(Q, np.floor(tau * bandwidth_arr[0] / delta), dtype=np.int64)
        hdisc = np.full(Q, bandwidth_arr[0], dtype=np.float64)
    else:
        raise ValueError(
            "'bandwidth' must be a scalar or an array of length 'gridsize'"
        )

    # Convert the 1-based bandwidth-level index (computed above, mirroring
    # R) to a 0-based index for use below.
    indic0 = indic - 1

    # Allocate space for the local-polynomial normal-equation components
    # (ss/Smat), the squared-kernel-weight moment components (uu/Umat),
    # and the resulting SS^T diagonal entries. Note: the Fortran routine
    # precomputes a flat kernel-weight lookup table "fkap" purely for
    # efficiency reasons; the value it stores at offset (k - j) for
    # bandwidth level i is always exactly
    # exp(-(delta*(k-j)/hdisc[i])**2 / 2), so that table is not needed here
    # and the kernel weight is instead computed directly, in closed form,
    # in the loop below (following the same pattern established for
    # F_locpol / F_sdiag).
    SSTd = np.zeros(M, dtype=np.float64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    uu = np.zeros((M, ppp), dtype=np.float64)

    powers_exp = np.arange(ppp)

    # Combine kernel weights and grid counts. As in F_sdiag, the Fortran
    # routine loops over data bins k, then over bandwidth levels i, then
    # over grid points j within L(i) of k subject to indic(j) == i; since
    # indic(j) selects a single bandwidth level per grid point j, this
    # triple loop is mathematically equivalent to, for each grid point j,
    # summing over data bins k within L(indic(j)) of j using j's own
    # discretised bandwidth level -- the per-grid-point windowing pattern
    # implemented directly below. The 'uu' moments use the *squared*
    # kernel weight w**2 (matching fkap(...)**2 in the Fortran code) but
    # the *same* polynomial factor 'fac' = (delta*(k-j))**(ii-1) as 'ss'
    # (fac itself is not squared).
    for jz in range(M):
        i0 = int(indic0[jz])
        L = int(Lvec[i0])
        h = hdisc[i0]

        k_lo = max(0, jz - L)
        k_hi = min(M - 1, jz + L)
        k_idx = np.arange(k_lo, k_hi + 1)

        offset = k_idx - jz
        w = np.exp(-((delta * offset / h) ** 2) / 2)
        powers = np.power((delta * offset)[:, None], powers_exp[None, :])

        ss[jz, :] = np.sum((xcounts[k_idx] * w)[:, None] * powers, axis=0)
        uu[jz, :] = np.sum((xcounts[k_idx] * (w ** 2))[:, None] * powers, axis=0)

    # Form the local-polynomial normal-equation matrix (Smat) and the
    # squared-kernel-weight moment matrix (Umat) at each grid point, and
    # compute the SS^T diagonal entry there. This mirrors the Fortran
    # dgefa/dgedi(job=01) full-matrix inversion of Smat followed by
    # 'SSTd(k) = sum_i sum_j Smat(1,i)*Umat(i,j)*Smat(j,1)'. Since Smat is
    # symmetric (Smat(i,j) depends only on i+j), so is its inverse, and
    # so 'Smat(1, :)' equals 'Smat(:, 1)' of the inverse; solving against
    # the first standard basis vector therefore yields the same vector
    # 'v' used on both sides of the sandwich product, which is
    # algebraically identical to, but cheaper than, forming the full
    # inverse.
    indss = np.arange(pp)[:, None] + np.arange(pp)[None, :]
    e1 = np.zeros(pp, dtype=np.float64)
    e1[0] = 1.0
    for kz in range(M):
        Smat = ss[kz, indss]
        Umat = uu[kz, indss]
        v = np.linalg.solve(Smat, e1)
        SSTd[kz] = v @ Umat @ v

    return {"x": gpoints, "y": SSTd}


def locpoly(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]] | None = None, drv: int = 0, degree: int | None = None, kernel: str = "normal", bandwidth: float | np.ndarray[Any, np.dtype[np.float64]] | None = None, gridsize: int = 401, bwdisc: int = 25, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # For computing a binned local polynomial regression estimator of a
    # univariate regression function or its derivative. The data are
    # discretised on an equally spaced grid, and the bandwidths are
    # discretised on a logarithmically spaced grid.
    #
    # This reimplements the Fortran routine F_locpol (KernSmooth/src/locpoly.f)
    # in pure NumPy, since no prior Python conversion of that routine exists.
    # For every grid point j, weighted local-polynomial normal-equation
    # components (Smat, Tvec) are formed using a Gaussian kernel with the
    # bandwidth-discretisation level assigned to j (via 'indic'), and the
    # resulting system is solved to obtain the drv-th derivative estimate
    # at j.
    #
    # Note: 'kernel' is accepted for API compatibility only, matching the R
    # function signature; the underlying Fortran routine implements only
    # the "normal" kernel, so this argument has no effect on the
    # computation (exactly as in the original R code, which never
    # references it inside the function body).

    # Install safeguard against non-positive bandwidths. In R, 'bandwidth'
    # has no default either, so a genuinely missing value is also treated
    # as an error here (mirroring the eventual `length(bandwidth)` failure
    # in R when the argument was never supplied).
    if bandwidth is None:
        raise ValueError("argument \"bandwidth\" is missing, with no default")
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.any(bandwidth_arr <= 0):
        raise ValueError("'bandwidth' must be strictly positive")

    drv = int(drv)
    if degree is None:
        degree = drv + 1
    else:
        degree = int(degree)

    x_arr = np.asarray(x, dtype=np.float64)

    if range_x is None and not binned:
        if y is None:
            extra = 0.05 * (np.max(x_arr) - np.min(x_arr))
            range_x = (float(np.min(x_arr) - extra), float(np.max(x_arr) + extra))
        else:
            range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename common variables.
    M = gridsize
    Q = int(bwdisc)
    a = float(range_x[0])
    b = float(range_x[1])
    pp = degree + 1
    ppp = 2 * degree + 1
    tau = 4

    # Decide whether a density estimate or regression estimate is required.
    if y is None:
        # Obtain density estimate. Note: as in the original R code, this
        # branch always bins the raw data via 'linbin', irrespective of
        # the 'binned' flag (which only affects the regression branch).
        n = x_arr.shape[0]
        gpoints = np.linspace(a, b, M)
        xcounts = linbin(x_arr, gpoints, truncate)
        ycounts = (M - 1) * xcounts / (n * (b - a))
        xcounts = np.ones(M, dtype=np.float64)
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

    # Set the bin width.
    delta = (b - a) / (M - 1)

    # Discretise the bandwidths.
    if bandwidth_arr.shape[0] == M:
        sorted_bw = np.sort(bandwidth_arr)
        hlow = sorted_bw[0]
        hupp = sorted_bw[M - 1]
        hdisc = np.exp(np.linspace(np.log(hlow), np.log(hupp), Q))

        # Determine value of L for each member of "hdisc".
        Lvec = np.floor(tau * hdisc / delta).astype(np.int64)

        # Determine index (1-based, as in R) of closest entry of "hdisc"
        # to each member of "bandwidth"; converted to 0-based just below.
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
    elif bandwidth_arr.shape[0] == 1:
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

    # Convert the 1-based bandwidth-level index (computed above, mirroring
    # R) to a 0-based index for use below.
    indic0 = indic - 1

    # Allocate space for the local-polynomial normal-equation components
    # and the final estimate. Note: the Fortran routine precomputes a flat
    # kernel-weight lookup table "fkap" purely for efficiency reasons; the
    # value it stores at offset (k - j) for bandwidth level i is always
    # exactly exp(-(delta*(k-j)/hdisc[i])**2 / 2), so that table is not
    # needed here and the kernel weight is instead computed directly, in
    # closed form, in the loop below.
    curvest = np.zeros(M, dtype=np.float64)
    ss = np.zeros((M, ppp), dtype=np.float64)
    tt = np.zeros((M, pp), dtype=np.float64)

    powers_exp = np.arange(ppp)

    # Combine kernel weights and grid counts. In the Fortran code, for a
    # fixed grid point j (1-based), the condition `indic(j).eq.i` inside
    # the loop over bandwidth levels i = 1..Q can only ever be satisfied by
    # the single level i = indic(j); the nested loop over i therefore
    # collapses to using grid point j's own discretised bandwidth level,
    # which is what is implemented directly below.
    for jz in range(M):
        i0 = int(indic0[jz])
        L = int(Lvec[i0])
        h = hdisc[i0]

        k_lo = max(0, jz - L)
        k_hi = min(M - 1, jz + L)
        k_idx = np.arange(k_lo, k_hi + 1)

        offset = k_idx - jz
        w = np.exp(-((delta * offset / h) ** 2) / 2)
        powers = np.power((delta * offset)[:, None], powers_exp[None, :])

        ss[jz, :] = np.sum((xcounts[k_idx] * w)[:, None] * powers, axis=0)
        tt[jz, :] = np.sum((ycounts[k_idx] * w)[:, None] * powers[:, :pp], axis=0)

    # Solve the local-polynomial normal equations at each grid point
    # (equivalent to the Fortran dgefa/dgesl LU solve).
    indss = np.arange(pp)[:, None] + np.arange(pp)[None, :]
    for kz in range(M):
        Smat = ss[kz, indss]
        Tvec = tt[kz, :].copy()
        sol = np.linalg.solve(Smat, Tvec)
        curvest[kz] = sol[drv]

    curvest = math.gamma(drv + 1) * curvest

    return {"x": gpoints, "y": curvest}


def bkfe(x: np.ndarray[Any, np.dtype[np.float64]], drv: int, bandwidth: float, gridsize: int = 401, range_x: tuple[float, float] | None = None, binned: bool = False, truncate: bool = True) -> float:
    # Binned kernel functional estimate.
    #
    # Estimates the functional theta_r = integral f^(r)(x) f(x) dx (r = drv)
    # by binning the data (via `linbin`), convolving the bin counts with
    # Hermite-polynomial-weighted Gaussian kernel weights using the FFT, and
    # combining the result with the bin counts. This follows the binned
    # approximation of Wand (1994) used by the R `bkfe` function.

    # Install safeguard against non-positive bandwidths.
    if bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    if range_x is None and not binned:
        x_arr = np.asarray(x, dtype=np.float64)
        range_x = (float(np.min(x_arr)), float(np.max(x_arr)))

    # Rename variables.
    M = gridsize
    a = range_x[0]
    b = range_x[1]
    h = bandwidth

    # Bin the data if not already binned.
    if not binned:
        gpoints = np.linspace(a, b, gridsize)
        gcounts = linbin(np.asarray(x, dtype=np.float64), gpoints, truncate)
    else:
        gcounts = np.asarray(x, dtype=np.float64)
        M = len(gcounts)
        gpoints = np.linspace(a, b, M)

    # Set the sample size and bin width.
    n = np.sum(gcounts)
    delta = (b - a) / (M - 1)

    # Obtain kernel weights.
    tau = 4 + drv
    L = int(min(int(np.floor(tau * h / delta)), M))

    if L == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
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

    # Now combine weights and counts to obtain estimate.
    # we need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log2(M + L + 1)))
    kappam_ext = np.concatenate(
        [kappam, np.zeros(P - 2 * L - 1), kappam[1:][::-1]]
    )
    Gcounts_ext = np.concatenate([gcounts, np.zeros(P - M)])
    kappam_fft = np.fft.fft(kappam_ext)
    Gcounts_fft = np.fft.fft(Gcounts_ext)

    # R's fft(z, inverse = TRUE) does NOT normalize by length P, so
    # R's Re(fft(z, inverse = TRUE)) / P is exactly NumPy's normalized
    # inverse transform np.fft.ifft(z) (which already divides by P).
    conv = np.real(np.fft.ifft(kappam_fft * Gcounts_fft))

    return float(np.sum(gcounts * conv[:M]) / (n ** 2))


def bkde(x: np.ndarray[Any, np.dtype[np.float64]], kernel: str = "normal", canonical: bool = False, bandwidth: float | None = None, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Binned kernel density estimate.
    #
    # Estimates the probability density function of a univariate data set
    # by binning the data (via `linbin`), convolving the bin counts with
    # kernel weights using the FFT, and combining the result with the bin
    # counts. This follows the binned approximation of Wand (1994) used by
    # the R `bkde` function.

    # Install safeguard against non-positive bandwidths.
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    allowed_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel not in allowed_kernels:
        raise ValueError(
            "'kernel' should be one of " + ", ".join(repr(k) for k in allowed_kernels)
        )

    # Rename common variables.
    x_arr = np.asarray(x, dtype=np.float64)
    n = x_arr.shape[0]
    M = gridsize

    # Set canonical scaling factors.
    del0_map = {
        "normal": (1.0 / (4.0 * np.pi)) ** (1.0 / 10.0),
        "box": (9.0 / 2.0) ** (1.0 / 5.0),
        "epanech": 15.0 ** (1.0 / 5.0),
        "biweight": 35.0 ** (1.0 / 5.0),
        "triweight": (9450.0 / 143.0) ** (1.0 / 5.0),
    }
    del0 = del0_map[kernel]

    if not isinstance(canonical, bool):
        raise ValueError("'canonical' must be a length-1 logical vector")

    # Set default bandwidth.
    if bandwidth is None:
        h = del0 * (243.0 / (35.0 * n)) ** (1.0 / 5.0) * np.sqrt(np.var(x_arr, ddof=1))
    elif canonical:
        h = del0 * bandwidth
    else:
        h = bandwidth

    # Set kernel support values.
    tau = 4.0 if kernel == "normal" else 1.0

    if range_x is None:
        range_x = (float(np.min(x_arr) - tau * h), float(np.max(x_arr) + tau * h))
    a = range_x[0]
    b = range_x[1]

    # Set up grid points and bin the data.
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    # Compute kernel weights.
    delta = (b - a) / (h * (M - 1))
    L = int(min(int(np.floor(tau / delta)), M))
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
    else:  # kernel == "triweight"
        kappa = 0.5 * beta.pdf(0.5 * (lvec * delta + 1), 4, 4) / (n * h)

    # Now combine weight and counts to obtain estimate.
    # we need P >= 2L+1, M: L <= M.
    P = int(2 ** np.ceil(np.log2(M + L + 1)))
    kappa = np.concatenate([kappa, np.zeros(P - 2 * L - 1), kappa[1:][::-1]])
    tot = np.sum(kappa) * (b - a) / (M - 1) * n  # should have total weight one
    gcounts = np.concatenate([gcounts, np.zeros(P - M)])
    kappa_fft = np.fft.fft(kappa / tot)
    gcounts_fft = np.fft.fft(gcounts)

    # R's fft(z, inverse = TRUE) does NOT normalize by length P, so
    # R's Re(fft(z, inverse = TRUE)) / P is exactly NumPy's normalized
    # inverse transform np.fft.ifft(z) (which already divides by P).
    y = np.real(np.fft.ifft(kappa_fft * gcounts_fft))[:M]

    return {"x": gpoints, "y": y}


def bkde2D(x: np.ndarray[Any, np.dtype[np.float64]], bandwidth: float | list[float] | tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]], gridsize: list[int] | tuple[int, int] | np.ndarray[Any, np.dtype[np.int64]] = (51, 51), range_x: list[tuple[float, float]] | tuple[tuple[float, float], tuple[float, float]] | None = None, truncate: bool = True) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
    # Binned approximation to a bivariate kernel density estimate.
    #
    # Bins the bivariate data set `x` on a rectangular grid (via
    # `linbin2D`), convolves the bin counts with a bivariate (product)
    # Gaussian kernel using a 2-D FFT, and returns the resulting density
    # estimate evaluated at the grid points. This mirrors the R
    # `bkde2D` function.

    # Install safeguard against non-positive bandwidths.
    bandwidth_arr = np.atleast_1d(np.asarray(bandwidth, dtype=np.float64))
    if np.min(bandwidth_arr) <= 0:
        raise ValueError("'bandwidth' must be strictly positive")

    # Rename common variables.
    x_arr = np.asarray(x, dtype=np.float64)
    n = x_arr.shape[0]
    M = np.asarray(gridsize, dtype=np.int64)
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction if only a single bandwidth is given.
    if bandwidth_arr.size == 1:
        h = np.array([bandwidth_arr[0], bandwidth_arr[0]], dtype=np.float64)
    else:
        h = bandwidth_arr.astype(np.float64)

    # If range_x is not specified then set it at its default value.
    if range_x is None:
        range_x = (
            (
                float(np.min(x_arr[:, 0]) - 1.5 * h[0]),
                float(np.max(x_arr[:, 0]) + 1.5 * h[0]),
            ),
            (
                float(np.min(x_arr[:, 1]) - 1.5 * h[1]),
                float(np.max(x_arr[:, 1]) + 1.5 * h[1]),
            ),
        )

    a = np.array([range_x[0][0], range_x[1][0]], dtype=np.float64)
    b = np.array([range_x[0][1], range_x[1][1]], dtype=np.float64)

    # Set up grid points and bin the data.
    gpoints1 = np.linspace(a[0], b[0], int(M[0]))
    gpoints2 = np.linspace(a[1], b[1], int(M[1]))

    gcounts = linbin2D(x_arr, gpoints1, gpoints2)

    # Compute kernel weights.
    L = np.zeros(2, dtype=np.int64)
    kapid: list[np.ndarray[Any, np.dtype[np.float64]]] = [
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    ]
    for idx in range(2):
        L[idx] = min(
            int(np.floor(tau * h[idx] * (int(M[idx]) - 1) / (b[idx] - a[idx]))),
            int(M[idx]) - 1,
        )
        lvecid = np.arange(0, int(L[idx]) + 1, dtype=np.float64)
        facid = (b[idx] - a[idx]) / (h[idx] * (int(M[idx]) - 1))
        z = norm.pdf(lvecid * facid) / h[idx]
        # c(z, rev(z[-1L])): z with its first element dropped, reversed,
        # appended after z. The sum is order-independent, so this equals
        # sum(z) + sum(z[1:]).
        tot = (np.sum(z) + np.sum(z[1:])) * facid * h[idx]
        kapid[idx] = z / tot

    # kapid[[1]] %*% t(kapid[[2]]) is an outer product of the two
    # 1-D kernel weight vectors.
    kapp = np.outer(kapid[0], kapid[1]) / n

    if int(np.min(L)) == 0:
        warnings.warn(
            "Binning grid too coarse for current (small) bandwidth: "
            "consider increasing 'gridsize'"
        )

    # Now combine weight and counts using the FFT to obtain estimate.
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
        # rows (P1-L1+1):P1 (R, 1-based) <- kapp rows (L1+1):2 (R, 1-based,
        # descending); 0-based this is rp[P1-L1:P1, :] <- kapp[L1:0:-1, :].
        rp[P1 - L1:P1, 0:L2 + 1] = kapp[L1:0:-1, :]
    if L2:
        # columns (P2-L2+1):P2 (R, 1-based) <- rp columns (L2+1):2
        # (R, 1-based, descending), using the rows already filled above;
        # 0-based this is rp[:, P2-L2:P2] <- rp[:, L2:0:-1].
        rp[:, P2 - L2:P2] = rp[:, L2:0:-1]
    # wrap-around version of "kapp"

    sp = np.zeros((P1, P2), dtype=np.float64)
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"

    rp_fft = np.fft.fft2(rp)  # Obtain FFT's of r and s
    sp_fft = np.fft.fft2(sp)
    # R's fft(z, inverse = TRUE) does NOT normalize by length P1*P2, so
    # R's Re(fft(z, inverse = TRUE)) / (P1*P2) is exactly NumPy's
    # normalized inverse transform np.fft.ifft2(z) (which already divides
    # by P1*P2).
    rp = np.real(np.fft.ifft2(rp_fft * sp_fft))[0:M1, 0:M2]
    # invert element-wise product of FFT's
    # and truncate and normalise it

    # Ensure that rp is non-negative.
    rp = np.where(rp > 0, rp, 0.0)

    return {"x1": gpoints1, "x2": gpoints2, "fhat": rp}


def dpih(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    # Direct plug-in bandwidth selector for histograms (Wand, 1996).
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    x_arr = np.asarray(x, dtype=np.float64)

    ## Rename variables
    n = x_arr.shape[0]
    M = gridsize
    if range_x is None:
        a = float(np.min(x_arr))
        b = float(np.max(x_arr))
    else:
        a = float(range_x[0])
        b = float(range_x[1])

    ## Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    ## Compute scale estimate
    allowed_scalest = ("minim", "stdev", "iqr")
    if scalest in allowed_scalest:
        scalest_choice = scalest
    else:
        candidates = [choice for choice in allowed_scalest if choice.startswith(scalest)]
        if len(candidates) == 1:
            scalest_choice = candidates[0]
        else:
            raise ValueError(
                "'arg' should be one of "
                + ", ".join(repr(choice) for choice in allowed_scalest)
            )

    if scalest_choice == "stdev":
        scale_est = float(np.sqrt(np.var(x_arr, ddof=1)))
    elif scalest_choice == "iqr":
        scale_est = float(
            (np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349
        )
    else:  # "minim"
        scale_est = float(
            min(
                (np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349,
                np.sqrt(np.var(x_arr, ddof=1)),
            )
        )

    if scale_est == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:
    x_mean = float(np.mean(x_arr))
    sx = (x_arr - x_mean) / scale_est
    sa = (a - x_mean) / scale_est
    sb = (b - x_mean) / scale_est

    ## Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

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

    return float(scale_est * hpi)


def dpik(x: np.ndarray[Any, np.dtype[np.float64]], scalest: str = "minim", level: int = 2, kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    # Direct plug-in bandwidth selector for kernel density estimation
    # (Sheather and Jones, 1991 / Wand and Jones, 1995).
    if level > 5:
        raise ValueError("Level should be between 0 and 5")

    allowed_kernels = ("normal", "box", "epanech", "biweight", "triweight")
    if kernel in allowed_kernels:
        kernel_choice = kernel
    else:
        candidates = [choice for choice in allowed_kernels if choice.startswith(kernel)]
        if len(candidates) == 1:
            kernel_choice = candidates[0]
        else:
            raise ValueError(
                "'arg' should be one of "
                + ", ".join(repr(choice) for choice in allowed_kernels)
            )

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

    x_arr = np.asarray(x, dtype=np.float64)

    ## Rename variables
    n = x_arr.shape[0]
    M = gridsize
    if range_x is None:
        a = float(np.min(x_arr))
        b = float(np.max(x_arr))
    else:
        a = float(range_x[0])
        b = float(range_x[1])

    ## Set up grid points and bin the data
    gpoints = np.linspace(a, b, M)
    gcounts = linbin(x_arr, gpoints, truncate)

    ## Compute scale estimate
    allowed_scalest = ("minim", "stdev", "iqr")
    if scalest in allowed_scalest:
        scalest_choice = scalest
    else:
        candidates = [choice for choice in allowed_scalest if choice.startswith(scalest)]
        if len(candidates) == 1:
            scalest_choice = candidates[0]
        else:
            raise ValueError(
                "'arg' should be one of "
                + ", ".join(repr(choice) for choice in allowed_scalest)
            )

    if scalest_choice == "stdev":
        scale_est = float(np.sqrt(np.var(x_arr, ddof=1)))
    elif scalest_choice == "iqr":
        scale_est = float(
            (np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349
        )
    else:  # "minim"
        scale_est = float(
            min(
                (np.quantile(x_arr, 0.75) - np.quantile(x_arr, 0.25)) / 1.349,
                np.sqrt(np.var(x_arr, ddof=1)),
            )
        )

    if scale_est == 0:
        raise ValueError("scale estimate is zero for input data")

    ## Replace input data by standardised data for numerical stability:
    x_mean = float(np.mean(x_arr))
    sx = (x_arr - x_mean) / scale_est
    sa = (a - x_mean) / scale_est
    sb = (b - x_mean) / scale_est

    ## Set up grid points and bin the data:
    gpoints = np.linspace(sa, sb, M)
    gcounts = linbin(sx, gpoints, truncate)

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

    return float(scale_est * del0 * (1 / (psi4hat * n)) ** (1 / 5))


def dpill(x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], blockmax: int = 5, divisor: int = 20, trim: float = 0.01, proptrun: float = 0.05, gridsize: int = 401, range_x: tuple[float, float] | None = None, truncate: bool = True) -> float:
    # Computes a direct plug-in selector of the
    # bandwidth for local linear regression as
    # described in the 1996 J. Amer. Statist. Assoc.
    # paper by Ruppert, Sheather and Wand.

    # Trim the 100(trim)% of the data from each end (in the x-direction).
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    n_full = x_sorted.shape[0]
    indlow = int(math.floor(trim * n_full))
    indupp = n_full - int(math.floor(trim * n_full))

    x = x_sorted[indlow:indupp]
    y = y_sorted[indlow:indupp]

    # Rename common parameters.
    # NOTE: R's default argument `range.x = range(x)` is a lazily evaluated
    # promise that is only forced here, AFTER `x` has been reassigned to the
    # trimmed vector above. Hence the default (when range_x is not supplied)
    # must be computed from the trimmed x, not the original input.
    n = x.shape[0]
    M = gridsize
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
    Nmax = max(min(int(math.floor(n / divisor)), blockmax), 1)
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

    llow = int(math.floor(proptrun * M))
    lupp = M - int(math.floor(proptrun * M))
    th22kn = float(np.sum((mddest[llow:lupp] ** 2) * xcounts[llow:lupp]) / n)

    # Estimate sigma^2 using a local linear fit
    # with a "direct plug-in" bandwidth: "lamseh".
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
    sigsqn = float(np.sum(y ** 2) - 2 * np.sum(mest * ycounts) + np.sum((mest ** 2) * xcounts))
    sigsqd = n - 2 * float(np.sum(Sdg * xcounts)) + float(np.sum(SSTdg * xcounts))
    sigsqkn = sigsqn / sigsqd

    # Combine to obtain final answer.
    return float((sigsqkn * (b - a) / (2 * math.sqrt(math.pi) * th22kn * n)) ** (1 / 5))

