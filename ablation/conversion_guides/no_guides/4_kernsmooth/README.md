# r2py_kernsmooth

A Python port of the R package [**KernSmooth**](https://cran.r-project.org/package=KernSmooth) (v. 2.23-26), which
implements the kernel-smoothing methods described in Wand, M.P. and Jones, M.C. (1995), *Kernel Smoothing*, Chapman
and Hall.

The package binds the original KernSmooth Fortran routines directly via [f2py](https://numpy.org/doc/stable/f2py/),
and reimplements the R-level driver logic (argument handling, FFT-based convolution, plug-in bandwidth selection) in
Python on top of NumPy and SciPy, so results match the R package to numerical precision.

## Installation

```bash
pip install r2py_kernsmooth
```

Building from source requires a Fortran compiler and a BLAS implementation (e.g. OpenBLAS); see `meson.build`.

## Functions

| Function | Description |
| --- | --- |
| `bkde` | Binned kernel density estimate |
| `bkde2D` | 2D binned kernel density estimate |
| `bkfe` | Binned kernel functional estimate |
| `dpih` | Direct plug-in histogram bin width selection |
| `dpik` | Direct plug-in bandwidth selection for kernel density estimation |
| `dpill` | Direct plug-in bandwidth selection for local linear regression |
| `locpoly` | Local polynomial regression / density derivative estimation |
| `linbin` / `linbin2D` | Linear binning of (1D / 2D) data onto a grid |
| `rlbin` | Linear binning of regression data |
| `blkest`, `cpblock`, `sdiag`, `sstdiag` | Internal estimation helpers used by `dpill` |

## Example

```python
import numpy as np
from r2py_kernsmooth import bkde, locpoly

x = np.random.normal(size=1000)
density = bkde(x)  # {"x": grid points, "y": density estimate}

y = x + np.random.normal(scale=0.5, size=1000)
fit = locpoly(x, y, bandwidth=0.3)  # {"x": grid points, "y": fitted curve}
```

## License

r2py_kernsmooth is distributed under the same "Unlimited" license as upstream KernSmooth: it may be used, copied,
modified, and redistributed for any purpose without restriction. See [LICENSE](LICENSE) for the full text.

## Attribution

r2py_kernsmooth is a derivative of the R package KernSmooth, originally authored by **Matt Wand**, with LINPACK
Fortran routines contributed by **Cleve Moler** and the R-language port maintained by **Brian Ripley**. The Python
port is authored by **Yufei Cai** (ycai9@nd.edu) and **Jun Li** (jun.li@nd.edu). See [NOTICE](NOTICE) for full
attribution details.
