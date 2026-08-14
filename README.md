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

## Citation

If you use `r2py_kernsmooth` in published work, please cite **all three** of
the following. The numerical core of `r2py_kernsmooth` is KernSmooth's own
Fortran code, retained unmodified — the results you obtain are produced by
Wand's implementation of the methods in Wand & Jones (1995).

1. Wand, M. (2025). *KernSmooth: Functions for Kernel Smoothing Supporting
   Wand & Jones (1995).* R package version 2.23-26. CRAN.
   https://cran.r-project.org/package=KernSmooth
   *(the original package, whose Fortran routines this port embeds)*

2. Cai, Y. and Li, J. *r2py: AI-Assisted Conversion of R Statistical Packages
   to Python.* (in preparation)
   *(the conversion method, and the validation evidence for this port)*

3. Cai, Y. and Li, J. (2026). *r2py_kernsmooth* (version 0.1.1)
   [Computer software]. PyPI. https://pypi.org/project/r2py_kernsmooth/
   *(the exact artifact executed — please cite the version you actually ran)*

For the underlying statistical methods, please also cite Wand, M. P. and
Jones, M. C. (1995). *Kernel Smoothing.* Chapman & Hall, London.
ISBN 9780412552700.

### BibTeX

```bibtex
@Manual{wand2025kernsmooth,
  title  = {KernSmooth: Functions for Kernel Smoothing Supporting
            Wand \& Jones (1995)},
  author = {Matt Wand},
  year   = {2025},
  note   = {R package version 2.23-26},
  url    = {https://CRAN.R-project.org/package=KernSmooth}
}

@Article{cai2026r2py,
  title   = {r2py: AI-Assisted Conversion of R Statistical Packages to Python},
  author  = {Cai, Yufei and Li, Jun},
  year    = {2026},
  note    = {In preparation}
}

@Misc{cai2026r2pykernsmooth,
  title  = {r2py\_kernsmooth},
  author = {Cai, Yufei and Li, Jun},
  year   = {2026},
  note   = {Python package version 0.1.1},
  url    = {https://pypi.org/project/r2py_kernsmooth/}
}

@Book{wandjones1995,
  title     = {Kernel Smoothing},
  author    = {Wand, M.~P. and Jones, M.~C.},
  publisher = {Chapman \& Hall},
  address   = {London},
  year      = {1995},
  isbn      = {9780412552700}
}
```

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
