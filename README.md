# python-KernSmooth

An AI-assisted port of the R package [**KernSmooth**](https://cran.r-project.org/package=KernSmooth) (v. 2.23-26) to
Python, together with the complete analysis, conversion, and validation pipeline used to produce it.

KernSmooth implements the kernel-smoothing methods of Wand, M.P. and Jones, M.C. (1995), *Kernel Smoothing*, Chapman
and Hall, and ships as a *recommended* package with every standard R installation. This repository documents and
executes a systematic, dependency-order translation of KernSmooth into an installable Python package,
**r2py_kernsmooth**, which compiles the original Fortran 77 computational core verbatim via `numpy.f2py` and
replaces the R wrapper layer with semantically equivalent, type-annotated Python functions built on NumPy and SciPy.

A full account of the project's methodology, findings, and validation results is given in the technical report at
[`docs/report/main.tex`](docs/report/main.tex) (compiled PDF: `docs/report/main.pdf`) and in the accompanying paper
at [`docs/paper/main.tex`](docs/paper/main.tex).

## Repository Structure

| Path | Contents |
| --- | --- |
| `KernSmooth/` | Unmodified R package source (v. 2.23-26). |
| `r2py_kernsmooth/` | The installable Python package; also mirrored as a standalone repository at [`github.com/r2py-project/r2py_kernsmooth`](https://github.com/r2py-project/r2py_kernsmooth) via `git subtree`. |
| `structural_analysis/` | Dependency-graph artefacts and per-function JSON analysis of the R source. |
| `language_dependency_analysis/` | Per-file R-to-Python language-dependency tables and the generated conversion guides. |
| `conversion_results/` | Per-function JSON translation artefacts produced by the conversion pipeline. |
| `docs/` | Architecture document, planning materials, per-phase summaries, the technical report, and the paper. |
| `.claude/agents/`, `.claude/commands/` | Sub-agent and skill specifications for the Claude Code-based conversion workflow. |
| `git_pull.sh`, `git_push.sh` | Cluster batch scripts synchronising this repository with the `r2py_kernsmooth` subtree remote. |
| `install_environments.sh` | Cluster batch script provisioning the `r-to-python` conda environment. |

## Conversion Methodology

The port was carried out in seven phases, each documented in detail in the technical report:

1. **Static analysis** of the R package's function-level dependency structure and Fortran integration conventions.
2. **Python build infrastructure**: a `meson-python` build backend compiling the original Fortran sources via
   `f2py`, with cross-platform BLAS discovery and `cibuildwheel`-based PyPI distribution.
3. **Language-dependency cataloguing**: systematic extraction of every R standard-library call site and generation
   of a dedicated Python translation guide for each, to avoid silent numerical discrepancies (e.g. FFT
   normalisation, sample-variance denominators, integer width).
4. **Automated function conversion**, in dependency order, from R to Python, followed by package assembly and
   correction of R-versus-f2py return-value semantics.
5. **Code-quality audit**: type-annotation completion, namespace hygiene, dead-code removal, and alignment of error
   and warning messages with the R source.
6. **Regression test infrastructure**: R test scripts ported to `pytest`/`rpy2`-based assertions that compare live
   output against the R reference implementation.
7. **Comprehensive test suite construction**: positive, negative, and edge-case tests generated for all seven public
   functions, closing out with 518 passing tests and zero failures against R KernSmooth 2.23 via `rpy2`.

## Installation

The Python package can be installed independently:

```bash
pip install r2py_kernsmooth
```

For local development from this repository (requires a Fortran compiler and a BLAS implementation such as
OpenBLAS):

```bash
cd r2py_kernsmooth
pip install --no-build-isolation .
```

See [`r2py_kernsmooth/README.md`](r2py_kernsmooth/README.md) for package usage and the list of public functions.

## Testing

```bash
python -m pytest r2py_kernsmooth/tests/ -q
```

The test suite requires `rpy2` and an R installation with the `KernSmooth` and `carData` packages, since tests
assert numerical agreement against live R reference values.

## License

The original content of this repository is distributed under an "Unlimited" license, matching the terms of upstream
KernSmooth. See [LICENSE](LICENSE) for details. The vendored `KernSmooth/` subdirectory retains its own upstream
license (see `KernSmooth/DESCRIPTION` and `KernSmooth/LICENCE.note`), and `r2py_kernsmooth/` carries its own
`LICENSE`/`NOTICE` pair for standalone distribution.

## Attribution

KernSmooth was originally authored by **Matt Wand**, with LINPACK Fortran routines contributed by **Cleve Moler**
and the R-language port maintained by **Brian Ripley**. The Python port and this project's conversion pipeline are
authored by **Yufei Cai** (ycai9@nd.edu) and **Jun Li** (jun.li@nd.edu).
