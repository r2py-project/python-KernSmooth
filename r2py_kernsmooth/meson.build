project('r2py_kernsmooth', 'c', 'fortran')
py_mod = import('python')
py = py_mod.find_installation(pure: false)
numpy_dep = dependency('numpy')
fc = meson.get_compiler('fortran')
blas_dep = dependency('blas', required: false, method: 'pkg-config')
if not blas_dep.found()
  blas_dep = fc.find_library('openblaso', required: false)
endif
if not blas_dep.found()
  blas_dep = fc.find_library('openblas', required: false)
endif
if not blas_dep.found()
  blas_dep = fc.find_library('blas', required: false)
endif
if not blas_dep.found()
  fs = import('fs')
  if fs.is_file('/usr/lib64/libopenblaso.so.0')
    blas_dep = declare_dependency(link_args: ['/usr/lib64/libopenblaso.so.0'])
  else
    error('BLAS not found. Install openblas-devel (Linux) or run: brew install openblas (macOS).')
  endif
endif
incdir_numpy = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
incdir_f2py = run_command(py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true).stdout().strip()
fortranobject_c = incdir_f2py / 'fortranobject.c'
numpy_include = include_directories(incdir_numpy, incdir_f2py)
py.install_sources(
  'r2py_kernsmooth/__init__.py',
  subdir: 'r2py_kernsmooth'
)
fortran_sources = files(
  'src/blkest.f',
  'src/cp.f',
  'src/dgedi.f',
  'src/dgefa.f',
  'src/dgesl.f',
  'src/dqrdc.f',
  'src/dqrsl.f',
  'src/linbin2D.f',
  'src/linbin.f',
  'src/locpoly.f',
  'src/rlbin.f',
  'src/sdiag.f',
  'src/sstdiag.f',
)
f2py_gen = custom_target(
  'f2py_wrapper',
  input: ['src/_KernSmooth.pyf'],
  output: ['_KernSmoothmodule.c', '_KernSmooth-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@']
)
py.extension_module(
  '_KernSmooth',
  [fortranobject_c, fortran_sources, f2py_gen],
  include_directories: numpy_include,
  install: true,
  subdir: 'r2py_kernsmooth',
  dependencies: [numpy_dep, blas_dep]
)