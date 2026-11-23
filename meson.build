# SPDX-License-Identifier: BSD-3-Clause
project('OpenDS', 'c',
  version : '0.4.0',
  default_options : ['c_std=c11', 'buildtype=release'],
)

version = meson.project_version().split('.')
add_project_arguments(
  '-DOPENDS_VERSION_MAJOR=' + version[0],
  '-DOPENDS_VERSION_MINOR=' + version[1],
  '-DOPENDS_VERSION_PATCH=' + version[2],
  language : 'c',
)

opends_inc = include_directories('include')
opends_internal_inc = include_directories('src')
tests_inc = include_directories('tests')

public_headers = files(
  'include/opends.h',
)
install_headers(public_headers)

libopends_ref = shared_library('opends_ref',
  'src/opends_common.c',
  'src/ref/opends_ref.c',
  'src/opends_sync_async.c',
  'src/opends_batch_async.c',
  include_directories : [opends_inc, opends_internal_inc],
  gnu_symbol_visibility : 'hidden',
  install : true,
)

opends_ref_dep = declare_dependency(
  include_directories : opends_inc,
  link_with : libopends_ref,
)

pkg = import('pkgconfig')
pkg.generate(libopends_ref,
  name : 'opends',
  description : 'OpenDS reference backend',
)

cc = meson.get_compiler('c')
cuda_dep = dependency('cuda', modules : ['cuda', 'cudart'], required : false)
cufile_dep = cc.find_library('cufile',
  dirs : ['/usr/local/cuda/lib64'],
  required : false,
)

cufile_enabled = cuda_dep.found() and cufile_dep.found()

if cufile_enabled
  libopends_cufile = shared_library('opends_cufile',
    'src/opends_common.c',
    'src/cufile/opends_cufile.c',
    include_directories : [opends_inc, opends_internal_inc],
    dependencies : [cuda_dep, cufile_dep],
    gnu_symbol_visibility : 'hidden',
    install : true,
  )

  opends_cufile_dep = declare_dependency(
    include_directories : opends_inc,
    link_with : libopends_cufile,
  )

  pkg.generate(libopends_cufile,
    name : 'opends_cufile',
    description : 'OpenDS cuFile backend (NVIDIA GPUDirect Storage)',
  )
endif

xnvme_dep = dependency('xnvme', version : '>=0.8.0', required : false)
xal_dep = dependency('xal', version : '>=0.4.0', required : false)

accel_backend = get_option('accel_backend')
accel_src = 'src/accel/opends_accel_@0@.c'.format(accel_backend)
accel_kernel_lang = 'cuda'
# Compile device code for the GPU present at build time. The nvcc default
# arch has no SASS for newer GPUs, and the driver rejects PTX from a newer
# toolkit, so the first kernel launch fails with CUDA error 222.
accel_kernel_args = ['-arch=native']
accel_sys_inc_dirs = []

if accel_backend == 'cuda'
  cuda_drv_dep = cc.find_library('cuda',
    dirs : [
      '/usr/local/cuda/lib64/stubs',
      '/usr/lib/x86_64-linux-gnu/stubs',
      '/usr/local/cuda/lib64',
    ],
    required : false,
  )

  # cuda.h ships with the CUDA toolkit (under .../targets/<arch>/include
  # on Ubuntu) but a minimal driver-dev package may put it on the system
  # include path instead. Probe toolkit locations first; fall back to a
  # system-wide cc.has_header check.
  foreach p : [
    '/usr/local/cuda/targets/x86_64-linux/include',
    '/usr/local/cuda/include',
  ]
    if accel_sys_inc_dirs.length() == 0 and \
        run_command('test', '-f', p / 'cuda.h', check : false).returncode() == 0
      accel_sys_inc_dirs += p
    endif
  endforeach
  have_cuda_h = accel_sys_inc_dirs.length() > 0 or cc.has_header('cuda.h')

  accel_deps = [cuda_dep, cuda_drv_dep]
  accel_kernel_src = 'src/accel/opends_bounce_kernel_cuda.cu'
  accel_runtime_found = cuda_dep.found() and cuda_drv_dep.found() and have_cuda_h
else
  error('accel_backend=@0@ has no opends_accel implementation; only cuda is wired up (see docs/vendor-port.md).'.format(accel_backend))
endif

accel_inc = []
if accel_sys_inc_dirs.length() > 0
  accel_inc = [include_directories(accel_sys_inc_dirs, is_system : true)]
endif

aisio_enabled = xnvme_dep.found() and accel_runtime_found and xal_dep.found()

if aisio_enabled
  # The backend links device code compiled from
  # src/accel/opends_bounce_kernel_cuda.cu. Adding its language only here
  # keeps a build without aisio free of nvcc.
  add_languages(accel_kernel_lang, required : true, native : false)

  aisio_inc_dirs = [opends_inc, opends_internal_inc] + accel_inc

  libopends_aisio = shared_library('opends_aisio',
    'src/opends_common.c',
    'src/aisio/opends_aisio.c',
    'src/aisio/opends_aisio_posix.c',
    'src/aisio/opends_aisio_config.c',
    'src/opends_sync_async.c',
    'src/opends_batch_async.c',
    accel_src,
    accel_kernel_src,
    include_directories : aisio_inc_dirs,
    dependencies : [xnvme_dep, xal_dep] + accel_deps,
    cuda_args : accel_kernel_args,
    gnu_symbol_visibility : 'hidden',
    install : true,
  )

  opends_aisio_dep = declare_dependency(
    include_directories : opends_inc,
    link_with : libopends_aisio,
  )

  pkg.generate(libopends_aisio,
    name : 'opends_aisio',
    description : 'OpenDS aisio backend (raw NVMe via xNVMe upcie P2P)',
    requires : ['xnvme'],
  )
endif

# Test binaries
threads_dep = dependency('threads')

ref_tests = [
  'smoke',
  'sync_read',
  'mt_read',
  'async_read',
  'batch_read',
]

foreach t : ref_tests
  name = 'test_@0@_ref'.format(t)
  executable(name,
    'tests/@0@.c'.format(name),
    dependencies : [opends_ref_dep, threads_dep],
  )
endforeach

# Prep binary: writes the shared pattern file consumed by every
# backend's test_sync_read_X.
executable('test_sync_read_prep',
  'tests/test_sync_read_prep.c',
)

if cufile_enabled
  executable('test_sync_read_cufile',
    'tests/test_sync_read_cufile.c',
    dependencies : [opends_cufile_dep, cuda_dep],
  )
else
  executable('test_sync_read_cufile', 'tests/skip.c')
endif

aisio_test_enabled = aisio_enabled and cuda_dep.found()

aisio_tests = [
  'sync_read',
  'mt_read',
  'async_read',
  'batch_read',
  'register_large',
  'coherency',
  'sparse_read',
  'stream_read',
  'sync_write',
  'stream_write',
  'raw_block',
]

if aisio_test_enabled
  aisio_test_inc_dirs = [tests_inc] + accel_inc
  aisio_test_deps = [opends_aisio_dep, cuda_dep, cuda_drv_dep, threads_dep]
endif

# The test of a backend that is not built is a stub that passes
foreach t : aisio_tests
  name = 'test_@0@_aisio'.format(t)
  if aisio_test_enabled
    executable(name,
      'tests/@0@.c'.format(name),
      include_directories : aisio_test_inc_dirs,
      dependencies : aisio_test_deps,
    )
  else
    executable(name, 'tests/skip.c')
  endif
endforeach

if aisio_test_enabled
  executable('sync_read_example',
    'examples/sync_read.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep, cuda_drv_dep],
  )

  executable('async_read_example',
    'examples/async_read.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep],
  )

  executable('stream_read_example',
    'examples/stream_read.cu',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep],
    cuda_args : accel_kernel_args,
  )
endif

summary({
  'Reference backend' : true,
  'cuFile backend' : cufile_enabled,
  'aisio backend' : aisio_enabled,
  'aisio accelerator vendor' : accel_backend,
}, section : 'Backends')
