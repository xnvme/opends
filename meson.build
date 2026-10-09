# SPDX-License-Identifier: BSD-3-Clause
project('OpenDS', 'c',
  version : '0.1.0',
  default_options : ['c_std=c11'],
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
  'src/opends_ref.c',
  'src/opends_sync_async.c',
  'src/opends_batch_async.c',
  include_directories : opends_inc,
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
    'src/opends_cufile.c',
    include_directories : opends_inc,
    dependencies : [cuda_dep, cufile_dep],
    install : true,
  )

  opends_cufile_dep = declare_dependency(
    include_directories : opends_inc,
    link_with : libopends_cufile,
  )
endif

xnvme_dep = dependency('xnvme', required : false)
xal_dep = dependency('xal', required : false)

accel_backend = get_option('accel_backend')
accel_src = 'src/ds_accel_@0@.c'.format(accel_backend)
accel_kernel_lang = 'cuda'
accel_kernel_args = []
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
  accel_kernel_src = 'src/ds_bounce_kernel_cuda.cu'
  accel_runtime_found = cuda_dep.found() and cuda_drv_dep.found() and have_cuda_h
else
  error('accel_backend=@0@ has no ds_accel implementation; only cuda is wired up (see docs/vendor-port.md).'.format(accel_backend))
endif

accel_inc = []
if accel_sys_inc_dirs.length() > 0
  accel_inc = [include_directories(accel_sys_inc_dirs, is_system : true)]
endif

aisio_enabled = xnvme_dep.found() and accel_runtime_found and xal_dep.found()

if aisio_enabled
  # The async path links a tiny D2D copy kernel compiled from
  # src/ds_bounce_kernel_cuda.cu. Adding the kernel language only when aisio is
  # actually built keeps ref-only builds free of an nvcc dependency.
  add_languages(accel_kernel_lang, required : true, native : false)

  aisio_inc_dirs = [opends_inc, opends_internal_inc] + accel_inc

  libopends_aisio = shared_library('opends_aisio',
    'src/opends_common.c',
    'src/opends_aisio.c',
    'src/opends_sync_async.c',
    'src/opends_batch_async.c',
    accel_src,
    accel_kernel_src,
    include_directories : aisio_inc_dirs,
    dependencies : [xnvme_dep, xal_dep] + accel_deps,
    cuda_args : accel_kernel_args,
    install : true,
  )

  opends_aisio_dep = declare_dependency(
    include_directories : opends_inc,
    link_with : libopends_aisio,
  )

  # Consumer-facing header: the file-to-LBA extent record external
  # benchmarks deserialize from the test-prep tool.
  install_headers(
    'src/ds_extent.h',
  )

  pkg.generate(libopends_aisio,
    name : 'opends_aisio',
    description : 'OpenDS aisio backend (raw NVMe via xNVMe upcie P2P)',
    requires : ['xnvme'],
  )
endif

# Test binaries
threads_dep = dependency('threads')

executable('test_smoke_ref',
  'tests/test_smoke_ref.c',
  dependencies : [opends_ref_dep],
)

# Prep binary: writes the shared pattern file consumed by every
# backend's test_sync_read_X, and the small extents record consumed by
# the aisio leg (which runs after the FS is unmounted).
executable('test_sync_read_prep',
  'tests/test_sync_read_prep.c',
  'tests/test_extents_io.c',
  include_directories : [opends_internal_inc, tests_inc],
)

executable('test_sync_read_ref',
  'tests/test_sync_read_ref.c',
  dependencies : [opends_ref_dep],
)

executable('test_mt_read_ref',
  'tests/test_mt_read_ref.c',
  dependencies : [opends_ref_dep, threads_dep],
)

executable('test_async_read_ref',
  'tests/test_async_read_ref.c',
  dependencies : [opends_ref_dep],
)

executable('test_batch_read_ref',
  'tests/test_batch_read_ref.c',
  dependencies : [opends_ref_dep],
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

if aisio_test_enabled
  aisio_test_inc_dirs = [opends_internal_inc, tests_inc] + accel_inc

  executable('test_sync_read_aisio',
    'tests/test_sync_read_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep],
  )
  executable('test_mt_read_aisio',
    'tests/test_mt_read_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep, cuda_drv_dep, threads_dep],
  )
  executable('test_async_read_aisio',
    'tests/test_async_read_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep, cuda_drv_dep],
  )
  executable('test_batch_read_aisio',
    'tests/test_batch_read_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep, cuda_drv_dep],
  )
  executable('test_register_large_aisio',
    'tests/test_register_large_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep],
  )
  executable('test_coherency_aisio',
    'tests/test_coherency_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep],
  )
  executable('test_stream_read_aisio',
    'tests/test_stream_read_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep, cuda_drv_dep],
  )
  executable('test_sync_write_aisio',
    'tests/test_sync_write_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep, cuda_drv_dep],
  )
  executable('test_stream_write_aisio',
    'tests/test_stream_write_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep, cuda_drv_dep],
  )
  executable('test_block_alloc_aisio',
    'tests/test_block_alloc_aisio.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep, cuda_drv_dep],
  )

  executable('aisio_homi_demo',
    'examples/aisio_homi_demo.c',
    include_directories : aisio_test_inc_dirs,
    dependencies : [opends_aisio_dep, cuda_dep, cuda_drv_dep],
  )
else
  executable('test_sync_read_aisio', 'tests/skip.c')
  executable('test_mt_read_aisio', 'tests/skip.c')
  executable('test_async_read_aisio', 'tests/skip.c')
  executable('test_batch_read_aisio', 'tests/skip.c')
  executable('test_register_large_aisio', 'tests/skip.c')
  executable('test_coherency_aisio', 'tests/skip.c')
  executable('test_stream_read_aisio', 'tests/skip.c')
  executable('test_sync_write_aisio', 'tests/skip.c')
  executable('test_stream_write_aisio', 'tests/skip.c')
  executable('test_block_alloc_aisio', 'tests/skip.c')
endif

summary({
  'Reference backend' : true,
  'cuFile backend' : cufile_enabled,
  'aisio backend' : aisio_enabled,
  'aisio accelerator vendor' : accel_backend,
}, section : 'Backends')
