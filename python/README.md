# OpenDS Python bindings

Thin ctypes binding over the OpenDS C ABI. No compiled extension; the
package tracks the C library by ABI and loads it at import.

## Usage

```python
import opends

buf = opends.alloc(4096)
with opends.OpenDSFile("data.bin", "r") as f:
    nbytes = f.read_sync(buf, size=4096, file_offset=0)
```

Methods are named after the C families: `read_sync`/`write_sync` block and
return the byte count. Buffers may be any
object exposing `__cuda_array_interface__`, `__array_interface__`, a
torch-style `data_ptr()`, the buffer protocol, or a `HostBuffer` from
`opends.alloc`. These are registered on first use and deregistered at driver
shutdown.

For the cuFile pattern of one large allocation indexed by offset, pass a
bare device pointer (`ctypes.c_void_p` or an `int` address) plus an
explicit `size` and `dev_offset`. A bare pointer has no discoverable
extent, so register the base allocation once up front:

```python
opends.register_buffer(base_ptr, nbytes)   # ctypes.c_void_p or int
with opends.OpenDSFile(path, "r") as f:
    f.read_sync(base_ptr, size=chunk, dev_offset=off)
```

## Migrating from cufile-python (GDS)

OpenDS mirrors the synchronous surface of NVIDIA's `cufile-python` (the
`CuFile` bindings consumers such as LMCache call through), so a GDS backend
ports with minimal change. The cufile-python pattern registers one base
allocation, holds a driver open, and issues blocking `read`/`write` against a
bare device pointer plus `dev_offset`:

```python
import ctypes
import cufile
from cufile.bindings import cuFileBufRegister, cuFileBufDeregister

driver = cufile.CuFileDriver()                          # hold the driver open
cuFileBufRegister(ctypes.c_void_p(base), nbytes, flags=0)

addr = ctypes.c_void_p(base)
with cufile.CuFile(path, "r", use_direct_io=True) as f:
    n = f.read(addr, size, file_offset=foff, dev_offset=doff)

cuFileBufDeregister(ctypes.c_void_p(base))
```

The OpenDS version keeps the argument shape; `read_sync`/`write_sync` are
blocking and return the byte count:

```python
import ctypes
import opends

opends.register_buffer(base, nbytes)                    # pins the driver open
addr = ctypes.c_void_p(base)
with opends.OpenDSFile(path, "r", use_direct_io=True) as f:
    n = f.read_sync(addr, size, file_offset=foff, dev_offset=doff)

opends.deregister_buffer(base)
```

Mapping at a glance:

| cufile-python | OpenDS |
| --- | --- |
| `import cufile` | `import opends` |
| `cufile.CuFileDriver()` | implicit; opened by `register_buffer`/`OpenDSFile` |
| `cufile.CuFile(path, "r", use_direct_io=dio)` | `opends.OpenDSFile(path, "r", use_direct_io=dio)` |
| `f.read(buf, size, file_offset=, dev_offset=)` | `f.read_sync(...)`, same arguments |
| `f.write(buf, size, file_offset=, dev_offset=)` | `f.write_sync(...)`, same arguments |
| `cuFileBufRegister(c_void_p(p), size, flags=0)` | `opends.register_buffer(p, size)` |
| `cuFileBufDeregister(c_void_p(p))` | `opends.deregister_buffer(p)` |

Two differences to note. There is no explicit driver object: OpenDS opens the
driver when you register a buffer or open a file and closes it when the last
of either is released, so the `CuFileDriver()` line has no OpenDS equivalent.
And `use_direct_io` defaults to `True`, since every backend requires O_DIRECT,
so it can be omitted unless overriding.

## Backend selection

- `OPENDS_BACKEND` selects `libopends_<backend>.so` (default `aisio`).
- `OPENDS_LIBRARY` overrides with an explicit path.

The loader also looks in `../build/` relative to the package, so an
in-tree `meson compile -C build` is picked up without installation.

## Test

```sh
cd python
OPENDS_BACKEND=ref PYTHONPATH=. python tests/test_file_ref.py   # or: pytest
```

The test runs against the reference backend and needs no GPU.
