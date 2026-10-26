# OpenDS Python bindings

Thin ctypes binding over the OpenDS C ABI. No compiled extension; the
package tracks the C library by ABI and loads it at import. The modules are
laid out by concern (`opends.driver`, `opends.buffer`, `opends.file`,
`opends.batch`, and `opends.cdll` for the raw prototypes); `import opends`
gives the convenience surface used below.

## Usage

```python
import opends

with opends.Driver():
    buf = opends.alloc(4096)
    with opends.OpenDSFile("data.bin", "r") as f:
        nbytes = f.read_sync(buf, size=4096, file_offset=0)
```

`examples/python_demo.py` at the repository root reads one file through every
mode (sync, async, stream, batch) into one registered torch CUDA tensor on the
aisio backend.

Methods are named after the C families: `read_sync`/`write_sync` block and
return the byte count, `read_async`/`write_async` return a `Future`, and
`read_stream`/`write_stream` enqueue on a CUDA stream. Buffers may be any
object exposing `__cuda_array_interface__`, `__array_interface__`, a
torch-style `data_ptr()`, the buffer protocol, or a `HostBuffer` from
`opends.alloc`. These are registered on first use and deregistered at driver
shutdown. `opends.alloc` is the backend's own allocator: host memory on ref,
device memory on GPU backends.

For the cuFile pattern of one large allocation indexed by offset, pass a
bare device pointer (`ctypes.c_void_p` or an `int` address) plus an
explicit `size` and `dev_offset`. A bare pointer has no discoverable
extent, so register the base allocation once up front:

```python
opends.register_buffer(base_ptr, nbytes)   # ctypes.c_void_p or int
with opends.OpenDSFile(path, "r") as f:
    f.read_sync(base_ptr, size=chunk, dev_offset=off)
```

## Driver

`opends.Driver()` opens the driver, as `cuFileDriverOpen` does, and must be
open before any file, buffer, stream or batch call; without it they raise
`OpenDSError` with `ErrorCode.DRIVER_NOT_INITIALIZED`. Driver objects are
counted, so independent components may each hold one; the C driver closes
when the last is closed, which drops every registration, so close files and
deregister buffers first. `get_properties()`, `use_count()` and
`set_max_direct_io_size()` pass through to the C driver calls. Failures raise
`OpenDSError`, whose `code` is an `opends.ErrorCode` member. A driver left
open is closed at exit and on SIGTERM/SIGINT; a framework that installs its
own SIGTERM handler calls `opends.cleanup()` from it.

```python
with opends.Driver():
    props = opends.get_properties()      # DriverProperties namedtuple
    print(props.max_direct_io_size)
```

## Async I/O

`read_async` and `write_async` take the same arguments as `read_sync` and
`write_sync` and return a `Future` without waiting. `Future.result()` blocks
until the operation completes and returns the byte count; `Future.done` polls.
Futures complete in any order. Keep the buffer and the Future alive until
`result()` has returned, and let operations complete before the Driver
closes: a Future awaited after that raises `DRIVER_NOT_INITIALIZED`. This is
the OpenDS primitive async form; cuFile's stream-ordered `ReadAsync`/
`WriteAsync` map to the stream family instead.

```python
with opends.OpenDSFile(path, "r") as f:
    futs = [
        f.read_async(buf, size=n, file_offset=i * n, dev_offset=i * n)
        for i in range(4)
    ]
    total = sum(fut.result() for fut in futs)
```

## Stream I/O

`read_stream` and `write_stream` are the cuFile `ReadAsync`/`WriteAsync`
counterparts: the operation is enqueued on a CUDA stream and completes in
stream order. `stream` accepts a raw handle (`int` or `ctypes.c_void_p`), a
`torch.cuda.Stream` or a `cupy.cuda.Stream`; `None` is the default stream.
The returned `StreamOp` holds the storage the backend reads and writes when
the stream reaches the operation, so keep it alive until the stream has been
synchronized; `result()` raises until then and returns the byte count after.
`register_stream` and
`deregister_stream` wrap `cuFileStreamRegister`/`Deregister`.

```python
opends.register_stream(stream)
with opends.OpenDSFile(path, "r") as f:
    op = f.read_stream(buf, size=n, file_offset=0, stream=stream)
    stream.synchronize()
    assert op.result() == n
opends.deregister_stream(stream)
```

## Batch I/O

`opends.Batch(max_nr)` wraps `cuFileBatchIOSetUp`: it holds up to `max_nr`
operations in flight, and a slot frees once `get_status` has delivered its
completion. `submit` takes `BatchOp` entries (file, buffer, size, offsets,
`opends.READ` or `opends.WRITE`, and any object as cookie) and raises
`OpenDSError` with `ErrorCode.BATCH_FULL` when they exceed the free slots.
`get_status(min_nr, max_nr, timeout)` waits for at least `min_nr` completions
(timeout in seconds, `None` without limit) and returns
`BatchEvent(cookie, status, ret)` tuples, each completion once; `status` is an
`opends.Status`. `cancel` reports undelivered completions as `CANCELED`.
Operations in a batch are unordered. A `Batch` is not thread-safe; guard one
shared between threads.

```python
with opends.OpenDSFile(path, "r") as f, opends.Batch(8) as batch:
    batch.submit(
        opends.BatchOp(f, buf, size=n, file_offset=i * n, dev_offset=i * n, cookie=i)
        for i in range(8)
    )
    for ev in batch.get_status(min_nr=8):
        assert ev.status == opends.Status.COMPLETE and ev.ret == n
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

driver = opends.Driver()                                # hold the driver open
opends.register_buffer(base, nbytes)
addr = ctypes.c_void_p(base)
with opends.OpenDSFile(path, "r", use_direct_io=True) as f:
    n = f.read_sync(addr, size, file_offset=foff, dev_offset=doff)

opends.deregister_buffer(base)
```

Mapping at a glance:

| cufile-python | OpenDS |
| --- | --- |
| `import cufile` | `import opends` |
| `cufile.CuFileDriver()` | `opends.Driver()` |
| `cuFileDriverGetProperties()` | `opends.get_properties()` |
| `cuFileDriverSetMaxDirectIOSize(n)` | `opends.set_max_direct_io_size(n)` |
| `cuFileUseCount()` | `opends.use_count()` |
| `cufile.CuFile(path, "r", use_direct_io=dio)` | `opends.OpenDSFile(path, "r", use_direct_io=dio)` |
| `f.read(buf, size, file_offset=, dev_offset=)` | `f.read_sync(...)`, same arguments |
| `f.write(buf, size, file_offset=, dev_offset=)` | `f.write_sync(...)`, same arguments |
| `cuFileBufRegister(c_void_p(p), size, flags=0)` | `opends.register_buffer(p, size)` |
| `cuFileBufDeregister(c_void_p(p))` | `opends.deregister_buffer(p)` |
| `cuFileReadAsync(fh, buf, &size, &foff, &doff, &n, stream)` | `op = f.read_stream(buf, size, file_offset=, dev_offset=, stream=)`; `op.result()` |
| `cuFileWriteAsync(...)` | `f.write_stream(...)`, same shape |
| `cuFileStreamRegister(stream, flags)` | `opends.register_stream(stream, flags)` |
| `cuFileStreamDeregister(stream)` | `opends.deregister_stream(stream)` |
| `cuFileBatchIOSetUp(&h, nr)` | `opends.Batch(nr)` |
| `cuFileBatchIOSubmit(h, nr, params, flags)` | `batch.submit(ops, flags)` with `opends.BatchOp` entries |
| `cuFileBatchIOGetStatus(h, min_nr, &nr, events, &timeout)` | `batch.get_status(min_nr, max_nr, timeout)` |
| `cuFileBatchIOCancel(h)` | `batch.cancel()` |
| `cuFileBatchIODestroy(h)` | `batch.close()`, or leave the `with` block |

One difference to note: `use_direct_io` defaults to `True`, since every
backend requires O_DIRECT, so it can be omitted unless overriding.

## Backend selection

- `OPENDS_BACKEND` selects `libopends_<backend>.so` (default `aisio`).
- `OPENDS_LIBRARY` overrides with an explicit path.

The loader also looks in `../build/` relative to the package, so an
in-tree `meson compile -C build` is picked up without installation.

## Test

```sh
cd python
python -m pytest tests    # conftest selects the ref backend
```

The suite runs against the reference backend and needs no GPU. It covers
the sync, async, stream and batch families and the driver calls.
`tests/test_aisio_cuda.py` runs the same families against the aisio backend
on a live HOMI stack and skips unless `OPENDS_BACKEND=aisio` and
`OPENDS_HOMI_MNT` are set and the CUDA libraries load; `scripts/run_tests.py`
runs it as the `python` test of the aisio suite.
