# SPDX-License-Identifier: BSD-3-Clause
"""Driver-level bindings against the ref backend."""

import mmap

import pytest

import opends


def test_requires_open_driver(tmp_path):
    # Everything that touches the C driver refuses without an open Driver,
    # as cuFile does before cuFileDriverOpen.
    assert opends.driver._driver_refs == 0
    path = tmp_path / "a"
    path.write_bytes(b"\0" * 4096)
    calls = [
        lambda: opends.OpenDSFile(str(path), "r"),
        lambda: opends.alloc(4096),
        lambda: opends.register_buffer(bytearray(4096)),
        lambda: opends.register_stream(0x10),
        lambda: opends.Batch(1),
        lambda: opends.get_properties(),
        lambda: opends.set_max_direct_io_size(1 << 20),
    ]
    for call in calls:
        with pytest.raises(opends.OpenDSError) as info:
            call()
        assert info.value.code is opends.ErrorCode.DRIVER_NOT_INITIALIZED
    assert opends.use_count() == 0


def test_driver_counts_holders():
    base = opends.driver._driver_refs
    a = opends.Driver()
    b = opends.Driver()
    assert opends.driver._driver_refs == base + 2
    a.close()
    a.close()
    assert opends.driver._driver_refs == base + 1
    b.close()
    assert opends.driver._driver_refs == base
    b.open()
    assert opends.driver._driver_refs == base + 1
    b.close()
    assert opends.driver._driver_refs == base


def test_file_does_not_hold_driver(driver, tmp_path):
    refs = opends.driver._driver_refs
    with opends.OpenDSFile(str(tmp_path / "a"), "w"):
        assert opends.driver._driver_refs == refs
    assert opends.driver._driver_refs == refs


def test_properties(driver):
    props = opends.get_properties()
    assert isinstance(props, opends.DriverProperties)
    assert (props.major_version, props.minor_version) == (1, 0)
    assert props.max_direct_io_size == 0


def test_set_max_direct_io_size(driver):
    opends.set_max_direct_io_size(1 << 20)


def test_use_count_tracks_handles(driver, tmp_path):
    base = opends.use_count()
    with opends.OpenDSFile(str(tmp_path / "a"), "w") as a:
        assert opends.use_count() == base + 1
        with opends.OpenDSFile(str(tmp_path / "b"), "w") as b:
            assert opends.use_count() == base + 2
            assert b.fileno() != a.fileno()
        assert opends.use_count() == base + 1
    assert opends.use_count() == base


def test_error_carries_code(driver, tmp_path):
    path = tmp_path / "plain"
    path.write_bytes(b"\0" * 4096)
    before = opends.use_count()
    with pytest.raises(opends.OpenDSError) as info:
        opends.OpenDSFile(str(path), "r", use_direct_io=False)
    assert info.value.code is opends.ErrorCode.DIO_NOT_SET
    assert "5020" in str(info.value)
    assert opends.use_count() == before


def test_register_deregister_buffer_and_stream(driver):
    buf = mmap.mmap(-1, 4096)
    opends.register_buffer(buf)
    with pytest.raises(opends.OpenDSError) as info:
        opends.register_buffer(buf)
    assert info.value.code is opends.ErrorCode.MEMORY_ALREADY_REGISTERED
    opends.deregister_buffer(buf)
    with pytest.raises(opends.OpenDSError) as info:
        opends.deregister_buffer(buf)
    assert info.value.code is opends.ErrorCode.MEMORY_NOT_REGISTERED
    opends.register_stream(0x1234)
    opends.deregister_stream(0x1234)
