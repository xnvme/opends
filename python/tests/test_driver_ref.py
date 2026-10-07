# SPDX-License-Identifier: BSD-3-Clause
"""Driver-level bindings against the ref backend."""

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
    ]
    for call in calls:
        with pytest.raises(opends.OpenDSError) as info:
            call()
        assert info.value.code is opends.ErrorCode.DRIVER_NOT_INITIALIZED


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


def test_error_carries_code(driver, tmp_path):
    path = tmp_path / "plain"
    path.write_bytes(b"\0" * 4096)
    with pytest.raises(opends.OpenDSError) as info:
        opends.OpenDSFile(str(path), "r", use_direct_io=False)
    assert info.value.code is opends.ErrorCode.DIO_NOT_SET
    assert "5020" in str(info.value)
