# SPDX-License-Identifier: BSD-3-Clause
"""Default the test suite to the ref backend and provide an open Driver.

On a GPU backend opends.alloc returns device memory, so the tests that
read HostBuffers through the host are skipped; only test_aisio_cuda runs.
"""

import os

os.environ.setdefault("OPENDS_BACKEND", "ref")

import pytest  # noqa: E402

import opends  # noqa: E402
from opends import cdll  # noqa: E402


@pytest.fixture
def driver():
    with opends.Driver() as drv:
        yield drv


def pytest_collection_modifyitems(items):
    if cdll.BACKEND == "ref":
        return
    skip = pytest.mark.skip(reason="reads HostBuffers through the host; ref only")
    for item in items:
        if item.module.__name__ != "test_aisio_cuda":
            item.add_marker(skip)
