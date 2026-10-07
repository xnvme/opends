# SPDX-License-Identifier: BSD-3-Clause
"""Default the test suite to the ref backend and provide an open Driver."""

import os

os.environ.setdefault("OPENDS_BACKEND", "ref")

import pytest  # noqa: E402

import opends  # noqa: E402


@pytest.fixture
def driver():
    with opends.Driver() as drv:
        yield drv
