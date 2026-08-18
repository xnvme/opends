# SPDX-License-Identifier: BSD-3-Clause
"""Default the test suite to the ref backend.
"""

import os

os.environ.setdefault("OPENDS_BACKEND", "ref")
