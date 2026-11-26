#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Install OpenDS target dependencies on the target.

Builds and installs xNVMe, xal, OpenDS, and fil in dependency order.
xNVMe is required for the aisio backend and ships the homi and qublk
tools that serve the controller. xal provides libxal, which OpenDS
links to read the extent index, and the xal-server tool that publishes
it. Assumes the tree has been synced to the target (rsync.py).

fil is bench-only and links opends_aisio, so it goes last. It once
needed its own xal prefix; it does not any more, since it asks for
xal >= 0.2.0 and the pinned xal satisfies that.

Names of the four limit the run to those. A hardware job gives "opends fil", since its guest has xnvme and xal.
"""

import argparse

from _helpers import ok, run_cijoe

ORDER = ["xnvme", "xal", "opends", "fil"]

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("names", nargs="*", metavar="NAME",
                    help=f"Build only these, of {', '.join(ORDER)}. Default: all.")
args = parser.parse_args()
unknown = set(args.names) - set(ORDER)
if unknown:
    parser.error(f"unknown: {', '.join(sorted(unknown))}")

for name in ORDER:
    if not args.names or name in args.names:
        run_cijoe(f"tasks/setup_{name}.yaml")

ok("setup_deps")
