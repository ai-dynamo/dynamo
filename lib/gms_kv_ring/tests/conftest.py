# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-test faulthandler watchdog for the suite.

If any test hangs longer than HANG_S, dump tracebacks of every
thread to stderr. Tests are individually short (<5s) so a 30s
watchdog is comfortably above the usual legitimate run. Longer stress
tests can use pytest-timeout markers; the watchdog follows those markers
so it does not dump noisy stacks for healthy scale tests."""

import faulthandler
import os
import sys

HANG_S = float(os.environ.get("GMS_KV_RING_TEST_WATCHDOG_S", "30"))


def _watchdog_seconds(item) -> float:
    marker = item.get_closest_marker("timeout")
    if marker and marker.args:
        try:
            return max(HANG_S, float(marker.args[0]))
        except (TypeError, ValueError):
            pass
    return HANG_S


def pytest_runtest_setup(item):
    faulthandler.dump_traceback_later(
        _watchdog_seconds(item), repeat=False, file=sys.stderr
    )


def pytest_runtest_teardown(item, nextitem):
    faulthandler.cancel_dump_traceback_later()
