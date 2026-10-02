# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Let vLLM capture NIXL transfer telemetry without exporting it."""

import logging
import os

logger = logging.getLogger(__name__)

# NIXL treats these as disabled. A set false value vetoes capture_telemetry=True.
_FALSE = frozenset({"n", "0", "no", "off", "false", "disable"})


def allow_nixl_telemetry_capture() -> bool:
    """Drop a false NIXL_TELEMETRY_ENABLE so capture is not vetoed.

    Unset enable plus no exporter and no dir selects NIXL's in-process NOP
    sink: get_xfer_telemetry works, and nothing is exported. Returns True
    when the environment was changed.
    """
    enable = os.environ.get("NIXL_TELEMETRY_ENABLE")
    if enable is None or enable.strip().lower() not in _FALSE:
        return False

    for name in (
        "NIXL_TELEMETRY_ENABLE",
        "NIXL_TELEMETRY_EXPORTER",
        "NIXL_TELEMETRY_DIR",
    ):
        os.environ.pop(name, None)
    logger.info("Cleared false NIXL_TELEMETRY_ENABLE; capture stays, export stays off")
    return True
