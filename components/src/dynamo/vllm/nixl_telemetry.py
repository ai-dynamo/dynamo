# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Let vLLM capture NIXL transfer telemetry without exporting it."""

import logging
import os

logger = logging.getLogger(__name__)

# NIXL compares these case-insensitively and does not trim whitespace.
_FALSE = frozenset({"n", "0", "no", "off", "false", "disable"})

# Plugin name. A set exporter is read before NIXL_TELEMETRY_DIR or a config file.
_NOP_EXPORTER = "NOP"


def allow_nixl_telemetry_capture() -> bool:
    """Replace a false NIXL_TELEMETRY_ENABLE with capture-only collection.

    NIXL 1.4 treats a set false value as a veto of capture_telemetry=True,
    so that value is removed. The exporter is pinned to NOP so 1.4 does not
    fall through to NIXL_TELEMETRY_DIR or a config file. Enable is left
    unset: NIXL 1.3.2, which the vLLM image builds, has no NOP plugin and
    loads an exporter only when enable is true. Returns True when the
    environment was changed.
    """
    enable = os.environ.get("NIXL_TELEMETRY_ENABLE")
    if enable is None or enable.lower() not in _FALSE:
        return False

    os.environ.pop("NIXL_TELEMETRY_ENABLE", None)
    os.environ["NIXL_TELEMETRY_EXPORTER"] = _NOP_EXPORTER
    os.environ.pop("NIXL_TELEMETRY_DIR", None)
    logger.info("Pinned NIXL telemetry to the NOP exporter")
    return True
