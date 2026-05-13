# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Optional

try:
    from sglang.srt.utils.network import NetworkAddress
except ImportError:
    NetworkAddress = None

SGLANG_WORKER_GROUP_ID_KEY = "sglang_worker_group_id"


def get_sglang_worker_group_id(server_args) -> Optional[str]:
    """Return the shared SGLang multi-node worker group id."""
    nnodes = getattr(server_args, "nnodes", 1) or 1
    if nnodes <= 1:
        return None

    dist_init_addr = getattr(server_args, "dist_init_addr", None)
    if not dist_init_addr:
        logging.warning(
            "SGLang multi-node worker group id requires dist_init_addr; "
            "falling back to local worker id behavior"
        )
        return None

    if NetworkAddress is None:
        logging.warning(
            "SGLang NetworkAddress is unavailable; cannot normalize "
            "dist_init_addr for worker group id"
        )
        return None

    try:
        parsed = NetworkAddress.parse(str(dist_init_addr).strip())
        resolved = parsed.resolved()
        return f"dist_init:{resolved.to_tcp()}"
    except Exception as e:
        logging.warning(
            "Failed to normalize SGLang dist_init_addr=%r for worker group id: %s",
            dist_init_addr,
            e,
        )
        return None
