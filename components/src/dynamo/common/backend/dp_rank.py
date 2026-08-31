# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DP-rank helpers shared across unified-backend engines.

The router encodes its rank decision in ``request.routing.dp_rank``.
Each engine validates it against the worker's local DP-rank slice and
forwards to the underlying inference engine. The helpers below capture
the extract-and-validate pattern so the same logic doesn't drift.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

from dynamo.common.backend.engine import GenerateRequest

logger = logging.getLogger(__name__)


def forced_dp_rank(request: GenerateRequest) -> Optional[int]:
    """Pull the router-forced ``dp_rank`` out of a request's ``routing``."""
    routing = cast("dict[str, Any]", request.get("routing") or {})
    rank = routing.get("dp_rank")
    return None if rank is None else int(rank)


def validate_global_dp_rank(
    dp_rank: Optional[int],
    dp_start: int,
    dp_size: int,
    backend_label: str,
) -> Optional[int]:
    """Bounds-check ``dp_rank`` against ``[dp_start, dp_start + dp_size)``;
    returns it when in range, else ``None`` (and warns)."""
    if dp_rank is None or dp_size <= 1:
        return None
    if not dp_start <= dp_rank < dp_start + dp_size:
        logger.warning(
            "Received DP rank %d outside [%d, %d); falling back to "
            "%s internal DP selection",
            dp_rank,
            dp_start,
            dp_start + dp_size,
            backend_label,
        )
        return None
    return dp_rank
