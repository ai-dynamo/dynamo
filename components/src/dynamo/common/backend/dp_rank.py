# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DP-rank helpers shared across unified-backend engines.

The router encodes its rank decision in ``request.routing.dp_rank``.
Each engine validates it against the local DP-rank slice the worker
owns and forwards it to the underlying inference engine. The helper
below captures the validation + warn-and-fall-back pattern so the same
logic doesn't drift across backends.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def validate_global_dp_rank(
    dp_rank: Optional[int],
    dp_start: int,
    dp_size: int,
    backend_label: str,
) -> Optional[int]:
    """Bounds-check a router-supplied global DP rank against this worker's
    slice ``[dp_start, dp_start + dp_size)``.

    Returns the (int-coerced) rank when in range. Returns ``None`` and
    emits a warning otherwise so the caller can fall back to the engine's
    internal load balancer.
    """
    if dp_rank is None or dp_size <= 1:
        return None
    rank = int(dp_rank)
    if not dp_start <= rank < dp_start + dp_size:
        logger.warning(
            "Received DP rank %d outside [%d, %d); falling back to "
            "%s internal DP selection",
            rank,
            dp_start,
            dp_start + dp_size,
            backend_label,
        )
        return None
    return rank
